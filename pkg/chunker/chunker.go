package chunker

import (
	"archive/zip"
	"bytes"
	"encoding/xml"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"unicode"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"
)

type Chunk struct {
	Text     string
	Filename string
}

// ignoredExts : extensions sans valeur RAG — binaires, assets, bruit de build
var ignoredExts = map[string]bool{
	// Binaires / médias
	".pyc": true, ".exe": true, ".dll": true, ".so": true,
	".zip": true, ".jpg": true, ".jpeg": true, ".png": true,
	".gif": true, ".svg": true, ".ico": true, ".mp3": true,
	".mp4": true, ".db": true, ".bin": true, ".woff": true,
	".woff2": true, ".swf": true, ".ttf": true, ".otf": true,
	// Formats Apple fermés — non parsables sans API propriétaire
	".pages": true, ".key": true, ".numbers": true,
	".plist": true, ".vpp": true, ".vux": true,
	// Xcode / projet macOS — bruit de build pur
	".pbxproj": true, ".xcscheme": true, ".xcworkspacedata": true,
	".xcsettings": true, ".xcbkptlist": true, ".xcworkspace": true,
	// Bruit web Doxygen — doc auto-générée sans valeur sémantique
	".js": true, ".css": true, ".map": true,
	// LaTeX Doxygen — mise en page pure, pas de contenu cours
	".sty": true, ".cls": true, ".bst": true, ".tex": true,
	// Config/build
	".yml": true, ".yaml": true, ".toml": true, ".lock": true,
	".log": true, ".tmp": true,
	// Packet Tracer — binaires réseau Cisco
	".pkt": true, ".pka": true,
	// Archives
	".gz": true, ".tar": true, ".rar": true, ".7z": true,
	// Autres formats sans valeur RAG
	".rtf": true, // mal parsé, bruit
	".bat": true, ".sh": true, // scripts de build — peu de valeur cours
}

var ignoredDirs = map[string]bool{
	"IRMP-master":            true,
	"IrEmetteur":             true,
	"IrEmetteurMotif":        true,
	"IrReceiver":             true,
	"Projet_no_cat":          true,
	"ProgrammesLoRa":         true,
	"secretcode":             true,
	"results_20260103_173142": true,
	"results_20260103_174118": true,
	"results_20260103_174559": true,
	"results_20260103_175414": true,
	"results_20260103_175822": true,
	"node_modules":           true,
	"__pycache__":            true,
	".git":                   true,
	"Visual pardigm projet":  true,
}

var ignoredFilenames = map[string]bool{
	"LICENSE.txt":  true,
	"LICENSE":      true,
	"CHANGELOG.md": true,
	"Makefile":     true,
	"Doxyfile":     true,
	"site.docx":    true,
	"make.bat":     true,
	"Dockerfile":   true,
	"datasheet TMP35_36_37.pdf":                       true,
	"DS18B20 Datasheet .pdf":                          true,
	"DS18B20.pdf":                                     true,
	"TSOP312 Infrared Receiver datasheet.pdf":         true,
	"SS49E Linear Hall-effect Sensor.pdf":             true,
	"3141 Ö 3144 HALL-EFFECT SWITCHES Datasheet .pdf": true,
	"DHT12 datasheet.pdf":                             true,
	"SS49e_Hall_Sensor_Datasheet.pdf":                 true,
	"TL1838 Infrared Receiver datasheet.pdf":          true,
	"datasheet LDR GL5528.pdf":                        true,
	"datasheet NTC MF52.pdf":                          true,
	"NTC MF52 datasheet.pdf":                          true,
	"5 mm Round White LED.pdf":                        true,
	"presentation_Modèle_expo.md":                     true,
	// Rapports personnels — contenu hors cours BTS
	"Claverie_Dimitri_Rapport_de_stage_2026.pdf":      true,
	// Mini-projets BTS — contenu hors sujet technique
	"TP_Sprint2.pdf":                                  true,
	"TP-Sprint1-Pilotage simple.pdf":                  true,
	"TP-Sprint3.pdf":                                  true,
	// Source parasite — GPIO générique qui noie les questions ADC/registres
	"Microcontrôleur 2-GPIO Les ports parallèles.pdf": true,
}

var extractedExts = map[string]bool{
	".pdf":  true,
	".docx": true,
	".pptx": true,
}

const minChunkRunes = 150

// detectDomain retourne un préfixe contextuel basé sur le chemin du fichier
// Ce préfixe améliore l'embedding en ancrant le chunk dans son domaine
func detectDomain(path string) string {
	lower := strings.ToLower(path)
	switch {
	case strings.Contains(lower, "atm") || strings.Contains(lower, "analog") ||
		strings.Contains(lower, "timer") || strings.Contains(lower, "adc") ||
		strings.Contains(lower, "khin") || strings.Contains(lower, "pwm") ||
		strings.Contains(lower, "datasheet"):
		return "[Domaine: Microcontrôleur ATmega328 AVR Électronique]"
	case strings.Contains(lower, "802") || strings.Contains(lower, "reseau") ||
		strings.Contains(lower, "réseau") || strings.Contains(lower, "cisco") ||
		strings.Contains(lower, "vlan") || strings.Contains(lower, "trunk") ||
		strings.Contains(lower, "gardon") || strings.Contains(lower, "mqtt"):
		return "[Domaine: Réseau Protocoles Sécurité]"
	case strings.Contains(lower, "ddos") || strings.Contains(lower, "slowloris") ||
		strings.Contains(lower, "contremesure") || strings.Contains(lower, "attaque"):
		return "[Domaine: Cybersécurité DDoS]"
	case strings.Contains(lower, "i2c") || strings.Contains(lower, "spi") ||
		strings.Contains(lower, "uart") || strings.Contains(lower, "serie") ||
		strings.Contains(lower, "infrarouge") || strings.Contains(lower, "liaison"):
		return "[Domaine: Communication Série Protocoles]"
	case strings.Contains(lower, "cpp") || strings.Contains(lower, "c_chap") ||
		strings.Contains(lower, "langage") || strings.Contains(lower, "algo"):
		return "[Domaine: Programmation C C++]"
	default:
		return ""
	}
}

func ChunkFile(path string, size, overlap int) ([]Chunk, error) {
	ext := strings.ToLower(filepath.Ext(path))
	// Normalise NFC — macOS stocke les noms de fichiers en NFD (e + combining accent)
	// ce qui casse les LIKE SQLite et la déduplication par filename
	base := norm.NFC.String(filepath.Base(path))

	if ignoredExts[ext] {
		return nil, nil
	}
	if ignoredFilenames[base] {
		return nil, nil
	}
	// Ignorer les fichiers cachés et les fichiers temporaires Word/Excel
	if strings.HasPrefix(base, ".") || strings.HasPrefix(base, "~$") {
		return nil, nil
	}

	var text string
	var err error

	switch ext {
	case ".pdf":
		text, err = extractPDF(path)
	case ".docx":
		text, err = extractDOCX(path)
	case ".pptx":
		text, err = extractPPTX(path)
	default:
		content, e := os.ReadFile(path)
		if e != nil {
			return nil, e
		}
		if isBinary(content) {
			return nil, nil
		}
		text = string(content)
	}

	if err != nil {
		fmt.Printf("⚠️  %s : %v\n", base, err)
		return nil, nil
	}

	text = strings.TrimSpace(text)
	if len(text) < 50 {
		return nil, nil
	}

	return splitSemantic(text, base, detectDomain(path), size, overlap), nil
}

// lastCompleteSentence retourne les derniers `maxRunes` caractères du texte
// en commençant à la première phrase complète (évite de couper au milieu d'une phrase).
// FIX : l'ancienne version retournait window entier si aucune ponctuation trouvée
// → overlap pouvait être aussi grand que le chunk précédent entier.
// Nouvelle version : cherche la DERNIÈRE ponctuation dans la fenêtre (pas la première)
// pour maximiser le contexte tout en démarrant sur une phrase propre.
// Si aucune ponctuation : retourne une chaîne vide (pas d'overlap plutôt qu'un overlap géant).
func lastCompleteSentence(text string, maxRunes int) string {
	runes := []rune(text)
	if len(runes) <= maxRunes {
		return text
	}
	window := []rune(text[len(text)-len(string(runes[len(runes)-maxRunes:]))-0:])
	window = runes[len(runes)-maxRunes:]

	// Cherche la DERNIÈRE ponctuation forte dans la fenêtre
	// pour démarrer l'overlap sur une phrase propre
	lastCut := -1
	for i, r := range window {
		if (r == '.' || r == '!' || r == '?') && i < len(window)-1 {
			lastCut = i
		}
	}

	if lastCut == -1 {
		// Aucune phrase complète dans la fenêtre — pas d'overlap
		// Vaut mieux pas d'overlap qu'un overlap qui duplique tout le chunk
		return ""
	}

	return strings.TrimSpace(string(window[lastCut+1:]))
}

func isSection(para string) bool {
	if strings.HasPrefix(para, "#") {
		return true
	}
	if strings.HasPrefix(para, "==") || strings.HasPrefix(para, "--") {
		return true
	}
	trimmed := strings.TrimSpace(para)
	if len(trimmed) > 0 {
		firstLine := strings.SplitN(trimmed, "\n", 2)[0]
		runes := []rune(firstLine)
		if len(runes) < 80 {
			if unicode.IsDigit(runes[0]) {
				rest := strings.TrimLeft(firstLine, "0123456789.")
				rest = strings.TrimSpace(rest)
				if len(rest) > 0 && unicode.IsUpper([]rune(rest)[0]) {
					return true
				}
			}
			upper := strings.ToUpper(firstLine)
			if firstLine == upper && len([]rune(firstLine)) > 5 && strings.ContainsAny(firstLine, " ") {
				return true
			}
		}
	}
	return false
}

func isCodeBlock(para string) bool {
	return strings.HasPrefix(para, "```") || strings.HasPrefix(para, "~~~")
}

func splitSemantic(text, filename, domainPrefix string, size, overlap int) []Chunk {
	text = strings.ReplaceAll(text, "\r\n", "\n")
	text = strings.ReplaceAll(text, "\r", "\n")

	lines := strings.Split(text, "\n")
	for i, l := range lines {
		fields := strings.Fields(l)
		lines[i] = strings.Join(fields, " ")
	}
	text = strings.Join(lines, "\n")

	paragraphs := strings.Split(text, "\n\n")

	var chunks []Chunk
	var buf strings.Builder

	flush := func() {
		s := strings.TrimSpace(buf.String())
		if utf8.RuneCountInString(s) >= minChunkRunes {
			text := s
			if domainPrefix != "" {
				text = domainPrefix + "\n" + s
			}
			chunks = append(chunks, Chunk{Text: text, Filename: filename})
		}
		buf.Reset()
	}

	addOverlap := func() {
		if len(chunks) > 0 && overlap > 0 {
			prev := lastCompleteSentence(chunks[len(chunks)-1].Text, overlap)
			if utf8.RuneCountInString(prev) >= 20 {
				buf.WriteString(prev)
				buf.WriteString("\n\n")
			}
		}
	}

	inCodeBlock := false

	for _, para := range paragraphs {
		para = strings.TrimSpace(para)
		if len(para) == 0 {
			continue
		}

		if isCodeBlock(para) {
			inCodeBlock = !inCodeBlock
		}
		if inCodeBlock || isCodeBlock(para) {
			buf.WriteString(para)
			buf.WriteString("\n\n")
			continue
		}

		paraRunes := utf8.RuneCountInString(para)
		bufRunes := utf8.RuneCountInString(buf.String())

		if isSection(para) && bufRunes >= minChunkRunes {
			flush()
		}

		if paraRunes > size {
			if bufRunes >= minChunkRunes {
				flush()
				addOverlap()
			}
			lines := strings.Split(para, "\n")
			for _, line := range lines {
				line = strings.TrimSpace(line)
				if len(line) == 0 {
					continue
				}
				lineRunes := utf8.RuneCountInString(line)
				currentRunes := utf8.RuneCountInString(buf.String())

				if currentRunes+lineRunes > size && currentRunes >= minChunkRunes {
					flush()
					addOverlap()
				}
				buf.WriteString(line)
				buf.WriteString("\n")
			}
			continue
		}

		if bufRunes+paraRunes > size && bufRunes >= minChunkRunes {
			flush()
			addOverlap()
		}

		buf.WriteString(para)
		buf.WriteString("\n\n")
	}

	flush()
	return chunks
}

func extractPDF(path string) (string, error) {
	pdftotextBin, err := exec.LookPath("pdftotext")
	if err != nil {
		for _, candidate := range []string{
			"/opt/homebrew/bin/pdftotext",
			"/usr/local/bin/pdftotext",
			"/usr/bin/pdftotext",
		} {
			if _, e := os.Stat(candidate); e == nil {
				pdftotextBin = candidate
				break
			}
		}
	}
	if pdftotextBin == "" {
		return "", fmt.Errorf("pdftotext introuvable — installe poppler: brew install poppler")
	}

	cmd := exec.Command(pdftotextBin, "-layout", "-enc", "UTF-8", path, "-")
	var out, stderr bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("pdftotext: %w — %s", err, stderr.String())
	}
	return out.String(), nil
}

func extractDOCX(path string) (string, error) {
	r, err := zip.OpenReader(path)
	if err != nil {
		return "", err
	}
	defer r.Close()

	for _, f := range r.File {
		if f.Name != "word/document.xml" {
			continue
		}
		rc, err := f.Open()
		if err != nil {
			return "", err
		}
		defer rc.Close()
		return extractXMLText(rc)
	}
	return "", fmt.Errorf("word/document.xml non trouvé")
}

func extractPPTX(path string) (string, error) {
	r, err := zip.OpenReader(path)
	if err != nil {
		return "", err
	}
	defer r.Close()

	var sb strings.Builder
	for _, f := range r.File {
		if !strings.HasPrefix(f.Name, "ppt/slides/slide") || !strings.HasSuffix(f.Name, ".xml") {
			continue
		}
		rc, err := f.Open()
		if err != nil {
			continue
		}
		text, err := extractXMLText(rc)
		rc.Close()
		if err != nil {
			continue
		}
		sb.WriteString(text)
		sb.WriteString("\n\n")
	}
	return sb.String(), nil
}

func extractXMLText(r interface{ Read([]byte) (int, error) }) (string, error) {
	var sb strings.Builder
	decoder := xml.NewDecoder(r)
	inText := false

	for {
		tok, err := decoder.Token()
		if err != nil {
			break
		}
		switch t := tok.(type) {
		case xml.StartElement:
			if t.Name.Local == "t" {
				inText = true
			}
		case xml.EndElement:
			if t.Name.Local == "t" {
				inText = false
			}
			if t.Name.Local == "p" {
				sb.WriteString("\n\n")
			}
		case xml.CharData:
			if inText {
				sb.Write(t)
				sb.WriteString(" ")
			}
		}
	}
	return sb.String(), nil
}

func isBinary(data []byte) bool {
	if len(data) == 0 {
		return false
	}
	sample := data
	if len(sample) > 512 {
		sample = sample[:512]
	}
	nonPrintable := 0
	for _, b := range sample {
		if b < 9 || (b > 13 && b < 32) {
			nonPrintable++
		}
	}
	return float64(nonPrintable)/float64(len(sample)) > 0.10
}

func ScanDir(root string) ([]string, error) {
	var files []string
	filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil || info == nil {
			return nil
		}
		base := filepath.Base(path)
		if info.IsDir() {
			if strings.HasPrefix(base, ".") || ignoredDirs[base] {
				return filepath.SkipDir
			}
			return nil
		}
		// Ignorer les fichiers cachés et les fichiers temporaires Word/Excel
		if strings.HasPrefix(base, ".") || strings.HasPrefix(base, "~$") {
			return nil
		}
		files = append(files, path)
		return nil
	})
	return files, nil
}