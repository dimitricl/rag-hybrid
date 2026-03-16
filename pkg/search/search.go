package search

import (
	"fmt"
	"strings"

	"rag-hybrid/pkg/client"
	"rag-hybrid/pkg/storage"
)

const minDisplayScore = -999.0 // Désactivé : le re-ranker cross-encoder gère le tri

type QuestionType int

const (
	QtRegister    QuestionType = iota
	QtCode
	QtConcept
	QtCalculation
	QtGeneral
)

type Engine struct {
	client *client.Client
	store  *storage.Store
}

func New(c *client.Client, s *storage.Store) *Engine {
	return &Engine{client: c, store: s}
}

func (e *Engine) Search(q string, k int) ([]storage.Chunk, error) {
	// mxbai-embed-large requiert un préfixe sur les queries (pas sur les documents)
	v, err := e.client.Embed([]string{"Represent this sentence for searching relevant passages: " + q}, "nomic-embed-text:latest")
	if err != nil {
		return nil, err
	}
	return e.store.SearchSmart(q, v[0], k)
}

// SearchWithVec retourne les chunks ET le vecteur calculé.
// Utilisé par rag-chat pour mettre le vecteur en cache et éviter
// le re-calcul réseau sur les questions répétées.
func (e *Engine) SearchWithVec(q string, k int) ([]storage.Chunk, []float32, error) {
	v, err := e.client.Embed([]string{"Represent this sentence for searching relevant passages: " + q}, "nomic-embed-text:latest")
	if err != nil {
		return nil, nil, err
	}
	chunks, err := e.store.SearchSmart(q, v[0], k)
	return chunks, v[0], err
}

// Store expose le store pour les recherches avec vecteur pré-calculé (cache).
func (e *Engine) Store() *storage.Store {
	return e.store
}

func ClassifyQuestion(q string) QuestionType {
	lower := strings.ToLower(q)

	registerKW := []string{
		"registre", "register", "adcsra", "admux", "adcsrb", "tccr", "timsk",
		"portb", "ddrd", "sreg", "spcr", "ucsr", "bit", "flag", "offset",
		"0x", "masque", "mask", "byte",
	}
	for _, kw := range registerKW {
		if strings.Contains(lower, kw) {
			return QtRegister
		}
	}

	codeKW := []string{
		"code", "programme", "fonction", "arduino", "c++", "python",
		"loop", "setup", "void", "int ", "return", "for ", "while",
		"exemple", "syntaxe", "implémenter", "écrire", "gpio", "configurer",
		"utiliser", "piloter", "lire",
	}
	for _, kw := range codeKW {
		if strings.Contains(lower, kw) {
			return QtCode
		}
	}

	calcKW := []string{
		"calcul", "calculer", "formule", "valeur", "fréquence", "tension",
		"résistance", "courant", "ohm", "volt", "hz", "mhz", "prescaler",
		"diviseur", "combien", "quelle valeur",
	}
	for _, kw := range calcKW {
		if strings.Contains(lower, kw) {
			return QtCalculation
		}
	}

	conceptKW := []string{
		"comment fonctionne", "qu'est-ce", "expliquer", "définir",
		"principe", "différence entre", "pourquoi", "quand utiliser",
	}
	for _, kw := range conceptKW {
		if strings.Contains(lower, kw) {
			return QtConcept
		}
	}

	return QtGeneral
}

// DetectPlatform analyse les chunks pour identifier la plateforme matérielle dominante.
func DetectPlatform(chunks []storage.Chunk) string {
	counts := map[string]int{"atmega": 0, "stm32": 0, "esp32": 0, "arduino": 0}
	keywords := map[string][]string{
		"atmega":  {"ATmega", "ATmega328", "avr/io", "DDRB", "DDRD", "PORTB", "PORTD", "PINB", "PIND", "TCCR", "ADCSRA", "ADMUX", "avr/interrupt"},
		"stm32":   {"STM32", "HAL_GPIO", "GPIOA", "GPIOB", "GPIOC", "RCC->", "HAL_Init", "NUCLEO", "stm32f", "HAL_Delay", "GPIO_InitTypeDef"},
		"esp32":   {"ESP32", "ESP-IDF", "esp_gpio", "gpio_set_direction", "esp_err"},
		"arduino": {"Arduino", "pinMode", "digitalWrite", "digitalRead", "analogRead", "Serial.", "delay("},
	}

	for _, chunk := range chunks {
		text := chunk.Text + chunk.Filename
		for platform, kws := range keywords {
			for _, kw := range kws {
				if strings.Contains(text, kw) {
					counts[platform]++
					break
				}
			}
		}
	}

	best := ""
	bestCount := 0
	for p, c := range counts {
		if c > bestCount {
			bestCount = c
			best = p
		}
	}

	if bestCount == 0 {
		return ""
	}

	labels := map[string]string{
		"atmega":  "ATmega328/AVR — utilise UNIQUEMENT les registres AVR (DDRx, PORTx, PINx, avr/io.h).",
		"stm32":   "STM32/HAL — utilise UNIQUEMENT l'API HAL STM32.",
		"esp32":   "ESP32 — utilise UNIQUEMENT l'API ESP-IDF ou Arduino-ESP32.",
		"arduino": "Arduino — utilise UNIQUEMENT les fonctions Arduino.",
	}

	secondCount := 0
	second := ""
	for p, c := range counts {
		if p != best && c > secondCount {
			secondCount = c
			second = p
		}
	}

	if secondCount > 0 && secondCount >= bestCount/2 && second != "" {
		return fmt.Sprintf("ATTENTION document mixte (%s ET %s détectés). Utilise PRIORITAIREMENT : %s", best, second, labels[best])
	}

	return labels[best]
}


// IsLargeModel retourne true pour les modèles capables de suivre des instructions complexes.
func IsLargeModel(model string) bool {
	largeModels := []string{"deepseek", "16b", "70b", "34b", "llama3", "gemma2", "qwen2.5:7b", "qwen2.5-coder:7b"}
	lower := strings.ToLower(model)
	for _, m := range largeModels {
		if strings.Contains(lower, m) {
			return true
		}
	}
	return false
}

// extractKeywords extrait les mots techniques d'une question (>3 chars, hors stop-words).
func extractKeywords(q string) []string {
	stop := map[string]bool{
		"comment": true, "quels": true, "quelle": true, "sont": true,
		"pour": true, "avec": true, "dans": true, "sur": true,
		"les": true, "des": true, "que": true, "qui": true,
		"le": true, "la": true, "de": true, "du": true, "en": true,
		"bits": true, "bit": true, "fonctionne": true, "utiliser": true,
		"faire": true, "vers": true, "une": true, "est": true,
	}
	var kws []string
	for _, w := range strings.Fields(strings.ToLower(q)) {
		w = strings.Trim(w, "?.,;:!()")
		if len([]rune(w)) > 3 && !stop[w] {
			kws = append(kws, w)
		}
	}
	return kws
}

// chunkRelevant vérifie qu'un chunk contient au moins 1 mot-clé de la question.
// Filtre post-rerank : évite qu'un fichier hors-sujet remonte malgré un bon score cross-encoder.
func chunkRelevant(chunk storage.Chunk, keywords []string) bool {
	if len(keywords) == 0 {
		return true
	}
	text := strings.ToLower(chunk.Text + " " + chunk.Filename)
	for _, kw := range keywords {
		if strings.Contains(text, kw) {
			return true
		}
	}
	return false
}

func BuildContext(results []storage.Chunk, q string) (string, []storage.Chunk, int, float32) {
	keywords := extractKeywords(q)
	var ctx strings.Builder
	var filtered []storage.Chunk
	included := 0
	var maxCosine float32

	var bestRRF float32
	for _, r := range results {
		if r.RRFRaw > bestRRF {
			bestRRF = r.RRFRaw
		}
	}

	seenFiles := make(map[string]bool)
	sourceNum := 1
	for _, r := range results {
		if r.Score < minDisplayScore {
			continue
		}
		// Filtre post-rerank : exclut les chunks sans aucun mot-clé de la question
		if !chunkRelevant(r, keywords) {
			continue
		}
		if r.RRFRaw > maxCosine {
			maxCosine = r.RRFRaw
		}
		ctx.WriteString(fmt.Sprintf(
			"=== SOURCE %d : %s ===\n%s\n\n",
			sourceNum, r.Filename, r.Text,
		))
		filtered = append(filtered, r)
		seenFiles[r.Filename] = true
		sourceNum++
		included++
	}
	return ctx.String(), filtered, len(seenFiles), maxCosine
}


// CheckCoherence vérifie que les sources couvrent réellement la question.
// Stratégie : détecte les paires de concepts clés dans la question.
// Si la question associe deux domaines distincts (ex: ADC + IR), vérifie
// qu'au moins un chunk les contient ENSEMBLE. Sinon → non documenté.
func CheckCoherence(q string, chunks []storage.Chunk) bool {
	if len(chunks) == 0 {
		return false
	}

	// Groupes de concepts techniques — si deux groupes distincts sont détectés
	// dans la question, ils doivent co-exister dans au moins un chunk.
	conceptGroups := [][]string{
		{"adc", "adcsra", "admux", "analogique", "numérique", "convertisseur"},
		{"infrarouge", "infrared", "ir ", "rc5", "nec", "télécommande"},
		{"uart", "série", "serial", "usart", "rx", "tx", "baud"},
		{"timer", "pwm", "tccr", "ocr", "compteur"},
		{"gpio", "ddrb", "portb", "pinb", "entrée", "sortie"},
		{"spi", "i2c", "twi", "scl", "sda", "mosi", "miso"},
		{"réseau", "ethernet", "tcp", "udp", "ip", "wifi", "zigbee", "lora"},
		{"ddos", "attaque", "slowloris", "exploit", "vulnérabilité"},
	}

	qLower := strings.ToLower(q)

	// Détecte quels groupes sont mentionnés dans la question
	var activeGroups []int
	for i, group := range conceptGroups {
		for _, kw := range group {
			if strings.Contains(qLower, kw) {
				activeGroups = append(activeGroups, i)
				break
			}
		}
	}

	// Si la question ne touche qu'un seul groupe (ou aucun) → pas de vérification croisée
	if len(activeGroups) < 2 {
		return true
	}

	// Vérifie qu'au moins un chunk contient des mots des DEUX groupes actifs
	for _, chunk := range chunks {
		text := strings.ToLower(chunk.Text)
		allPresent := true
		for _, gIdx := range activeGroups {
			found := false
			for _, kw := range conceptGroups[gIdx] {
				if strings.Contains(text, kw) {
					found = true
					break
				}
			}
			if !found {
				allPresent = false
				break
			}
		}
		if allPresent {
			return true
		}
	}

	// Aucun chunk ne couvre tous les groupes → question hors domaine
	return false
}

func buildPrompt(ctxStr, q, model string, chunks []storage.Chunk) string {
	qt := ClassifyQuestion(q)
	platformHint := DetectPlatform(chunks)

	platformLine := ""
	if platformHint != "" {
		platformLine = "\nCONTRAINTE DE PLATEFORME : " + platformHint
	}

	// LE BOUCLIER ANTI-HALLUCINATION UNIVERSEL (Appliqué à TOUTES les questions)
	systemPrompt := `Tu es un robot d'extraction strict pour BTS CIEL. Tu n'es PAS un professeur.
RÈGLE ABSOLUE 1 : Tu lis les sources fournies.
RÈGLE ABSOLUE 2 : Si la question associe des concepts qui ne sont pas explicitement liés dans les sources pour accomplir la tâche, tu DOIS répondre EXACTEMENT et UNIQUEMENT : "Désolé, cette opération n'est pas décrite dans le cours."
RÈGLE ABSOLUE 3 : AUCUNE connaissance externe. AUCUNE déduction. AUCUNE adaptation de méthodologie.`

	var specificPrompt string

	switch qt {
	case QtRegister:
		specificPrompt = "Format attendu : Analyse des registres. Pour chaque bit, donne le nom exact et la description depuis la source. Cite le fichier. Si un bit manque : 'non documenté'."
	case QtCode:
		specificPrompt = fmt.Sprintf("Format attendu : Programmation.%s\nFournis des explications ou du code basés UNIQUEMENT sur les sources. Ne comble pas les trous avec tes connaissances.", platformLine)
	case QtCalculation:
		specificPrompt = "Format attendu : Calcul étape par étape. Utilise UNIQUEMENT les formules et valeurs des sources. Vérifie les unités."
	default:
		specificPrompt = "Format attendu : Réponse textuelle structurée basée UNIQUEMENT sur les sources. Cite les fichiers."
	}

	return fmt.Sprintf(`%s

%s

SOURCES :
%s

QUESTION : %s

⚠️ INSTRUCTION FINALE CRITIQUE : Si la question demande d'associer deux choses (ex: ADC et Infrarouge) et que les sources ne disent pas EXPLICITEMENT comment les utiliser ensemble, tu as l'interdiction absolue d'expliquer quoi que ce soit. Tu DOIS écrire UNIQUEMENT : "Désolé, cette opération n'est pas décrite dans le cours."

RÉPONSE :`, systemPrompt, specificPrompt, ctxStr, q)
}

func (e *Engine) Ask(q string) (string, error) {
	return e.AskWithModel(q, "mistral:7b-instruct")
}

func (e *Engine) AskWithModel(q, model string) (string, error) {
	res, err := e.Search(q, 3)
	if err != nil {
		return "", err
	}
	ctxStr, _, included, _ := BuildContext(res, q)
	if included == 0 {
		return "Aucune source suffisamment pertinente trouvée. Essaie de reformuler.", nil
	}
	if !CheckCoherence(q, res) {
		return "Cette combinaison de concepts n'est pas documentée dans les sources indexées.", nil
	}
	return e.client.Generate(buildPrompt(ctxStr, q, model, res), model)
}

func (e *Engine) AskStream(q string) (<-chan string, error) {
	return e.AskStreamWithModel(q, "mistral:7b-instruct")
}

func (e *Engine) AskStreamWithModel(q, model string) (<-chan string, error) {
	res, err := e.Search(q, 3)
	if err != nil {
		return nil, err
	}
	ctxStr, _, included, _ := BuildContext(res, q)
	if included == 0 {
		out := make(chan string, 1)
		out <- "Aucune source suffisamment pertinente trouvée. Essaie de reformuler."
		close(out)
		return out, nil
	}
	if !CheckCoherence(q, res) {
		out := make(chan string, 1)
		out <- "Cette combinaison de concepts n'est pas documentée dans les sources indexées."
		close(out)
		return out, nil
	}
	return e.client.GenerateStream(buildPrompt(ctxStr, q, model, res), model)
}