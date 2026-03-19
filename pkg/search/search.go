package search

import (
	"context"
	"fmt"
	"strings"

	"rag-hybrid/pkg/client"
	"rag-hybrid/pkg/storage"
)

const minDisplayScore = -999.0

// EXPORTÉ : Utilisé par main.go pour l'affichage
type QuestionType int

const (
	QtRegister QuestionType = iota
	QtCode
	QtConcept
	QtCalculation
	QtGeneral
)

type Engine struct {
	client     *client.Client
	store      *storage.Store
	embedModel string
}

func New(c *client.Client, s *storage.Store, embedModel string) *Engine {
	if embedModel == "" {
		embedModel = "nomic-embed-text:latest"
	}
	return &Engine{client: c, store: s, embedModel: embedModel}
}

// EXPORTÉ : Permet à main.go d'accéder à la DB
func (e *Engine) Store() *storage.Store {
	return e.store
}

func (e *Engine) Search(q string, k int) ([]storage.Chunk, error) {
	v, err := e.client.Embed([]string{"query: " + q}, e.embedModel)
	if err != nil {
		return nil, err
	}
	return e.store.SearchSmart(q, v[0], k)
}

// EXPORTÉ : Restauration de SearchWithVec pour le cache de rag-chat
func (e *Engine) SearchWithVec(q string, k int) ([]storage.Chunk, []float32, error) {
	v, err := e.client.Embed([]string{"query: " + q}, e.embedModel)
	if err != nil {
		return nil, nil, err
	}
	res, err := e.store.SearchSmart(q, v[0], k)
	return res, v[0], err
}

// EXPORTÉ : Restauration de IsLargeModel pour rag-web
func IsLargeModel(model string) bool {
	largeModels := []string{"deepseek", "16b", "70b", "34b", "llama3"}
	lower := strings.ToLower(model)
	for _, m := range largeModels {
		if strings.Contains(lower, m) {
			return true
		}
	}
	return false
}

// EXPORTÉ : ClassifyQuestion en majuscule pour rag-web
//
// FIX ordre des checks : QtRegister AVANT QtCode
// Avant ce fix, "comment lire l'ADCSRA" matchait "lire" → QtCode
// au lieu de "ADCSRA" → QtRegister, car codeKW était évalué en premier.
// Règle : plus le type est précis/technique, plus il est évalué tôt.
func ClassifyQuestion(q string) QuestionType {
	lower := strings.ToLower(q)

	// 1. Registres — le plus spécifique, priorité absolue
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

	// 2. Calcul — avant code pour éviter "calculer une fréquence en C"
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

	// 3. Concept — avant code pour "expliquer comment fonctionne le SPI"
	conceptKW := []string{
		"comment fonctionne", "qu'est-ce", "expliquer", "définir",
		"principe", "différence entre", "pourquoi", "quand utiliser",
	}
	for _, kw := range conceptKW {
		if strings.Contains(lower, kw) {
			return QtConcept
		}
	}

	// 4. Code — en dernier parmi les types spécialisés
	// "utiliser" retiré : trop générique, causait des faux positifs sur des questions registre/concept
	codeKW := []string{
		"code", "programme", "fonction", "arduino", "c++", "python",
		"loop", "setup", "void", "int ", "return", "for ", "while",
		"exemple", "syntaxe", "implémenter", "écrire", "gpio", "configurer",
		"piloter", "lire",
	}
	for _, kw := range codeKW {
		if strings.Contains(lower, kw) {
			return QtCode
		}
	}

	return QtGeneral
}

func detectPlatform(chunks []storage.Chunk) string {
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
		"atmega":  "ATmega328/AVR — utilise UNIQUEMENT les registres AVR.",
		"stm32":   "STM32/HAL — utilise UNIQUEMENT l'API HAL STM32.",
		"esp32":   "ESP32 — utilise UNIQUEMENT l'API ESP-IDF ou Arduino-ESP32.",
		"arduino": "Arduino — utilise UNIQUEMENT les fonctions Arduino.",
	}

	return labels[best]
}

func CheckCoherence(q string, chunks []storage.Chunk) bool {
	conceptGroups := [][]string{
		{"adc", "adcsra", "admux", "analogique", "numérique", "convertisseur"},
		{"infrarouge", "infrared", "ir ", "rc5", "nec", "télécommande"},
		{"uart", "série", "serial", "usart", "rx", "tx", "baud"},
		{"timer", "pwm", "tccr", "ocr", "compteur", "rapport cyclique"},
		{"gpio", "ddrb", "portb", "pinb", "entrée", "sortie"},
		{"spi", "i2c", "twi", "scl", "sda", "mosi", "miso"},
		{"réseau", "ethernet", "tcp", "udp", "ip", "wifi", "zigbee", "lora", "802.1x", "vlan", "switch"},
		{"ddos", "attaque", "slowloris", "exploit", "vulnérabilité", "fail2ban", "iptables"},
	}

	qLower := strings.ToLower(q)
	var foundGroups []int

	for i, group := range conceptGroups {
		for _, word := range group {
			if strings.Contains(qLower, word) {
				foundGroups = append(foundGroups, i)
				break
			}
		}
	}

	// FIX : logique inversée
	// 0 groupe connu = question hors-domaine total → bloquer
	// 1 groupe connu = question mono-domaine → laisser passer
	// 2+ groupes connus = vérifier que les chunks couvrent tous les domaines
	if len(foundGroups) == 0 {
		return false
	}
	if len(foundGroups) == 1 {
		return true
	}

	for _, chunk := range chunks {
		chunkLower := strings.ToLower(chunk.Text)
		matchCount := 0
		for _, groupIdx := range foundGroups {
			groupFound := false
			for _, word := range conceptGroups[groupIdx] {
				if strings.Contains(chunkLower, word) {
					groupFound = true
					break
				}
			}
			if groupFound {
				matchCount++
			}
		}
		if matchCount == len(foundGroups) {
			return true
		}
	}
	return false
}

// EXPORTÉ : Signature corrigée pour renvoyer (string, []storage.Chunk, int, float32)
func BuildContext(results []storage.Chunk, q string) (string, []storage.Chunk, int, float32) {
	var ctx strings.Builder
	included := 0
	var maxCosine float32

	var filteredChunks []storage.Chunk

	seenFiles := make(map[string]bool)
	seenTexts := make(map[string]bool) // Déduplique les chunks au contenu identique
	sourceNum := 1
	for _, r := range results {
		if r.Score < minDisplayScore {
			continue
		}
		// Ignore les chunks au contenu quasi-identique (premiers 100 chars)
		textKey := r.Text
		if len(textKey) > 100 {
			textKey = textKey[:100]
		}
		if seenTexts[textKey] {
			continue
		}
		seenTexts[textKey] = true
		if r.RRFRaw > maxCosine {
			maxCosine = r.RRFRaw
		}
		ctx.WriteString(fmt.Sprintf(
			"<source id=\"%d\" filename=\"%s\">\n%s\n</source>\n\n",
			sourceNum, r.Filename, r.Text,
		))

		seenFiles[r.Filename] = true
		filteredChunks = append(filteredChunks, r)

		sourceNum++
		included++
	}
	return ctx.String(), filteredChunks, len(seenFiles), maxCosine
}

func buildPrompt(ctxStr, q, model string, chunks []storage.Chunk) string {
	qt := ClassifyQuestion(q)
	platformHint := detectPlatform(chunks)

	platformLine := ""
	if platformHint != "" {
		platformLine = "\nCONTRAINTE DE PLATEFORME : " + platformHint
	}

	systemPrompt := `Tu es un robot d'extraction strict pour BTS CIEL. Tu n'es PAS un professeur.
RÈGLE ABSOLUE 1 : Tu lis les sources fournies.
RÈGLE ABSOLUE 2 : Si la question associe des concepts qui ne sont pas explicitement liés dans les sources pour accomplir la tâche, tu DOIS répondre EXACTEMENT et UNIQUEMENT : "Désolé, cette opération n'est pas décrite dans le cours."
RÈGLE ABSOLUE 3 : AUCUNE connaissance externe. AUCUNE déduction. AUCUNE adaptation de méthodologie.

EXEMPLE DE COMPORTEMENT ATTENDU :
Question: Comment utiliser le port USB pour mesurer la vitesse du vent ?
Ta Réponse: Désolé, cette opération n'est pas décrite dans le cours.`

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
		return "Désolé, cette opération n'est pas décrite dans le cours.", nil
	}
	return e.client.Generate(buildPrompt(ctxStr, q, model, res), model)
}

func (e *Engine) AskStream(q string) (<-chan string, error) {
	return e.AskStreamWithModel(context.Background(), q, "mistral:7b-instruct")
}

func (e *Engine) AskStreamWithModel(ctx context.Context, q, model string) (<-chan string, error) {
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
		out <- "Désolé, cette opération n'est pas décrite dans le cours."
		close(out)
		return out, nil
	}

	return e.client.GenerateStream(ctx, buildPrompt(ctxStr, q, model, res), model)
}

func (e *Engine) EmbedModel() string {
	return e.embedModel
}
