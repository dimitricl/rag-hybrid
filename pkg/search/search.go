package search

import (
	"context"
	"fmt"
	"strings"

	"rag-hybrid/pkg/client"
	"rag-hybrid/pkg/config"
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
	client        *client.Client
	store         *storage.Store
	embedModel    string
	minScore      float32 // seuil de cohérence sémantique — depuis config.yaml (min_score)
	contextChunks int     // nb de chunks envoyés au LLM — depuis config.yaml (context_chunks)
}

func New(c *client.Client, s *storage.Store, embedModel string) *Engine {
	if embedModel == "" {
		embedModel = "nomic-embed-text:latest"
	}
	return &Engine{client: c, store: s, embedModel: embedModel, minScore: 0.30, contextChunks: 3}
}

// NewWithMinScore crée un Engine avec le seuil de cohérence issu de config.yaml.
func NewWithMinScore(c *client.Client, s *storage.Store, embedModel string, minScore float32) *Engine {
	e := New(c, s, embedModel)
	if minScore > 0 {
		e.minScore = minScore
	}
	return e
}

// NewWithConfig crée un Engine avec tous les paramètres issus de config.yaml.
func NewWithConfig(c *client.Client, s *storage.Store, cfg config.RAGConfig) *Engine {
	e := New(c, s, cfg.EmbedModel)
	if cfg.MinScore >= 0 {
		e.minScore = cfg.MinScore
	}
	if cfg.ContextChunks > 0 {
		e.contextChunks = cfg.ContextChunks
	}
	return e
}

// EXPORTÉ : Permet à main.go d'accéder à la DB
func (e *Engine) Store() *storage.Store {
	return e.store
}

func normalizeQuery(q string) string {
	q = strings.TrimSpace(q)
	// Supprime la ponctuation finale
	for len(q) > 0 && strings.ContainsRune(".,;:!?", rune(q[len(q)-1])) {
		q = q[:len(q)-1]
	}
	return q
}

func (e *Engine) Search(q string, k int) ([]storage.Chunk, error) {
	q = normalizeQuery(q)
	v, err := e.client.Embed([]string{"Represent this sentence for searching relevant passages: " + q}, e.embedModel)
	if err != nil {
		return nil, err
	}
	return e.store.SearchSmart(q, v[0], k)
}

// EXPORTÉ : Restauration de SearchWithVec pour le cache de rag-chat
func (e *Engine) SearchWithVec(q string, k int) ([]storage.Chunk, []float32, error) {
	v, err := e.client.Embed([]string{"Represent this sentence for searching relevant passages: " + q}, e.embedModel)
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
		"diviseur", "combien", "quelle valeur", "résolution", "échantillonnage",
		"quantum", "can ", "adc ",
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
		"refs", "adlar", "mux", "tension de référence",
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

// CheckCoherence vérifie que les chunks retournés sont suffisamment pertinents
// pour justifier une réponse. Utilise le score RRF normalisé du meilleur chunk.
//
// Ancienne approche : keywords hardcodés par domaine → bloquait des questions
// valides dont les termes exacts n'étaient pas dans la liste (ex: "contremesures",
// "I2C" avec des chunks non-littéraux).
//
// Nouvelle approche : si au moins un chunk dépasse minScore, on laisse passer.
// Le modèle a déjà les instructions pour répondre "non documenté" si les sources
// ne couvrent pas la question — pas besoin d'un garde-fou keyword en plus.
func CheckCoherence(q string, chunks []storage.Chunk, minScore float32) bool {
	for _, c := range chunks {
		if c.Score >= minScore {
			return true
		}
	}
	return false
}

// EXPORTÉ : Signature corrigée pour renvoyer (string, []storage.Chunk, int, float32)
func BuildContext(results []storage.Chunk, q string) (string, []storage.Chunk, int, float32) {
	var ctx strings.Builder
	included := 0

	var filteredChunks []storage.Chunk

	seenFiles := make(map[string]bool)
	seenTexts := make(map[string]bool) // Déduplique les chunks au contenu identique
	var maxRRF float32                 // Score RRF brut du meilleur chunk (utilisé par l'appelant)
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
		if r.RRFRaw > maxRRF {
			maxRRF = r.RRFRaw
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
	return ctx.String(), filteredChunks, len(seenFiles), maxRRF
}

func buildPrompt(ctxStr, q, model string, chunks []storage.Chunk) string {
	qt := ClassifyQuestion(q)
	platformHint := detectPlatform(chunks)

	platformLine := ""
	if platformHint != "" {
		platformLine = "\nCONTRAINTE DE PLATEFORME : " + platformHint
	}

	systemPrompt := `Tu es un assistant technique expert pour le BTS CIEL. Ton rôle est d'extraire et de synthétiser des informations à partir des sources fournies.

RÈGLES STRICTES :
1. RÉPONSE BASÉE SUR LES SOURCES : Ta réponse doit s'appuyer exclusivement sur les sources fournies ci-dessous.
2. SYNTHÈSE AUTORISÉE : Tu peux (et dois) synthétiser les informations si elles sont présentes dans plusieurs sources pour répondre de manière complète.
3. CITATIONS : Cite tes sources UNE SEULE FOIS à la fin de ta réponse au format "Sources : [nom_fichier1, nom_fichier2]". N'insère AUCUN [Source N] dans le corps du texte.
4. ABSENCE D'INFORMATION : Si les sources ne contiennent pas l'information demandée, réponds exactement : "Désolé, cette opération n'est pas décrite dans le cours."
5. AUCUNE CONNAISSANCE EXTERNE : Ne complète pas les manques avec tes propres connaissances.`

	var specificPrompt string

	switch qt {
	case QtRegister:
		specificPrompt = "INSTRUCTIONS REGISTRES : Liste les bits, donne leur nom exact et leur description. Ne cite PAS les sources dans le corps du texte. Si un bit est mentionné mais non décrit : 'non documenté'."
	case QtCode:
		specificPrompt = fmt.Sprintf("INSTRUCTIONS PROGRAMMATION : %s\nFournis des explications ou du code basés sur les sources. Structure clairement ton code.", platformLine)
	case QtCalculation:
		specificPrompt = "INSTRUCTIONS CALCUL : Détaille chaque étape du calcul. Utilise exclusivement les formules et valeurs présentes dans les sources. Précise les unités."
	default:
		specificPrompt = "INSTRUCTIONS : Fournis une réponse structurée et synthétique basée sur les sources."
	}

	return fmt.Sprintf(`%s

%s

SOURCES :
%s

QUESTION : %s

⚠️ INSTRUCTION FINALE : Analyse attentivement les sources. Si elles couvrent le sujet (même partiellement ou via des termes connexes), synthétise une réponse précise. Cite les sources UNE SEULE FOIS à la fin, format "Sources : [fichier1, fichier2]". Ne refuse de répondre que si les sources sont totalement muettes sur le sujet.

RÉPONSE :`, systemPrompt, specificPrompt, ctxStr, q)
}


func isSmallTalk(q string) bool {
	lower := strings.ToLower(strings.TrimSpace(q))
	triggers := []string{
		"hello", "bonjour", "salut", "hi", "hey", "coucou",
		"merci", "thanks", "thank you",
		"ca va", "comment tu vas", "comment vas-tu",
		"qui es-tu", "qui es tu", "tu es quoi",
		"tu t appelles", "ton nom",
		"au revoir", "bye", "a bientot", "ciao",
		"bien joue", "bravo", "super", "cool", "nickel",
		"aide", "help", "que sais-tu faire", "que peux-tu faire",
	}
	for _, t := range triggers {
		if strings.Contains(lower, t) {
			return true
		}
	}
	return false
}

func smallTalkPrompt(q string) string {
	return fmt.Sprintf(`Tu es un assistant pour etudiants BTS CIEL. Tu es sympa, direct et naturel.
Si on te salue, reponds chaleureusement. Si on te remercie, sois modeste.
Si on te demande ce que tu fais, explique que tu aides sur les cours BTS CIEL (electro, reseau, cyber).
Reposes courtes et naturelles, 2-3 phrases max.

Message : %s

Reponse :`, q)
}

func (e *Engine) Ask(q string) (string, error) {
	return e.AskWithModel(q, "mistral:7b-instruct")
}

func (e *Engine) AskWithModel(q, model string) (string, error) {
	res, err := e.Search(q, e.contextChunks)
	if err != nil {
		return "", err
	}
	ctxStr, _, included, _ := BuildContext(res, q)
	if included == 0 {
		return "Aucune source suffisamment pertinente trouvée. Essaie de reformuler.", nil
	}
	if !CheckCoherence(q, res, e.minScore) {
		return "Désolé, cette opération n'est pas décrite dans le cours.", nil
	}
	return e.client.Generate(buildPrompt(ctxStr, q, model, res), model)
}

func (e *Engine) AskStream(q string) (<-chan string, error) {
	return e.AskStreamWithModel(context.Background(), q, "mistral:7b-instruct")
}

func (e *Engine) AskStreamWithModel(ctx context.Context, q, model string) (<-chan string, error) {
	if isSmallTalk(q) {
		return e.client.GenerateStream(ctx, smallTalkPrompt(q), model)
	}
	res, err := e.Search(q, e.contextChunks)
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

	if !CheckCoherence(q, res, e.minScore) {
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
