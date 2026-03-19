#!/usr/bin/env python3
"""
rag_test.py — Suite de tests automatisés pour le RAG hybride BTS CIEL
Envoie des questions à l'API RAG, analyse les réponses, génère un rapport PDF.

Usage:
    python3 rag_test.py
    python3 rag_test.py --url http://localhost:8080 --model deepseek-coder-v2:16b
"""

import argparse
import json
import time
import urllib.request
import urllib.error
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

# ─── Configuration des tests ────────────────────────────────────────────────

TESTS = [
    # (catégorie, question, mots_clés_attendus_dans_réponse, sources_attendues)
    # mots_clés : si au moins 1 présent → PASS
    # sources_attendues : si au moins 1 présente dans les sources retournées → PASS

    # Registres AVR
    ("Registres AVR", "quels sont les bits du registre ADMUX ?",
     ["REFS", "ADLAR", "MUX", "tension de référence"],
     ["Microcontrôleur 3", "ATMega328", "convertisseur"]),

    ("Registres AVR", "comment fonctionne le registre ADCSRA ?",
     ["ADEN", "ADSC", "ADATE", "ADIF", "ADIE", "ADPS"],
     ["ATMega328", "ADC_grove", "convertisseur"]),

    # Code AVR
    ("Code AVR", "comment configurer un timer en mode PWM sur ATmega328 ?",
     ["TCCR2A", "TCCR2B", "OCR2A", "DDRB", "FastPWM", "prédivision"],
     ["timer2pwm", "timer2output", "PWM"]),

    ("Code AVR", "comment lire une valeur analogique sur ATmega328 ?",
     ["analogRead", "ADMUX", "ADC", "analogique", "conversion"],
     ["ADC_grove", "ADC_simple", "ADC_rapide", "convertisseur"]),

    # Réseau / Sécurité
    ("Réseau", "qu'est-ce que le protocole 802.1X ?",
     ["authentification", "port", "RADIUS", "EAP", "commutateur"],
     ["802.1X", "protocole"]),

    ("Sécurité", "comment fonctionne une attaque Slowloris ?",
     ["connexion", "HTTP", "lente", "DoS", "serveur", "ressources"],
     ["Slowloris", "DDoS", "Défenses"]),

    ("Réseau", "comment configurer un switch Cisco en mode trunk ?",
     ["switchport", "trunk", "VLAN", "configure terminal"],
     ["packet-tracer", "switch"]),

    # Concepts
    ("Concepts", "quelle est la différence entre interruption et polling ?",
     ["interruption", "polling", "attente", "ISR", "CPU"],
     ["ButtonIHM", "GPIO", "ATMega328", "timer", "trace", "ino", "cpp"]),

    ("Concepts", "comment fonctionne le protocole I2C ?",
     ["SDA", "SCL", "maître", "esclave", "adresse", "horloge"],
     ["Communication série", "I2C", "série", "protocole", "liaison"]),

    # Questions pièges — le fallback répond en LLM pur (hors corpus)
    ("Piège", "comment configurer l'ADC pour émettre une trame infrarouge ?",
     ["général", "connaissances", "assistant", "infrarouge", "ADC"],
     []),  # aucune source attendue

    ("Piège", "comment utiliser le protocole 802.1X pour faire du PWM ?",
     ["général", "connaissances", "assistant", "802.1X", "PWM", "protocole"],
     []),

    ("Piège", "comment configurer l'UART pour envoyer des trames Ethernet ?",
     ["général", "connaissances", "assistant", "UART", "Ethernet"],
     []),
]

# ─── Client RAG ─────────────────────────────────────────────────────────────

def ask_rag(question: str, url: str, model: str, timeout: int = 60) -> dict:
    """
    Envoie une question au rag-web et retourne la réponse complète.
    Gère le streaming SSE de rag-web.
    """
    payload = json.dumps({"q": question, "model": model}).encode()
    req = urllib.request.Request(
        f"{url}/api/chat-stream",
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )

    result = {
        "question": question,
        "answer": "",
        "sources": [],
        "search_ms": 0,
        "gen_s": 0.0,
        "total_s": 0.0,
        "error": None,
    }

    try:
        t_start = time.time()
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    ev = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # Format SSE rag-web :
                # token  : {"type":"token","text":"..."}
                # sources: {"type":"sources","sources":[...],"metrics":{...}}
                etype = ev.get("type", "")
                if etype == "token":
                    result["answer"] += ev.get("text", "")
                elif etype == "sources":
                    result["sources"] = [
                        s.get("filename", "") for s in ev.get("sources", [])
                    ]
                    m = ev.get("metrics", {})
                    try:
                        st = str(m.get("search_time", "0ms")).replace("ms", "")
                        result["search_ms"] = int(float(st)) if st else 0
                        result["gen_s"] = float(str(m.get("gen_time", "0s")).replace("s", ""))
                        result["total_s"] = float(str(m.get("total_time", "0s")).replace("s", ""))
                    except (ValueError, AttributeError):
                        pass

        result["total_s"] = round(time.time() - t_start, 1)

    except urllib.error.URLError as e:
        result["error"] = f"Connexion impossible : {e}"
    except Exception as e:
        result["error"] = str(e)

    return result


# ─── Évaluation ─────────────────────────────────────────────────────────────

def evaluate(result: dict, keywords: list, expected_sources: list) -> dict:
    """Évalue un résultat de test."""
    answer_lower = result["answer"].lower()
    sources_lower = [s.lower() for s in result["sources"]]

    # Vérifie les mots-clés
    keyword_hit = any(kw.lower() in answer_lower for kw in keywords) if keywords else True
    keyword_found = [kw for kw in keywords if kw.lower() in answer_lower]

    # Vérifie les sources (si attendues)
    source_hit = True
    source_found = []
    if expected_sources:
        source_hit = any(
            any(exp.lower() in src for src in sources_lower)
            for exp in expected_sources
        )
        source_found = [
            exp for exp in expected_sources
            if any(exp.lower() in src for src in sources_lower)
        ]

    # Statut global
    if result["error"]:
        status = "ERROR"
    elif keyword_hit and source_hit:
        status = "PASS"
    elif keyword_hit or source_hit:
        status = "PARTIAL"
    else:
        status = "FAIL"

    return {
        "status": status,
        "keyword_hit": keyword_hit,
        "keyword_found": keyword_found,
        "source_hit": source_hit,
        "source_found": source_found,
    }


# ─── Génération PDF ──────────────────────────────────────────────────────────

STATUS_COLORS = {
    "PASS":    colors.HexColor("#2ea043"),
    "PARTIAL": colors.HexColor("#d29922"),
    "FAIL":    colors.HexColor("#f85149"),
    "ERROR":   colors.HexColor("#8b949e"),
}

STATUS_BG = {
    "PASS":    colors.HexColor("#0d1117"),
    "PARTIAL": colors.HexColor("#0d1117"),
    "FAIL":    colors.HexColor("#0d1117"),
    "ERROR":   colors.HexColor("#0d1117"),
}


def build_pdf(all_results: list, output_path: str, model: str, url: str):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )

    styles = getSampleStyleSheet()

    # Styles custom
    title_style = ParagraphStyle(
        "Title", parent=styles["Title"],
        fontSize=20, textColor=colors.HexColor("#58a6ff"),
        spaceAfter=6,
    )
    sub_style = ParagraphStyle(
        "Sub", parent=styles["Normal"],
        fontSize=9, textColor=colors.HexColor("#8b949e"),
        spaceAfter=16,
    )
    h2_style = ParagraphStyle(
        "H2", parent=styles["Heading2"],
        fontSize=13, textColor=colors.HexColor("#e6edf3"),
        spaceBefore=14, spaceAfter=6,
        borderPad=4,
    )
    question_style = ParagraphStyle(
        "Q", parent=styles["Normal"],
        fontSize=10, textColor=colors.HexColor("#79c0ff"),
        fontName="Helvetica-Bold", spaceAfter=4,
    )
    answer_style = ParagraphStyle(
        "A", parent=styles["Normal"],
        fontSize=8, textColor=colors.HexColor("#c9d1d9"),
        leading=12, spaceAfter=4,
    )
    meta_style = ParagraphStyle(
        "M", parent=styles["Normal"],
        fontSize=7, textColor=colors.HexColor("#8b949e"),
        spaceAfter=8,
    )
    error_style = ParagraphStyle(
        "E", parent=styles["Normal"],
        fontSize=8, textColor=colors.HexColor("#f85149"),
        spaceAfter=8,
    )

    story = []

    # ── En-tête ──
    story.append(Paragraph("RAG Hybrid — Rapport de Tests", title_style))
    now = datetime.now().strftime("%d/%m/%Y %H:%M")
    story.append(Paragraph(
        f"Généré le {now} &nbsp;|&nbsp; Modèle : {model} &nbsp;|&nbsp; URL : {url}",
        sub_style
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#30363d")))
    story.append(Spacer(1, 12))

    # ── Résumé global ──
    counts = {"PASS": 0, "PARTIAL": 0, "FAIL": 0, "ERROR": 0}
    total_time = 0.0
    for r in all_results:
        counts[r["eval"]["status"]] += 1
        total_time += r["result"].get("total_s", 0)

    total = len(all_results)
    pass_rate = round(counts["PASS"] / total * 100) if total else 0

    summary_data = [
        ["Tests", "PASS", "PARTIAL", "FAIL", "ERROR", "Taux PASS", "Temps moyen"],
        [
            str(total),
            str(counts["PASS"]),
            str(counts["PARTIAL"]),
            str(counts["FAIL"]),
            str(counts["ERROR"]),
            f"{pass_rate}%",
            f"{total_time/total:.1f}s" if total else "—",
        ]
    ]
    summary_table = Table(summary_data, colWidths=[2.2*cm]*7)
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#161b22")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#8b949e")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#0d1117")),
        ("TEXTCOLOR", (1, 1), (1, 1), STATUS_COLORS["PASS"]),
        ("TEXTCOLOR", (2, 1), (2, 1), STATUS_COLORS["PARTIAL"]),
        ("TEXTCOLOR", (3, 1), (3, 1), STATUS_COLORS["FAIL"]),
        ("TEXTCOLOR", (4, 1), (4, 1), STATUS_COLORS["ERROR"]),
        ("TEXTCOLOR", (0, 1), (0, 1), colors.HexColor("#e6edf3")),
        ("TEXTCOLOR", (5, 1), (6, 1), colors.HexColor("#e6edf3")),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWHEIGHT", (0, 0), (-1, -1), 18),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#30363d")),
        ("ROUNDEDCORNERS", [4, 4, 4, 4]),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 20))

    # ── Tableau récapitulatif par catégorie ──
    story.append(Paragraph("Récapitulatif par catégorie", h2_style))

    cats = {}
    for r in all_results:
        cat = r["category"]
        if cat not in cats:
            cats[cat] = {"PASS": 0, "PARTIAL": 0, "FAIL": 0, "ERROR": 0}
        cats[cat][r["eval"]["status"]] += 1

    cat_data = [["Catégorie", "PASS", "PARTIAL", "FAIL", "Total"]]
    for cat, c in cats.items():
        t = sum(c.values())
        cat_data.append([cat, str(c["PASS"]), str(c["PARTIAL"]), str(c["FAIL"]), str(t)])

    cat_table = Table(cat_data, colWidths=[5*cm, 2*cm, 2.5*cm, 2*cm, 2*cm])
    cat_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#161b22")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#8b949e")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#0d1117")),
        ("TEXTCOLOR", (0, 1), (0, -1), colors.HexColor("#e6edf3")),
        ("TEXTCOLOR", (1, 1), (1, -1), STATUS_COLORS["PASS"]),
        ("TEXTCOLOR", (2, 1), (2, -1), STATUS_COLORS["PARTIAL"]),
        ("TEXTCOLOR", (3, 1), (3, -1), STATUS_COLORS["FAIL"]),
        ("TEXTCOLOR", (4, 1), (4, -1), colors.HexColor("#8b949e")),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWHEIGHT", (0, 0), (-1, -1), 16),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#30363d")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#0d1117"), colors.HexColor("#0d1117")]),
    ]))
    story.append(cat_table)
    story.append(Spacer(1, 20))

    # ── Détail par test ──
    story.append(PageBreak())
    story.append(Paragraph("Détail des tests", h2_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#30363d")))
    story.append(Spacer(1, 8))

    current_cat = None
    for i, r in enumerate(all_results):
        cat = r["category"]
        result = r["result"]
        ev = r["eval"]
        status = ev["status"]

        if cat != current_cat:
            current_cat = cat
            story.append(Paragraph(f"■ {cat}", ParagraphStyle(
                "CatH", parent=styles["Normal"],
                fontSize=11, textColor=colors.HexColor("#58a6ff"),
                fontName="Helvetica-Bold",
                spaceBefore=10, spaceAfter=4,
            )))

        # Badge statut + question
        badge_color = STATUS_COLORS[status]
        story.append(Paragraph(
            f'<font color="{badge_color.hexval()}">[{status}]</font> '
            f'<font color="#e6edf3">{result["question"]}</font>',
            question_style
        ))

        if result.get("error"):
            story.append(Paragraph(f"Erreur : {result['error']}", error_style))
        else:
            # Réponse tronquée
            answer = result["answer"].strip()
            if len(answer) > 400:
                answer = answer[:400] + "…"
            # Nettoie les caractères problématiques pour ReportLab
            answer = answer.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            story.append(Paragraph(answer, answer_style))

            # Sources + métriques
            sources_str = ", ".join(result["sources"]) if result["sources"] else "aucune"
            kw_str = ", ".join(ev["keyword_found"]) if ev["keyword_found"] else "aucun"
            story.append(Paragraph(
                f"Sources : {sources_str} &nbsp;|&nbsp; "
                f"Mots-clés trouvés : {kw_str} &nbsp;|&nbsp; "
                f"Temps : {result.get('total_s', 0):.1f}s",
                meta_style
            ))

        story.append(HRFlowable(
            width="100%", thickness=0.3,
            color=colors.HexColor("#21262d"),
            spaceAfter=6
        ))

    doc.build(story)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Suite de tests RAG → PDF")
    parser.add_argument("--url", default="http://localhost:8080",
                        help="URL du rag-web (défaut: http://localhost:8080)")
    parser.add_argument("--model", default="deepseek-coder-v2:16b",
                        help="Modèle LLM à utiliser")
    parser.add_argument("--output", default="rag_rapport_tests.pdf",
                        help="Nom du fichier PDF de sortie")
    parser.add_argument("--timeout", type=int, default=90,
                        help="Timeout par question en secondes")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  RAG Test Suite — {len(TESTS)} tests")
    print(f"  URL    : {args.url}")
    print(f"  Modèle : {args.model}")
    print(f"{'='*60}\n")

    all_results = []
    for i, (cat, question, keywords, expected_sources) in enumerate(TESTS):
        print(f"[{i+1:02d}/{len(TESTS)}] {cat} — {question[:55]}…")
        result = ask_rag(question, args.url, args.model, args.timeout)
        ev = evaluate(result, keywords, expected_sources)

        status_sym = {"PASS": "✅", "PARTIAL": "⚠️ ", "FAIL": "❌", "ERROR": "💥"}.get(ev["status"], "?")
        print(f"       {status_sym} {ev['status']} — {result.get('total_s', 0):.1f}s")

        all_results.append({
            "category": cat,
            "result": result,
            "eval": ev,
        })

    # Résumé console
    counts = {"PASS": 0, "PARTIAL": 0, "FAIL": 0, "ERROR": 0}
    for r in all_results:
        counts[r["eval"]["status"]] += 1

    print(f"\n{'='*60}")
    print(f"  PASS: {counts['PASS']} | PARTIAL: {counts['PARTIAL']} | "
          f"FAIL: {counts['FAIL']} | ERROR: {counts['ERROR']}")
    print(f"  Taux PASS : {counts['PASS']/len(TESTS)*100:.0f}%")
    print(f"{'='*60}\n")

    # Génère le PDF
    print(f"Génération du rapport PDF : {args.output}")
    build_pdf(all_results, args.output, args.model, args.url)
    print(f"Rapport sauvegardé : {args.output}\n")


if __name__ == "__main__":
    main()
