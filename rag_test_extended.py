#!/usr/bin/env python3
"""
rag_test_extended.py — Suite de tests étendue pour le RAG hybride BTS CIEL
20 questions naturelles, comparaison multi-modèles, scores cosine, graphe latence.

Usage:
    python3 rag_test_extended.py
    python3 rag_test_extended.py --models mistral:7b-instruct,deepseek-coder-v2:16b
    python3 rag_test_extended.py --models mistral:7b-instruct --output mon_rapport.pdf
"""

import argparse
import json
import time
import urllib.request
import urllib.error
from datetime import datetime
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, Image
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.graphics.shapes import Drawing, Rect, String, Line
from reportlab.graphics import renderPDF

# ─── Tests ────────────────────────────────────────────────────────────────────
# Format : (catégorie, question, mots_clés_attendus, sources_attendues)
# mots_clés : au moins 1 présent → PASS partiel
# sources   : au moins 1 présente → PASS partiel
# Les deux → PASS complet

TESTS = [
    # --- Registres AVR ---
    ("Registres AVR",
     "quels sont les bits du registre ADMUX ?",
     ["REFS", "ADLAR", "MUX", "tension de référence"],
     ["ATmega328", "convertisseur", "ADC"]),

    ("Registres AVR",
     "à quoi sert le bit ADEN dans ADCSRA ?",
     ["ADEN", "enable", "activer", "convertisseur", "ADC"],
     ["ATmega328", "ADC", "convertisseur"]),

    # --- Code AVR ---
    ("Code AVR",
     "comment configurer un timer en mode PWM sur ATmega328 ?",
     ["TCCR2A", "TCCR2B", "OCR2A", "PWM", "prédivision"],
     ["timer2pwm", "timer2output", "ATmega328"]),

    ("Code AVR",
     "comment lire une valeur analogique sur ATmega328 ?",
     ["ADC", "analogique", "conversion", "ADMUX"],
     ["ADC", "convertisseur", "ATmega328"]),

    ("Code AVR",
     "comment configurer une interruption externe sur ATmega328 ?",
     ["INT0", "INT1", "EICRA", "sei", "ISR", "interruption"],
     ["ATmega328", "GPIO", "interruption"]),

    # --- Réseau / Sécurité ---
    ("Réseau",
     "comment configurer un switch Cisco en mode trunk ?",
     ["switchport", "trunk", "VLAN", "configure terminal"],
     ["packet-tracer", "switch", "cisco"]),

    ("Réseau",
     "qu'est-ce que le protocole 802.1X ?",
     ["authentification", "port", "RADIUS", "EAP"],
     ["802.1X", "protocole", "réseau"]),

    ("Sécurité",
     "comment fonctionne une attaque Slowloris ?",
     ["connexion", "HTTP", "DoS", "serveur", "lente"],
     ["Slowloris", "DDoS", "Défenses"]),

    ("Sécurité",
     "quelles sont les contremesures contre une attaque DDoS ?",
     ["fail2ban", "iptables", "limite", "filtrage", "blocage"],
     ["DDoS", "Défenses", "attaque"]),

    # --- Concepts électronique ---
    ("Concepts",
     "comment fonctionne le protocole I2C ?",
     ["SDA", "SCL", "maître", "esclave", "adresse"],
     ["I2C", "série", "protocole", "liaison"]),

    ("Concepts",
     "quelle est la différence entre interruption et polling ?",
     ["interruption", "polling", "attente", "CPU"],
     ["GPIO", "ATmega328", "timer", "interruption"]),

    ("Concepts",
     "comment fonctionne un convertisseur ADC ?",
     ["analogique", "numérique", "résolution", "tension", "échantillonnage"],
     ["ADC", "convertisseur", "ATmega328"]),

    # --- Questions naturelles (ce qu'un étudiant poserait vraiment) ---
    ("Question naturelle",
     "c'est quoi la différence entre Arduino et ATmega ?",
     ["Arduino", "ATmega", "microcontrôleur", "bibliothèque", "registre"],
     ["ATmega328", "Arduino", "GPIO"]),

    ("Question naturelle",
     "comment déboguer un programme sur ATmega ?",
     ["debug", "UART", "série", "oscilloscope", "breakpoint", "débogage"],
     ["ATmega328", "UART", "série"]),

    ("Question naturelle",
     "pourquoi mon PWM ne fonctionne pas ?",
     ["TCCR", "OCR", "DDR", "broche", "fréquence", "PWM"],
     ["timer", "PWM", "ATmega328"]),

    # --- Calculs ---
    ("Calcul",
     "comment calculer la fréquence d'un timer ATmega avec prescaler 64 ?",
     ["fréquence", "prescaler", "formule", "16MHz", "Hz"],
     ["timer", "ATmega328", "fréquence"]),

    ("Calcul",
     "quelle est la résolution d'un ADC 10 bits avec Vref 5V ?",
     ["résolution", "10 bits", "5V", "mV", "1024"],
     ["ADC", "convertisseur", "ATmega328"]),

    # --- Pièges / Hors corpus ---
    ("Piège",
     "comment configurer l'ADC pour émettre une trame infrarouge ?",
     ["général", "connaissances", "assistant", "infrarouge", "ADC"],
     []),

    ("Piège",
     "comment utiliser le protocole 802.1X pour faire du PWM ?",
     ["général", "connaissances", "assistant", "802.1X", "PWM"],
     []),

    ("Piège",
     "comment programmer une IA avec TensorFlow sur ATmega ?",
     ["général", "connaissances", "assistant", "TensorFlow", "IA"],
     []),
]

# ─── Couleurs ─────────────────────────────────────────────────────────────────
STATUS_COLORS = {
    "PASS":    colors.HexColor("#2ea043"),
    "PARTIAL": colors.HexColor("#d29922"),
    "FAIL":    colors.HexColor("#f85149"),
    "ERROR":   colors.HexColor("#8b949e"),
}
BG_DARK   = colors.HexColor("#0d1117")
BG_HEADER = colors.HexColor("#161b22")
BG_ALT    = colors.HexColor("#0d1117")
TEXT_MAIN = colors.HexColor("#e6edf3")
TEXT_DIM  = colors.HexColor("#8b949e")
TEXT_BLUE = colors.HexColor("#58a6ff")
BORDER    = colors.HexColor("#30363d")

# ─── Client RAG ───────────────────────────────────────────────────────────────

def ask_rag(question: str, url: str, model: str, timeout: int = 90) -> dict:
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
        "source_scores": {},   # filename → score cosine
        "search_ms": 0,
        "gen_s": 0.0,
        "total_s": 0.0,
        "fallback": False,
        "fallback_reason": "",
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

                etype = ev.get("type", "")
                if etype == "token":
                    result["answer"] += ev.get("text", "")
                elif etype == "fallback":
                    result["fallback"] = True
                    result["fallback_reason"] = ev.get("reason", "")
                elif etype == "sources":
                    srcs = ev.get("sources", [])
                    result["sources"] = [s.get("filename", "") for s in srcs]
                    result["source_scores"] = {
                        s.get("filename", ""): round(float(s.get("score", 0)), 3)
                        for s in srcs
                    }
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


# ─── Évaluation ───────────────────────────────────────────────────────────────

def evaluate(result: dict, keywords: list, expected_sources: list) -> dict:
    answer_lower = result["answer"].lower()
    sources_lower = [s.lower() for s in result["sources"]]

    keyword_hit = any(kw.lower() in answer_lower for kw in keywords) if keywords else True
    keyword_found = [kw for kw in keywords if kw.lower() in answer_lower]

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


# ─── Graphe latence par catégorie (ReportLab natif) ──────────────────────────

def build_latency_chart(all_results_per_model: dict, width=16*cm, height=6*cm) -> Drawing:
    """Graphe à barres horizontales : latence moyenne par catégorie par modèle."""
    cats = list(dict.fromkeys(r["category"] for runs in all_results_per_model.values() for r in runs))
    models = list(all_results_per_model.keys())

    # Calcul latences moyennes
    lat = {m: {} for m in models}
    for m, runs in all_results_per_model.items():
        for cat in cats:
            times = [r["result"]["total_s"] for r in runs if r["category"] == cat]
            lat[m][cat] = round(sum(times)/len(times), 1) if times else 0

    max_lat = max((v for m in lat.values() for v in m.values()), default=1)

    model_colors = [
        colors.HexColor("#58a6ff"),
        colors.HexColor("#2ea043"),
        colors.HexColor("#d29922"),
        colors.HexColor("#f85149"),
    ]

    padding_left = 4.5*cm
    padding_right = 0.5*cm
    padding_top = 0.8*cm
    padding_bottom = 1.2*cm
    bar_area_w = float(width) - float(padding_left) - float(padding_right)
    bar_area_h = float(height) - float(padding_top) - float(padding_bottom)

    n_cats = len(cats)
    group_h = bar_area_h / max(n_cats, 1)
    bar_h = min(group_h / (len(models) + 1), 12)

    d = Drawing(float(width), float(height))

    # Fond
    d.add(Rect(0, 0, float(width), float(height),
               fillColor=BG_DARK, strokeColor=None))

    # Axe vertical (ligne)
    d.add(Line(
        float(padding_left), float(padding_bottom),
        float(padding_left), float(padding_bottom) + bar_area_h,
        strokeColor=BORDER, strokeWidth=0.5
    ))

    # Graduations X
    for pct in [0, 25, 50, 75, 100]:
        x = float(padding_left) + bar_area_w * pct / 100
        val = max_lat * pct / 100
        d.add(Line(x, float(padding_bottom), x, float(padding_bottom) + bar_area_h,
                   strokeColor=BORDER, strokeWidth=0.3))
        d.add(String(x, float(padding_bottom) - 10, f"{val:.0f}s",
                     fontSize=6, fillColor=TEXT_DIM,
                     textAnchor="middle"))

    # Barres
    for ci, cat in enumerate(reversed(cats)):
        y_group = float(padding_bottom) + ci * group_h + group_h * 0.1

        # Label catégorie
        d.add(String(float(padding_left) - 4, y_group + bar_h * len(models) / 2,
                     cat[:20],
                     fontSize=6.5, fillColor=TEXT_MAIN,
                     textAnchor="end"))

        for mi, model in enumerate(models):
            val = lat[model].get(cat, 0)
            bar_w = bar_area_w * val / max_lat if max_lat > 0 else 0
            y_bar = y_group + mi * (bar_h + 1)
            mc = model_colors[mi % len(model_colors)]

            d.add(Rect(float(padding_left), y_bar, bar_w, bar_h,
                       fillColor=mc, strokeColor=None))

            if val > 0:
                d.add(String(float(padding_left) + bar_w + 2, y_bar + bar_h * 0.3,
                             f"{val:.1f}s",
                             fontSize=5.5, fillColor=mc))

    # Légende modèles
    lx = float(padding_left)
    for mi, model in enumerate(models):
        mc = model_colors[mi % len(model_colors)]
        d.add(Rect(lx, 2, 8, 6, fillColor=mc, strokeColor=None))
        short = model.split(":")[0][:20]
        d.add(String(lx + 10, 3, short, fontSize=6, fillColor=TEXT_MAIN))
        lx += len(short) * 4.5 + 18

    return d


# ─── Tableau comparaison multi-modèles ────────────────────────────────────────

def build_comparison_table(all_results_per_model: dict, styles_obj) -> Table:
    """Tableau côte à côte : pour chaque test, statut par modèle."""
    models = list(all_results_per_model.keys())
    first_model_results = list(all_results_per_model.values())[0]

    header = ["#", "Question"] + [m.split(":")[0][:12] for m in models]
    col_widths = [0.6*cm, 7.5*cm] + [2.2*cm] * len(models)

    data = [header]
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), BG_HEADER),
        ("TEXTCOLOR", (0, 0), (-1, 0), TEXT_DIM),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ALIGN", (1, 0), (1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.3, BORDER),
        ("ROWHEIGHT", (0, 0), (-1, -1), 14),
    ]

    for i, ref_r in enumerate(first_model_results):
        q = ref_r["result"]["question"]
        short_q = q[:55] + "…" if len(q) > 55 else q
        row = [str(i+1), short_q]

        for mi, model in enumerate(models):
            runs = all_results_per_model[model]
            if i < len(runs):
                st = runs[i]["eval"]["status"]
                row.append(st)
                sc = STATUS_COLORS.get(st, TEXT_DIM)
                col = mi + 2
                style_cmds.append(("TEXTCOLOR", (col, i+1), (col, i+1), sc))
                style_cmds.append(("FONTNAME", (col, i+1), (col, i+1), "Helvetica-Bold"))
            else:
                row.append("—")

        # Alternance de fond
        bg = colors.HexColor("#111820") if i % 2 == 0 else BG_DARK
        style_cmds.append(("BACKGROUND", (0, i+1), (-1, i+1), bg))
        style_cmds.append(("TEXTCOLOR", (0, i+1), (1, i+1), TEXT_MAIN))

        data.append(row)

    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle(style_cmds))
    return t


# ─── Tableau scores cosine ────────────────────────────────────────────────────

def build_cosine_table(all_results_per_model: dict) -> Table:
    """Score cosine moyen par catégorie par modèle."""
    models = list(all_results_per_model.keys())
    cats = list(dict.fromkeys(r["category"] for runs in all_results_per_model.values() for r in runs))

    header = ["Catégorie"] + [m.split(":")[0][:14] for m in models]
    col_widths = [4*cm] + [3*cm] * len(models)

    data = [header]
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), BG_HEADER),
        ("TEXTCOLOR", (0, 0), (-1, 0), TEXT_DIM),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.3, BORDER),
        ("ROWHEIGHT", (0, 0), (-1, -1), 16),
    ]

    for ci, cat in enumerate(cats):
        row = [cat]
        for model in models:
            runs = all_results_per_model[model]
            scores = []
            for r in runs:
                if r["category"] == cat:
                    scores.extend(r["result"]["source_scores"].values())
            avg = round(sum(scores)/len(scores), 3) if scores else 0
            row.append(f"{avg:.3f}")

            # Couleur selon score
            col = models.index(model) + 1
            if avg >= 0.7:
                sc = STATUS_COLORS["PASS"]
            elif avg >= 0.4:
                sc = STATUS_COLORS["PARTIAL"]
            elif avg > 0:
                sc = STATUS_COLORS["FAIL"]
            else:
                sc = TEXT_DIM
            style_cmds.append(("TEXTCOLOR", (col, ci+1), (col, ci+1), sc))
            style_cmds.append(("FONTNAME", (col, ci+1), (col, ci+1), "Helvetica-Bold"))

        bg = colors.HexColor("#111820") if ci % 2 == 0 else BG_DARK
        style_cmds.append(("BACKGROUND", (0, ci+1), (-1, ci+1), bg))
        style_cmds.append(("TEXTCOLOR", (0, ci+1), (0, ci+1), TEXT_MAIN))
        data.append(row)

    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle(style_cmds))
    return t


# ─── Génération PDF ───────────────────────────────────────────────────────────

def build_pdf(all_results_per_model: dict, output_path: str, url: str):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=1.8*cm, rightMargin=1.8*cm,
        topMargin=1.8*cm, bottomMargin=1.8*cm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle("T", parent=styles["Title"],
        fontSize=18, textColor=TEXT_BLUE, spaceAfter=4)
    sub_style = ParagraphStyle("S", parent=styles["Normal"],
        fontSize=8, textColor=TEXT_DIM, spaceAfter=12)
    h2_style = ParagraphStyle("H2", parent=styles["Heading2"],
        fontSize=11, textColor=TEXT_MAIN, spaceBefore=14, spaceAfter=6)
    answer_style = ParagraphStyle("A", parent=styles["Normal"],
        fontSize=7, textColor=colors.HexColor("#c9d1d9"), leading=10, spaceAfter=3)
    meta_style = ParagraphStyle("M", parent=styles["Normal"],
        fontSize=6.5, textColor=TEXT_DIM, spaceAfter=6)
    question_style = ParagraphStyle("Q", parent=styles["Normal"],
        fontSize=9, textColor=colors.HexColor("#79c0ff"),
        fontName="Helvetica-Bold", spaceAfter=3)

    story = []
    models = list(all_results_per_model.keys())
    now = datetime.now().strftime("%d/%m/%Y %H:%M")

    # ── En-tête ──
    story.append(Paragraph("RAG Hybrid — Rapport de Tests Étendu", title_style))
    story.append(Paragraph(
        f"Généré le {now} &nbsp;|&nbsp; URL : {url} &nbsp;|&nbsp; "
        f"Modèles : {', '.join(m.split(':')[0] for m in models)} &nbsp;|&nbsp; "
        f"{len(TESTS)} tests",
        sub_style
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=BORDER))
    story.append(Spacer(1, 10))

    # ── Résumé global par modèle ──
    story.append(Paragraph("Résumé global", h2_style))

    summary_header = ["Modèle", "Tests", "PASS", "PARTIAL", "FAIL", "ERROR", "Taux PASS", "Temps moy.", "Fallback"]
    summary_data = [summary_header]
    for model, runs in all_results_per_model.items():
        counts = {"PASS": 0, "PARTIAL": 0, "FAIL": 0, "ERROR": 0}
        total_t = 0.0
        fallback_n = 0
        for r in runs:
            counts[r["eval"]["status"]] += 1
            total_t += r["result"].get("total_s", 0)
            if r["result"].get("fallback"):
                fallback_n += 1
        n = len(runs)
        summary_data.append([
            model.split(":")[0][:16],
            str(n),
            str(counts["PASS"]),
            str(counts["PARTIAL"]),
            str(counts["FAIL"]),
            str(counts["ERROR"]),
            f"{counts['PASS']/n*100:.0f}%" if n else "—",
            f"{total_t/n:.1f}s" if n else "—",
            str(fallback_n),
        ])

    col_w = [3.5*cm, 1.2*cm, 1.2*cm, 1.5*cm, 1.2*cm, 1.2*cm, 1.8*cm, 1.8*cm, 1.5*cm]
    st = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BG_HEADER),
        ("TEXTCOLOR", (0, 0), (-1, 0), TEXT_DIM),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BACKGROUND", (0, 1), (-1, -1), BG_DARK),
        ("TEXTCOLOR", (0, 1), (-1, -1), TEXT_MAIN),
        ("TEXTCOLOR", (2, 1), (2, -1), STATUS_COLORS["PASS"]),
        ("TEXTCOLOR", (3, 1), (3, -1), STATUS_COLORS["PARTIAL"]),
        ("TEXTCOLOR", (4, 1), (4, -1), STATUS_COLORS["FAIL"]),
        ("FONTNAME", (2, 1), (4, -1), "Helvetica-Bold"),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWHEIGHT", (0, 0), (-1, -1), 16),
        ("GRID", (0, 0), (-1, -1), 0.3, BORDER),
    ])
    story.append(Table(summary_data, colWidths=col_w, style=st))
    story.append(Spacer(1, 16))

    # ── Graphe latence ──
    story.append(Paragraph("Latence moyenne par catégorie", h2_style))
    chart = build_latency_chart(all_results_per_model, width=16.5*cm, height=7*cm)
    story.append(chart)
    story.append(Spacer(1, 16))

    # ── Scores cosine ──
    story.append(Paragraph("Score de confiance moyen des sources (cosine)", h2_style))
    story.append(Paragraph(
        "Score normalisé 0→1. ≥0.7 = très pertinent (vert), 0.4–0.7 = pertinent (orange), <0.4 = faible (rouge), 0 = fallback/aucune source.",
        ParagraphStyle("note", parent=styles["Normal"], fontSize=7, textColor=TEXT_DIM, spaceAfter=6)
    ))
    story.append(build_cosine_table(all_results_per_model))
    story.append(Spacer(1, 16))

    # ── Comparaison multi-modèles ──
    story.append(PageBreak())
    story.append(Paragraph("Comparaison multi-modèles — statut par test", h2_style))
    story.append(build_comparison_table(all_results_per_model, styles))
    story.append(Spacer(1, 20))

    # ── Détail par modèle ──
    for model, runs in all_results_per_model.items():
        story.append(PageBreak())
        story.append(Paragraph(f"Détail — {model}", h2_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER))
        story.append(Spacer(1, 6))

        current_cat = None
        for r in runs:
            cat = r["category"]
            result = r["result"]
            ev = r["eval"]
            status = ev["status"]

            if cat != current_cat:
                current_cat = cat
                story.append(Paragraph(f"■ {cat}", ParagraphStyle(
                    "CH", parent=styles["Normal"],
                    fontSize=10, textColor=TEXT_BLUE,
                    fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=3,
                )))

            badge = STATUS_COLORS[status]
            fallback_tag = ' <font color="#d29922">[HORS CORPUS]</font>' if result.get("fallback") else ""
            story.append(Paragraph(
                f'<font color="{badge.hexval()}">[{status}]</font>{fallback_tag} '
                f'<font color="#e6edf3">{result["question"]}</font>',
                question_style
            ))

            if result.get("error"):
                story.append(Paragraph(f"Erreur : {result['error']}", ParagraphStyle(
                    "E", parent=styles["Normal"], fontSize=7,
                    textColor=STATUS_COLORS["ERROR"], spaceAfter=4)))
            else:
                answer = result["answer"].strip()[:350]
                if len(result["answer"].strip()) > 350:
                    answer += "…"
                answer = answer.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                story.append(Paragraph(answer, answer_style))

                # Sources + scores cosine
                src_parts = []
                for fn, sc in result["source_scores"].items():
                    col = "#2ea043" if sc >= 0.7 else "#d29922" if sc >= 0.4 else "#f85149"
                    src_parts.append(f'<font color="{col}">{fn[:30]} ({sc:.2f})</font>')
                sources_str = " | ".join(src_parts) if src_parts else "aucune (fallback)"

                kw_str = ", ".join(ev["keyword_found"]) if ev["keyword_found"] else "aucun"
                story.append(Paragraph(
                    f"Sources : {sources_str}<br/>"
                    f"Mots-clés : {kw_str} &nbsp;|&nbsp; "
                    f"Recherche : {result.get('search_ms', 0)}ms &nbsp;|&nbsp; "
                    f"Total : {result.get('total_s', 0):.1f}s",
                    meta_style
                ))

            story.append(HRFlowable(width="100%", thickness=0.2,
                                    color=colors.HexColor("#21262d"), spaceAfter=4))

    doc.build(story)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Suite de tests RAG étendue → PDF")
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--models", default="mistral:7b-instruct",
                        help="Modèles séparés par virgule (ex: mistral:7b-instruct,deepseek-coder-v2:16b)")
    parser.add_argument("--output", default="rag_rapport_extended.pdf")
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]

    print(f"\n{'='*65}")
    print(f"  RAG Extended Test Suite — {len(TESTS)} tests × {len(models)} modèle(s)")
    print(f"  URL     : {args.url}")
    print(f"  Modèles : {', '.join(models)}")
    print(f"{'='*65}\n")

    all_results_per_model = {}

    for model in models:
        print(f"\n── Modèle : {model} ──")
        runs = []
        for i, (cat, question, keywords, expected_sources) in enumerate(TESTS):
            print(f"  [{i+1:02d}/{len(TESTS)}] {cat} — {question[:50]}…")
            result = ask_rag(question, args.url, model, args.timeout)
            ev = evaluate(result, keywords, expected_sources)
            sym = {"PASS": "✅", "PARTIAL": "⚠️ ", "FAIL": "❌", "ERROR": "💥"}.get(ev["status"], "?")
            fb = " [FALLBACK]" if result.get("fallback") else ""
            print(f"         {sym} {ev['status']}{fb} — {result.get('total_s', 0):.1f}s")
            runs.append({"category": cat, "result": result, "eval": ev})
        all_results_per_model[model] = runs

        counts = {"PASS": 0, "PARTIAL": 0, "FAIL": 0, "ERROR": 0}
        for r in runs:
            counts[r["eval"]["status"]] += 1
        print(f"\n  → PASS:{counts['PASS']} PARTIAL:{counts['PARTIAL']} FAIL:{counts['FAIL']} | {counts['PASS']/len(TESTS)*100:.0f}%")

    print(f"\n{'='*65}")
    print(f"Génération PDF : {args.output}")
    build_pdf(all_results_per_model, args.output, args.url)
    print(f"Rapport sauvegardé : {args.output}")


if __name__ == "__main__":
    main()
