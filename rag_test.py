#!/usr/bin/env python3
"""
rag_test.py — Suite de tests automatisés pour le RAG hybride BTS CIEL

Tests externalisés dans tests.yaml. Supporte la comparaison multi-modèles
et génère un rapport PDF complet.

Usage:
    python3 rag_test.py
    python3 rag_test.py --models mistral:7b-instruct,deepseek-coder-v2:16b
    python3 rag_test.py --tests mes_tests.yaml --output mon_rapport.pdf
"""

import argparse
import json
import os
import time
import urllib.request
import urllib.error
from datetime import datetime

try:
    import yaml
except ImportError:
    print("⚠️  PyYAML manquant — installe avec : pip3 install pyyaml --break-system-packages")
    raise

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak,
)
from reportlab.graphics.shapes import Drawing, Rect, String, Line
from reportlab.graphics import renderPDF

# ─── Couleurs ──────────────────────────────────────────────────────────────────
STATUS_COLORS = {
    "PASS":    colors.HexColor("#2ea043"),
    "PARTIAL": colors.HexColor("#d29922"),
    "FAIL":    colors.HexColor("#f85149"),
    "ERROR":   colors.HexColor("#8b949e"),
}
BG_DARK   = colors.HexColor("#0d1117")
BG_HEADER = colors.HexColor("#161b22")
TEXT_MAIN = colors.HexColor("#e6edf3")
TEXT_DIM  = colors.HexColor("#8b949e")
TEXT_BLUE = colors.HexColor("#58a6ff")
BORDER    = colors.HexColor("#30363d")

# ─── Chargement des tests ──────────────────────────────────────────────────────

def load_tests(path: str) -> list:
    """Charge les cas de test depuis un fichier YAML."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    tests = []
    for t in raw:
        tests.append({
            "category":        t["category"],
            "type":            t.get("type", "normal"),
            "question":        t["question"],
            "keywords":        t.get("keywords", []),
            "sources":         t.get("sources", []),
            "refusal_patterns": t.get("refusal_patterns", [
                "pas decrit", "pas décrit", "non documenté", "désolé", "desole",
                "ne figure pas", "hors corpus", "not described",
            ]),
        })
    return tests

# ─── Client RAG ───────────────────────────────────────────────────────────────

def ask_rag(question: str, url: str, model: str, timeout: int = 90, num_predict: int = 0) -> dict:
    """Envoie une question au rag-web via SSE et retourne la réponse complète."""
    payload = json.dumps({"q": question, "model": model, "num_predict": num_predict}).encode()
    req = urllib.request.Request(
        f"{url}/api/chat-stream",
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    result = {
        "question": question, "answer": "", "sources": [],
        "source_scores": {}, "search_ms": 0, "gen_s": 0.0,
        "total_s": 0.0, "fallback": False, "error": None,
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
                        result["gen_s"]   = float(str(m.get("gen_time",   "0s")).replace("s", ""))
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

def evaluate(result: dict, test: dict) -> dict:
    """
    Évalue un résultat selon le type de test.

    normal : PASS si keyword_hit ET source_hit, PARTIAL si l'un seulement, FAIL sinon.
    trap   : PASS si la réponse contient un refusal_pattern (le RAG a bien refusé).
             FAIL si le RAG a répondu à une question hors corpus.
    """
    if result["error"]:
        return {"status": "ERROR", "detail": result["error"],
                "keyword_found": [], "source_found": []}

    answer_lower = result["answer"].lower()

    # ── Test piège ──
    if test["type"] == "trap":
        refused = any(p.lower() in answer_lower for p in test["refusal_patterns"])
        if refused or result.get("fallback"):
            return {"status": "PASS",
                    "detail": "Refus correct ✓",
                    "keyword_found": [], "source_found": []}
        return {"status": "FAIL",
                "detail": "Le RAG a répondu au lieu de refuser ✗",
                "keyword_found": [], "source_found": []}

    # ── Test normal ──
    sources_lower    = [s.lower() for s in result["sources"]]
    keywords         = test.get("keywords", [])
    expected_sources = test.get("sources", [])

    keyword_found = [kw for kw in keywords if kw.lower() in answer_lower]
    keyword_hit   = bool(keyword_found) if keywords else True

    source_found  = [
        exp for exp in expected_sources
        if any(exp.lower() in src for src in sources_lower)
    ]
    source_hit = bool(source_found) if expected_sources else True

    if keyword_hit and source_hit:
        status = "PASS"
    elif keyword_hit or source_hit:
        status = "PARTIAL"
    else:
        status = "FAIL"

    return {"status": status, "detail": "",
            "keyword_found": keyword_found, "source_found": source_found}

# ─── Graphe latence ───────────────────────────────────────────────────────────

def build_latency_chart(all_results_per_model: dict,
                        width=16.5*cm, height=7*cm) -> Drawing:
    cats   = list(dict.fromkeys(
        r["category"] for runs in all_results_per_model.values() for r in runs
    ))
    models = list(all_results_per_model.keys())
    lat    = {m: {} for m in models}
    for m, runs in all_results_per_model.items():
        for cat in cats:
            times = [r["result"]["total_s"] for r in runs if r["category"] == cat]
            lat[m][cat] = round(sum(times) / len(times), 1) if times else 0

    max_lat = max((v for m in lat.values() for v in m.values()), default=1) or 1
    model_colors = [
        colors.HexColor("#58a6ff"), colors.HexColor("#2ea043"),
        colors.HexColor("#d29922"), colors.HexColor("#f85149"),
    ]
    pl, pr, pt, pb = 4.5*cm, 0.5*cm, 0.8*cm, 1.2*cm
    bw = float(width) - float(pl) - float(pr)
    bh = float(height) - float(pt) - float(pb)
    group_h = bh / max(len(cats), 1)
    bar_h   = min(group_h / (len(models) + 1), 12)

    d = Drawing(float(width), float(height))
    d.add(Rect(0, 0, float(width), float(height), fillColor=BG_DARK, strokeColor=None))
    d.add(Line(float(pl), float(pb), float(pl), float(pb) + bh,
               strokeColor=BORDER, strokeWidth=0.5))
    for pct in [0, 25, 50, 75, 100]:
        x = float(pl) + bw * pct / 100
        d.add(Line(x, float(pb), x, float(pb) + bh, strokeColor=BORDER, strokeWidth=0.3))
        d.add(String(x, float(pb) - 10, f"{max_lat * pct / 100:.0f}s",
                     fontSize=6, fillColor=TEXT_DIM, textAnchor="middle"))
    for ci, cat in enumerate(reversed(cats)):
        y_group = float(pb) + ci * group_h + group_h * 0.1
        d.add(String(float(pl) - 4, y_group + bar_h * len(models) / 2, cat[:22],
                     fontSize=6.5, fillColor=TEXT_MAIN, textAnchor="end"))
        for mi, model in enumerate(models):
            val = lat[model].get(cat, 0)
            w   = bw * val / max_lat
            yb  = y_group + mi * (bar_h + 1)
            mc  = model_colors[mi % len(model_colors)]
            d.add(Rect(float(pl), yb, w, bar_h, fillColor=mc, strokeColor=None))
            if val > 0:
                d.add(String(float(pl) + w + 2, yb + bar_h * 0.3,
                             f"{val:.1f}s", fontSize=5.5, fillColor=mc))
    lx = float(pl)
    for mi, model in enumerate(models):
        mc    = model_colors[mi % len(model_colors)]
        short = model.split(":")[0][:20]
        d.add(Rect(lx, 2, 8, 6, fillColor=mc, strokeColor=None))
        d.add(String(lx + 10, 3, short, fontSize=6, fillColor=TEXT_MAIN))
        lx += len(short) * 4.5 + 18
    return d

# ─── Tableaux PDF ─────────────────────────────────────────────────────────────

def _base_style():
    return [
        ("BACKGROUND", (0, 0), (-1, 0),  BG_HEADER),
        ("TEXTCOLOR",  (0, 0), (-1, 0),  TEXT_DIM),
        ("FONTNAME",   (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 8),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("GRID",       (0, 0), (-1, -1), 0.3, BORDER),
        ("ROWHEIGHT",  (0, 0), (-1, -1), 15),
    ]


def build_summary_table(all_results_per_model: dict) -> Table:
    header = ["Modèle", "Tests", "PASS", "PARTIAL", "FAIL", "ERROR",
              "Taux PASS", "Temps moy.", "Pièges OK"]
    data   = [header]
    for model, runs in all_results_per_model.items():
        counts = {"PASS": 0, "PARTIAL": 0, "FAIL": 0, "ERROR": 0}
        total_t = traps_ok = traps_total = 0
        for r in runs:
            counts[r["eval"]["status"]] += 1
            total_t += r["result"].get("total_s", 0)
            if r.get("test_type") == "trap":
                traps_total += 1
                if r["eval"]["status"] == "PASS":
                    traps_ok += 1
        n = len(runs)
        data.append([
            model.split(":")[0][:16], str(n),
            str(counts["PASS"]), str(counts["PARTIAL"]),
            str(counts["FAIL"]), str(counts["ERROR"]),
            f"{counts['PASS']/n*100:.0f}%" if n else "—",
            f"{total_t/n:.1f}s"            if n else "—",
            f"{traps_ok}/{traps_total}"    if traps_total else "—",
        ])
    col_w = [3.5*cm, 1.2*cm, 1.2*cm, 1.5*cm, 1.2*cm, 1.2*cm, 1.8*cm, 1.8*cm, 1.8*cm]
    cmds  = _base_style() + [
        ("ALIGN",      (0, 0),  (0, -1),  "LEFT"),
        ("BACKGROUND", (0, 1),  (-1, -1), BG_DARK),
        ("TEXTCOLOR",  (0, 1),  (-1, -1), TEXT_MAIN),
        ("TEXTCOLOR",  (2, 1),  (2, -1),  STATUS_COLORS["PASS"]),
        ("TEXTCOLOR",  (3, 1),  (3, -1),  STATUS_COLORS["PARTIAL"]),
        ("TEXTCOLOR",  (4, 1),  (4, -1),  STATUS_COLORS["FAIL"]),
        ("FONTNAME",   (2, 1),  (4, -1),  "Helvetica-Bold"),
    ]
    return Table(data, colWidths=col_w, style=TableStyle(cmds))


def build_comparison_table(all_results_per_model: dict) -> Table:
    models = list(all_results_per_model.keys())
    first  = list(all_results_per_model.values())[0]
    header = ["#", "Question"] + [m.split(":")[0][:12] for m in models]
    col_w  = [0.6*cm, 8*cm] + [2*cm] * len(models)
    data   = [header]
    cmds   = _base_style() + [
        ("ALIGN",    (1, 0), (1, -1), "LEFT"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
    ]
    for i, ref_r in enumerate(first):
        q       = ref_r["result"]["question"]
        is_trap = ref_r.get("test_type") == "trap"
        row     = [str(i+1), (q[:58] + "…") if len(q) > 58 else q]
        bg      = colors.HexColor("#111820") if i % 2 == 0 else BG_DARK
        cmds.append(("BACKGROUND", (0, i+1), (-1, i+1), bg))
        cmds.append(("TEXTCOLOR",  (0, i+1), (1, i+1),
                     colors.HexColor("#79c0ff") if is_trap else TEXT_MAIN))
        for mi, model in enumerate(models):
            runs = all_results_per_model[model]
            if i < len(runs):
                st  = runs[i]["eval"]["status"]
                col = mi + 2
                row.append(st)
                cmds.append(("TEXTCOLOR", (col, i+1), (col, i+1),
                             STATUS_COLORS.get(st, TEXT_DIM)))
                cmds.append(("FONTNAME",  (col, i+1), (col, i+1), "Helvetica-Bold"))
            else:
                row.append("—")
        data.append(row)
    return Table(data, colWidths=col_w, style=TableStyle(cmds))


def build_cosine_table(all_results_per_model: dict) -> Table:
    models = list(all_results_per_model.keys())
    cats   = list(dict.fromkeys(
        r["category"] for runs in all_results_per_model.values() for r in runs
    ))
    header = ["Catégorie"] + [m.split(":")[0][:14] for m in models]
    col_w  = [4*cm] + [3*cm] * len(models)
    data   = [header]
    cmds   = _base_style() + [("ALIGN", (0, 0), (0, -1), "LEFT")]
    for ci, cat in enumerate(cats):
        row = [cat]
        for model in models:
            runs   = all_results_per_model[model]
            scores = [
                sc for r in runs if r["category"] == cat
                for sc in r["result"]["source_scores"].values()
            ]
            avg = round(sum(scores) / len(scores), 3) if scores else 0
            row.append(f"{avg:.3f}")
            col = models.index(model) + 1
            sc  = (STATUS_COLORS["PASS"]    if avg >= 0.7 else
                   STATUS_COLORS["PARTIAL"] if avg >= 0.4 else
                   STATUS_COLORS["FAIL"]    if avg > 0   else TEXT_DIM)
            cmds.append(("TEXTCOLOR", (col, ci+1), (col, ci+1), sc))
            cmds.append(("FONTNAME",  (col, ci+1), (col, ci+1), "Helvetica-Bold"))
        bg = colors.HexColor("#111820") if ci % 2 == 0 else BG_DARK
        cmds.append(("BACKGROUND", (0, ci+1), (-1, ci+1), bg))
        cmds.append(("TEXTCOLOR",  (0, ci+1), (0, ci+1), TEXT_MAIN))
        data.append(row)
    return Table(data, colWidths=col_w, style=TableStyle(cmds))

# ─── Génération PDF ───────────────────────────────────────────────────────────

def build_pdf(all_results_per_model: dict, output_path: str,
              url: str, tests: list):
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=1.8*cm, rightMargin=1.8*cm,
        topMargin=1.8*cm,  bottomMargin=1.8*cm,
    )
    S = getSampleStyleSheet()

    def ps(name, **kw):
        return ParagraphStyle(name, parent=S["Normal"], **kw)

    title_s  = ps("T",  fontSize=18, textColor=TEXT_BLUE,
                  fontName="Helvetica-Bold", spaceAfter=4)
    sub_s    = ps("S",  fontSize=8,  textColor=TEXT_DIM,   spaceAfter=12)
    h2_s     = ps("H2", fontSize=11, textColor=TEXT_MAIN,
                  fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=6)
    note_s   = ps("N",  fontSize=7,  textColor=TEXT_DIM,   spaceAfter=6)
    q_s      = ps("Q",  fontSize=9,  textColor=colors.HexColor("#79c0ff"),
                  fontName="Helvetica-Bold", spaceAfter=3)
    answer_s = ps("A",  fontSize=7,  textColor=colors.HexColor("#c9d1d9"),
                  leading=10, spaceAfter=3)
    meta_s   = ps("M",  fontSize=6.5, textColor=TEXT_DIM,  spaceAfter=6)

    models = list(all_results_per_model.keys())
    now    = datetime.now().strftime("%d/%m/%Y %H:%M")
    story  = []

    # ── En-tête ──
    story.append(Paragraph("RAG Hybrid — Rapport de Tests", title_s))
    story.append(Paragraph(
        f"Généré le {now} &nbsp;|&nbsp; URL : {url} &nbsp;|&nbsp; "
        f"Modèles : {', '.join(m.split(':')[0] for m in models)} &nbsp;|&nbsp; "
        f"{len(tests)} tests",
        sub_s,
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=BORDER))
    story.append(Spacer(1, 10))

    # ── Résumé global ──
    story.append(Paragraph("Résumé global", h2_s))
    story.append(build_summary_table(all_results_per_model))
    story.append(Spacer(1, 16))

    # ── Graphe latence ──
    story.append(Paragraph("Latence moyenne par catégorie", h2_s))
    story.append(build_latency_chart(all_results_per_model))
    story.append(Spacer(1, 16))

    # ── Scores cosine ──
    story.append(Paragraph("Score de confiance moyen des sources (cosine)", h2_s))
    story.append(Paragraph(
        "≥0.7 = très pertinent (vert) · 0.4–0.7 = pertinent (orange) · "
        "<0.4 = faible (rouge) · 0 = aucune source / fallback",
        note_s,
    ))
    story.append(build_cosine_table(all_results_per_model))
    story.append(Spacer(1, 16))

    # ── Comparaison multi-modèles ──
    story.append(PageBreak())
    story.append(Paragraph("Comparaison multi-modèles — statut par test", h2_s))
    story.append(Paragraph(
        "Questions en bleu = tests pièges (le RAG doit refuser de répondre).",
        note_s,
    ))
    story.append(build_comparison_table(all_results_per_model))
    story.append(Spacer(1, 20))

    # ── Détail par modèle ──
    for model, runs in all_results_per_model.items():
        story.append(PageBreak())
        story.append(Paragraph(f"Détail — {model}", h2_s))
        story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER))
        story.append(Spacer(1, 6))

        current_cat = None
        for r in runs:
            cat     = r["category"]
            result  = r["result"]
            ev      = r["eval"]
            status  = ev["status"]
            is_trap = r.get("test_type") == "trap"

            if cat != current_cat:
                current_cat = cat
                story.append(Paragraph(f"■ {cat}", ps(
                    "CH", fontSize=10, textColor=TEXT_BLUE,
                    fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=3,
                )))

            badge    = STATUS_COLORS[status]
            trap_tag = ' <font color="#79c0ff">[PIÈGE]</font>' if is_trap else ""
            story.append(Paragraph(
                f'<font color="{badge.hexval()}">[{status}]</font>{trap_tag} '
                f'<font color="#e6edf3">{result["question"]}</font>',
                q_s,
            ))

            if result.get("error"):
                story.append(Paragraph(f"Erreur : {result['error']}",
                    ps("E", fontSize=7, textColor=STATUS_COLORS["ERROR"], spaceAfter=4)))
            else:
                ans = result["answer"].strip()
                if len(ans) > 400:
                    ans = ans[:400] + "…"
                ans = ans.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                story.append(Paragraph(ans, answer_s))

                if is_trap:
                    dc = "#2ea043" if status == "PASS" else "#f85149"
                    story.append(Paragraph(
                        f'<font color="{dc}">{ev.get("detail", "")}</font>', meta_s))
                else:
                    src_parts = []
                    for fn, sc in result["source_scores"].items():
                        c = "#2ea043" if sc >= 0.7 else "#d29922" if sc >= 0.4 else "#f85149"
                        src_parts.append(f'<font color="{c}">{fn[:28]} ({sc:.2f})</font>')
                    kw_str = ", ".join(ev.get("keyword_found", [])) or "aucun"
                    story.append(Paragraph(
                        f"Sources : {' | '.join(src_parts) or 'aucune'}<br/>"
                        f"Mots-clés : {kw_str} &nbsp;|&nbsp; "
                        f"Recherche : {result.get('search_ms', 0)}ms &nbsp;|&nbsp; "
                        f"Total : {result.get('total_s', 0):.1f}s",
                        meta_s,
                    ))

            story.append(HRFlowable(
                width="100%", thickness=0.2,
                color=colors.HexColor("#21262d"), spaceAfter=4,
            ))

    doc.build(story)

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Suite de tests RAG → PDF")
    parser.add_argument("--url",     default="http://localhost:8080")
    parser.add_argument("--models",  default="mistral:7b-instruct",
                        help="Modèles séparés par virgule")
    parser.add_argument("--tests",   default="tests.yaml",
                        help="Fichier YAML de tests (défaut: tests.yaml)")
    parser.add_argument("--output",  default="rag_rapport.pdf")
    parser.add_argument("--timeout",     type=int, default=90)
    parser.add_argument("--num-predict", type=int, default=0, dest="num_predict",
                        help="Limite tokens LLM (0=défaut config). Ex: 256 pour tests rapides")
    parser.add_argument("--debug",   action="store_true",
                        help="Affiche pourquoi chaque test est PARTIAL/FAIL")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run",
                        help="Affiche les réponses brutes sans générer de PDF")
    args = parser.parse_args()

    tests_path = args.tests
    if not os.path.isabs(tests_path):
        tests_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), tests_path
        )
    if not os.path.exists(tests_path):
        print(f"❌ Fichier de tests introuvable : {tests_path}")
        raise SystemExit(1)

    tests  = load_tests(tests_path)
    models = [m.strip() for m in args.models.split(",")]

    print(f"\n{'='*65}")
    print(f"  RAG Test Suite — {len(tests)} tests × {len(models)} modèle(s)")
    print(f"  URL     : {args.url}")
    print(f"  Modèles : {', '.join(models)}")
    print(f"  Tests   : {tests_path}")
    print(f"{'='*65}\n")

    all_results_per_model = {}

    for model in models:
        print(f"\n── Modèle : {model} ──")
        runs = []
        t_model_start = time.time()
        for i, test in enumerate(tests):
            is_trap = test["type"] == "trap"
            tag     = " [PIÈGE]" if is_trap else ""
            print(f"  [{i+1:02d}/{len(tests)}] {test['category']}{tag} — "
                  f"{test['question'][:50]}…")
            result = ask_rag(test["question"], args.url, model, args.timeout, args.num_predict)
            ev     = evaluate(result, test)
            sym    = {"PASS": "✅", "PARTIAL": "⚠️ ", "FAIL": "❌",
                      "ERROR": "💥"}.get(ev["status"], "?")
            elapsed  = time.time() - t_model_start
            done     = i + 1
            avg_t    = elapsed / done
            remaining = avg_t * (len(tests) - done)
            eta_str  = f"  ETA ~{remaining/60:.0f}m{remaining%60:.0f}s" if done < len(tests) else ""
            print(f"         {sym} {ev['status']} — {result.get('total_s', 0):.1f}s{eta_str}")

            # --debug : détail des keywords/sources manquants
            if args.debug and ev["status"] in ("PARTIAL", "FAIL") and not is_trap:
                kw_miss = [kw for kw in test.get("keywords", [])
                           if kw.lower() not in result["answer"].lower()]
                src_miss = [s for s in test.get("sources", [])
                            if not any(s.lower() in src.lower() for src in result["sources"])]
                if kw_miss:
                    print(f"         ↳ keywords manquants : {kw_miss}")
                if src_miss:
                    print(f"         ↳ sources manquantes : {src_miss}")
                if result["sources"]:
                    print(f"         ↳ sources reçues     : {result['sources']}")

            if is_trap and ev["status"] == "FAIL":
                print(f"         ↳ {ev.get('detail', '')}")

            # --dry-run : affiche la réponse brute
            if args.dry_run:
                ans = result["answer"].strip()[:300]
                print(f"         ┌─ réponse : {ans}{'…' if len(result['answer'])>300 else ''}")

            runs.append({
                "category":  test["category"],
                "test_type": test["type"],
                "result":    result,
                "eval":      ev,
            })
        all_results_per_model[model] = runs

        counts      = {"PASS": 0, "PARTIAL": 0, "FAIL": 0, "ERROR": 0}
        traps       = [r for r in runs if r["test_type"] == "trap"]
        traps_ok    = sum(1 for r in traps if r["eval"]["status"] == "PASS")
        for r in runs:
            counts[r["eval"]["status"]] += 1
        print(f"\n  → PASS:{counts['PASS']} PARTIAL:{counts['PARTIAL']} "
              f"FAIL:{counts['FAIL']} | {counts['PASS']/len(tests)*100:.0f}% "
              f"| Pièges: {traps_ok}/{len(traps)}")

    print(f"\n{'='*65}")
    if args.dry_run:
        print("Mode --dry-run : pas de PDF généré.")
    else:
        print(f"Génération PDF : {args.output}")
        build_pdf(all_results_per_model, args.output, args.url, tests)
        print(f"✅ Rapport sauvegardé : {args.output}\n")


if __name__ == "__main__":
    main()
