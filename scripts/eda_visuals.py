"""Generate NLP-focused EDA statistics and dependency-free SVG charts.

Run after ``scripts/acquire.py`` and ``scripts/eda.py``.  The script reads the
acquired annotations and the machine-readable EDA report, then writes
``reports/eda_nlp.json`` and SVG files under ``reports/figures``.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    from .acquire import ROOT, write_json
    from .eda import iter_records
except ImportError:
    from acquire import ROOT, write_json
    from eda import iter_records


TOKEN = re.compile(r"[^\W\d_]+(?:[’'-][^\W\d_]+)*", re.UNICODE)
PT_STOPWORDS = set("""
a à agora ainda algo algumas alguns ali ambos antes ao aos após aquela aquelas aquele aqueles
aquilo as às assim até através cada com como contra contudo da daquela daquele dar das de dela
dele deles demais depois desde desta deste deve devem do dos e é ela elas ele eles em entre era
eram essa essas esse esses esta está estão estas estava estavam este estes eu foi foram há isso
isto já lhe lhes mais mas me mesmo meu minha muito na não nas nem no nos nós nossa nossas nosso
nossos num numa o os ou para pela pelas pelo pelos por porque qual quando que quem se sem sendo
ser será seu seus só sua suas também tem têm tendo toda todas todo todos um uma umas uns você
vocês
""".split())
EN_STOPWORDS = set("""
a about after again against all am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has
have having he her here hers herself him himself his how i if in into is it its itself just me
more most my myself no nor not now of off on once only or other our ours ourselves out over own
same she should so some such than that the their theirs them themselves then there these they
this those through to too under until up very was we were what when where which while who whom
why will with would you your yours yourself yourselves
""".split())

COLORS = {
    "fake": "#d95f5f",
    "true": "#3f8f72",
    "Supported": "#3f8f72",
    "Refuted": "#d95f5f",
    "Not Enough Evidence": "#d6a63c",
    "Conflicting Evidence/Cherrypicking": "#8064a2",
    "SUPPORTS": "#3f8f72",
    "REFUTES": "#d95f5f",
    "NOT ENOUGH INFO": "#d6a63c",
    "SEM RÓTULO": "#8a929a",
    "mean": "#4063a8",
    "p50": "#57a6a1",
    "p95": "#d68c45",
}


def tokens(text: str) -> list[str]:
    return TOKEN.findall(text.casefold())


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    location = (len(ordered) - 1) * q
    lo, hi = math.floor(location), math.ceil(location)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (location - lo)


def describe(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "min": round(min(values), 3),
        "p25": round(percentile(values, .25), 3),
        "p50": round(percentile(values, .50), 3),
        "p95": round(percentile(values, .95), 3),
        "p99": round(percentile(values, .99), 3),
        "max": round(max(values), 3),
        "mean": round(statistics.fmean(values), 3),
        "std_population": round(statistics.pstdev(values), 3),
    }


def auc_binary(rows: list[tuple[int, str]], positive: str) -> float:
    """Mann-Whitney AUC for a numeric score, with average ranks for ties."""
    ordered = sorted(rows)
    positives = sum(label == positive for _, label in ordered)
    negatives = len(ordered) - positives
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2
        rank_sum += average_rank * sum(label == positive for _, label in ordered[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def best_length_threshold(rows: list[tuple[int, str]]) -> dict:
    """Diagnostic only: predict true above a word-count threshold."""
    grouped = defaultdict(Counter)
    for length, label in rows:
        grouped[length][label] += 1
    ordered = sorted(grouped)
    correct = sum(label == "true" for _, label in rows)
    best = (correct, ordered[0] - 1)
    for length in ordered:
        correct += grouped[length]["fake"] - grouped[length]["true"]
        if correct > best[0]:
            best = (correct, length)
    return {
        "rule": "predict true when words > threshold; diagnostic fitted on all records",
        "threshold_words": best[1],
        "apparent_accuracy": round(best[0] / len(rows), 4),
        "length_auc_true_as_positive": round(auc_binary(rows, "true"), 4),
    }


def histogram(values: list[int], edges: list[int]) -> list[float]:
    counts = [0] * (len(edges) - 1)
    for value in values:
        for index in range(len(edges) - 1):
            if edges[index] <= value < edges[index + 1] or (
                    index == len(edges) - 2 and value == edges[index + 1]):
                counts[index] += 1
                break
    total = len(values)
    return [round(100 * count / total, 3) for count in counts]


def analyze_fakebr(fakebr_dir: Path) -> dict:
    result = {}
    doc_frequency = {"fake": Counter(), "true": Counter()}
    term_frequency = {"fake": Counter(), "true": Counter()}
    document_counts = Counter()
    edges = [0, 50, 100, 150, 200, 300, 500, 750, 1000, 1500, 2500, 4000, 8000]
    for variant in ("full_texts", "size_normalized_texts"):
        lengths = defaultdict(list)
        style = defaultdict(lambda: defaultdict(list))
        rows = []
        for label in ("fake", "true"):
            for path in sorted((fakebr_dir / variant / label).glob("*.txt")):
                text = path.read_text(encoding="utf-8-sig")
                word_list = tokens(text)
                length = len(text.split())
                lengths[label].append(length)
                rows.append((length, label))
                letters = [char for char in text if char.isalpha()]
                style[label]["uppercase_letter_ratio"].append(
                    sum(char.isupper() for char in letters) / len(letters) if letters else 0)
                scale = max(len(text), 1) / 1000
                style[label]["exclamation_per_1000_chars"].append(text.count("!") / scale)
                style[label]["question_per_1000_chars"].append(text.count("?") / scale)
                style[label]["urls_per_document"].append(len(re.findall(r"https?://|www\.", text.casefold())))
                style[label]["mean_word_characters"].append(
                    statistics.fmean(map(len, word_list)) if word_list else 0)
                if variant == "size_normalized_texts":
                    content = [word for word in word_list if len(word) >= 3 and word not in PT_STOPWORDS]
                    term_frequency[label].update(content)
                    doc_frequency[label].update(set(content))
                    document_counts[label] += 1
        result[variant] = {
            "word_count_by_label": {label: describe(values) for label, values in lengths.items()},
            "style_by_label": {
                label: {metric: describe(values) for metric, values in metrics.items()}
                for label, metrics in style.items()
            },
            "length_diagnostic": best_length_threshold(rows),
            "histogram": {
                "edges": edges,
                "percent_by_label": {label: histogram(values, edges) for label, values in lengths.items()},
            },
        }

    vocabulary = set(term_frequency["fake"]) | set(term_frequency["true"])
    candidates = []
    for word in vocabulary:
        fake_df, true_df = doc_frequency["fake"][word], doc_frequency["true"][word]
        if fake_df + true_df < 30:
            continue
        fake_odds = (fake_df + .5) / (document_counts["fake"] - fake_df + .5)
        true_odds = (true_df + .5) / (document_counts["true"] - true_df + .5)
        candidates.append((math.log2(fake_odds / true_odds), word, fake_df, true_df))
    candidates.sort()
    discriminative = candidates[:10] + candidates[-10:]
    result["size_normalized_texts"]["lexical"] = {
        "token_count_by_label": {label: sum(counter.values()) for label, counter in term_frequency.items()},
        "vocabulary_size": len(vocabulary),
        "top_terms_by_label": {
            label: counter.most_common(20) for label, counter in term_frequency.items()
        },
        "document_frequency_log_odds": [
            {"term": word, "log2_odds_fake_vs_true": round(score, 3),
             "documents_fake": fake_df, "documents_true": true_df}
            for score, word, fake_df, true_df in discriminative
        ],
        "method": "NFKC-independent Unicode word regex; casefold; Portuguese stopword removal; "
                  "document-frequency log-odds with 0.5 smoothing; minimum 30 documents.",
    }
    return result


def analyze_fakerecogna(path: Path) -> dict:
    import csv

    lengths = defaultdict(list)
    rows = []
    edges = [0, 20, 40, 60, 80, 100, 150, 200, 300, 500, 1000, 2500]
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            label = "fake" if str(row.get("Label")).strip() == "1" else "true"
            length = len(str(row.get("Noticia") or "").split())
            lengths[label].append(length)
            rows.append((length, label))
    return {
        "word_count_by_label": {label: describe(values) for label, values in lengths.items()},
        "length_diagnostic": best_length_threshold(rows),
        "histogram": {
            "edges": edges,
            "percent_by_label": {label: histogram(values, edges) for label, values in lengths.items()},
        },
    }


def analyze_claim_lengths(data_dir: Path) -> dict:
    result = {}
    for dataset, suffix, splits in (
        ("averitec", ".json", ("train", "dev")),
        ("feverous", ".jsonl", ("train", "dev", "test")),
    ):
        result[dataset] = {}
        for split in splits:
            by_label = defaultdict(list)
            path = data_dir / dataset / f"{split}{suffix}"
            for row in iter_records(path):
                label = row.get("label") or "SEM RÓTULO"
                by_label[label].append(len(row["claim"].split()))
            result[dataset][split] = {
                "word_count_by_label": {label: describe(values) for label, values in by_label.items()}
            }
    return result


def svg_text(value) -> str:
    return html.escape(str(value), quote=True)


def save_svg(path: Path, width: int, height: int, title: str, description: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
<title id="title">{svg_text(title)}</title><desc id="desc">{svg_text(description)}</desc>
<style>
text {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #263238; }}
.title {{ font-size: 24px; font-weight: 700; }} .subtitle {{ font-size: 13px; fill: #5f6b72; }}
.label {{ font-size: 13px; }} .value {{ font-size: 12px; font-weight: 600; }}
.axis {{ stroke: #9aa5aa; stroke-width: 1; }} .grid {{ stroke: #e5e9eb; stroke-width: 1; }}
</style>
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="30" y="36" class="title">{svg_text(title)}</text>
<text x="30" y="60" class="subtitle">{svg_text(description)}</text>
{body}
</svg>'''
    path.write_text(content, encoding="utf-8")


def horizontal_bars(path: Path, title: str, description: str, items: list[dict],
                    maximum: float | None = None, value_suffix: str = "") -> None:
    # Keep generous margins for long Portuguese labels and values.  Several
    # charts mix counts and percentages, so the value labels need more room
    # than a conventional axis-only plot.
    width, left, right, top, row_h = 1250, 390, 230, 90, 32
    height = top + row_h * len(items) + 45
    chart_width = width - left - right
    maximum = maximum or max(item["value"] for item in items) or 1
    body = []
    for tick in range(6):
        value = maximum * tick / 5
        x = left + chart_width * tick / 5
        body.append(f'<line x1="{x:.1f}" y1="{top-12}" x2="{x:.1f}" y2="{height-35}" class="grid"/>')
        body.append(f'<text x="{x:.1f}" y="{height-14}" text-anchor="middle" class="subtitle">{value:.0f}{svg_text(value_suffix)}</text>')
    for index, item in enumerate(items):
        y = top + index * row_h
        bar_width = chart_width * item["value"] / maximum
        if item["value"] > 0:
            bar_width = max(bar_width, 2)
        color = item.get("color", "#4063a8")
        body.append(f'<text x="{left-10}" y="{y+18}" text-anchor="end" class="label">{svg_text(item["label"])}</text>')
        body.append(f'<rect x="{left}" y="{y+4}" width="{bar_width:.1f}" height="20" rx="3" fill="{color}"/>')
        display = item.get("display", f'{item["value"]:.1f}{value_suffix}')
        text_x = min(left + bar_width + 7, width - right + 4)
        body.append(f'<text x="{text_x:.1f}" y="{y+19}" class="value">{svg_text(display)}</text>')
    save_svg(path, width, height, title, description, "\n".join(body))


def histogram_chart(path: Path, fakebr: dict) -> None:
    width, height = 1200, 650
    panels = [("full_texts", "Textos completos"), ("size_normalized_texts", "Textos normalizados")]
    body = []
    for panel_index, (key, heading) in enumerate(panels):
        x0 = 70 + panel_index * 590
        y0, chart_w, chart_h = 130, 510, 390
        hist = fakebr[key]["histogram"]
        edges = hist["edges"]
        values = hist["percent_by_label"]
        maximum = max(max(series) for series in values.values())
        body.append(f'<text x="{x0+chart_w/2}" y="100" text-anchor="middle" class="label" font-weight="700">{heading}</text>')
        for tick in range(5):
            pct = maximum * tick / 4
            y = y0 + chart_h - chart_h * tick / 4
            body.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+chart_w}" y2="{y:.1f}" class="grid"/>')
            body.append(f'<text x="{x0-8}" y="{y+4:.1f}" text-anchor="end" class="subtitle">{pct:.0f}%</text>')
        group_w = chart_w / (len(edges) - 1)
        for index in range(len(edges) - 1):
            for label_index, label in enumerate(("fake", "true")):
                value = values[label][index]
                bar_w = group_w * .34
                x = x0 + index * group_w + group_w * (.12 + label_index * .40)
                bar_h = chart_h * value / maximum
                body.append(f'<rect x="{x:.1f}" y="{y0+chart_h-bar_h:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{COLORS[label]}" opacity=".88"/>')
            label = f'{edges[index]}–{edges[index+1]}'
            body.append(f'<text x="{x0+(index+.5)*group_w:.1f}" y="{y0+chart_h+18}" text-anchor="end" transform="rotate(-45 {x0+(index+.5)*group_w:.1f} {y0+chart_h+18})" class="subtitle">{label}</text>')
        body.append(f'<line x1="{x0}" y1="{y0+chart_h}" x2="{x0+chart_w}" y2="{y0+chart_h}" class="axis"/>')
    body += [
        f'<rect x="440" y="595" width="14" height="14" fill="{COLORS["fake"]}"/><text x="460" y="607" class="label">fake</text>',
        f'<rect x="550" y="595" width="14" height="14" fill="{COLORS["true"]}"/><text x="570" y="607" class="label">true</text>',
    ]
    save_svg(path, width, height, "Distribuição do comprimento no Fake.br",
             "Percentual de documentos por faixa de palavras separadas por espaços; não são tokens de modelo.",
             "\n".join(body))


def diverging_terms_chart(path: Path, lexical: dict) -> None:
    terms = sorted(lexical["document_frequency_log_odds"], key=lambda item: item["log2_odds_fake_vs_true"])
    width, left, top, row_h = 1100, 550, 90, 27
    height = top + row_h * len(terms) + 55
    max_abs = max(abs(item["log2_odds_fake_vs_true"]) for item in terms)
    half = 420
    body = [f'<line x1="{left}" y1="{top-10}" x2="{left}" y2="{height-40}" class="axis"/>']
    for index, item in enumerate(terms):
        score = item["log2_odds_fake_vs_true"]
        y = top + index * row_h
        bar_w = half * abs(score) / max_abs
        x = left if score >= 0 else left - bar_w
        color = COLORS["fake"] if score >= 0 else COLORS["true"]
        body.append(f'<rect x="{x:.1f}" y="{y+3}" width="{bar_w:.1f}" height="18" rx="2" fill="{color}"/>')
        body.append(f'<text x="{left-8 if score < 0 else left+8}" y="{y+17}" text-anchor="end" class="label">{svg_text(item["term"])}</text>' if score < 0 else
                    f'<text x="{left+8}" y="{y+17}" class="label">{svg_text(item["term"])}</text>')
        value_x = x - 7 if score < 0 else x + bar_w + 7
        anchor = "end" if score < 0 else "start"
        body.append(f'<text x="{value_x:.1f}" y="{y+17}" text-anchor="{anchor}" class="value">{score:+.2f}</text>')
    body.append(f'<text x="{left-half/2}" y="{height-15}" text-anchor="middle" class="subtitle">mais associado a true</text>')
    body.append(f'<text x="{left+half/2}" y="{height-15}" text-anchor="middle" class="subtitle">mais associado a fake</text>')
    save_svg(path, width, height, "Termos que mais diferenciam as classes do Fake.br",
             "Log2 das chances de presença por documento na variante normalizada; associação não implica causalidade.",
             "\n".join(body))


def generate_charts(output_dir: Path, base_report: dict, nlp: dict) -> None:
    figures = output_dir / "figures"
    horizontal_bars(
        figures / "01_tamanho_datasets.svg",
        "Tamanho das coleções adquiridas",
        "Fake.br conta notícias únicas; suas duas variantes representam os mesmos 7.200 documentos.",
        [
            {"label": "Fake.br", "value": 7200, "display": "7.200", "color": "#4063a8"},
            {"label": "FakeRecogna 2.0", "value": 52800, "display": "52.800", "color": "#d68c45"},
            {"label": "AVeriTeC (train + dev)", "value": 3568, "display": "3.568", "color": "#8064a2"},
            {"label": "FEVEROUS (train + dev + test)", "value": 87026, "display": "87.026", "color": "#57a6a1"},
        ],
    )

    labels = []
    for dataset, split, caption in (
        ("fakebr", "full_texts", "Fake.br"),
        ("fakerecogna2", "extractive", "FakeRecogna 2.0"),
        ("averitec", "train", "AVeriTeC train"),
        ("averitec", "dev", "AVeriTeC dev"),
        ("feverous", "train", "FEVEROUS train"),
        ("feverous", "dev", "FEVEROUS dev"),
    ):
        profile = base_report[dataset][split]
        for label, percentage in profile["percent_of_labeled"].items():
            if percentage is not None and profile["labels"][label]:
                labels.append({"label": f"{caption} · {label}", "value": percentage,
                               "display": f"{percentage:.2f}%", "color": COLORS.get(label, "#4063a8")})
    horizontal_bars(
        figures / "02_distribuicao_classes.svg",
        "Distribuição de classes",
        "Percentual dentro de cada split rotulado; as taxonomias têm significados diferentes.",
        labels, maximum=100, value_suffix="%",
    )

    length_items = []
    for variant, caption in (("full_texts", "Completo"), ("size_normalized_texts", "Normalizado")):
        for label in ("fake", "true"):
            stats = nlp["fakebr"][variant]["word_count_by_label"][label]
            length_items.extend([
                {"label": f"{caption} · {label} · média", "value": stats["mean"],
                 "display": f'{stats["mean"]:.1f}', "color": COLORS["mean"]},
                {"label": f"{caption} · {label} · mediana", "value": stats["p50"],
                 "display": f'{stats["p50"]:.1f}', "color": COLORS["p50"]},
            ])
    horizontal_bars(
        figures / "03_fakebr_comprimento_classe.svg",
        "Fake.br: comprimento por variante e classe",
        "Média e mediana de palavras; a normalização praticamente remove a diferença central entre as classes.",
        length_items,
    )
    histogram_chart(figures / "04_fakebr_distribuicao_comprimento.svg", nlp["fakebr"])

    claim_items = []
    for dataset, split, caption in (("averitec", "train", "AVeriTeC"), ("feverous", "train", "FEVEROUS")):
        for label, stats in nlp["claim_lengths"][dataset][split]["word_count_by_label"].items():
            for metric in ("p50", "p95"):
                claim_items.append({"label": f"{caption} · {label} · {metric.upper()}",
                                    "value": stats[metric], "display": f'{stats[metric]:.1f}',
                                    "color": COLORS[metric]})
    horizontal_bars(
        figures / "05_comprimento_claims.svg",
        "Comprimento das alegações por classe",
        "Mediana (P50) e percentil 95 em palavras nos splits de treino.",
        claim_items,
    )
    diverging_terms_chart(
        figures / "06_fakebr_termos_discriminativos.svg",
        nlp["fakebr"]["size_normalized_texts"]["lexical"],
    )

    source_items = []
    source_categories = base_report["fakebr"]["full_texts"]["categories"]
    for label in ("fake", "true"):
        domains = source_categories[f"source_domain_{label}"]
        total = sum(domains.values())
        for domain, count in list(domains.items())[:5]:
            percentage = 100 * count / total
            source_items.append({"label": f"{label} · {domain or '<sem domínio>'}",
                                 "value": percentage, "display": f"{count} ({percentage:.1f}%)",
                                 "color": COLORS[label]})
    horizontal_bars(
        figures / "07_fakebr_concentracao_fontes.svg",
        "Fake.br: concentração por domínio de origem",
        "Participação dentro de cada classe; forte separação de fontes pode produzir vazamento e baixa generalização.",
        source_items, maximum=100, value_suffix="%",
    )

    fakerecogna_length_items = []
    for label in ("fake", "true"):
        stats = nlp["fakerecogna2"]["word_count_by_label"][label]
        for metric, caption, color in (("mean", "média", "#4063a8"), ("p50", "mediana", "#57a6a1"), ("p95", "P95", "#d68c45")):
            fakerecogna_length_items.append({
                "label": f"{label} · {caption}", "value": stats[metric],
                "display": f'{stats[metric]:.1f}', "color": color,
            })
    horizontal_bars(
        figures / "08_fakerecogna_comprimento_classe.svg",
        "FakeRecogna 2.0: comprimento por classe",
        "Média, mediana e percentil 95 dos resumos extrativos, em palavras separadas por espaços.",
        fakerecogna_length_items,
    )

    fakerecogna_sources = []
    source_categories = base_report["fakerecogna2"]["extractive"]["categories"]
    for label in ("fake", "true"):
        domains = source_categories[f"source_domain_{label}"]
        total = sum(domains.values())
        for source, count in list(domains.items())[:6]:
            percentage = 100 * count / total
            fakerecogna_sources.append({
                "label": f"{label} · {source}", "value": percentage,
                "display": f"{count} ({percentage:.1f}%)", "color": COLORS[label],
            })
    horizontal_bars(
        figures / "09_fakerecogna_concentracao_fontes.svg",
        "FakeRecogna 2.0: principais domínios por classe",
        "Participação dentro de cada classe; domínios ausentes são mantidos como uma categoria de qualidade.",
        fakerecogna_sources, maximum=100, value_suffix="%",
    )

    categories = source_categories["category_all"]
    category_total = sum(categories.values())
    category_items = [
        {"label": category, "value": 100 * count / category_total,
         "display": f"{count} ({100 * count / category_total:.1f}%)", "color": "#d68c45"}
        for category, count in list(categories.items())[:12]
    ]
    horizontal_bars(
        figures / "10_fakerecogna_categorias.svg",
        "FakeRecogna 2.0: categorias mais frequentes",
        "Distribuição conjunta das classes; grafias originais foram normalizadas apenas para minúsculas.",
        category_items, maximum=max(item["value"] for item in category_items), value_suffix="%",
    )

    averitec_categories = base_report["averitec"]["train"]["categories"]
    averitec_metadata_items = []
    for key, caption, color in (
        ("location_ISO_code", "local", "#4063a8"),
        ("claim_types", "tipo", "#8064a2"),
        ("fact_checking_strategies", "estratégia", "#d68c45"),
    ):
        counts = averitec_categories[key]
        total = sum(counts.values())
        for value, count in list(counts.items())[:5]:
            percentage = 100 * count / total
            averitec_metadata_items.append({
                "label": f"{caption} · {value}", "value": percentage,
                "display": f"{count} ({percentage:.1f}%)", "color": color,
            })
    horizontal_bars(
        figures / "11_averitec_metadados.svg",
        "AVeriTeC: metadados mais frequentes no treino",
        "Cinco itens mais frequentes em cada campo; porcentagens calculadas dentro do respectivo campo.",
        averitec_metadata_items, maximum=100, value_suffix="%",
    )

    averitec_answer_items = []
    for key, caption, color, limit in (
        ("answer_type", "tipo de resposta", "#8064a2", 4),
        ("source_medium", "meio da fonte", "#3f8f72", 6),
    ):
        counts = averitec_categories[key]
        total = sum(counts.values())
        for value, count in list(counts.items())[:limit]:
            percentage = 100 * count / total
            averitec_answer_items.append({
                "label": f"{caption} · {value or '<missing>'}", "value": percentage,
                "display": f"{count} ({percentage:.1f}%)", "color": color,
            })
    horizontal_bars(
        figures / "15_averitec_respostas_fontes.svg",
        "AVeriTeC: tipos de resposta e meios das fontes",
        "Distribuição das 8.479 respostas do treino; porcentagens calculadas dentro de cada campo.",
        averitec_answer_items, maximum=100, value_suffix="%",
    )

    averitec = base_report["averitec"]["train"]["metrics"]
    averitec_items = []
    for metric, caption in (
        ("questions_per_claim", "perguntas por alegação"),
        ("answers_per_claim", "respostas por alegação"),
        ("justification_words", "palavras na justificativa"),
    ):
        for stat, color in (("p50", "#57a6a1"), ("p95", "#d68c45")):
            value = averitec[metric][stat]
            averitec_items.append({
                "label": f"{caption} · {stat.upper()}", "value": value,
                "display": f"{value:.1f}", "color": color,
            })
    horizontal_bars(
        figures / "12_averitec_estrutura_investigativa.svg",
        "AVeriTeC: estrutura investigativa do treino",
        "Mediana e percentil 95 de perguntas, respostas e extensão das justificativas.",
        averitec_items,
    )

    feverous_categories = base_report["feverous"]["train"]["categories"]
    challenges = feverous_categories["challenge"]
    challenge_total = sum(challenges.values())
    challenge_items = [
        {"label": challenge, "value": 100 * count / challenge_total,
         "display": f"{count} ({100 * count / challenge_total:.1f}%)", "color": "#57a6a1"}
        for challenge, count in challenges.items()
    ]
    horizontal_bars(
        figures / "13_feverous_desafios_verificacao.svg",
        "FEVEROUS: tipos de desafio no treino",
        "Proporção das 71.291 alegações por dificuldade principal de verificação.",
        challenge_items, maximum=max(item["value"] for item in challenge_items), value_suffix="%",
    )

    evidence_types = feverous_categories["evidence_element_type"]
    evidence_total = sum(evidence_types.values())
    evidence_type_items = [
        {"label": evidence_type, "value": 100 * count / evidence_total,
         "display": f"{count} ({100 * count / evidence_total:.1f}%)", "color": "#57a6a1"}
        for evidence_type, count in evidence_types.items()
    ]
    horizontal_bars(
        figures / "14_feverous_tipos_evidencia.svg",
        "FEVEROUS: tipos de elementos de evidência",
        "Participação entre todas as referências anotadas no treino; um exemplo pode conter vários elementos.",
        evidence_type_items, maximum=100, value_suffix="%",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--fakebr-dir", type=Path, default=ROOT / "Fake.br-Corpus")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    base_path = args.output_dir / "eda.json"
    if not base_path.exists():
        raise FileNotFoundError("Run scripts/eda.py before scripts/eda_visuals.py")
    base_report = json.loads(base_path.read_text(encoding="utf-8"))
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "All records; whitespace lengths; Unicode regex for lexical analysis; dependency-free SVG output.",
        "fakebr": analyze_fakebr(args.fakebr_dir),
        "fakerecogna2": analyze_fakerecogna(
            args.data_dir / "fakerecogna2" / "fakerecogna_extrativo.csv"
        ),
        "claim_lengths": analyze_claim_lengths(args.data_dir),
    }
    write_json(args.output_dir / "eda_nlp.json", report)
    generate_charts(args.output_dir, base_report, report)
    print(f"NLP EDA saved to {args.output_dir / 'eda_nlp.json'}", flush=True)
    print(f"SVG charts saved to {args.output_dir / 'figures'}", flush=True)


if __name__ == "__main__":
    main()
