"""Build the BrainFMBench docs source.

Copies docs/ into .docs_build/, then generates the pages that come from the repo itself:
the landing page with live numbers, one page per model and per dataset, the results page,
the submission reference (from CONTRIBUTING.md) and the release notes (from CHANGELOG.md).
Writes mkdocs.gen.yml with the generated pages in the navigation. Then run:

    python scripts/build_docs.py && mkdocs build -f mkdocs.gen.yml
"""
import csv
import datetime
import glob
import json
import os
import re
import shutil
import sys

import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import make_leaderboard as L  # noqa: E402

SRC = os.path.join(REPO, "docs")
OUT = os.path.join(REPO, ".docs_build")
GH = "https://github.com/epics-lab/BrainFMBench"
SITE = L.PAGES_URL

RBC = "https://reprobrainchart.github.io/docs/datasets/"
# names as given on each dataset's Reproducible Brain Charts page; citations match labels/README.md
DATASETS = {
    "NKI": dict(full="Rockland Sample", rbc="NKI",
                cite="Nooner, K. B. et al. The NKI-Rockland Sample: a model for accelerating the pace of discovery science in psychiatry. *Front. Neurosci.* 6, 152 (2012).",
                doi="https://doi.org/10.3389/fnins.2012.00152",
                note="The only cohort here with adults. Mental health scores exist for 139 children only."),
    "HBN": dict(full="Healthy Brain Network", rbc="HBN",
                cite="Alexander, L. M. et al. An open resource for transdiagnostic research in pediatric mental health and learning disorders. *Sci. Data* 4, 170181 (2017).",
                doi="https://doi.org/10.1038/sdata.2017.181", note=""),
    "PNC": dict(full="Philadelphia Neurodevelopmental Cohort", rbc="PNC",
                cite="Satterthwaite, T. D. et al. Neuroimaging of the Philadelphia Neurodevelopmental Cohort. *NeuroImage* 86, 544–553 (2014).",
                doi="https://doi.org/10.1016/j.neuroimage.2013.07.064",
                note="The largest cohort here, and the only one with the p factor zero vs nonzero task."),
    "BHRC": dict(full="Brazil High Risk Cohort", rbc="BHRC",
                 cite="Salum, G. A. et al. High risk cohort study for psychiatric disorders in childhood: rationale, design, methods and preliminary results. *Int. J. Methods Psychiatr. Res.* 24, 58–73 (2015).",
                 doi="https://doi.org/10.1002/mpr.1459", note=""),
    "CCNP": dict(full="Developmental Chinese Color Nest Project", rbc="dCCNP",
                 cite="Liu, S. et al. Chinese Color Nest Project: an accelerated longitudinal brain-mind cohort. *Dev. Cogn. Neurosci.* 52, 101020 (2021).",
                 doi="https://doi.org/10.1016/j.dcn.2021.101020", note=""),
}
RBC_CITE = ("Shafiei, G. et al. Reproducible Brain Charts: An open data resource for mapping brain development "
            "and its associations with mental health. *Neuron* 113(22), 3758–3779 (2025). "
            "[doi](https://doi.org/10.1016/j.neuron.2025.08.026)")
TASK_NAME = {t: n for t, n, _ in L.OV_TASKS}
IS_CLS = {t: c for t, _, c in L.OV_TASKS}
PREP_CHIP = {"turboprep": ("rigid", ""), "turboprep-affine": ("affine", "affine"), "cat12": ("CAT12", "cat12"),
             "freesurfer": ("FreeSurfer", "freesurfer"), "raw": ("raw", "raw")}


def chip(prep):
    label, cls = PREP_CHIP.get(prep, (prep, ""))
    return f'<span class="bfb-chip {cls}">{label}</span>'


def slugify(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def write(rel, text):
    p = os.path.join(OUT, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(text)


# ---------------------------------------------------------------- data
def load_models():
    """name -> {"folders": {prep: folder}, "meta": yaml of the native folder}"""
    out = {}
    for y in sorted(glob.glob(os.path.join(REPO, "models", "*", "model.yaml"))):
        folder = os.path.basename(os.path.dirname(y))
        meta = yaml.safe_load(open(y)) or {}
        name = meta.get("name") or folder
        if not os.path.isfile(os.path.join(os.path.dirname(y), "results.json")):
            continue
        e = out.setdefault(name, {"folders": {}, "metas": {}})
        prep = meta.get("preprocessing", "turboprep")
        e["folders"][prep] = folder
        e["metas"][prep] = meta
    return out


def native_prep(m, preps):
    return L._NATIVE.get(m, (None, None))[0] if m in L._NATIVE else sorted(preps)[0]


def build_tables(rows):
    """(probe, ds, task) -> {model: (mean, std)} on each model's own preprocessing, plus the same for every prep."""
    itable = {}
    for r in rows:
        itable.setdefault((r.get("probe", "rf"), r["model"], r.get("preprocessing", "turboprep"), r["dataset"]), {})[r["task"]] = r
    L._set_native(itable)
    native, allprep = {}, {}
    for (probe, m, prep, ds), tasks in itable.items():
        for t, r in tasks.items():
            if t not in IS_CLS:
                continue
            key = "test_acc" if IS_CLS[t] else "test_mae"
            if r.get(key + "_mean") is None:
                continue
            v = (r[key + "_mean"], r.get(key + "_std") or 0.0)
            allprep.setdefault((probe, ds, t, prep), {})[m] = v
            if L._in_view(m, prep, "native"):
                native.setdefault((probe, ds, t), {})[m] = v
    return native, allprep


def gain(ds, t, v):
    key = "test_acc" if IS_CLS[t] else "test_mae"
    ch = L._ov_chance(ds, t, key)
    if ch is None:
        return None
    return v - ch if IS_CLS[t] else (ch - v) / ch


def ranked(native, probe, ds, t):
    """Models sorted best first, and whether the row has signal (best gain >= 0.1)."""
    d = native.get((probe, ds, t), {})
    order = sorted(d, key=lambda m: -d[m][0] if IS_CLS[t] else d[m][0])
    gains = [gain(ds, t, d[m][0]) for m in order]
    best = max([g for g in gains if g is not None], default=0)
    return order, d, best >= 0.1


def fmt(t, v):
    mu, sd = v
    return f"{mu:.3f} ± {sd:.3f}" if IS_CLS[t] else f"{mu:.2f} ± {sd:.2f}"


def metric_name(t):
    return "bal acc ↑" if IS_CLS[t] else "MAE ↓"


def labels_info(ds):
    rows = list(csv.DictReader(open(os.path.join(REPO, "labels", f"{ds}.csv"))))
    ages = np.array([float(r["age"]) for r in rows if r.get("age")])
    fem = sum(1 for r in rows if r.get("sex") == "1")
    cov = {c: sum(1 for r in rows if (r.get(c) or "").strip()) for c in rows[0] if c != "subject_id"}
    left = sum(1 for r in rows if r.get("handedness") == "1")
    return dict(n=len(rows), ages=ages, female=fem, cov=cov, left=left)


def age_plot(ds, ages, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 2.2), dpi=160)
    ax.hist(ages, bins=30, color="#5fa58c", edgecolor="#1f4e3d", linewidth=0.6)
    ax.set_xlabel("Age, years", color="#1f3b4d")
    ax.set_ylabel("Subjects", color="#1f3b4d")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors="#1f3b4d")
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    fig.tight_layout()
    fig.savefig(path, transparent=True)
    plt.close(fig)


def social_card(src, out, line):
    """1200x630 link preview: title band over the top of the overview heatmap"""
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib.font_manager as fm
    if not os.path.isfile(src):
        return
    im = Image.open(src).convert("RGB")
    W, H = im.size
    heat = im.crop((0, int(H * 0.038), W, int(H * 0.25)))
    card = Image.new("RGB", (1200, 630), "#1f4e3d")
    d = ImageDraw.Draw(card)
    bold = fm.findfont(fm.FontProperties(family="DejaVu Sans", weight="bold"))
    reg = fm.findfont(fm.FontProperties(family="DejaVu Sans"))
    d.text((56, 44), "BrainFMBench", font=ImageFont.truetype(bold, 64), fill="white")
    d.text((58, 124), "Living benchmark of brain MRI foundation models", font=ImageFont.truetype(reg, 30), fill="#bfd9cf")
    d.text((58, 170), line, font=ImageFont.truetype(reg, 24), fill="#f2c14e")
    panel = Image.new("RGB", (1200, 400), "white")
    h = min(400, int(heat.height * 1160 / heat.width))
    panel.paste(heat.resize((1160, h), Image.LANCZOS), (20, (400 - h) // 2))
    card.paste(panel, (0, 230))
    card.save(out, optimize=True)


# ---------------------------------------------------------------- paper tables
TABLE_TASKS = [("sex", "Sex"), ("age", "Age"), ("bmi", "BMI")]


def overall_order(native, probe):
    """models in leaderboard order: mean relative gain over rows where some model is clearly above chance"""
    sc = {}
    for ds in L.DATASETS:
        for t, _, _ in L.OV_TASKS:
            d = native.get((probe, ds, t), {})
            g = {m: gain(ds, t, v[0]) for m, v in d.items()}
            g = {m: x for m, x in g.items() if x is not None}
            if not g or max(g.values()) < 0.1:
                continue
            best = max(g.values())
            for m, x in g.items():
                sc.setdefault(m, []).append(max(x / best, 0))
    return sorted(sc, key=lambda m: -sum(sc[m]) / len(sc[m]))


def _tex(s):
    for a, b in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#")):
        s = s.replace(a, b)
    return s


def latex_table(native, probe, highlight=None):
    """booktabs table: every model on sex, age and BMI in all five cohorts; the highlighted model goes last, as in a paper"""
    plabel = {"linear": "linear probe", "rf": "random forest probe"}[probe]
    order = overall_order(native, probe)
    fs = [m for m in order if m.startswith("FS ")]
    fm = [m for m in order if m not in fs and m != highlight]
    blocks = [fm, [m for m in fs if m != highlight]] + ([[highlight]] if highlight else [])
    cols = [(ds, t) for ds in L.DATASETS for t, _ in TABLE_TASKS]
    flat = {c for c in cols if not ranked(native, probe, *c)[2]}
    best = {}
    for ds, t in cols:
        d = native.get((probe, ds, t), {})
        vals = sorted({round(v[0], 3 if IS_CLS[t] else 2) for v in d.values()}, reverse=IS_CLS[t])
        best[(ds, t)] = vals[:2]

    def cell(m, ds, t):
        v = native.get((probe, ds, t), {}).get(m)
        if v is None:
            return "--"
        r = round(v[0], 3 if IS_CLS[t] else 2)
        txt = f"{r:.3f}" if IS_CLS[t] else f"{r:.2f}"
        b = best[(ds, t)]
        if r == b[0]:
            return f"\\textbf{{{txt}}}"
        if len(b) > 1 and r == b[1]:
            return f"\\underline{{{txt}}}"
        return txt

    today = datetime.date.today()
    n = len(L.DATASETS)
    out = [f"% BrainFMBench comparison, {plabel}, generated {today.isoformat()} from {SITE}",
           "% Needs \\usepackage{booktabs} and \\usepackage{graphicx}",
           "\\begin{table*}[t]", "\\centering",
           f"\\caption{{Comparison on BrainFMBench~\\cite{{encin2026comparative}}, {plabel} on frozen features. "
           "Held out test mean over 5 seeds, each model on its own preprocessing. "
           "Sex: balanced accuracy ($\\uparrow$); age: mean absolute error in years ($\\downarrow$); "
           "BMI: mean absolute error ($\\downarrow$). Best in bold, second underlined. "
           f"Results as of {today.strftime('%B %-d, %Y')}."
           + (" $^\\dagger$Every model is near chance." if flat else "") + "}",
           "\\label{tab:brainfmbench}",
           "\\resizebox{\\textwidth}{!}{%",
           "\\begin{tabular}{l" + " ccc" * n + "}", "\\toprule",
           " & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{ds}}}" for ds in L.DATASETS) + " \\\\",
           " ".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(n)),
           "Model & " + " & ".join(lab + ("$^\\dagger$" if (ds, t) in flat else "")
                                   for ds in L.DATASETS for t, lab in TABLE_TASKS) + " \\\\"]
    for blk in blocks:
        if not blk:
            continue
        out.append("\\midrule")
        for m in blk:
            out.append(_tex(m) + " & " + " & ".join(cell(m, ds, t) for ds, t in cols) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}", ""]
    return "\n".join(out)


def bibtex():
    """the citation block from the README, so the docs never drift from it"""
    txt = open(os.path.join(REPO, "README.md")).read()
    m = re.search(r"```bibtex\n(.*?)```", txt, re.S)
    return m.group(1).strip() if m else ""


def paper_section(name, slug, native):
    """name None gives the table of every model in leaderboard order, for the leaderboard page"""
    intro = (f"A ready to paste LaTeX table comparing {name} with every model on BrainFMBench, with {name} in the last row. " if name else
             "A ready to paste LaTeX table of every model on BrainFMBench, in leaderboard order. "
             "Each model page has a version with that model in the last row. ")
    lines = ["## Use in your paper", "",
             intro + "It needs `booktabs` and `graphicx`. Numbers are a snapshot of today's leaderboard; please cite BrainFMBench when you use them.", ""]
    for probe, plabel in (("linear", "Linear probe"), ("rf", "Random forest")):
        tex = latex_table(native, probe, highlight=name)
        write(f"assets/tables/{slug}_{probe}.tex", tex)
        lines += [f'=== "{plabel}"', "", "    ```latex"] + ["    " + ln for ln in tex.rstrip().splitlines()] + ["    ```", "",
                  f"    [:material-download: Download {slug}_{probe}.tex](../assets/tables/{slug}_{probe}.tex){{ .md-button download }}", ""]
    lines += ["```bibtex", bibtex(), "```", ""]
    return lines


# ---------------------------------------------------------------- pages
def page_model(name, e, native, allprep):
    preps = list(e["folders"])
    nat = native_prep(name, preps)
    meta = e["metas"][nat]
    folder = e["folders"][nat]
    links = " ".join(f"[{lab}]({url}){{ .md-button }}" for lab, url in L._links(meta))
    desc = f"{name} on BrainFMBench: frozen embedding scores and ranks for sex, age, BMI and mental health on five open brain MRI cohorts."
    lines = ["---", f"description: {json.dumps(desc)}", "---", "", f"# {name}", "", " ".join(chip(p) for p in [nat] + [p for p in preps if p != nat]), ""]
    if meta.get("description"):
        lines += [f"**{meta['description']}**", ""]
    if links:
        lines += [links, ""]
    facts = [("Authors", ", ".join(meta.get("authors") or []) if isinstance(meta.get("authors"), list) else meta.get("authors")),
             ("Submitted by", meta.get("submitted_by")),
             ("Architecture", meta.get("architecture")),
             ("Embedding size", meta.get("embedding_dim")),
             ("Input", meta.get("preprocessing_label") or PREP_CHIP.get(nat, (nat,))[0]),
             ("Note", meta.get("note")),
             ("Source", f"[`models/{folder}`]({GH}/tree/main/models/{folder})")]
    lines += ["| | |", "|---|---|"] + [f"| **{k}** | {v} |" for k, v in facts if v not in (None, "", [])] + [""]

    lines += ["## Results", "",
              f"Held out test, mean ± std over 5 seeds, on this model's own preprocessing. Rank is among all models on their own preprocessing. "
              f"Rows marked *near chance* do not count toward the [ranking](../protocol.md#ranking). Compare every model on the [interactive leaderboard]({SITE}).", ""]
    for probe, plabel in (("linear", "Linear probe"), ("rf", "Random forest")):
        lines += [f'=== "{plabel}"', "", "    | Dataset | Task | Score | Rank |", "    |---|---|---|---|"]
        for ds in L.DATASETS:
            for t, tname, _ in L.OV_TASKS:
                order, d, sig = ranked(native, probe, ds, t)
                if name not in d:
                    continue
                r = order.index(name) + 1
                score = f"{fmt(t, d[name])} <span class='bfb-muted'>{metric_name(t)}</span>"
                rank = f"**{r}** of {len(order)}" if sig else f"<span class='bfb-muted'>{r} of {len(order)}, near chance</span>"
                lines.append(f"    | {ds} | {tname} | {score} | {rank} |")
        lines.append("")

    other = [p for p in preps if p != nat and p in ("turboprep", "turboprep-affine")]
    if other:
        o = other[0]
        lines += ["## Registration sensitivity", "",
                  f"The same model on {PREP_CHIP[o][0]} instead of {PREP_CHIP[nat][0]} registration, linear probe.", "",
                  f"| Dataset | Task | {PREP_CHIP[nat][0]} | {PREP_CHIP[o][0]} |", "|---|---|---|---|"]
        for ds in L.DATASETS:
            for t, tname, _ in L.OV_TASKS[:3]:
                a = allprep.get(("linear", ds, t, nat), {}).get(name)
                b = allprep.get(("linear", ds, t, o), {}).get(name)
                if a and b:
                    lines.append(f"| {ds} | {tname} | {fmt(t, a)} | {fmt(t, b)} |")
        lines.append("")
    lines += paper_section(name, folder, native)
    return "\n".join(lines)


def page_models_index(models, native):
    lines = ["# Models", "",
             "Every model on the leaderboard, run as a frozen encoder on the same subjects. "
             "Cards list each model's own preprocessing and its best linear probe rank on a task with real signal.", "",
             '<div class="grid cards" markdown>', ""]
    for name in sorted(models, key=lambda n: (n.startswith("FS "), n.lower())):
        e = models[name]
        nat = native_prep(name, list(e["folders"]))
        meta = e["metas"][nat]
        best = None
        for ds in L.DATASETS:
            for t, tname, _ in L.OV_TASKS:
                order, d, sig = ranked(native, "linear", ds, t)
                if sig and name in d:
                    r = order.index(name) + 1
                    if best is None or r < best[0]:
                        best = (r, ds, tname)
        slug = e["slug"]
        desc = (meta.get("description") or "").split(";")[0]
        lines += [f"-   **[{name}]({slug}.md)** {chip(nat)}", "", "    ---", "",
                  f"    {desc}", ""]
        if best:
            lines += [f"    :material-trophy-outline: best rank **{best[0]}** on {best[1]} {best[2].lower()}", ""]
    lines += ["</div>", ""]
    return "\n".join(lines)


def page_dataset(ds, info, native, img_rel):
    m = DATASETS[ds]
    ages = info["ages"]
    desc = f"{ds}, the {m['full']}, on BrainFMBench: subjects, ages, labels and the top brain MRI foundation models per task."
    lines = ["---", f"description: {json.dumps(desc)}", "---", "", f"# {ds}: {m['full']}", "",
             f"[Dataset page on Reproducible Brain Charts :material-open-in-new:]({RBC}{m['rbc']}){{ .md-button }}", ""]
    if m["note"]:
        lines += [m["note"], ""]
    lines += [
             '<div class="bfb-stats">',
             f'<div class="bfb-stat"><b>{info["n"]:,}</b><span>subjects</span></div>',
             f'<div class="bfb-stat"><b>{ages.min():.0f}–{ages.max():.0f}</b><span>age range, years</span></div>',
             f'<div class="bfb-stat"><b>{ages.mean():.1f}</b><span>mean age</span></div>',
             f'<div class="bfb-stat"><b>{100 * info["female"] / info["n"]:.0f}%</b><span>female</span></div>',
             "</div>", "",
             f"![Age distribution of {ds}]({img_rel}){{ .bfb-fig }}", "",
             "## Labels", "", "| Task | Subjects with a label |", "|---|---|"]
    for t, tname, _ in L.OV_TASKS:
        if t in info["cov"]:
            extra = f", {info['left']} left handed" if t == "handedness" else ""
            lines.append(f"| {tname} | {info['cov'][t]:,}{extra} |")
    lines += ["", "Inputs: rigid and affine turboprep, CAT12 and raw T1w, plus FreeSurfer features from the Reproducible Brain Charts outputs.", "",
              "## Top models", "", "Linear probe, held out test, each model on its own preprocessing. Only tasks where some model is clearly above chance.", "",
              "| Task | 1st | 2nd | 3rd |", "|---|---|---|---|"]
    any_sig = False
    for t, tname, _ in L.OV_TASKS:
        order, d, sig = ranked(native, "linear", ds, t)
        if not sig:
            continue
        any_sig = True
        cells = [f"{mm}<br><span class='bfb-muted'>{fmt(t, d[mm])}</span>" for mm in order[:3]]
        lines.append(f"| {tname} <span class='bfb-muted'>{metric_name(t)}</span> | " + " | ".join(cells) + " |")
    if not any_sig:
        lines = lines[:-2] + ["Every model is near chance on every task here for now."]
    lines += ["", "## Cite", "", f"- {m['cite']} [doi]({m['doi']})", f"- {RBC_CITE}", "",
              f"Labels come from [Reproducible Brain Charts](https://reprobrainchart.github.io/) under CC BY-NC-SA 4.0, see [`labels/README.md`]({GH}/blob/main/labels/README.md).", ""]
    return "\n".join(lines)


def page_datasets_index(infos):
    lines = ["# Datasets", "",
             "Five cohorts from [Reproducible Brain Charts](https://reprobrainchart.github.io/), shared openly without a data use agreement. "
             "Every subject passes RBC T1w quality control, and every model sees the same subjects.", "",
             '<div class="grid cards" markdown>', ""]
    for ds, m in DATASETS.items():
        i = infos[ds]
        lines += [f"-   **[{ds}]({ds.lower()}.md)** · {m['full']}", "", "    ---", "",
                  f"    {i['n']:,} subjects · ages {i['ages'].min():.0f}–{i['ages'].max():.0f} · [RBC page]({RBC}{m['rbc']})", ""]
    lines += ["</div>", "", "See also [Tasks](../tasks.md) and [Preprocessing](../preprocessing.md).", ""]
    return "\n".join(lines)


def page_results(native):
    lines = ["# Leaderboard", "",
             f"[Open the interactive leaderboard :material-open-in-new:]({SITE}){{ .md-button .md-button--primary }} "
             "[Full tables :material-table:](https://github.com/epics-lab/BrainFMBench/blob/main/LEADERBOARD.md){ .md-button }", "",
             "## Overview", "",
             "Linear probe, held out test, every model on its own preprocessing. Color is the improvement over chance relative to the best model in the row; "
             "pale rows are near chance. Columns are ordered automatically, see [Ranking](../protocol.md#ranking).", "",
             "![Overview heatmap](../assets/fig/leaderboard_overview.png){ .bfb-fig }", "",
             "## Top three per task", "", "Linear probe, tasks where some model is clearly above chance.", "",
             "| Dataset | Task | 1st | 2nd | 3rd |", "|---|---|---|---|---|"]
    for ds in L.DATASETS:
        for t, tname, _ in L.OV_TASKS:
            order, d, sig = ranked(native, "linear", ds, t)
            if sig:
                lines.append(f"| {ds} | {tname} | " + " | ".join(order[:3] + [""] * (3 - len(order[:3]))) + " |")
    lines += ["", "## Distributions", "",
              "Boxes are 5 fold cross validation over 5 seeds, points are the held out results.", "",
              "![Boxplots](../assets/fig/leaderboard_boxplots.png){ .bfb-fig }", ""]
    lines += paper_section(None, "all", native)
    return "\n".join(lines)


def page_index(models, infos, native):
    n_models = len(models)
    n_fm = sum(1 for m in models if not m.startswith("FS "))
    n_sub = sum(i["n"] for i in infos.values())
    today = datetime.date.today().strftime("%B %-d, %Y")
    lines = [
        "---", "hide:", "  - navigation", "  - toc", "---", "",
        '<div class="bfb-hero" markdown>',
        "# BrainFMBench",
        "A living benchmark for structural brain MRI foundation models. Every model runs frozen on the same subjects, "
        "and the same simple probes read sex, age, BMI and mental health from its features.",
        "",
        '<div class="bfb-actions" markdown>',
        f"[Leaderboard :material-arrow-right:]({SITE}){{ .md-button .md-button--primary }}",
        "[Submit a model](submit/index.md){ .md-button }",
        "[Paper](https://papers.miccai.org/miccai-2026-sat/paper/MedAGI_025.pdf){ .md-button }",
        "</div>", "</div>", "",
        '<div class="bfb-stats">',
        f'<div class="bfb-stat"><b>{n_fm}</b><span>foundation models</span></div>',
        f'<div class="bfb-stat"><b>{n_models - n_fm}</b><span>FreeSurfer baselines</span></div>',
        f'<div class="bfb-stat"><b>{len(infos)}</b><span>datasets</span></div>',
        f'<div class="bfb-stat"><b>{n_sub:,}</b><span>subjects</span></div>',
        f'<div class="bfb-stat"><b>{len(L.OV_TASKS)}</b><span>tasks</span></div>',
        '<div class="bfb-stat"><b>2</b><span>probes</span></div>',
        "</div>", "",
        '<div class="grid cards" markdown>', "",
        "-   :material-chart-box-outline:{ .lg } **[Leaderboard](results/index.md)**", "", "    ---", "",
        "    The overview heatmap, the top three per task and the full distributions.", "",
        "-   :material-cube-outline:{ .lg } **[Models](models/index.md)**", "", "    ---", "",
        "    One page per model: links, input, embedding size, and every score with its rank.", "",
        "-   :material-account-group-outline:{ .lg } **[Datasets](datasets/index.md)**", "", "    ---", "",
        "    Five Reproducible Brain Charts cohorts across three countries, with ages, labels and citations.", "",
        "-   :material-source-pull:{ .lg } **[Submit a model](submit/index.md)**", "", "    ---", "",
        "    Three files and a pull request. Extraction runs on our cluster, scoring runs in public.", "",
        "-   :material-scale-balance:{ .lg } **[Protocol](protocol.md)**", "", "    ---", "",
        "    Splits, seeds, probes, metrics, and how the ranking ignores rows at chance.", "",
        "-   :material-tag-outline:{ .lg } **[Releases](releases.md)**", "", "    ---", "",
        "    Quarterly releases with every change listed. Next release planned for March 2027.", "",
        "</div>", "",
        "## The leaderboard today", "",
        f"[![Overview heatmap](assets/fig/leaderboard_overview.png){{ .bfb-fig }}]({SITE})", "",
    ]
    lines += [f"<p class='bfb-muted'>Numbers on this site are rebuilt from the repository on every update. Last build {today}.</p>", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------- main
def main():
    shutil.rmtree(OUT, ignore_errors=True)
    shutil.copytree(SRC, OUT)
    os.makedirs(os.path.join(OUT, "assets", "fig"), exist_ok=True)
    for f in ("leaderboard_overview.png", "leaderboard_boxplots.png"):
        if os.path.isfile(os.path.join(REPO, f)):
            shutil.copy(os.path.join(REPO, f), os.path.join(OUT, "assets", "fig", f))

    native, allprep = build_tables(L.load_rows())
    models = load_models()
    for name, e in models.items():
        e["slug"] = e["folders"][native_prep(name, list(e["folders"]))]
    infos = {ds: labels_info(ds) for ds in DATASETS}

    write("index.md", page_index(models, infos, native))
    write("results/index.md", page_results(native))
    write("models/index.md", page_models_index(models, native))
    for name, e in models.items():
        write(f"models/{e['slug']}.md", page_model(name, e, native, allprep))
    write("datasets/index.md", page_datasets_index(infos))
    for ds, info in infos.items():
        age_plot(ds, info["ages"], os.path.join(OUT, "assets", "fig", f"age_{ds}.png"))
        write(f"datasets/{ds.lower()}.md", page_dataset(ds, info, native, f"../assets/fig/age_{ds}.png"))

    ref = open(os.path.join(REPO, "CONTRIBUTING.md")).read()
    ref = ref.replace("# Contributing a model to BrainFMBench", "# Reference", 1)
    ref = ref.replace("[CHANGELOG.md](CHANGELOG.md)", "[Releases](../releases.md)")
    ref = ref.replace("See `example-submission/`", f"See [`example-submission/`]({GH}/tree/main/example-submission)")
    write("submit/reference.md", ref)
    cl = open(os.path.join(REPO, "CHANGELOG.md")).read().replace("# Changelog", "# Releases", 1)
    cl = cl.replace("(CONTRIBUTING.md#versioning)", "(protocol.md#versioning)")
    write("releases.md", cl)

    n_fm = sum(1 for m in models if not m.startswith("FS "))
    social_card(os.path.join(REPO, "leaderboard_overview.png"), os.path.join(OUT, "assets", "social.png"),
                f"{n_fm} foundation models · {len(infos)} open cohorts · {sum(i['n'] for i in infos.values()):,} subjects · {len(L.OV_TASKS)} tasks")

    cfg = open(os.path.join(REPO, "mkdocs.yml")).read()
    cfg = cfg.replace("\nextra:\n", "\nextra:\n  jsonld: " + json.dumps(L.jsonld(datetime.date.today().isoformat())) + "\n", 1)
    mnav = "".join(f"      - {n}: models/{models[n]['slug']}.md\n"
                   for n in sorted(models, key=lambda n: (n.startswith("FS "), n.lower())))
    dnav = "".join(f"      - {ds}: datasets/{ds.lower()}.md\n" for ds in DATASETS)
    cfg = cfg.replace("      - models/index.md\n", "      - models/index.md\n" + mnav, 1)
    cfg = cfg.replace("      - datasets/index.md\n", "      - datasets/index.md\n" + dnav, 1)
    open(os.path.join(REPO, "mkdocs.gen.yml"), "w").write(cfg)
    print(f"docs source built: {len(models)} models, {len(infos)} datasets -> {OUT}")


if __name__ == "__main__":
    main()
