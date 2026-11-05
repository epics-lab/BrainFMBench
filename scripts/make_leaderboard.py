#!/usr/bin/env python3

import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error
from sklearn.dummy import DummyClassifier, DummyRegressor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASETS = ["HBN", "NKI"]
SEEDS = [0, 1, 2, 3, 42]

# model display order + exact paper colors
MODEL_ORDER = [
    "AnatCL (Global)", "AnatCL (Local)", "BrainIAC", "SwinBrain",
    "3D-Neuro-SimCLR", "Dinov3D-neuro", "un-CNN", "FS Schaefer", "FS aparc",
]
MODEL_COLORS = {
    "AnatCL (Global)": "#BA68C8",
    "AnatCL (Local)":  "#4A148C",
    "BrainIAC":        "#E67E22",
    "SwinBrain":       "#1ABC9C",
    "3D-Neuro-SimCLR": "#E91E63",
    "Dinov3D-neuro":   "#7CB342",
    "un-CNN":          "#1E88E5",
    "FS Schaefer":     "#959EA7",
    "FS aparc":        "#22282d",
}
DISPLAY_NAMES = {
    "AnatCL (Global)": "AnatCL\n(Global)",
    "AnatCL (Local)":  "AnatCL\n(Local)",
    "BrainIAC":        "BrainIAC",
    "SwinBrain":       "SwinBrain",
    "3D-Neuro-SimCLR": "3D-Neuro\nSimCLR",
    "Dinov3D-neuro":   "Dinov3D\nneuro",
    "un-CNN":          "un-CNN",
    "FS Schaefer":     "FS:\nSchaefer",
    "FS aparc":        "FS:\naparc",
}

# (row_label, task, is_cls, metric_ylabel, box_key, dot_key)
BOX_ROWS = [
    ("Sex Classification", "sex", True,  "Balanced Accuracy", "cv_acc_values", "test_acc_seeds"),
    ("Age Prediction",     "age", False, "MAE (years)",       "cv_mae_values", "test_mae_seeds"),
    ("BMI Prediction",     "bmi", False, "MAE",               "cv_mae_values", "test_mae_seeds"),
]

SEP_COLOR = "black"; SEP_LW = 1.5
XTICK_FS = 16; YTICK_FS = 11; YTITLE_FS = 18; DS_FS = 24; LEGEND_FS = 13


def fmt(mean, std, places=3):
    if mean is None:
        return "-"
    return f"{mean:.{places}f} ± {std:.{places}f}"


def load_rows():
    rows = []
    for path in sorted(glob.glob(os.path.join(REPO, "models", "*", "results.json"))):
        try:
            rows.extend(json.load(open(path)))
        except Exception as e:
            print(f"skip {path}: {e}")
    return rows


def dummy_baseline(dataset, task, is_cls):
    lab_path = os.path.join(REPO, "labels", dataset + ".csv")
    if not os.path.isfile(lab_path):
        return None
    lab = pd.read_csv(lab_path)
    if task not in lab.columns:
        return None
    y = lab[task].to_numpy()
    y = y[~pd.isna(y)]
    if len(y) == 0:
        return None
    vals = []
    for seed in SEEDS:
        if is_cls:
            y2 = y.astype(int)
            cv, te = train_test_split(np.arange(len(y2)), test_size=int(0.1 * len(y2)),
                                      stratify=y2, random_state=seed)
            d = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(cv), 1)), y2[cv])
            vals.append(balanced_accuracy_score(y2[te], d.predict(np.zeros((len(te), 1)))))
        else:
            y2 = y.astype(float)
            cv, te = train_test_split(np.arange(len(y2)), test_size=int(0.1 * len(y2)),
                                      random_state=seed)
            d = DummyRegressor(strategy="mean").fit(np.zeros((len(cv), 1)), y2[cv])
            vals.append(mean_absolute_error(y2[te], d.predict(np.zeros((len(te), 1)))))
    return float(np.mean(vals))



import matplotlib.cm as _cm
import matplotlib.colors as _mcolors

# All models present across results (known keep paper colors; new ones get
# auto-assigned distinct colors + appended to the order, so a contributor's
# model always appears in the figure and legend).
def _all_models_in(table):
    return {m for (m, _ds) in table.keys()}

def _png_v():
    import hashlib
    try:
        return hashlib.sha1(open(os.path.join(REPO, "leaderboard_boxplots.png"), "rb").read()).hexdigest()[:8]
    except OSError:
        return "0"


def _order_and_colors(table):
    present = _all_models_in(table)
    known = [m for m in MODEL_ORDER if m in present]
    unknown = sorted(m for m in present if m not in MODEL_COLORS)
    colors = dict(MODEL_COLORS)
    cmap = matplotlib.colormaps["tab20"]
    pool = [_mcolors.to_hex(matplotlib.colormaps["tab10"](k)) for k in range(10)] + [_mcolors.to_hex(cmap(k)) for k in range(1, 20, 2)]
    def _gap(c):
        r = _mcolors.to_rgb(c)
        return min(sum((x - y) ** 2 for x, y in zip(r, _mcolors.to_rgb(u)))
                   for u in colors.values())
    for m in unknown:
        free = [c for c in pool if c not in colors.values()]
        strong = [c for c in free if c in pool[:10]]
        pick = max(strong or free, key=_gap)
        colors[m] = pick
    return known + unknown, colors


def _collect_group(table, dataset, box_key, dot_key, task, order, colors):
    positions, box_data, box_colors, heldout_data, xpos, xlbl = [], [], [], [], [], []
    pos = 1
    for m in order:
        row = table.get((m, dataset), {}).get(task)
        if not row:
            continue
        vals = row.get(box_key)
        if not vals:
            continue
        hvals = row.get(dot_key) or []
        positions.append(pos); box_data.append(np.asarray(vals, float))
        box_colors.append(colors[m])
        heldout_data.append(np.asarray(hvals, float))
        xpos.append(pos); xlbl.append(DISPLAY_NAMES.get(m, m.replace(' ', chr(10))))
        pos += 1
    return positions, box_data, box_colors, heldout_data, xpos, xlbl


def _draw_boxes_core(ax, positions, box_data, box_colors, heldout_data, xpos, xlbl, bv, ylim):
    bp = ax.boxplot(box_data, positions=positions, widths=0.6,
                    patch_artist=True, showfliers=False,
                    boxprops={"linewidth": 1.2},
                    whiskerprops={"linewidth": 1.0},
                    capprops={"linewidth": 1.0},
                    medianprops={"color": "black", "linewidth": 1.5})
    for patch, col in zip(bp["boxes"], box_colors):
        patch.set_facecolor(col); patch.set_edgecolor(col); patch.set_alpha(0.55)

    for px, hvals, col in zip(positions, heldout_data, box_colors):
        if len(hvals) == 0:
            continue
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(hvals))
        ax.scatter(px + jitter, hvals, color=col, edgecolors="black",
                   linewidths=0.6, s=30, zorder=5, alpha=0.85)

    if bv is not None and np.isfinite(bv):
        ax.axhline(bv, color="#AAAAAA", linestyle=":", linewidth=2, zorder=0)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.3)
    for sp in ax.spines.values():
        sp.set_linewidth(SEP_LW); sp.set_color(SEP_COLOR)

    ax.set_xticks(xpos)
    ax.set_xticklabels(xlbl, rotation=90, fontsize=XTICK_FS, ha="center")
    ax.tick_params(axis="x", pad=3)


def make_boxplots(table, out_png):
    order, colors = _order_and_colors(table)
    fig = plt.figure(figsize=(16, 20))
    outer = GridSpec(3, 1, figure=fig, hspace=0.55,
                     left=0.10, right=0.97, top=0.94, bottom=0.12)

    # shared sex y-limits
    _sex_vals, _sex_bvs = [], []
    for ds in DATASETS:
        p, d, c, h, xp, xl = _collect_group(table, ds, "cv_acc_values", "test_acc_seeds", "sex", order, colors)
        if d:
            _sex_vals.extend(np.concatenate(d))
            for hv in h:
                if len(hv):
                    _sex_vals.extend(hv)
            bv = dummy_baseline(ds, "sex", True)
            if bv is not None:
                _sex_bvs.append(bv)
    if _sex_vals:
        gmin = min(min(_sex_vals), min(_sex_bvs) if _sex_bvs else min(_sex_vals))
        gmax = max(_sex_vals)
        dr = gmax - gmin if gmax > gmin else 0.01
        sex_ylim = (gmin - dr * 0.08, gmax + dr * 0.20)
    else:
        sex_ylim = (0.45, 0.95)

    shared_sex_ax = None
    wratios = [1, 1]
    for row_i, (row_label, task, is_cls, metric_ylabel, box_key, dot_key) in enumerate(BOX_ROWS):
        inner = GridSpecFromSubplotSpec(1, len(DATASETS), subplot_spec=outer[row_i],
                                        wspace=0.05, width_ratios=wratios)
        for col_i, ds in enumerate(DATASETS):
            p, d, c, h, xp, xl = _collect_group(table, ds, box_key, dot_key, task, order, colors)

            if is_cls:
                if col_i == 0:
                    ax = fig.add_subplot(inner[col_i]); shared_sex_ax = ax
                else:
                    ax = fig.add_subplot(inner[col_i], sharey=shared_sex_ax)
            else:
                ax = fig.add_subplot(inner[col_i])

            if not d:
                ax.set_visible(False); continue

            bv = dummy_baseline(ds, task, is_cls)
            if is_cls:
                ylim = sex_ylim
            else:
                vals = list(np.concatenate(d))
                for hv in h:
                    if len(hv):
                        vals += list(hv)
                if bv is not None and np.isfinite(bv):
                    vals.append(bv)
                lo, hi = min(vals), max(vals)
                rng = hi - lo if hi > lo else 1.0
                ylim = (max(0, lo - rng * 0.08), hi + rng * 0.20)

            _draw_boxes_core(ax, p, d, c, h, xp, xl, bv=bv, ylim=ylim)
            ax.set_xlim(min(p) - 0.7, max(p) + 0.7)

            if row_i == 0:
                ax.text(0.5, 1.04, ds, transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=DS_FS, fontweight="bold")

            if col_i == 0:
                ax.tick_params(axis="y", labelsize=YTICK_FS, left=True, labelleft=True)
                ax.set_ylabel(row_label + "\n" + metric_ylabel,
                              fontsize=YTITLE_FS, weight="bold", labelpad=8, linespacing=1.5)
            else:
                ax.tick_params(axis="y", labelsize=YTICK_FS,
                               left=True, labelleft=(not is_cls))

    legend_handles = [mpatches.Patch(color=colors[m], label=m) for m in order]
    legend_handles += [
        Line2D([0], [0], color="#AAAAAA", linewidth=2, linestyle=":", label="Chance / dummy"),
        Line2D([0], [0], color="white", marker="o", markerfacecolor="gray",
               markeredgecolor="black", markersize=7, linestyle="None", label="Held-out seeds"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5, fontsize=LEGEND_FS,
               frameon=True, bbox_to_anchor=(0.5, 0.03),
               handlelength=2.2, handletextpad=0.6, columnspacing=1.4)

    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


PAGES_URL = "https://www.epics-lab.ca/BrainFMBench/"


def _box_stats(v):
    # same quartiles and 1.5 IQR whiskers as matplotlib's boxplot
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    iqr = q3 - q1
    return q1, med, q3, v[v >= q1 - 1.5 * iqr].min(), v[v <= q3 + 1.5 * iqr].max()


def make_interactive(table, out_html, extra_html=""):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    order, colors = _order_and_colors(table)
    ncol = len(DATASETS)
    fig = make_subplots(rows=len(BOX_ROWS), cols=ncol,
                        subplot_titles=DATASETS + [""] * ncol * (len(BOX_ROWS) - 1),
                        horizontal_spacing=0.05, vertical_spacing=0.09)
    in_legend = set()

    for r, (row_label, task, is_cls, metric_ylabel, box_key, dot_key) in enumerate(BOX_ROWS, 1):
        for c, ds in enumerate(DATASETS, 1):
            cats = []
            for m in order:
                row = table.get((m, ds), {}).get(task)
                if not row or not row.get(box_key):
                    continue
                lbl = DISPLAY_NAMES.get(m, m).replace("\n", "<br>")
                cats.append(lbl)
                rgb = _mcolors.to_rgb(colors[m])
                fill = "rgba(%d,%d,%d,0.55)" % tuple(int(255 * x) for x in rgb)
                q1, med, q3, lo, hi = _box_stats(np.asarray(row[box_key], float))
                fig.add_trace(go.Box(
                    x=[lbl], q1=[q1], median=[med], q3=[q3], lowerfence=[lo], upperfence=[hi],
                    name=m, legendgroup=m, showlegend=m not in in_legend,
                    fillcolor=fill, line=dict(color=colors[m], width=1.5),
                ), row=r, col=c)
                in_legend.add(m)

                hvals = row.get(dot_key) or []
                if hvals:
                    fig.add_trace(go.Box(
                        x=[lbl] * len(hvals), y=hvals, name=m, legendgroup=m, showlegend=False,
                        boxpoints="all", jitter=0.5, pointpos=0, hoveron="points",
                        fillcolor="rgba(0,0,0,0)", line=dict(color="rgba(0,0,0,0)"),
                        marker=dict(color=colors[m], size=8, line=dict(color="black", width=1)),
                        text=[f"seed {s}" for s in SEEDS[:len(hvals)]],
                        hovertemplate=f"<b>{m}</b><br>held out test, %{{text}}<br>"
                                      f"{metric_ylabel}: %{{y:.3f}}<extra></extra>",
                    ), row=r, col=c)

            fig.update_xaxes(categoryorder="array", categoryarray=cats, row=r, col=c)
            bv = dummy_baseline(ds, task, is_cls)
            if bv is not None and np.isfinite(bv):
                fig.add_hline(y=bv, line=dict(color="#AAAAAA", dash="dot", width=2), row=r, col=c)
            if c == 1:
                fig.update_yaxes(title_text=f"<b>{row_label}</b><br>{metric_ylabel}", row=r, col=c)
            elif is_cls:
                first = (r - 1) * ncol + 1
                fig.update_yaxes(matches="y" if first == 1 else f"y{first}", row=r, col=c)

    fig.update_annotations(font_size=20)
    fig.update_layout(
        template="plotly_white", boxmode="overlay", height=430 * len(BOX_ROWS),
        title=dict(text="BrainFMBench leaderboard<br><sup>Boxes: 5 fold CV over 5 seeds. "
                        "Points: held out test per seed. Dotted line: chance or dummy baseline. "
                        "Click legend entries to hide or show models.</sup>"),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.06, yanchor="top"),
        margin=dict(l=90, r=20, t=110, b=60),
    )

    try:
        static = go.Figure(fig)
        static.update_layout(title_text="BrainFMBench leaderboard", width=1300)
        static.write_image(os.path.join(REPO, "leaderboard_boxplots.png"), scale=2)
        print("wrote plotly png")
    except Exception as e:
        print(f"plotly png export failed, keeping matplotlib png: {e}")
    plot = fig.to_html(full_html=False, include_plotlyjs="cdn",
                       config={"responsive": True, "displaylogo": False})
    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    with open(out_html, "w") as f:
        f.write(
            "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            "<title>BrainFMBench Leaderboard</title>"
            "<style>body{font-family:system-ui,sans-serif;max-width:1400px;margin:0 auto;padding:16px;color:#222}table{border-collapse:collapse;margin:8px 0 20px;font-size:14px}th,td{border:1px solid #ddd;padding:6px 10px;text-align:left}th{background:#f4f6f8}tr:nth-child(even) td{background:#fafbfc}.sub{color:#666;font-weight:normal;font-size:0.9em}.note{max-width:900px;color:#444}th.on{background:#e3e9f0}.arr{color:#889;font-size:0.85em;margin-left:4px}th.on .arr{color:#1a5fb4}th[data-col]:hover{background:#eaeef3}tr.grp th{background:#e8edf2;text-align:left;font-size:13px;letter-spacing:.02em}th.ds{text-align:center}td.best{font-weight:600}</style>"
            "</head><body>\n"
            "<p><a href=\"https://github.com/epics-lab/BrainFMBench\">epics-lab/BrainFMBench</a>"
            " &middot; auto generated from <code>models/*/results.json</code></p>\n"
            + plot + "\n" + extra_html + "\n</body></html>\n")
    print(f"Wrote {out_html}")


TASK_TABLES = [
    ("sex", "Sex classification", "balanced accuracy, higher is better", "test_acc", 3, True),
    ("age", "Age prediction", "MAE in years, lower is better", "test_mae", 2, False),
    ("bmi", "BMI prediction", "MAE, lower is better", "test_mae", 2, False),
]


def load_meta():
    import yaml
    meta = {}
    for path in sorted(glob.glob(os.path.join(REPO, "models", "*", "model.yaml"))):
        try:
            y = yaml.safe_load(open(path)) or {}
        except Exception as e:
            print(f"skip {path}: {e}")
            continue
        if y.get("name"):
            meta[y["name"]] = y
    return meta


def _links(y):
    out = []
    for key, label in (("paper", "paper"), ("model_card", "model card"), ("code", "code")):
        v = str(y.get(key) or "").split()[0] if y.get(key) else ""
        if v.startswith("http") and "private" not in str(y.get(key)):
            out.append((label, v))
    return out


PROBES = [("linear", "Linear probe"), ("rf", "RandomForest probe")]
TASK_SPEC = {task: dict(key=key, places=places, higher=higher, title=title)
             for task, title, unit, key, places, higher in TASK_TABLES}
SHORT = {"sex": "Sex (acc ↑)", "age": "Age (MAE ↓)", "bmi": "BMI (MAE ↓)"}


def build_entries(rows, meta):
    """One leaderboard row per (probe, model, preprocessing): {(dataset, task): (mean, text)}."""
    ent = {}
    for r in rows:
        spec = TASK_SPEC.get(r.get("task"))
        if spec is None:
            continue
        m = r["model"]
        prep = r.get("preprocessing") or meta.get(m, {}).get("preprocessing", "turboprep")
        mu = r.get(spec["key"] + "_mean")
        if mu is None:
            continue
        cell = (mu, fmt(mu, r.get(spec["key"] + "_std"), spec["places"]))
        ent.setdefault((r.get("probe", "rf"), m, prep), {})[(r["dataset"], r["task"])] = cell
    return ent


def _cols():
    return [(ds, t) for ds in DATASETS for t, *_ in TASK_TABLES]


def _best(ent):
    """Best mean per (probe, dataset, task)."""
    best = {}
    for (probe, _m, _p), cells in ent.items():
        for (ds, t), (mu, _txt) in cells.items():
            k = (probe, ds, t)
            hi = TASK_SPEC[t]["higher"]
            if k not in best or (mu > best[k] if hi else mu < best[k]):
                best[k] = mu
    return best


def _probe_keys(ent, probe):
    return sorted((k for k in ent if k[0] == probe), key=lambda k: (k[1].lower(), k[2]))


def task_tables_md(ent, meta):
    cols, best = _cols(), _best(ent)
    lines = ["| Probe | Model | Preprocessing | " + " | ".join(f"{ds} {SHORT[t]}" for ds, t in cols) + " |",
             "|---|---|---|" + "---|" * len(cols)]
    for probe, label in PROBES:
        for k in _probe_keys(ent, probe):
            _, m, prep = k
            name = m + (" †" if meta.get(m, {}).get("note") else "")
            vals = []
            for ds, t in cols:
                mu, txt = ent[k].get((ds, t), (None, "-"))
                vals.append(f"**{txt}**" if mu is not None and mu == best.get((probe, ds, t)) else txt)
            lines.append(f"| {label.split()[0]} | {name} | {prep} | " + " | ".join(vals) + " |")
    return lines + ["", "_Bold: best per column within each probe._", ""]


def notes_md(meta):
    return [f"† **{n}:** {y['note'].strip()}" for n, y in sorted(meta.items()) if y.get("note")]


def models_md(meta, present):
    lines = ["| Model | Preprocessing | Dim | Links |", "|---|---|---|---|"]
    for n in sorted(present):
        y = meta.get(n, {})
        links = " · ".join(f"[{l}]({u})" for l, u in _links(y)) or "-"
        lines.append(f"| {n} | {(y.get('preprocessing_label') or y.get('preprocessing', '-'))} | {y.get('embedding_dim', '-')} | {links} |")
    return lines


def tables_html(ent, meta, present):
    import html as _h
    e = _h.escape
    cols, best = _cols(), _best(ent)
    ncol = 2 + len(cols)
    ntask = len(TASK_TABLES)
    out = ["<h2>Leaderboard</h2>",
           "<p class=\"sub\">Held out test, mean ± std over 5 seeds. Rows are grouped by probe; "
           "click a column to sort within each probe, click again to reverse. Bold marks the best "
           "value per column within a probe. There is no single overall score.</p>",
           "<table id=\"lb\"><thead><tr>"
           "<th rowspan=\"2\" data-col=\"0\" data-dir=\"1\">Model <span class=\"arr\">&#8645;</span></th>"
           "<th rowspan=\"2\" data-col=\"1\" data-dir=\"1\">Preprocessing <span class=\"arr\">&#8645;</span></th>"
           + "".join(f"<th colspan=\"{ntask}\" class=\"ds\">{e(ds)}</th>" for ds in DATASETS) + "</tr><tr>"
           + "".join(f"<th data-col=\"{k}\" data-dir=\"{-1 if TASK_SPEC[t]['higher'] else 1}\">"
                     f"{e(SHORT[t])} <span class=\"arr\">&#8645;</span></th>"
                     for k, (ds, t) in enumerate(cols, 2)) + "</tr></thead>"]
    for probe, label in PROBES:
        keys = _probe_keys(ent, probe)
        if not keys:
            continue
        out.append(f"<tbody><tr class=\"grp\"><th colspan=\"{ncol}\">{e(label)}</th></tr>")
        for k in keys:
            _, m, prep = k
            name = e(m) + (" &dagger;" if meta.get(m, {}).get("note") else "")
            tds = []
            for ds, t in cols:
                mu, txt = ent[k].get((ds, t), (None, "-"))
                key = "" if mu is None else repr(mu)
                b = mu is not None and mu == best.get((probe, ds, t))
                tds.append(f"<td data-v=\"{key}\"{' class=\"best\"' if b else ''}>{e(txt)}</td>")
            out.append(f"<tr><td data-v=\"{e(m.lower())}\">{name}</td>"
                       f"<td data-v=\"{e(prep)}\">{e(prep)}</td>{''.join(tds)}</tr>")
        out.append("</tbody>")
    out.append("</table>")
    out.append("""<script>
document.querySelectorAll('#lb th[data-col]').forEach(function (th) {
  th.style.cursor = 'pointer';
  th.addEventListener('click', function () {
    var c = +th.dataset.col, d = +th.dataset.dir;
    document.querySelectorAll('#lb tbody').forEach(function (tb) {
      var rows = Array.from(tb.rows).filter(function (r) { return !r.classList.contains('grp'); });
      rows.sort(function (a, b) {
        var x = a.cells[c].dataset.v, y = b.cells[c].dataset.v;
        if (x === '') return 1; if (y === '') return -1;
        var nx = parseFloat(x), ny = parseFloat(y);
        return (isNaN(nx) ? x.localeCompare(y) : nx - ny) * d;
      });
      rows.forEach(function (r) { tb.appendChild(r); });
    });
    document.querySelectorAll('#lb th').forEach(function (h) {
      h.classList.remove('on');
      var a = h.querySelector('.arr'); if (a) a.innerHTML = '&#8645;';
    });
    th.classList.add('on'); th.querySelector('.arr').innerHTML = d > 0 ? '&#9650;' : '&#9660;';
    th.dataset.dir = -d;
  });
});
</script>""")
    for n, y in sorted(meta.items()):
        if y.get("note"):
            out.append(f"<p class=\"note\">&dagger; <b>{e(n)}:</b> {e(y['note'].strip())}</p>")
    out.append("<h2>Models</h2><table><thead><tr><th>Model</th><th>Preprocessing</th>"
               "<th>Dim</th><th>Links</th></tr></thead><tbody>")
    for n in sorted(present):
        y = meta.get(n, {})
        links = " &middot; ".join(f"<a href=\"{e(u)}\">{e(l)}</a>" for l, u in _links(y)) or "-"
        out.append(f"<tr><td>{e(n)}</td><td>{e(str((y.get('preprocessing_label') or y.get('preprocessing', '-'))))}</td>"
                   f"<td>{e(str(y.get('embedding_dim', '-')))}</td><td>{links}</td></tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def main():
    rows = load_rows()
    table = {}
    for r in rows:
        if r.get("probe", "rf") != "rf":   # figures show the RandomForest probe
            continue
        table.setdefault((r["model"], r["dataset"]), {})[r["task"]] = r

    display = []
    for (model, dataset), tasks in table.items():
        sex = tasks.get("sex", {}); age = tasks.get("age", {}); bmi = tasks.get("bmi", {})
        display.append({
            "model": model, "dataset": dataset,
            "sex_acc": sex.get("test_acc_mean"),
            "sex_cell": fmt(sex.get("test_acc_mean"), sex.get("test_acc_std")),
            "age_cell": fmt(age.get("test_mae_mean"), age.get("test_mae_std"), 2),
            "bmi_cell": fmt(bmi.get("test_mae_mean"), bmi.get("test_mae_std"), 2),
        })
    display.sort(key=lambda d: (d["sex_acc"] is not None, d["sex_acc"] or 0), reverse=True)

    fig_path = os.path.join(REPO, "leaderboard_boxplots.png")
    try:
        make_boxplots(table, fig_path)
        fig_ok = True
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"figure generation failed: {e}")
        fig_ok = False

    meta = load_meta()
    present = {m for (m, _ds) in table}
    ent = build_entries(rows, meta)
    try:
        make_interactive(table, os.path.join(REPO, "site", "index.html"),
                         tables_html(ent, meta, present))
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"interactive figure failed: {e}")

    lines = [
        "# BrainFMBench Leaderboard", "",
        "Downstream probing of frozen features (RandomForest, held-out test).",
        "Sex = balanced accuracy (higher is better); Age / BMI = MAE (lower is better).", "",
    ]
    if fig_ok:
        lines += [
            f"[![Leaderboard boxplots](leaderboard_boxplots.png?v={_png_v()})]({PAGES_URL})", "",
            "_Boxes show 5-fold cross-validation across 5 seeds (25 values); "
            "overlaid points show held-out test results for the same 5 seeds._", "",
            f"**[Interactive version]({PAGES_URL})**", "",
        ]
    lines += [f"_Held out test, mean ± std over 5 seeds. There is no single overall score; "
              f"sort by task on the [interactive site]({PAGES_URL})._", ""]
    lines += task_tables_md(ent, meta) + notes_md(meta) + ["", "### Models", ""] + models_md(meta, present)
    lines += ["", "_Auto-generated from `models/*/results.json`. Do not edit by hand._"]

    with open(os.path.join(REPO, "LEADERBOARD.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote LEADERBOARD.md  ({len(ent)} leaderboard rows)")

    # inject leaderboard (figure + table) into README.md between markers
    readme = os.path.join(REPO, "README.md")
    if os.path.isfile(readme):
        blk = ["<!-- LEADERBOARD:START -->", ""]
        if fig_ok:
            blk += [f"[![Leaderboard boxplots](leaderboard_boxplots.png?v={_png_v()})]({PAGES_URL})", "",
                    f"**[Interactive version]({PAGES_URL})**", ""]
        blk += [f"_Held out test, mean ± std over 5 seeds. There is no single overall score; "
                f"sort by task on the [interactive site]({PAGES_URL})._", ""]
        blk += task_tables_md(ent, meta) + notes_md(meta)
        blk += ["", "Model details, papers and model cards: [`LEADERBOARD.md`](LEADERBOARD.md#models).",
                "", "<!-- LEADERBOARD:END -->"]
        blk_text = "\n".join(blk)
        c = open(readme).read()
        a, b = "<!-- LEADERBOARD:START -->", "<!-- LEADERBOARD:END -->"
        if a in c and b in c:
            c = c.split(a)[0] + blk_text + c.split(b)[1]
            open(readme, "w").write(c)
            print(f"Injected leaderboard into {readme}")
            import glob, re
            root = os.path.dirname(os.path.abspath(readme))
            n = len(glob.glob(os.path.join(root, "models", "*", "results.json")))
            c2 = re.sub(r"_Models-\d+-", f"_Models-{n}-", c)
            if c2 != c:
                open(readme, "w").write(c2)
                print(f"Models badge -> {n}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
