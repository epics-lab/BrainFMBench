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

def _order_and_colors(table):
    present = _all_models_in(table)
    known = [m for m in MODEL_ORDER if m in present]
    unknown = sorted(m for m in present if m not in MODEL_COLORS)
    colors = dict(MODEL_COLORS)
    if unknown:
        cmap = matplotlib.colormaps["tab20"]
        for i, m in enumerate(unknown):
            colors[m] = _mcolors.to_hex(cmap(i % 20))
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


def make_interactive(table, out_html):
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

    plot = fig.to_html(full_html=False, include_plotlyjs="cdn",
                       config={"responsive": True, "displaylogo": False})
    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    with open(out_html, "w") as f:
        f.write(
            "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            "<title>BrainFMBench Leaderboard</title>"
            "<style>body{font-family:system-ui,sans-serif;max-width:1400px;margin:0 auto;padding:16px}</style>"
            "</head><body>\n"
            "<p><a href=\"https://github.com/epics-lab/BrainFMBench\">epics-lab/BrainFMBench</a>"
            " &middot; auto generated from <code>models/*/results.json</code></p>\n"
            + plot + "\n</body></html>\n")
    print(f"Wrote {out_html}")


def main():
    rows = load_rows()
    table = {}
    for r in rows:
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

    try:
        make_interactive(table, os.path.join(REPO, "site", "index.html"))
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
            f"[![Leaderboard boxplots](leaderboard_boxplots.png)]({PAGES_URL})", "",
            "_Boxes show 5-fold cross-validation across 5 seeds (25 values); "
            "overlaid points show held-out test results for the same 5 seeds._", "",
            f"**[Interactive version]({PAGES_URL})**", "",
        ]
    lines += [
        "| Model | Dataset | Sex (acc) | Age (MAE) | BMI (MAE) |",
        "|-------|---------|-----------|-----------|-----------|",
    ]
    for d in display:
        lines.append(f"| {d['model']} | {d['dataset']} | "
                     f"{d['sex_cell']} | {d['age_cell']} | {d['bmi_cell']} |")
    lines += ["", "_Auto-generated from `models/*/results.json`. Do not edit by hand._"]

    with open(os.path.join(REPO, "LEADERBOARD.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote LEADERBOARD.md  ({len(display)} rows)")

    # inject leaderboard (figure + table) into README.md between markers
    readme = os.path.join(REPO, "README.md")
    if os.path.isfile(readme):
        blk = ["<!-- LEADERBOARD:START -->", ""]
        if fig_ok:
            blk += [f"[![Leaderboard boxplots](leaderboard_boxplots.png)]({PAGES_URL})", "",
                    f"**[Interactive version]({PAGES_URL})**", ""]
        blk += ["| Model | Dataset | Sex (acc) | Age (MAE) | BMI (MAE) |",
                "|-------|---------|-----------|-----------|-----------|"]
        for d in display:
            blk.append(f"| {d['model']} | {d['dataset']} | "
                       f"{d['sex_cell']} | {d['age_cell']} | {d['bmi_cell']} |")
        blk += ["", "_Ranking varies by task and dataset; there is no single overall winner._",
                "", "<!-- LEADERBOARD:END -->"]
        blk_text = "\n".join(blk)
        c = open(readme).read()
        a, b = "<!-- LEADERBOARD:START -->", "<!-- LEADERBOARD:END -->"
        if a in c and b in c:
            c = c.split(a)[0] + blk_text + c.split(b)[1]
            open(readme, "w").write(c)
            print(f"Injected leaderboard into {readme}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
