"""Render precomputed native CSV tables. No simulation or error analysis."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analytic_panels import render as render_analytic_panels

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--results", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.labelsize": 10, "legend.fontsize": 8,
                     "pdf.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False, "lines.linewidth": 1.3})
colors = ["#243e54", "#b45135", "#408b78", "#9b729f", "#b18a37"]
markers = ["o", "s", "^", "D", "v"]

def spin_label(n):
    return "$N=" + format(n, ",").replace(",", "{,}") + "$"

def read(name):
    return np.atleast_1d(np.genfromtxt(args.results / f"{name}.csv", names=True, delimiter=","))

def canvas(width=6.5, height=3.8):
    fig, ax = plt.subplots(figsize=(width, height))
    fig.subplots_adjust(left=.13, right=.98, bottom=.16, top=.95)
    ax.grid(alpha=.16)
    return fig, ax

def save(fig, name):
    fig.savefig(args.output / f"fig_{name}.pdf")
    fig.savefig(args.output / f"fig_{name}.png", dpi=180)
    plt.close(fig)

def fid_axes(ax, xmax):
    ax.axhline(0, color=".65", lw=.6, zorder=0)
    ax.set(xlabel="Dimensionless time $t$", ylabel="$F(t)$", xlim=(0, xmax))

# Analytical comparisons use a shared canvas with aligned panel axes.
d = read("truncation")
fig, ax = canvas()
ax.loglog(d["K"], d["actual_trace_distance"], "o-", color=colors[0], label="Measured state distance")
ax.loglog(d["K"], d["total_error_bound"], "s--", color=colors[1], label="Total allowance")
ax.set(xlabel="Retained-coefficient limit $K$", ylabel="State distance or allowance")
ax.legend(frameon=False)
save(fig, "truncation_bound")

fig, axes = plt.subplots(2, 1, figsize=(3.4, 3.8), sharex=True, layout="constrained")
for ax, n, panel in zip(axes, [1000, 16], "ab"):
    f = read(f"fid{n}_fine")
    if n == 1000:
        c = read("fid1000_coarse")
        ax.fill_between(f["time"], f["lower"], f["upper"], color="#70aacc", label="Error interval")
        ax.plot(c["time"], c["F"], color=colors[0], label="Coarse")
    else:
        c = read("fulltrace16")
        ax.plot(c["time"], c["F"], color=colors[0], label="Full trace")
    ax.plot(f["time"][::4], f["F"][::4], "o", ms=3.5, mfc="white", mec=colors[1], label="Fine")
    fid_axes(ax, 2)
    ax.set_ylim(-.22, 1.1)
    ax.text(.04, .12, f"({panel})", transform=ax.transAxes)
    ax.legend(title=spin_label(n), loc="upper right", frameon=False)
axes[0].set_xlabel("")
save(fig, "fid_t2_rev")

render_analytic_panels(read("pair"), read("complete"), read("ising"),
                      [(dt, read(f"xy_{step}")) for dt, step in
                       [(.08, ".08"), (.04, ".04"), (.02, ".02")]], args.output)

b = read("bounds")
spread = read("threshold_envelope")
fig, ax = canvas(6.8, 4.7)
old, new = b["time"] <= 1.5, b["time"] <= 2
ax.fill_between(b["time"][old], b["lower_M6"][old], b["upper_M8"][old], color="#a4d9e4", alpha=.4, label="$M_6$–$M_8$ enclosure")
ax.fill_between(b["time"][new], b["lower_M10"][new], b["upper_M12"][new], color="#6b9dd2", alpha=.4, label="$M_{10}$–$M_{12}$ enclosure")
ax.fill_between(spread["time"], spread["lower"], spread["upper"], color=".5", alpha=.25, label="Threshold spread")
for eta, color, marker in [("1e-4", "black", "o"), ("3e-5", colors[1], "o"), ("1e-5", colors[2], "x")]:
    d = read(f"ring_.04_{eta}")
    threshold_label = {"1e-4": "10^{-4}", "3e-5": "3\\times10^{-5}", "1e-5": "10^{-5}"}[eta]
    ax.plot(d["time"], d["fid_average"], color=color, marker=marker, markevery=3, ms=3.5, mfc="white", label=f"TPGP, $\\eta={threshold_label}$")
ax.plot(b["time"], b["Abragam_M2_M4_approximation"], "--", color=colors[3], label="Abragam moment match")
fid_axes(ax, 3)
ax.set_ylim(-.55, 1.05)
ax.legend(frameon=False, fontsize=8, loc="upper right")
save(fig, "moment_bounds")

d, p = read("lanczos"), read("krylov_tpgp")
fig, ax = canvas(7, 3.9)
ax.fill_between(d["time"], d["lower95"], d["upper95"], color="#9db9d4", alpha=.5, label="Lanczos pointwise 95% interval")
ax.plot(d["time"], d["krylov_mean"], color=colors[0], label="Lanczos mean")
ax.plot(p["time"][::3], p["fid_average"][::3], "o", mfc="white", mec=colors[1], ms=4, label="TPGP")
fid_axes(ax, 3.2)
ax.legend(frameon=False, title="$N=16$")
save(fig, "krylov_comparison")

fig, ax = canvas(7, 3.9)
for i, n in enumerate([1000, 24, 16, 12, 8]):
    d = read(f"finite_{n}_1e-4")
    ax.plot(d["time"], d["fid_average"], color=colors[i], marker=markers[i], markevery=(i, 9), ms=4, mfc="white", label=spin_label(n))
fid_axes(ax, 3.2)
ax.legend(frameon=False, ncol=2)
save(fig, "finite_size_collapse")

fig, ax = canvas(7, 4)
for i, (name, label) in enumerate([("cubic6_2_long", "Two shells, $\\eta=10^{-4}$"),
                                  ("cubic6_1_long", "One shell, $\\eta=10^{-4}$"),
                                  ("cubic6_coarse_long", "$\\tau=0.04$"),
                                  ("cubic6_tight_long", "$\\eta=10^{-5}$")]):
    d = read(name)
    ax.plot(d["time"], d["fid"], color=colors[i], marker=markers[i], markevery=(i, 5), ms=4, mfc="white", label=label)
fid_axes(ax, 1)
ax.legend(frameon=False, title="$N=216$, field [001]")
save(fig, "dipolar3d")

d = read("growth")
fig, ax = canvas()
ax.plot(d["time"], d["untruncated_terms"], color=colors[0], label="Unrestricted")
ax.plot(d["time"], d["topK_terms"], color=colors[1], label="$K=100$")
ax.set(xlabel="Dimensionless time $t$", ylabel="Retained Pauli coefficients")
right = ax.twinx()
right.spines["right"].set_visible(True)
right.plot(d["time"], d["topK_cumulative_removal_bound"], ":", color="#a53131", label="Removal contribution")
right.set_ylabel("Accumulated removal contribution", color="#a53131")
right.tick_params(axis="y", colors="#a53131")
fig.subplots_adjust(right=.85)
handles, labels = ax.get_legend_handles_labels()
rh, rl = right.get_legend_handles_labels()
ax.legend(handles + rh, labels + rl, frameon=False, loc="upper left", bbox_to_anchor=(.22, .86))
save(fig, "term_growth")
