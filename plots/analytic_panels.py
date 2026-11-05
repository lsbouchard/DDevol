"""Publication panels from precomputed native tables; no numerical evolution."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    return np.atleast_1d(np.genfromtxt(path, names=True, delimiter=","))


def render(pair, complete, ising, xy_tables, output):
    """xy_tables contains (step, table) pairs; retain all three refinements."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    style = {"font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9,
             "ytick.labelsize": 9, "legend.fontsize": 8.5, "pdf.fonttype": 42,
             "axes.spines.top": False, "axes.spines.right": False,
             "lines.linewidth": 1.25}
    with plt.rc_context(style):
        for large, stem in [(False, "analytic_small_panels"),
                            (True, "analytic_large_panels")]:
            fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0))
            # A 0.49-inch axes gap: about half the former separate-canvas gap.
            fig.subplots_adjust(left=.095, right=.97, bottom=.18, top=.77,
                                wspace=.17)
            for i, ax in enumerate(axes):
                pos = ax.get_position()
                fig.text(pos.x0, .955, f"({chr(97+i)})", fontsize=11,
                         fontweight="bold", va="top")
                ax.axhline(0, color=".65", lw=.6, zorder=0)
                ax.grid(alpha=.14, linewidth=.5)
                ax.yaxis.labelpad = 1
                if large:
                    fig.text(pos.x1, .955, "$N=1{,}000$", ha="right", va="top")
                if large and i == 1:
                    base = xy_tables[-1][1]
                    ax.plot(base["time"], base["finite_N_exact"], color="#254962",
                            label="Exact finite chain")
                    for k, (dt, d) in enumerate(xy_tables):
                        stride = max(3, len(d)//18)
                        ix = np.arange(k, len(d), stride)
                        ax.plot(d["time"][ix], d["TPGP_correlation"][ix],
                                ["o", "s", "^"][k], ms=4, mfc="white",
                                mec=["#b45135", "#408b78", "#825e9c"][k],
                                label=fr"TPGP, $\tau={dt:g}$")
                    ax.set(xlim=(0, 20), xlabel="$Jt$", ylabel="Boundary correlation")
                    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.015),
                              ncol=2, frameon=False, columnspacing=.8,
                              handlelength=1.3, borderaxespad=0)
                else:
                    d = ising if large else [pair, complete][i]
                    ax.plot(d["time"], d["analytic_FID"], color="#254962",
                            label="Analytical")
                    stride = max(1, len(d)//36)
                    ax.plot(d["time"][::stride], d["TPGP_FID"][::stride], "o",
                            ms=4, mfc="white", mec="#b45135", label="TPGP")
                    ax.set(xlim=(0, float(d["time"][-1])),
                           xlabel="Dimensionless time $t$", ylabel="$F(t)$")
                    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.025),
                              ncol=2, frameon=False, borderaxespad=0)
            fig.savefig(output / f"fig_{stem}.pdf")
            fig.savefig(output / f"fig_{stem}.png", dpi=220)
            plt.close(fig)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(read(args.results / "pair.csv"), read(args.results / "complete.csv"),
           read(args.results / "ising.csv"),
           [(dt, read(args.results / f"xy_{dt:.2f}.csv".replace("xy_0.", "xy_.")))
            for dt in [.08, .04, .02]],
           args.output)
