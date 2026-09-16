# %% [markdown]
# Matrix-multiplication benchmark analysis
# Reads the CSV produced by matrix_bench, writes compact summary tables,
# estimates the CPU/GPU crossover point, and saves report-ready plots.

# %%
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# %% [markdown]
# Command-line arguments

# %%
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/results.csv")
    parser.add_argument("--outdir", default="analysis")
    parser.add_argument("--gpu-name", default="Tesla T4")
    return parser.parse_args()


# %% [markdown]
# Helpers

# %%
def gpu_total_ms(frame: pd.DataFrame) -> pd.Series:
    return frame["total_gpu_ms"]


def cpu_total_ms(frame: pd.DataFrame) -> pd.Series:
    return frame["total_wall_ms"]


def method_family(method: str) -> str:
    if method.startswith("cpu_"):
        return "CPU"
    if method.startswith("cublas"):
        return "cuBLAS"
    if method.startswith("cusparse"):
        return "cuSPARSE"
    if method.startswith("cuda_tiled"):
        return "CUDA tiled"
    if method.startswith("cuda_naive"):
        return "CUDA naive"
    return method


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def prepare_clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    clean["family"] = clean["method"].map(method_family)
    clean["required_total_ms"] = clean.apply(
        lambda row: row["total_wall_ms"] if row["family"] == "CPU" else row["total_gpu_ms"],
        axis=1,
    )
    clean["verified"] = clean["verified"].fillna(1).astype(int)
    clean = clean.loc[clean["verified"] == 1].copy()
    return clean


def best_curve(frame: pd.DataFrame, family: str, ycol: str) -> pd.DataFrame:
    part = frame[frame["family"] == family]
    if part.empty:
        return pd.DataFrame(columns=["N", ycol])
    return part.groupby("N", as_index=False)[ycol].min().sort_values("N")


def min_curve(frame: pd.DataFrame, ycol: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["N", ycol])
    return frame.groupby("N", as_index=False)[ycol].min().sort_values("N")


def safe_positive(series: pd.Series) -> pd.Series:
    return series.where(series > 0)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    if not df.empty:
        df.to_csv(path, index=False)


# %% [markdown]
# Plot functions

# %%
def plot_total_time_by_dtype(clean: pd.DataFrame, outdir: Path) -> list[str]:
    created: list[str] = []
    for dtype, part in clean.groupby("data_type"):
        plt.figure(figsize=(9, 6))
        for family, group in part.groupby("family"):
            curve = group.groupby("N", as_index=False)["required_total_ms"].min().sort_values("N")
            if curve.empty:
                continue
            plt.plot(curve["N"], curve["required_total_ms"], marker="o", label=family)
        plt.xscale("log", base=2)
        plt.yscale("log")
        plt.xlabel("Matrix size N")
        plt.ylabel("Total time (ms)")
        plt.title(f"Total execution time - {dtype}")
        plt.grid(True, which="both", linewidth=0.4)
        plt.legend()
        plt.tight_layout()
        name = f"total_execution_time_{dtype}.png"
        plt.savefig(outdir / name, dpi=180)
        plt.close()
        created.append(name)
    return created


def plot_time_breakdown(clean: pd.DataFrame, outdir: Path) -> list[str]:
    created: list[str] = []
    gpu_families = ["CUDA naive", "CUDA tiled", "cuBLAS", "cuSPARSE"]
    for dtype, part in clean.groupby("data_type"):
        for family in gpu_families:
            fam = part[part["family"] == family]
            if fam.empty:
                continue
            curve = fam.groupby("N", as_index=False)[["h2d_ms", "compute_ms", "d2h_ms", "setup_ms"]].min().sort_values("N")
            if curve.empty:
                continue
            x = curve["N"]
            h2d = curve["h2d_ms"]
            kernel = curve["compute_ms"]
            d2h = curve["d2h_ms"]
            allocation = curve["setup_ms"]
            plt.figure(figsize=(9, 6))
            plt.stackplot(x, h2d, kernel, d2h, allocation, labels=["H2D transfer", "Kernel", "D2H transfer", "Allocation"])
            plt.xscale("log", base=2)
            plt.xlabel("Matrix size N")
            plt.ylabel("Time (ms)")
            plt.title(f"Time breakdown - {family} / {dtype}")
            plt.grid(True, which="both", linewidth=0.3)
            plt.legend(loc="upper left")
            plt.tight_layout()
            name = f"time_breakdown_{slugify(family)}_{dtype}.png"
            plt.savefig(outdir / name, dpi=180)
            plt.close()
            created.append(name)
    return created


def compute_crossover_rows(clean: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dtype, part in clean.groupby("data_type"):
        best_cpu_curve = min_curve(part[part["family"] == "CPU"], "required_total_ms")
        cublas_curve = best_curve(part, "cuBLAS", "required_total_ms")
        first_n = None
        first_cpu = math.nan
        first_gpu = math.nan
        speedup = math.nan
        merged = best_cpu_curve.merge(cublas_curve, on="N", how="inner", suffixes=("_cpu", "_gpu"))
        for _, row in merged.sort_values("N").iterrows():
            if row["required_total_ms_gpu"] < row["required_total_ms_cpu"]:
                first_n = int(row["N"])
                first_cpu = float(row["required_total_ms_cpu"])
                first_gpu = float(row["required_total_ms_gpu"])
                speedup = first_cpu / first_gpu if first_gpu > 0 else math.nan
                break
        rows.append(
            {
                "data_type": dtype,
                "first_tested_N": first_n if first_n is not None else "not observed",
                "best_cpu_total_ms": first_cpu if first_n is not None else "",
                "cublas_total_ms": first_gpu if first_n is not None else "",
                "speedup": speedup if first_n is not None else "",
            }
        )
    return pd.DataFrame(rows)


def plot_crossover(clean: pd.DataFrame, outdir: Path) -> tuple[pd.DataFrame, list[str]]:
    created: list[str] = []
    summary = compute_crossover_rows(clean)
    save_csv(summary, outdir / "crossover_summary.csv")

    for dtype, part in clean.groupby("data_type"):
        best_cpu_curve = min_curve(part[part["family"] == "CPU"], "required_total_ms")
        cublas_curve = best_curve(part, "cuBLAS", "required_total_ms")
        tiled_curve = best_curve(part, "CUDA tiled", "required_total_ms")
        if best_cpu_curve.empty or cublas_curve.empty:
            continue

        plt.figure(figsize=(9, 6))
        plt.plot(best_cpu_curve["N"], best_cpu_curve["required_total_ms"], marker="o", label="Best CPU")
        plt.plot(cublas_curve["N"], cublas_curve["required_total_ms"], marker="o", label="cuBLAS")
        if not tiled_curve.empty:
            plt.plot(tiled_curve["N"], tiled_curve["required_total_ms"], marker="o", label="CUDA tiled")

        merged = best_cpu_curve.merge(cublas_curve, on="N", how="inner", suffixes=("_cpu", "_gpu")).sort_values("N")
        crossover_n = None
        crossover_cpu = None
        crossover_gpu = None
        for _, row in merged.iterrows():
            if row["required_total_ms_gpu"] < row["required_total_ms_cpu"]:
                crossover_n = int(row["N"])
                crossover_cpu = float(row["required_total_ms_cpu"])
                crossover_gpu = float(row["required_total_ms_gpu"])
                break

        if crossover_n is not None:
            plt.axvline(crossover_n, linestyle="--", linewidth=1)
            plt.scatter([crossover_n], [crossover_gpu], zorder=5)
            plt.annotate(
                f"crossover @ N={crossover_n}",
                xy=(crossover_n, crossover_gpu),
                xytext=(10, 10),
                textcoords="offset points",
            )

        plt.xscale("log", base=2)
        plt.yscale("log")
        plt.xlabel("Matrix size N")
        plt.ylabel("Total time (ms)")
        plt.title(f"GPU vs CPU cross over point (total time basis) - {dtype}")
        plt.grid(True, which="both", linewidth=0.4)
        plt.legend()
        plt.tight_layout()
        name = f"gpu_vs_cpu_cross_over_point_total_time_basis_{dtype}.png"
        plt.savefig(outdir / name, dpi=180)
        plt.close()
        created.append(name)
    return summary, created


def plot_throughput(clean: pd.DataFrame, outdir: Path, gpu_name: str) -> list[str]:
    created: list[str] = []
    gpu_families = ["CUDA naive", "CUDA tiled", "cuBLAS", "cuSPARSE"]
    for dtype, part in clean.groupby("data_type"):
        plt.figure(figsize=(9, 6))
        plotted = False
        for family in gpu_families:
            fam = part[part["family"] == family]
            if fam.empty:
                continue
            curve = fam.groupby("N", as_index=False)["compute_gops"].max().sort_values("N")
            curve = curve[curve["compute_gops"] > 0]
            if curve.empty:
                continue
            plt.plot(curve["N"], curve["compute_gops"], marker="o", label=family)
            plotted = True
        if not plotted:
            plt.close()
            continue
        plt.xscale("log", base=2)
        plt.xlabel("Matrix size N")
        plt.ylabel("Throughput (GOPS)")
        plt.title(f"Compute throughput (kernel time only) - {gpu_name} / {dtype}")
        plt.grid(True, which="both", linewidth=0.4)
        plt.legend()
        plt.tight_layout()
        name = f"compute_throughput_kernel_time_only_{slugify(gpu_name)}_{dtype}.png"
        plt.savefig(outdir / name, dpi=180)
        plt.close()
        created.append(name)
    return created


def write_report(clean: pd.DataFrame, outdir: Path, crossover: pd.DataFrame, figure_names: list[str]) -> None:
    summary_columns = [
        "N",
        "data_type",
        "method",
        "h2d_ms",
        "compute_ms",
        "d2h_ms",
        "setup_ms",
        "required_total_ms",
        "launch_overhead_us",
        "compute_gops",
        "end_to_end_gops",
        "actual_sparsity_a",
    ]
    save_csv(
        clean[summary_columns].sort_values(["data_type", "N", "method"]),
        outdir / "benchmark_summary.csv",
    )

    lines = [
        "# Automatic benchmark report",
        "",
        "## CPU vs GPU crossover summary",
        "",
        crossover.to_markdown(index=False) if not crossover.empty else "Not enough data.",
        "",
        "## Notes",
        "",
        "- GPU total time is taken from `total_gpu_ms` and includes H2D, compute/library call, and D2H.",
        "- CPU total time is taken from `total_wall_ms`.",
        "- `setup_ms` is shown separately; for cuSPARSE it includes dense-to-CSR preparation.",
        "- `compute_gops` is based on kernel/library compute time only.",
        "",
        "## Generated figures",
        "",
    ]
    lines.extend([f"- {name}" for name in sorted(figure_names)])
    (outdir / "REPORT_AUTO.md").write_text("\n".join(lines), encoding="utf-8")


# %% [markdown]
# Main analysis

# %%
def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if df.empty:
        raise SystemExit("The input CSV is empty.")

    clean = prepare_clean_frame(df)
    if clean.empty:
        raise SystemExit("No verified rows were found in the input CSV.")

    figure_names: list[str] = []
    figure_names.extend(plot_total_time_by_dtype(clean, outdir))
    figure_names.extend(plot_time_breakdown(clean, outdir))
    crossover, crossover_figures = plot_crossover(clean, outdir)
    figure_names.extend(crossover_figures)
    figure_names.extend(plot_throughput(clean, outdir, args.gpu_name))

    # Save a compact table of GPU timing components for each data type.
    for dtype, part in clean.groupby("data_type"):
        gpu = part[part["family"] != "CPU"].copy()
        if gpu.empty:
            continue
        pivot = gpu.pivot_table(
            index=["N", "family"],
            values=["h2d_ms", "compute_ms", "d2h_ms", "setup_ms", "required_total_ms", "compute_gops"],
            aggfunc="min",
        ).reset_index().sort_values(["family", "N"])
        save_csv(pivot, outdir / f"gpu_components_{dtype}.csv")

    write_report(clean, outdir, crossover, figure_names)

    print("Created figures:")
    for name in sorted(figure_names):
        print(f" - {name}")
    if not crossover.empty:
        print("\nCrossover summary:")
        print(crossover.to_string(index=False))
    print(f"\nAnalysis files written to: {outdir.resolve()}")


# %%
if __name__ == "__main__":
    main()
