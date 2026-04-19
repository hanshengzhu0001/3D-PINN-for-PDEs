"""Generate a PDF report comparing diffusion solutions against oracle numerics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from fpdf import FPDF

from .benchmark import NonlinearElliptic1D
from .dataset import OracleSolutionDataset, generate_oracle_dataset
from .pipeline import DiffusionProjectorSolver


Color = Tuple[int, int, int]


def _tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _summarize(values: List[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def _format_metric(value: float) -> str:
    return f"{value:.3e}"


def _select_sample_indices(guided_errors: List[float]) -> List[int]:
    values = np.asarray(guided_errors, dtype=np.float64)
    order = np.argsort(values)
    candidate_positions = [0, len(order) // 2, len(order) - 1]
    selected: List[int] = []
    for position in candidate_positions:
        index = int(order[position])
        if index not in selected:
            selected.append(index)
    return selected


class ReportPDF(FPDF):
    def header(self) -> None:
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 8, "Diffusion Approximation vs Oracle Numerical Solution", ln=1)
        self.set_font("Helvetica", "", 9)
        self.cell(0, 5, "CHONKNORIS nonlinear elliptic benchmark", ln=1)
        self.ln(2)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Helvetica", "", 8)
        self.cell(0, 5, f"Page {self.page_no()}", align="C")


def _draw_text_block(
    pdf: ReportPDF,
    x: float,
    y: float,
    width: float,
    lines: Sequence[str],
    font_size: int = 10,
) -> float:
    pdf.set_xy(x, y)
    pdf.set_font("Helvetica", "", font_size)
    for line in lines:
        pdf.multi_cell(width, 5, line)
    return pdf.get_y()


def _draw_metric_table(
    pdf: ReportPDF,
    x: float,
    y: float,
    rows: Sequence[Tuple[str, str, str, str, str]],
    headers: Sequence[str] = ("Metric", "Mean", "Median", "P95", "Max"),
) -> float:
    col_widths = [38, 33, 33, 33, 33]
    pdf.set_xy(x, y)
    pdf.set_font("Helvetica", "B", 9)
    for width, header in zip(col_widths, headers):
        pdf.cell(width, 7, header, border=1, align="C")
    pdf.ln(7)
    pdf.set_font("Helvetica", "", 9)
    for row in rows:
        pdf.set_x(x)
        for width, value in zip(col_widths, row):
            pdf.cell(width, 7, value, border=1, align="C")
        pdf.ln(7)
    return pdf.get_y()


def _draw_detail_panel(
    pdf: ReportPDF,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    lines: Sequence[str],
) -> None:
    pdf.set_draw_color(0, 0, 0)
    pdf.rect(x, y, width, height)
    pdf.set_xy(x, y + 2)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(width, 6, title, align="C")
    pdf.set_xy(x + 4, y + 12)
    pdf.set_font("Helvetica", "", 9)
    for line in lines:
        if pdf.get_y() > y + height - 8:
            break
        pdf.multi_cell(width - 8, 5, line)


def _plot_series(
    pdf: ReportPDF,
    x_values: np.ndarray,
    series: Sequence[Tuple[str, np.ndarray, Color]],
    box: Tuple[float, float, float, float],
    title: str,
) -> None:
    x0, y0, width, height = box
    legend_cols = min(3, max(1, len(series)))
    legend_rows = int(math.ceil(len(series) / legend_cols))
    title_band = 10
    footer_band = 8
    legend_band = 7 * legend_rows + 4
    plot_left = x0 + 10
    plot_top = y0 + title_band + 4
    plot_width = width - 20
    plot_height = height - title_band - footer_band - legend_band - 10

    pdf.set_draw_color(0, 0, 0)
    pdf.rect(x0, y0, width, height)
    pdf.set_xy(x0, y0 + 2)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(width, 6, title, align="C")
    pdf.rect(plot_left, plot_top, plot_width, plot_height)

    all_values = np.concatenate([np.asarray(values, dtype=np.float64) for _, values, _ in series])
    y_min = float(all_values.min())
    y_max = float(all_values.max())
    if abs(y_max - y_min) < 1.0e-12:
        y_min -= 1.0
        y_max += 1.0
    else:
        margin = 0.05 * (y_max - y_min)
        y_min -= margin
        y_max += margin

    x_min = float(x_values.min())
    x_max = float(x_values.max())
    if abs(x_max - x_min) < 1.0e-12:
        x_max = x_min + 1.0

    def scale_point(x_value: float, y_value: float) -> Tuple[float, float]:
        x_scaled = plot_left + (x_value - x_min) / (x_max - x_min) * plot_width
        y_scaled = plot_top + plot_height - (y_value - y_min) / (y_max - y_min) * plot_height
        return x_scaled, y_scaled

    for _, values, color in series:
        pdf.set_draw_color(*color)
        points = [scale_point(float(x_val), float(y_val)) for x_val, y_val in zip(x_values, values)]
        for start, end in zip(points[:-1], points[1:]):
            pdf.line(start[0], start[1], end[0], end[1])

    legend_start_x = x0 + 12
    legend_y = y0 + height - legend_band + 2
    pdf.set_font("Helvetica", "", 8)
    for idx, (name, _, color) in enumerate(series):
        row = idx // legend_cols
        col = idx % legend_cols
        legend_x = legend_start_x + col * ((width - 24) / legend_cols)
        row_y = legend_y + row * 6
        pdf.set_draw_color(*color)
        pdf.line(legend_x, row_y, legend_x + 8, row_y)
        pdf.set_xy(legend_x + 10, row_y - 3)
        pdf.cell(28, 5, name)

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_xy(plot_left, plot_top + plot_height + 1)
    pdf.cell(plot_width, 4, f"x in [{x_min:.2f}, {x_max:.2f}]", align="C")
    pdf.set_xy(x0 + width - 52, y0 + 2)
    pdf.cell(46, 4, f"y in [{y_min:.2e}, {y_max:.2e}]", align="R")


def _draw_residual_history(
    pdf: ReportPDF,
    history: Sequence[float],
    box: Tuple[float, float, float, float],
) -> None:
    iterations = np.arange(1, len(history) + 1, dtype=np.float64)
    values = np.log10(np.asarray(history, dtype=np.float64) + 1.0e-30)
    _plot_series(
        pdf,
        iterations,
        [("log10 residual", values, (85, 168, 104))],
        box,
        "Projector residual history",
    )


def generate_report(
    checkpoint_path: str,
    out_path: str,
    num_samples: Optional[int] = None,
    guidance_mode: Optional[str] = None,
    guidance_strength: Optional[float] = None,
    guidance_start_fraction: Optional[float] = None,
    guidance_lambda: Optional[float] = None,
    projector_iterations: Optional[int] = None,
    projector_tolerance: Optional[float] = None,
) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    solver = DiffusionProjectorSolver.from_checkpoint(checkpoint_path, device=device)
    dataset_path = generate_oracle_dataset(solver.config)
    val_dataset = OracleSolutionDataset(dataset_path, split="val")
    benchmark = NonlinearElliptic1D(solver.config.benchmark, device=device)

    n_eval = num_samples or min(solver.config.sampling.num_eval_samples, len(val_dataset))
    active_guidance = (
        solver.config.sampling.guidance_mode if guidance_mode is None else guidance_mode
    )

    records = []
    for index in range(n_eval):
        sample = val_dataset[index]
        forcing_u = sample["u_phys"].to(device)
        oracle_v = sample["v_phys"].to(device, dtype=torch.float64).unsqueeze(0)
        seed = solver.config.sampling.seed + index
        init_generator = torch.Generator(device=device)
        init_generator.manual_seed(seed)
        diffusion_generator = torch.Generator(device=device)
        diffusion_generator.manual_seed(seed)
        guided_generator = torch.Generator(device=device)
        guided_generator.manual_seed(seed)
        projected_generator = torch.Generator(device=device)
        projected_generator.manual_seed(seed)
        x_init = torch.randn(
            (1, 1, solver.config.benchmark.nx), generator=init_generator, device=device
        )

        diffusion_v, _ = solver.sample(
            forcing_u,
            guidance_mode="none",
            projector=False,
            x_init=x_init,
            generator=diffusion_generator,
        )
        guided_v, _ = solver.sample(
            forcing_u,
            guidance_mode=active_guidance,
            guidance_strength=guidance_strength,
            guidance_start_fraction=guidance_start_fraction,
            guidance_lambda=guidance_lambda,
            projector=False,
            x_init=x_init,
            generator=guided_generator,
        )
        projected_v, projector_result = solver.sample(
            forcing_u,
            guidance_mode=active_guidance,
            guidance_strength=guidance_strength,
            guidance_start_fraction=guidance_start_fraction,
            guidance_lambda=guidance_lambda,
            projector=True,
            projector_iterations=projector_iterations,
            projector_tolerance=projector_tolerance,
            x_init=x_init,
            generator=projected_generator,
        )

        forcing_batch = forcing_u.unsqueeze(0).to(torch.float64)
        records.append(
            {
                "index": index,
                "x": _tensor_to_numpy(benchmark.x),
                "forcing_u": _tensor_to_numpy(forcing_u.to(torch.float64)),
                "oracle_v": _tensor_to_numpy(oracle_v.squeeze(0)),
                "diffusion_v": _tensor_to_numpy(diffusion_v.squeeze(0)),
                "guided_v": _tensor_to_numpy(guided_v.squeeze(0)),
                "projected_v": _tensor_to_numpy(projected_v.squeeze(0)),
                "diffusion_error": float(benchmark.relative_l2_error(diffusion_v, oracle_v).item()),
                "guided_error": float(benchmark.relative_l2_error(guided_v, oracle_v).item()),
                "projected_error": float(benchmark.relative_l2_error(projected_v, oracle_v).item()),
                "diffusion_residual": float(benchmark.residual_norm(forcing_batch, diffusion_v).item()),
                "guided_residual": float(benchmark.residual_norm(forcing_batch, guided_v).item()),
                "projected_residual": float(benchmark.residual_norm(forcing_batch, projected_v).item()),
                "projector_history": list(projector_result.residual_history),
                "projector_lambda_history": list(projector_result.lambda_history),
                "projector_alpha_history": list(projector_result.alpha_history),
                "projector_converged": bool(projector_result.converged),
            }
        )

    summary = {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "dataset_path": str(Path(dataset_path).resolve()),
        "num_samples": n_eval,
        "guidance_mode": active_guidance,
        "guidance_strength": (
            solver.config.sampling.guidance_strength
            if guidance_strength is None
            else guidance_strength
        ),
        "guidance_start_fraction": (
            solver.config.sampling.guidance_start_fraction
            if guidance_start_fraction is None
            else guidance_start_fraction
        ),
        "guidance_lambda": (
            solver.config.sampling.guidance_lambda
            if guidance_lambda is None
            else guidance_lambda
        ),
        "projector_iterations": (
            solver.config.sampling.projector_iterations
            if projector_iterations is None
            else projector_iterations
        ),
        "diffusion_error": _summarize([record["diffusion_error"] for record in records]),
        "guided_error": _summarize([record["guided_error"] for record in records]),
        "projected_error": _summarize([record["projected_error"] for record in records]),
        "diffusion_residual": _summarize([record["diffusion_residual"] for record in records]),
        "guided_residual": _summarize([record["guided_residual"] for record in records]),
        "projected_residual": _summarize([record["projected_residual"] for record in records]),
        "projection_steps": _summarize([len(record["projector_history"]) for record in records]),
        "projection_converged_rate": float(
            np.mean([float(record["projector_converged"]) for record in records])
        ),
        "projected_error_count_gt_1e-6": int(
            np.sum([record["projected_error"] > 1.0e-6 for record in records])
        ),
        "projected_residual_count_gt_1e-10": int(
            np.sum([record["projected_residual"] > 1.0e-10 for record in records])
        ),
        "projected_residual_count_gt_1e-8": int(
            np.sum([record["projected_residual"] > 1.0e-8 for record in records])
        ),
    }

    report_path = Path(out_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = ReportPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)

    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Diffusion Approximation vs Actual Numerical Results", ln=1)
    intro_lines = [
        "Benchmark PDE: -Delta v + kappa v^3 = u with periodic boundary conditions.",
        f"Benchmark constants: kappa={solver.config.benchmark.kappa:.0f}, N_x={solver.config.benchmark.nx}, validation samples={summary['num_samples']}.",
        "Actual numerical results are oracle float64 LM/NK solutions generated by the repo's numerical solver.",
        f"Checkpoint: {summary['checkpoint']}",
        f"Guidance / projector: mode={summary['guidance_mode']}, strength={summary['guidance_strength']:.2e}, "
        f"start_fraction={summary['guidance_start_fraction']:.2f}, lambda={summary['guidance_lambda']:.2e}, "
        f"projector_iterations={summary['projector_iterations']}.",
    ]
    current_y = _draw_text_block(pdf, 15, 35, 180, intro_lines, font_size=10)

    rows = [
        (
            "Diffusion error",
            _format_metric(summary["diffusion_error"]["mean"]),
            _format_metric(summary["diffusion_error"]["median"]),
            _format_metric(summary["diffusion_error"]["p95"]),
            _format_metric(summary["diffusion_error"]["max"]),
        ),
        (
            "Guided error",
            _format_metric(summary["guided_error"]["mean"]),
            _format_metric(summary["guided_error"]["median"]),
            _format_metric(summary["guided_error"]["p95"]),
            _format_metric(summary["guided_error"]["max"]),
        ),
        (
            "Projected error",
            _format_metric(summary["projected_error"]["mean"]),
            _format_metric(summary["projected_error"]["median"]),
            _format_metric(summary["projected_error"]["p95"]),
            _format_metric(summary["projected_error"]["max"]),
        ),
        (
            "Diffusion residual",
            _format_metric(summary["diffusion_residual"]["mean"]),
            _format_metric(summary["diffusion_residual"]["median"]),
            _format_metric(summary["diffusion_residual"]["p95"]),
            _format_metric(summary["diffusion_residual"]["max"]),
        ),
        (
            "Guided residual",
            _format_metric(summary["guided_residual"]["mean"]),
            _format_metric(summary["guided_residual"]["median"]),
            _format_metric(summary["guided_residual"]["p95"]),
            _format_metric(summary["guided_residual"]["max"]),
        ),
        (
            "Projected residual",
            _format_metric(summary["projected_residual"]["mean"]),
            _format_metric(summary["projected_residual"]["median"]),
            _format_metric(summary["projected_residual"]["p95"]),
            _format_metric(summary["projected_residual"]["max"]),
        ),
    ]
    current_y = _draw_metric_table(pdf, 15, current_y + 6, rows)

    notes = [
        f"Projected error count > 1e-6: {summary['projected_error_count_gt_1e-6']}",
        f"Projected residual count > 1e-10: {summary['projected_residual_count_gt_1e-10']}",
        f"Projected residual count > 1e-8: {summary['projected_residual_count_gt_1e-8']}",
        f"Projection converged rate (strict tolerance flag): {summary['projection_converged_rate']:.3f}",
        f"Projection steps mean / median / p95: "
        f"{summary['projection_steps']['mean']:.2f} / "
        f"{summary['projection_steps']['median']:.2f} / "
        f"{summary['projection_steps']['p95']:.2f}",
    ]
    _draw_detail_panel(pdf, 15, current_y + 6, 180, 42, "Key validation notes", notes)

    pdf.add_page()
    comparison_lines = [
        "Interpretation:",
        f"1. Diffusion-only approximation reaches mean relative L2 error {_format_metric(summary['diffusion_error']['mean'])}.",
        f"2. Reverse-time PDE guidance improves the approximation to {_format_metric(summary['guided_error']['mean'])}.",
        f"3. The final deterministic LM/NK projector reduces the mean error to {_format_metric(summary['projected_error']['mean'])} and the mean PDE residual to {_format_metric(summary['projected_residual']['mean'])}.",
        "",
        "This report therefore separates three quantities on the same validation set:",
        "- Diffusion prior only",
        "- Guided diffusion approximation",
        "- Final projected solution after float64 numerical cleanup",
        "",
        "The oracle numerical solution v* is the ground-truth reference used for all relative error calculations.",
        "The PDE residual norms are evaluated directly from F(u, v) = -Delta v + kappa v^3 - u.",
    ]
    current_y = _draw_text_block(pdf, 15, 25, 180, comparison_lines, font_size=11)
    method_rows = [
        (
            "Diffusion",
            _format_metric(summary["diffusion_error"]["mean"]),
            _format_metric(summary["diffusion_residual"]["mean"]),
            "-",
            "-",
        ),
        (
            "Guided",
            _format_metric(summary["guided_error"]["mean"]),
            _format_metric(summary["guided_residual"]["mean"]),
            f"{summary['guidance_mode']}",
            f"{summary['guidance_strength']:.1e}",
        ),
        (
            "Projected",
            _format_metric(summary["projected_error"]["mean"]),
            _format_metric(summary["projected_residual"]["mean"]),
            f"{summary['projector_iterations']} steps",
            f"{summary['projection_converged_rate']:.2f}",
        ),
    ]
    _draw_metric_table(
        pdf,
        15,
        current_y + 6,
        method_rows,
        headers=("Stage", "Mean error", "Mean residual", "Control", "Value"),
    )

    selected_indices = _select_sample_indices([record["guided_error"] for record in records])
    for selected in selected_indices:
        record = records[selected]
        pdf.add_page("L")
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, f"Representative sample {record['index']}", ln=1)
        _plot_series(
            pdf,
            record["x"],
            [("forcing u", record["forcing_u"], (129, 114, 179))],
            (10, 26, 132, 76),
            "Forcing",
        )
        _plot_series(
            pdf,
            record["x"],
            [
                ("oracle", record["oracle_v"], (0, 0, 0)),
                ("diffusion", record["diffusion_v"], (76, 114, 176)),
                ("guided", record["guided_v"], (221, 132, 82)),
                ("projected", record["projected_v"], (85, 168, 104)),
            ],
            (155, 26, 132, 76),
            "Solution comparison",
        )
        _draw_residual_history(
            pdf,
            record["projector_history"],
            (10, 112, 132, 76),
        )
        detail_lines = [
            f"Diffusion error: {_format_metric(record['diffusion_error'])}",
            f"Guided error: {_format_metric(record['guided_error'])}",
            f"Projected error: {_format_metric(record['projected_error'])}",
            f"Diffusion residual: {_format_metric(record['diffusion_residual'])}",
            f"Guided residual: {_format_metric(record['guided_residual'])}",
            f"Projected residual: {_format_metric(record['projected_residual'])}",
            f"Projector steps: {len(record['projector_history'])}",
            f"Strict convergence flag: {record['projector_converged']}",
            f"Final lambda: {_format_metric(record['projector_lambda_history'][-1])}",
            f"Final alpha: {_format_metric(record['projector_alpha_history'][-1])}",
        ]
        _draw_detail_panel(pdf, 155, 112, 132, 76, "Sample metrics", detail_lines)

    pdf.output(str(report_path))
    summary_json_path = report_path.with_suffix(".json")
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file")
    parser.add_argument("--out", required=True, help="Path to the output PDF report")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=128,
        help="Number of validation samples to include in the report",
    )
    parser.add_argument(
        "--guidance-mode",
        default=None,
        help="Override guidance mode used during report generation",
    )
    parser.add_argument(
        "--guidance-strength",
        type=float,
        default=None,
        help="Override guidance strength used during report generation",
    )
    parser.add_argument(
        "--guidance-start-fraction",
        type=float,
        default=None,
        help="Override when reverse-time guidance ramps in",
    )
    parser.add_argument(
        "--guidance-lambda",
        type=float,
        default=None,
        help="Override Gauss-Newton guidance damping",
    )
    parser.add_argument(
        "--projector-iterations",
        type=int,
        default=None,
        help="Override final projector iteration count",
    )
    parser.add_argument(
        "--projector-tolerance",
        type=float,
        default=None,
        help="Override final projector tolerance",
    )
    args = parser.parse_args()

    summary = generate_report(
        args.checkpoint,
        args.out,
        num_samples=args.num_samples,
        guidance_mode=args.guidance_mode,
        guidance_strength=args.guidance_strength,
        guidance_start_fraction=args.guidance_start_fraction,
        guidance_lambda=args.guidance_lambda,
        projector_iterations=args.projector_iterations,
        projector_tolerance=args.projector_tolerance,
    )
    print(f"Saved PDF report to {Path(args.out).resolve()}")
    print(f"Projected error mean: {summary['projected_error']['mean']:.6e}")
    print(f"Projected residual mean: {summary['projected_residual']['mean']:.6e}")


if __name__ == "__main__":
    main()
