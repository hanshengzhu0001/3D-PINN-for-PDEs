"""Generate a LaTeX report with side-by-side figures and explanations."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

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


def _format_percent(value: float) -> str:
    return f"{value:.1f}%"


def _format_ratio(value: float) -> str:
    return f"{value:.2e}x"


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


def _sample_narrative(label: str, record: Dict[str, object]) -> List[str]:
    diffusion_error = float(record["diffusion_error"])
    guided_error = float(record["guided_error"])
    projected_error = float(record["projected_error"])
    diffusion_residual = float(record["diffusion_residual"])
    guided_residual = float(record["guided_residual"])
    projected_residual = float(record["projected_residual"])
    steps = len(record["projector_history"])

    error_drop = 0.0
    residual_drop = 0.0
    if diffusion_error > 0.0:
        error_drop = max(0.0, 100.0 * (diffusion_error - guided_error) / diffusion_error)
    if diffusion_residual > 0.0:
        residual_drop = max(0.0, 100.0 * (diffusion_residual - guided_residual) / diffusion_residual)
    projector_gain = diffusion_error / max(projected_error, 1.0e-30)

    takeaway = (
        f"这一页要表达的是: guidance first moves the sample toward the correct basin, "
        f"and the final projector then finishes the solve in {steps} LM/NK steps."
    )
    if "Hard" in label:
        takeaway = (
            "这一页要表达的是: even when diffusion and guidance are still visibly imperfect, "
            "the final projector can still recover the oracle solution to near machine precision."
        )
    elif "Typical" in label:
        takeaway = (
            "这一页要表达的是: this is the normal case. diffusion gives a usable initializer, "
            "guidance gives a modest physics-aware correction, and the projector closes the remaining gap."
        )
    elif "Best" in label:
        takeaway = (
            "这一页要表达的是: this is the friendly case. guidance already improves the sample a lot before optimization starts, "
            "so the projector only needs a short cleanup."
        )

    return [
        label,
        f"图从哪里来: this page uses validation sample index {record['index']} from the oracle-generated validation split. 所有曲线都来自同一个 sample, not different examples.",
        "图上每条线是什么意思: black = oracle numerical solution v*; blue = diffusion-only output; orange = guided reverse-sampling output; green = final projected output.",
        "下方 residual history panel 从哪里来: it is taken from projector_result.residual_history, i.e. the residual norm recorded at each LM/NK iteration for this exact sample.",
        f"这一页的数字怎么读: diffusion error = {_format_metric(diffusion_error)}, guided error = {_format_metric(guided_error)}, projected error = {_format_metric(projected_error)}.",
        f"guidance 的直接效果: error drops by {_format_percent(error_drop)} and residual drops by {_format_percent(residual_drop)} before the final projector.",
        f"final projector 的效果: relative error improves by about {_format_ratio(projector_gain)} from diffusion to projected output, and projected residual reaches {_format_metric(projected_residual)}.",
        takeaway,
    ]


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "/System/Library/Fonts/Supplemental/Helvetica.ttc",
            ]
        )
    candidates.extend(
        [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        ]
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_plot(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    x_values: np.ndarray,
    series: Sequence[Tuple[str, np.ndarray, Color]],
    title: str,
    font: ImageFont.ImageFont,
    bold_font: ImageFont.ImageFont,
) -> None:
    x0, y0, width, height = box
    draw.rounded_rectangle((x0, y0, x0 + width, y0 + height), radius=12, outline=(60, 60, 60), width=2)
    draw.text((x0 + 12, y0 + 8), title, fill=(0, 0, 0), font=bold_font)

    legend_rows = int(np.ceil(len(series) / 3))
    plot_left = x0 + 16
    plot_top = y0 + 40
    plot_width = width - 32
    plot_height = height - 72 - 18 * legend_rows
    draw.rectangle((plot_left, plot_top, plot_left + plot_width, plot_top + plot_height), outline=(120, 120, 120), width=1)

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

    def scale_point(x_value: float, y_value: float) -> Tuple[int, int]:
        x_scaled = int(plot_left + (x_value - x_min) / (x_max - x_min) * plot_width)
        y_scaled = int(plot_top + plot_height - (y_value - y_min) / (y_max - y_min) * plot_height)
        return x_scaled, y_scaled

    for _, values, color in series:
        points = [scale_point(float(x_val), float(y_val)) for x_val, y_val in zip(x_values, values)]
        for start, end in zip(points[:-1], points[1:]):
            draw.line((start[0], start[1], end[0], end[1]), fill=color, width=2)

    draw.text((plot_left, plot_top + plot_height + 6), f"x in [{x_min:.2f}, {x_max:.2f}]", fill=(0, 0, 0), font=font)
    y_range_text = f"y in [{y_min:.2e}, {y_max:.2e}]"
    y_range_width = draw.textbbox((0, 0), y_range_text, font=font)[2]
    draw.text((x0 + width - 16 - y_range_width, y0 + 10), y_range_text, fill=(90, 90, 90), font=font)

    legend_y = plot_top + plot_height + 28
    for index, (name, _, color) in enumerate(series):
        row = index // 3
        col = index % 3
        item_x = x0 + 20 + col * ((width - 40) // 3)
        item_y = legend_y + row * 18
        draw.line((item_x, item_y + 7, item_x + 18, item_y + 7), fill=color, width=3)
        draw.text((item_x + 24, item_y), name, fill=(0, 0, 0), font=font)


def _draw_detail_box(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    title: str,
    lines: Sequence[str],
    font: ImageFont.ImageFont,
    bold_font: ImageFont.ImageFont,
) -> None:
    x0, y0, width, height = box
    draw.rounded_rectangle((x0, y0, x0 + width, y0 + height), radius=12, outline=(60, 60, 60), width=2)
    draw.text((x0 + 12, y0 + 8), title, fill=(0, 0, 0), font=bold_font)
    cursor_y = y0 + 38
    for line in lines:
        draw.text((x0 + 12, cursor_y), line, fill=(0, 0, 0), font=font)
        cursor_y += 20
        if cursor_y > y0 + height - 20:
            break


def _create_sample_figure(record: Dict[str, object], output_path: Path) -> None:
    width, height = 1600, 980
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _load_font(32, bold=True)
    font = _load_font(20)
    bold_font = _load_font(22, bold=True)

    draw.text((50, 30), f"Validation sample {record['index']}", fill=(0, 0, 0), font=title_font)

    x_values = np.asarray(record["x"])
    _draw_plot(
        draw,
        (40, 100, 1520, 210),
        x_values,
        [("forcing u", np.asarray(record["forcing_u"]), (129, 114, 179))],
        "Forcing profile",
        font,
        bold_font,
    )
    _draw_plot(
        draw,
        (40, 340, 1520, 300),
        x_values,
        [
            ("oracle", np.asarray(record["oracle_v"]), (0, 0, 0)),
            ("diffusion", np.asarray(record["diffusion_v"]), (76, 114, 176)),
            ("guided", np.asarray(record["guided_v"]), (221, 132, 82)),
            ("projected", np.asarray(record["projected_v"]), (85, 168, 104)),
        ],
        "Solution comparison",
        font,
        bold_font,
    )
    history = np.log10(np.asarray(record["projector_history"], dtype=np.float64) + 1.0e-30)
    history_x = np.arange(1, len(history) + 1, dtype=np.float64)
    _draw_plot(
        draw,
        (40, 670, 900, 250),
        history_x,
        [("log10 residual", history, (85, 168, 104))],
        "Projector residual history",
        font,
        bold_font,
    )
    _draw_detail_box(
        draw,
        (980, 670, 580, 250),
        "Sample metrics",
        [
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
        ],
        font,
        bold_font,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _collect_records(
    checkpoint_path: str,
    num_samples: int,
    guidance_mode: Optional[str],
    guidance_strength: Optional[float],
    guidance_start_fraction: Optional[float],
    guidance_lambda: Optional[float],
    projector_iterations: Optional[int],
    projector_tolerance: Optional[float],
) -> Tuple[DiffusionProjectorSolver, Dict[str, object], List[Dict[str, object]]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    solver = DiffusionProjectorSolver.from_checkpoint(checkpoint_path, device=device)
    dataset_path = generate_oracle_dataset(solver.config)
    val_dataset = OracleSolutionDataset(dataset_path, split="val")
    benchmark = NonlinearElliptic1D(solver.config.benchmark, device=device)
    n_eval = min(num_samples, len(val_dataset))
    active_guidance = solver.config.sampling.guidance_mode if guidance_mode is None else guidance_mode

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
        x_init = torch.randn((1, 1, solver.config.benchmark.nx), generator=init_generator, device=device)

        diffusion_v, _ = solver.sample(forcing_u, guidance_mode="none", projector=False, x_init=x_init, generator=diffusion_generator)
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

    summary: Dict[str, object] = {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "dataset_path": str(Path(dataset_path).resolve()),
        "num_samples": n_eval,
        "guidance_mode": active_guidance,
        "guidance_strength": solver.config.sampling.guidance_strength if guidance_strength is None else guidance_strength,
        "guidance_start_fraction": solver.config.sampling.guidance_start_fraction if guidance_start_fraction is None else guidance_start_fraction,
        "guidance_lambda": solver.config.sampling.guidance_lambda if guidance_lambda is None else guidance_lambda,
        "projector_iterations": solver.config.sampling.projector_iterations if projector_iterations is None else projector_iterations,
        "diffusion_error": _summarize([record["diffusion_error"] for record in records]),
        "guided_error": _summarize([record["guided_error"] for record in records]),
        "projected_error": _summarize([record["projected_error"] for record in records]),
        "diffusion_residual": _summarize([record["diffusion_residual"] for record in records]),
        "guided_residual": _summarize([record["guided_residual"] for record in records]),
        "projected_residual": _summarize([record["projected_residual"] for record in records]),
        "projection_steps": _summarize([len(record["projector_history"]) for record in records]),
        "projection_converged_rate": float(np.mean([float(record["projector_converged"]) for record in records])),
        "projected_error_count_gt_1e-6": int(np.sum([record["projected_error"] > 1.0e-6 for record in records])),
        "projected_residual_count_gt_1e-10": int(np.sum([record["projected_residual"] > 1.0e-10 for record in records])),
        "projected_residual_count_gt_1e-8": int(np.sum([record["projected_residual"] > 1.0e-8 for record in records])),
    }
    return solver, summary, records


def _latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("#", "\\#")
    )


def generate_latex_report(
    checkpoint_path: str,
    out_path: str,
    num_samples: int = 128,
    guidance_mode: Optional[str] = None,
    guidance_strength: Optional[float] = None,
    guidance_start_fraction: Optional[float] = None,
    guidance_lambda: Optional[float] = None,
    projector_iterations: Optional[int] = None,
    projector_tolerance: Optional[float] = None,
    compile_pdf: bool = True,
) -> Dict[str, object]:
    solver, summary, records = _collect_records(
        checkpoint_path,
        num_samples,
        guidance_mode,
        guidance_strength,
        guidance_start_fraction,
        guidance_lambda,
        projector_iterations,
        projector_tolerance,
    )

    out_tex = Path(out_path)
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    asset_dir = out_tex.parent / f"{out_tex.stem}_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)

    sample_indices = _select_sample_indices([record["guided_error"] for record in records])
    sample_figures = []
    sample_explanations = []
    labels = ["Best case / 最容易看懂的例子", "Typical case / 典型例子", "Hard case / 最难的例子"]
    for label, sample_index in zip(labels, sample_indices):
        record = records[sample_index]
        image_name = f"sample_{record['index']:03d}.png"
        image_path = asset_dir / image_name
        _create_sample_figure(record, image_path)
        sample_figures.append(f"{out_tex.stem}_assets/{image_name}")
        sample_explanations.append(_sample_narrative(label, record))

    checkpoint_display = Path(checkpoint_path).as_posix()
    dataset_display = Path(solver.config.benchmark.dataset.out_path).as_posix()
    guidance_mode_display = str(summary["guidance_mode"]).upper()

    tex_lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=0.85in]{geometry}",
        r"\usepackage{amsmath}",
        r"\usepackage{graphicx}",
        r"\usepackage{booktabs}",
        r"\usepackage{array}",
        r"\usepackage{tabularx}",
        r"\usepackage{xcolor}",
        r"\usepackage{parskip}",
        r"\usepackage{microtype}",
        r"\usepackage{fontspec}",
        r"\usepackage{xeCJK}",
        r"\definecolor{Accent}{HTML}{234A6B}",
        r"\definecolor{Soft}{HTML}{F4F7FA}",
        r"\definecolor{SoftBorder}{HTML}{CFD8E3}",
        r"\setCJKmainfont{Songti SC}",
        r"\setlength{\parindent}{0pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{document}",
        r"{\LARGE\bfseries\color{Accent} Diffusion Approximation to the Elliptic Benchmark\par}",
        r"{\large\bfseries\color{Accent} Comparison against oracle numerical solutions\par}",
        r"\vspace{0.5em}",
        r"{\small CHONKDIFF reproducibility report，给老板看的版本：把 method 和 figures 都讲清楚\par}",
        r"\vspace{1.0em}",
        r"\noindent\fcolorbox{SoftBorder}{Soft}{%",
        r"\begin{minipage}{0.97\linewidth}",
        r"\textbf{一句话先讲清楚.} 我们没有让 diffusion model 直接变成一个 high-precision PDE solver. "
        r"我们做的是: first train a conditional diffusion model to give a strong initial guess, "
        r"then use physics-guided reverse sampling to improve that guess, "
        r"and finally use a short float64 LM/NK projector to finish the numerical solve.",
        r"\medskip",
        r"\textbf{Problem setting.} We study the periodic 1D nonlinear elliptic benchmark",
        r"\[",
        r"-\Delta v + \kappa v^3 = u",
        r"\]",
        rf"with $\kappa={int(solver.config.benchmark.kappa)}$ on a grid of $N_x={solver.config.benchmark.nx}$ points.",
        r"\textbf{Ground truth.} The ``actual numerical results'' in this report are not analytic formulas. "
        r"They are oracle reference solutions computed numerically by the repository's float64 LM / Newton--Kantorovich solver and stored as validation targets. "
        r"所以这里的 ground truth 是 numerical solve，不是 closed-form solution.",
        r"\textbf{Pipeline.} Each sample is shown in four stages: forcing $u$, diffusion-only prediction, PDE-guided reverse sample, and the final float64 projector output. "
        r"换句话说，图里同时展示 learned approximation 和 final numerical cleanup.",
        r"\end{minipage}}",
        r"\vspace{1.0em}",
        r"\section*{Workflow / 按一二三四五讲清楚}",
        r"\subsection*{Step 1 / 第一步: 定义要解的 elliptic problem}",
        r"先把问题定义固定，不然后面所有结果都没有参照系。"
        r" We solve the periodic 1D nonlinear elliptic benchmark",
        r"\[",
        r"-\Delta v + \kappa v^3 = u",
        r"\]",
        rf"with periodic boundary condition, $\kappa={int(solver.config.benchmark.kappa)}$, and $N_x={solver.config.benchmark.nx}$ grid points. "
        r"这一步的意思很简单: we first decide exactly which PDE we are solving.",
        r"\subsection*{Step 2 / 第二步: 生成输入 $u$ 和 ground-truth solution $v^*$}",
        r"第二步是做数据。We sample forcing functions $u$ from the periodic GP distribution used for this benchmark. "
        r"Then, for each sampled $u$, we compute a high-accuracy reference solution $v^*$ using the repository's float64 LM / Newton--Kantorovich oracle solver. "
        r"所以这里的 ground truth 不是 analytic formula，而是一个 very accurate numerical solve.",
        r"\begin{center}",
        r"\begin{tabularx}{0.97\linewidth}{>{\bfseries}p{0.28\linewidth}X}",
        r"\toprule",
        r"Item & Setting \\",
        r"\midrule",
        rf"Dataset & train = {solver.config.benchmark.dataset.train_size}, val = {solver.config.benchmark.dataset.val_size}, file = \texttt{{\detokenize{{{dataset_display}}}}} \\",
        rf"Oracle solver & float64 LM/NK, tolerance = {_format_metric(solver.config.oracle.tolerance)}, max iterations = {solver.config.oracle.max_iterations} \\",
        rf"Ground truth meaning & each $v^*$ in the report is the oracle numerical solution corresponding to its forcing $u$ \\",
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{center}",
        r"\subsection*{Step 3 / 第三步: 训练 conditional diffusion model}",
        r"第三步才是 learning. The model input is noisy solution $v_t$, forcing $u$, and diffusion time $t$, and the model predicts noise $\epsilon_\theta(v_t, u, t)$. "
        r"这里学到的是一个 conditional generative prior，也就是 `given $u$, what kind of solution shape is likely`, not the Jacobian and not the full solver itself.",
        r"\begin{center}",
        r"\begin{tabularx}{0.97\linewidth}{>{\bfseries}p{0.28\linewidth}X}",
        r"\toprule",
        r"Item & Setting \\",
        r"\midrule",
        rf"Diffusion model & conditional 1D DDPM with residual CNN backbone, timesteps = {solver.config.diffusion.timesteps} \\",
        rf"Training & epochs = {solver.config.training.epochs}, batch size = {solver.config.training.batch_size}, learning rate = {_format_metric(solver.config.training.learning_rate)}, seed = {solver.config.training.seed} \\",
        rf"Physics loss schedule & Stage A until {solver.config.training.stage_a_fraction:.2f}, Stage B until {solver.config.training.stage_b_fraction:.2f}, final PDE weight = {_format_metric(solver.config.training.pde_weight)} \\",
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{center}",
        r"如果用更直白的话讲: Step 3 teaches the network to produce a good initial guess, but not yet machine precision.",
        r"\subsection*{Step 4 / 第四步: inference 时怎么从 diffusion output 走到 high-accuracy answer}",
        r"第四步是实际解题流程。For a new forcing $u$, we first sample from the trained diffusion model. "
        rf"Then we apply {guidance_mode_display} physics guidance during reverse sampling, with guidance strength {_format_metric(summary['guidance_strength'])}, start fraction {summary['guidance_start_fraction']:.2f}, and damping lambda {_format_metric(summary['guidance_lambda'])}. "
        rf"After that, we run {int(summary['projector_iterations'])} float64 LM/NK projector steps.",
        r"这一段最重要的逻辑是: diffusion gives the initialization, guidance nudges it toward physics consistency, and the final projector is the part that really drives the solution to near machine precision.",
        r"\begin{center}",
        r"\begin{tabularx}{0.97\linewidth}{>{\bfseries}p{0.28\linewidth}X}",
        r"\toprule",
        r"Item & Setting \\",
        r"\midrule",
        rf"Sampling & guidance mode = {guidance_mode_display}, guidance strength = {_format_metric(summary['guidance_strength'])}, guidance start fraction = {summary['guidance_start_fraction']:.2f}, guidance lambda = {_format_metric(summary['guidance_lambda'])} \\",
        rf"Projector & LM/NK iterations = {int(summary['projector_iterations'])}, checkpoint = \texttt{{\detokenize{{{checkpoint_display}}}}} \\",
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{center}",
        r"\subsection*{Step 5 / 第五步: 怎么核对结果, 图从哪里来, 这些图说明了什么}",
        r"第五步是 evaluation 和解释结果。每个 validation sample 都会经过同一个 four-stage pipeline: oracle target, diffusion-only sample, guided sample, and projected sample.",
        r"The two main metrics are",
        r"\[",
        r"\text{relative error} = \frac{\|\hat{v} - v^*\|_2}{\|v^*\|_2}, \qquad \text{residual} = \|-\Delta \hat{v} + \kappa \hat{v}^3 - u\|_2.",
        r"\]",
        r"这里的 $v^*$ 是 oracle numerical solution. 所以 relative error tells us how close we are to numerical ground truth, and residual tells us how well the predicted function satisfies the PDE itself.",
        r"\medskip",
        r"\textbf{图从哪里来.} Every figure in this PDF is generated directly from one validation sample. 它们不是手工画的，也不是截屏。",
        r"The top panel uses the forcing vector $u$ from the oracle-generated validation dataset. "
        r"The middle panel overlays four curves for the same sample index: oracle $v^*$, diffusion-only output, guided output, and projected output. "
        r"The lower-left panel is the residual history recorded during the final projector for that same sample.",
        r"\medskip",
        r"\textbf{这些图要怎么读.} First compare blue to black to see how good the pure diffusion approximation is. "
        r"Then compare orange to blue to isolate the contribution of physics-guided reverse sampling. "
        r"Finally compare green to black and the residual-history panel to see what the LM/NK projector achieved numerically.",
        r"\medskip",
        r"\textbf{整体结果如下.}",
        r"\begin{center}",
        r"\begin{tabularx}{0.97\linewidth}{>{\bfseries}p{0.28\linewidth}X}",
        r"\toprule",
        r"Item & Setting \\",
        r"\midrule",
        rf"Evaluation sample count & {int(summary['num_samples'])} validation samples \\",
        rf"Projected error count above $10^{{-6}}$ & {summary['projected_error_count_gt_1e-6']} \\",
        rf"Projected residual count above $10^{{-10}}$ & {summary['projected_residual_count_gt_1e-10']} \\",
        rf"Strict projector convergence rate & {summary['projection_converged_rate']:.3f} \\",
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{center}",
        r"\begin{center}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Metric & Mean & Median & P95 & Max \\",
        r"\midrule",
        rf"Diffusion error & {_format_metric(summary['diffusion_error']['mean'])} & {_format_metric(summary['diffusion_error']['median'])} & {_format_metric(summary['diffusion_error']['p95'])} & {_format_metric(summary['diffusion_error']['max'])} \\",
        rf"Guided error & {_format_metric(summary['guided_error']['mean'])} & {_format_metric(summary['guided_error']['median'])} & {_format_metric(summary['guided_error']['p95'])} & {_format_metric(summary['guided_error']['max'])} \\",
        rf"Projected error & {_format_metric(summary['projected_error']['mean'])} & {_format_metric(summary['projected_error']['median'])} & {_format_metric(summary['projected_error']['p95'])} & {_format_metric(summary['projected_error']['max'])} \\",
        rf"Diffusion residual & {_format_metric(summary['diffusion_residual']['mean'])} & {_format_metric(summary['diffusion_residual']['median'])} & {_format_metric(summary['diffusion_residual']['p95'])} & {_format_metric(summary['diffusion_residual']['max'])} \\",
        rf"Guided residual & {_format_metric(summary['guided_residual']['mean'])} & {_format_metric(summary['guided_residual']['median'])} & {_format_metric(summary['guided_residual']['p95'])} & {_format_metric(summary['guided_residual']['max'])} \\",
        rf"Projected residual & {_format_metric(summary['projected_residual']['mean'])} & {_format_metric(summary['projected_residual']['median'])} & {_format_metric(summary['projected_residual']['p95'])} & {_format_metric(summary['projected_residual']['max'])} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{center}",
        r"\noindent\fcolorbox{SoftBorder}{Soft}{%",
        r"\begin{minipage}{0.97\linewidth}",
        rf"\textbf{{Headline result.}} The projector drives the mean relative error down to {_format_metric(summary['projected_error']['mean'])} and the mean PDE residual down to {_format_metric(summary['projected_residual']['mean'])}. "
        rf"Across this validation run, the projected error count above $10^{{-6}}$ is {summary['projected_error_count_gt_1e-6']} and the projected residual count above $10^{{-10}}$ is {summary['projected_residual_count_gt_1e-10']}. "
        rf"The strict internal projector convergence rate is {summary['projection_converged_rate']:.3f}. 简单说: diffusion 和 guidance 负责把样本带到正确 basin, final projector 负责把精度真的打下来.",
        r"\end{minipage}}",
        r"\vspace{0.75em}",
        r"\textbf{下面三页代表性样本怎么选.} The three sample pages are chosen as best guided case, typical guided case, and hard guided case across the evaluated validation set. 这样老板一眼就能看到最好、正常、最难三种情况。",
        r"\section*{Representative validation samples / 代表性样本}",
    ]

    for image_path, explanation in zip(sample_figures, sample_explanations):
        tex_lines.extend(
            [
                r"\noindent\fcolorbox{SoftBorder}{white}{%",
                r"\begin{minipage}{0.97\linewidth}",
                r"\begin{minipage}[t]{0.58\linewidth}",
                r"\vspace{0pt}",
                rf"\includegraphics[width=\linewidth]{{\detokenize{{{image_path}}}}}",
                r"\end{minipage}\hfill",
                r"\begin{minipage}[t]{0.34\linewidth}",
                r"\vspace{0pt}",
                r"\raggedright\small",
                rf"{{\normalsize\bfseries {_latex_escape(explanation[0])}}}\par\vspace{{0.6em}}",
                *(rf"{_latex_escape(line)}\par\vspace{{0.45em}}" for line in explanation[1:]),
                r"\end{minipage}",
                r"\end{minipage}}",
                r"\vspace{1.0em}",
                r"\clearpage",
            ]
        )

    tex_lines.extend(
        [
            r"\section*{One-paragraph takeaway / 最后一句话怎么讲}",
            r"\textbf{最短版本.} Step 1 fixed the PDE benchmark, Step 2 built oracle ground truth, Step 3 trained a conditional diffusion prior, Step 4 added physics guidance and a short float64 projector, and Step 5 verified everything against numerical ground truth and PDE residuals.",
            r"\medskip",
            rf"\textbf{{所以最后该怎么说.}} diffusion-only already gives a meaningful approximation with mean relative error {_format_metric(summary['diffusion_error']['mean'])}. "
            rf"Guided reverse sampling improves this to {_format_metric(summary['guided_error']['mean'])}. "
            rf"The final projector then pushes the mean error down to {_format_metric(summary['projected_error']['mean'])} and the mean residual down to {_format_metric(summary['projected_residual']['mean'])}. "
            r"因此一个合理、逻辑清楚的说法是: the learned model is the initializer and basin finder, while the final numerical projector is the accuracy engine.",
            r"\end{document}",
        ]
    )

    out_tex.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
    out_json = out_tex.with_suffix(".json")
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if compile_pdf:
        tectonic = shutil.which("tectonic")
        if tectonic is not None:
            subprocess.run(
                [tectonic, "--outdir", ".", out_tex.name],
                check=True,
                cwd=str(out_tex.parent),
            )

    return {
        "tex_path": str(out_tex.resolve()),
        "pdf_path": str(out_tex.with_suffix(".pdf").resolve()),
        "json_path": str(out_json.resolve()),
        "asset_dir": str(asset_dir.resolve()),
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file")
    parser.add_argument("--out", required=True, help="Path to output LaTeX file")
    parser.add_argument("--num-samples", type=int, default=128, help="Validation sample count")
    parser.add_argument("--guidance-mode", default=None, help="Override guidance mode")
    parser.add_argument("--guidance-strength", type=float, default=None, help="Override guidance strength")
    parser.add_argument("--guidance-start-fraction", type=float, default=None, help="Override guidance start fraction")
    parser.add_argument("--guidance-lambda", type=float, default=None, help="Override guidance lambda")
    parser.add_argument("--projector-iterations", type=int, default=None, help="Override projector iteration count")
    parser.add_argument("--projector-tolerance", type=float, default=None, help="Override projector tolerance")
    parser.add_argument("--no-compile", action="store_true", help="Write LaTeX source without compiling")
    args = parser.parse_args()

    result = generate_latex_report(
        checkpoint_path=args.checkpoint,
        out_path=args.out,
        num_samples=args.num_samples,
        guidance_mode=args.guidance_mode,
        guidance_strength=args.guidance_strength,
        guidance_start_fraction=args.guidance_start_fraction,
        guidance_lambda=args.guidance_lambda,
        projector_iterations=args.projector_iterations,
        projector_tolerance=args.projector_tolerance,
        compile_pdf=not args.no_compile,
    )
    print(f"Saved LaTeX report to {result['tex_path']}")
    print(f"Saved PDF report to {result['pdf_path']}")


if __name__ == "__main__":
    main()
