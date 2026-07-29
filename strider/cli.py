"""Command-line interface for STRIDER inference."""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from strider import __version__, load_model
from strider.io import SpectrumInput, load_inputs
from strider.model_info import MODEL_FILENAME, model_id, verify_model
from strider.results import compact_result, json_safe, redshift_quality_label


_REPO_MODEL = Path(__file__).resolve().parent.parent / "models" / MODEL_FILENAME


def _default_model() -> Path:
    """Model path: STRIDER_MODEL, else the clone's models/ directory."""
    override = os.environ.get("STRIDER_MODEL")
    if override:
        return Path(override)
    return _REPO_MODEL if _REPO_MODEL.is_file() else Path("models") / MODEL_FILENAME


def _add_model_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        "--checkpoint",
        dest="model",
        type=Path,
        default=_default_model(),
        help="STRIDER model file (default: models/strider.pt or STRIDER_MODEL).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="strider", description="STRIDER inference")
    parser.add_argument("--version", action="version", version=f"STRIDER {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check-model", help="Verify the production model")
    _add_model_argument(check)

    classify = commands.add_parser("classify", help="Classify one spectrum or time series")
    classify.add_argument(
        "input", nargs="+", type=Path, help="One input file, or one file per epoch"
    )
    _add_model_argument(classify)
    classify.add_argument(
        "--phase",
        nargs="+",
        type=float,
        help="Rest-frame phase in days; give one value per input file.",
    )
    classify.add_argument("--device", default="cpu", help="Torch device: cpu or cuda")
    classify.add_argument("--z-prior", nargs=2, type=float, metavar=("MEAN", "SIGMA"))
    classify.add_argument("--z-window", nargs=2, type=float, metavar=("MIN", "MAX"))
    classify.add_argument(
        "--wave-range", nargs=2, type=float, metavar=("MIN_AA", "MAX_AA")
    )
    classify.add_argument("--condition-class", help="Also report p(z | class, spectrum)")
    classify.add_argument(
        "--wavelength-unit",
        choices=("auto", "angstrom", "micron"),
        default="auto",
    )
    phase_group = classify.add_mutually_exclusive_group()
    phase_group.add_argument("--latest-phase", type=float)
    phase_group.add_argument("--phase-range", nargs=2, type=float, metavar=("MIN", "MAX"))
    phase_group.add_argument("--phase-list", nargs="+", type=float, metavar="PHASE")
    classify.add_argument("--uncalibrated", action="store_true")
    classify.add_argument("--top-k", type=int, default=5)
    classify.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show input coverage, the gold-Ia cut, and redshift details",
    )
    classify.add_argument(
        "--output-text",
        type=Path,
        help="Write the concise overall result as text",
    )
    classify.add_argument("--output-json", type=Path, help="Write the compact result")
    classify.add_argument("--diagnostics-json", type=Path, help="Write the full result")
    classify.add_argument("--plot", type=Path, help="Write an evidence-map PNG")
    classify.add_argument(
        "--plot-evolution",
        type=Path,
        help="Write a cumulative evidence GIF, one frame per added spectrum",
    )
    classify.add_argument(
        "--plot-epochs",
        type=Path,
        help="Write one cumulative evidence-map PNG per epoch to a directory",
    )
    classify.add_argument(
        "--plot-confidence",
        type=Path,
        help="Write Ia versus non-Ia probability against cumulative spectral S/N",
    )
    classify.add_argument("--json", action="store_true", help="Print compact JSON")
    return parser


def select_phases(
    data: SpectrumInput,
    *,
    latest: float | None,
    phase_range: Sequence[float] | None,
    phase_list: Sequence[float] | None,
) -> tuple[SpectrumInput, str | None]:
    phases = np.atleast_1d(np.asarray(data.phase, dtype=np.float32))
    flux = np.asarray(data.flux)
    if flux.ndim == 1:
        flux = flux[None, :]
    if flux.shape[0] != phases.size:
        raise ValueError("Flux epoch count does not match phase count")

    mask = np.ones(phases.size, dtype=bool)
    description = None
    if latest is not None:
        mask = phases <= float(latest)
        description = f"phase <= {latest:g} d"
    elif phase_range is not None:
        lo, hi = sorted(float(value) for value in phase_range)
        mask = (phases >= lo) & (phases <= hi)
        description = f"{lo:g} <= phase <= {hi:g} d"
    elif phase_list is not None:
        wanted = np.asarray(phase_list, dtype=np.float32)
        mask = np.any(np.isclose(phases[:, None], wanted[None, :], atol=1e-4), axis=1)
        description = "phase in {" + ", ".join(f"{value:g}" for value in wanted) + "} d"
    if not mask.any():
        raise ValueError(f"Phase selection {description} leaves no spectra")

    error = None
    if data.flux_err is not None:
        error_array = np.asarray(data.flux_err)
        if error_array.ndim == 1:
            error_array = error_array[None, :]
        error = error_array[mask]
    metadata = dict(data.metadata)
    if description:
        metadata["phase_selection"] = description
    return (
        SpectrumInput(
            wavelength=data.wavelength,
            flux=flux[mask],
            phase=phases[mask],
            flux_err=error,
            metadata=metadata,
        ),
        description,
    )


def save_evidence_map(output: dict[str, Any], metadata: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cache = path.parent / ".matplotlib-cache"
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    try:
        import cmasher  # noqa: F401
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Evidence maps require matplotlib and cmasher") from exc
    from strider.plotting.evidence_map import evidence_map

    fig = plt.figure(figsize=(12.2, 9.4), dpi=160)
    fig.patch.set_facecolor("white")
    evidence_map(fig, out=output, meta=metadata)
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def save_evidence_evolution(
    outputs: Sequence[dict[str, Any]],
    metadata: dict[str, Any],
    path: Path,
) -> None:
    if path.suffix.lower() != ".gif":
        raise ValueError("--plot-evolution output must end in .gif")
    path.parent.mkdir(parents=True, exist_ok=True)
    cache = path.parent / ".matplotlib-cache"
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    try:
        import cmasher  # noqa: F401
        import matplotlib.animation as animation
        import matplotlib.pyplot as plt
        from PIL import Image  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Evidence GIFs require matplotlib, cmasher and Pillow"
        ) from exc
    from strider.plotting.evidence_map import evidence_map

    total = len(outputs)
    if total == 0:
        raise ValueError("Evidence evolution needs at least one spectrum")
    fig = plt.figure(figsize=(12.2, 9.4), dpi=120)
    fig.patch.set_facecolor("white")

    def draw(frame_index: int) -> tuple[()]:
        output = outputs[frame_index]
        phases = np.asarray(output["spectra"]["phase_days"], dtype=float)
        frame_metadata = dict(metadata)
        frame_metadata["frame_label"] = (
            f"{frame_index + 1} of {total} spectra"
            f" · through phase {np.nanmax(phases):+.0f} d"
        )
        evidence_map(fig, out=output, meta=frame_metadata)
        return ()

    frame_indices = list(range(total)) + ([total - 1] if total > 1 else [])
    movie = animation.FuncAnimation(
        fig,
        draw,
        frames=frame_indices,
        interval=1250,
        repeat_delay=1500,
        blit=False,
    )
    movie.save(
        path,
        writer=animation.PillowWriter(fps=0.8),
        dpi=120,
        savefig_kwargs={"facecolor": "white"},
    )
    plt.close(fig)


def save_evidence_epochs(
    outputs: Sequence[dict[str, Any]],
    metadata: dict[str, Any],
    directory: Path,
) -> list[Path]:
    """Write the cumulative evidence map after each added spectrum."""
    if not outputs:
        raise ValueError("Epoch evidence maps need at least one spectrum")
    directory.mkdir(parents=True, exist_ok=True)
    cache = directory / ".matplotlib-cache"
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    try:
        import cmasher  # noqa: F401
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Evidence maps require matplotlib and cmasher") from exc
    from strider.plotting.evidence_map import evidence_map

    total = len(outputs)
    paths: list[Path] = []
    for frame_index, output in enumerate(outputs):
        phases = np.asarray(output["spectra"]["phase_days"], dtype=float)
        latest_phase = float(np.nanmax(phases))
        phase_token = (
            f"n{abs(latest_phase):.1f}"
            if latest_phase < 0
            else f"p{latest_phase:.1f}"
        )
        frame_metadata = dict(metadata)
        frame_metadata["frame_label"] = (
            f"{frame_index + 1} of {total} spectra"
            f" · through phase {latest_phase:+.0f} d"
        )
        path = directory / f"epoch_{frame_index + 1:02d}_phases_{phase_token}d.png"
        fig = plt.figure(figsize=(12.2, 9.4), dpi=160)
        fig.patch.set_facecolor("white")
        evidence_map(fig, out=output, meta=frame_metadata)
        fig.savefig(path, facecolor=fig.get_facecolor())
        plt.close(fig)
        paths.append(path)
    return paths


def save_confidence_plot(
    outputs: Sequence[dict[str, Any]],
    data: SpectrumInput,
    path: Path,
) -> None:
    if data.flux_err is None:
        raise ValueError("--plot-confidence needs flux_err in the input")
    path.parent.mkdir(parents=True, exist_ok=True)
    cache = path.parent / ".matplotlib-cache"
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Confidence plots require matplotlib") from exc
    from strider.plotting.confidence import confidence_curve

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=160)
    confidence_curve(
        ax,
        outputs=outputs,
        flux=data.flux,
        flux_err=data.flux_err,
        phase=data.phase,
        true_class=data.metadata.get("true_class"),
    )
    object_id = str(data.metadata.get("object", "input"))
    ax.set_title(object_id, loc="left", fontsize=14, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(path, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _classify(
    model: Any,
    data: SpectrumInput,
    args: argparse.Namespace,
    *,
    return_evidence: bool,
) -> dict[str, Any]:
    return model.classify(
        wavelength=data.wavelength,
        flux=data.flux,
        phase=data.phase,
        flux_err=data.flux_err,
        z_prior=tuple(args.z_prior) if args.z_prior else None,
        z_window=tuple(args.z_window) if args.z_window else None,
        wave_range=tuple(args.wave_range) if args.wave_range else None,
        condition_class=args.condition_class,
        wavelength_unit=args.wavelength_unit,
        calibrated=not args.uncalibrated,
        return_joint=return_evidence,
        return_inputs=return_evidence,
    )


def _evolution_outputs(
    model: Any,
    data: SpectrumInput,
    args: argparse.Namespace,
    final_output: dict[str, Any],
) -> list[dict[str, Any]]:
    phases = np.atleast_1d(np.asarray(data.phase, dtype=np.float32))
    flux = np.atleast_2d(np.asarray(data.flux))
    error = None if data.flux_err is None else np.atleast_2d(np.asarray(data.flux_err))
    order = np.argsort(phases)
    outputs: list[dict[str, Any]] = []
    for count in range(1, len(order)):
        selected = order[:count]
        partial = SpectrumInput(
            wavelength=data.wavelength,
            flux=flux[selected],
            phase=phases[selected],
            flux_err=None if error is None else error[selected],
            metadata=data.metadata,
        )
        outputs.append(_classify(model, partial, args, return_evidence=True))
    outputs.append(final_output)
    return outputs


def print_summary(
    output: dict[str, Any], metadata: dict[str, Any], top_k: int, *, verbose: bool = False
) -> None:
    result = compact_result(output, metadata=metadata, checkpoint=MODEL_FILENAME)
    classification = result["classification"]
    redshift = result["redshift"]
    input_block = result["input"]
    probabilities = classification["probabilities"]
    ranked = [
        (item["class"], item["probability"])
        for item in classification["top_classes"]
    ]
    if top_k > len(ranked):
        raw_scores = np.asarray(output["p_class_uncal_with_controls"], dtype=float)
        ranked = [
            (name, probabilities[name])
            for name in np.asarray(output["class_names"])[np.argsort(raw_scores)[::-1]]
        ]
    top_k = max(1, min(int(top_k), len(ranked)))

    phase_days = list(input_block["phase_days"])
    if len(phase_days) == 1:
        phase_text = f"phase {phase_days[0]:+.1f} d"
    else:
        phase_text = f"phase {min(phase_days):+.1f} to {max(phase_days):+.1f} d"

    print(f"STRIDER  {result['object_id']}")
    print("─" * 50)
    print(f"Input      {input_block['n_spectra']} spectra, {phase_text}")
    if verbose:
        wave_lo, wave_hi = input_block["wavelength_range_aa"]
        q = input_block["quality"]
        errors = "yes" if q["flux_error_supplied"] else "no"
        print(f"           {wave_lo:.0f}-{wave_hi:.0f} A, "
              f"model coverage {q['model_wavelength_coverage_fraction']:.0%}, "
              f"flux errors {errors}")
    print()

    class_name = classification["class"]
    print(f"Class      {class_name:<8} {classification['confidence']:.3f}")
    if verbose:
        if class_name != "Ia":
            print(f"           {'P(Ia)':<8} {classification['p_Ia']:.3f}")
        if classification["gold_Ia_selected"]:
            print("           in the gold-Ia cosmology sample")
        elif class_name == "Ia":
            print("           not in the gold-Ia sample (a stricter cut than the "
                  "class probability)")

    quality = redshift.get("quality")
    print(f"Redshift   z = {redshift['z_STRIDER']:.4f}   "
          f"68% [{redshift['z_p16']:.4f}, {redshift['z_p84']:.4f}]   "
          f"{redshift_quality_label(quality)}")
    if verbose:
        interval_kind = "calibrated" if redshift["interval_calibration"].get("applied") else "raw"
        print(f"           {interval_kind} interval")
        if result["controls"]["prior_or_window_supplied"]:
            spectra_z = redshift["spectra_only"]["z_S"]
            comparison = "prior" if result["controls"]["z_prior"] is not None else "redshift window"
            print(f"           spectra only z = {spectra_z:.4f} before {comparison}")
        if quality == "ambiguous":
            print(f"           about a 5% chance the true redshift is above "
                  f"z = {redshift['z_p95']:.2f}")
    print()

    label = f"Top {top_k}"
    for index, (name, probability) in enumerate(ranked[:top_k]):
        prefix = f"{label:<10}" if index == 0 else " " * 10
        print(f"{prefix} {name:<6} {probability:.3f}")


def summary_text(
    output: dict[str, Any],
    metadata: dict[str, Any],
    top_k: int,
    *,
    verbose: bool = False,
) -> str:
    buffer = StringIO()
    with redirect_stdout(buffer):
        print_summary(output, metadata, top_k, verbose=verbose)
    return buffer.getvalue()


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2) + "\n")


def run_classify(args: argparse.Namespace) -> int:
    model_path = args.model.expanduser().resolve()
    # load_model rejects a missing/bad file; the provenance hash is taken in compact_result.
    data = load_inputs(args.input, phases=args.phase)
    data, _ = select_phases(
        data,
        latest=args.latest_phase,
        phase_range=args.phase_range,
        phase_list=args.phase_list,
    )

    model = load_model(model_path, device=args.device)
    need_evidence = (
        args.plot is not None
        or args.plot_evolution is not None
        or args.plot_epochs is not None
    )
    output = _classify(model, data, args, return_evidence=need_evidence)
    compact = compact_result(output, metadata=data.metadata, checkpoint=model_path)
    summary = summary_text(output, data.metadata, args.top_k, verbose=args.verbose)
    if not args.json:
        print(summary, end="")
    if args.output_text:
        _write_text(args.output_text, summary)
        if not args.json:
            print(f"\nWrote summary: {args.output_text}")
    if args.output_json:
        _write_json(args.output_json, compact)
        if not args.json:
            print(f"\nWrote result : {args.output_json}")
    if args.diagnostics_json:
        _write_json(args.diagnostics_json, output)
        if not args.json:
            print(f"Wrote details: {args.diagnostics_json}")
    if args.plot:
        save_evidence_map(output, data.metadata, args.plot)
        if not args.json:
            print(f"Wrote plot   : {args.plot}")
    evolution = None
    if args.plot_evolution or args.plot_epochs or args.plot_confidence:
        evolution = _evolution_outputs(model, data, args, output)
    if args.plot_evolution:
        assert evolution is not None
        save_evidence_evolution(evolution, data.metadata, args.plot_evolution)
        if not args.json:
            print(f"Wrote GIF    : {args.plot_evolution}")
    if args.plot_epochs:
        assert evolution is not None
        paths = save_evidence_epochs(evolution, data.metadata, args.plot_epochs)
        if not args.json:
            print(f"Wrote epochs : {args.plot_epochs} ({len(paths)} PNGs)")
    if args.plot_confidence:
        assert evolution is not None
        save_confidence_plot(evolution, data, args.plot_confidence)
        if not args.json:
            print(f"Wrote S/N    : {args.plot_confidence}")
    if args.json:
        print(json.dumps(json_safe(compact), indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "check-model":
            path = args.model.expanduser().resolve()
            digest = verify_model(path)
            model = load_model(path, device="cpu")
            n_classes = len(model.metadata.class_names)
            print("STRIDER model verified")
            print(f"  model     {model_id(n_classes)}")
            print(f"  file      {path}")
            print(f"  sha256    {digest[:12]}...")
            print(f"  classes   {n_classes}")
            print(f"  redshift  {model.metadata.z_min:g} - {model.metadata.z_max:g}")
            return 0
        return run_classify(args)
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"strider: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
