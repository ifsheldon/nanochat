# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Render recorded RLM training curves, optionally comparing original NanoChat logs."""

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
import numpy as np


BETA = 0.9
MEDIAN_WINDOW = 31
BLUE, ORANGE, INK = "#2563A6", "#CF6518", "#263442"


@dataclass(frozen=True)
class RlmUpdate:
    update: int
    mean: float
    last_batch: float
    cycles: tuple[float, ...]
    seconds: float
    frozen: bool | None
    attempt: str

    @classmethod
    def from_record(cls, record, attempt):
        row = cls(
            update=record["completed_updates"], mean=float(record["loss"]),
            last_batch=float(record["last_batch_loss"]),
            cycles=tuple(float(value) for value in record["cycle_losses"]),
            seconds=float(record["duration_seconds"]),
            frozen=record.get("input_embedding_frozen"), attempt=attempt,
        )
        values = (row.mean, row.last_batch, row.seconds, *row.cycles)
        if not row.cycles or not np.isfinite(values).all() or row.seconds <= 0:
            raise ValueError("Losses must be finite, cycle losses nonempty, and durations positive")
        if type(row.update) is not int or row.update < 1:
            raise ValueError("completed_updates must be a positive integer")
        if row.frozen is not None and type(row.frozen) is not bool:
            raise ValueError("input_embedding_frozen must be a boolean when recorded")
        return row


def source_record(path, data):
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}


def require_contiguous(updates, label):
    if not updates:
        raise ValueError(f"No completed updates found in {label}")
    if any(right != left + 1 for left, right in zip(updates, updates[1:])):
        raise ValueError(f"{label} has gaps, duplicates, or out-of-order updates; select the authoritative history")


def read_run(root, through_update):
    rows, sources, contracts = [], [], []
    ignored = 0
    for path in sorted(root.glob("attempts/*/updates.jsonl")):
        data = path.read_bytes()
        complete = data[:data.rfind(b"\n") + 1]
        ignored += len(data) - len(complete)
        batch = []
        for number, line in enumerate(complete.splitlines(), 1):
            if not line.strip():
                continue
            try:
                batch.append(RlmUpdate.from_record(json.loads(line), path.parent.name))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{path}:{number}: {error}") from error
        rows.extend(batch)
        sources.append({**source_record(path, complete), "records": len(batch),
                        "ignored_partial_trailing_bytes": len(data) - len(complete)})
        invocation = path.with_name("invocation.json")
        if batch and invocation.exists() and (through_update is None or batch[0].update <= through_update):
            data = invocation.read_bytes()
            config = json.loads(data).get("contract", {}).get("training_config", {})
            contracts.append({**source_record(invocation, data), "training_config": config})
    if through_update is not None:
        if not any(row.update == through_update for row in rows):
            raise ValueError(f"RLM update {through_update} is not available")
        rows = [row for row in rows if row.update <= through_update]
    require_contiguous([row.update for row in rows], str(root))
    if any(len(row.cycles) != len(rows[0].cycles) for row in rows):
        raise ValueError("Cycle count changes within the selected history; plot those configurations separately")
    return rows, sources, contracts, ignored


def read_original(path):
    data = path.read_bytes()
    pattern = re.compile(r"^step\s+(\d+)/(\d+).*?\| loss:\s+([\d.eE+-]+).*?\| dt:\s+([\d.eE+-]+)ms")
    rows = []
    for number, line in enumerate(data.decode().splitlines(), 1):
        line = re.sub(r"\x1b\[[0-9;]*m", "", line).strip()
        match = pattern.match(line)
        if match:
            rows.append((int(match[1]) + 1, float(match[3]), float(match[4]) / 1000))
        elif re.match(r"^step\s+\d+/", line):
            raise ValueError(f"{path}:{number}: unrecognized base_train metric record")
    require_contiguous([row[0] for row in rows], str(path))
    values = np.asarray(rows)
    if not np.isfinite(values).all() or np.any(values[:, 2] <= 0):
        raise ValueError(f"{path}: original losses must be finite and durations positive")
    return values, source_record(path, data)


def ema(values):
    result = []
    value = 0.0
    for count, raw in enumerate(values, 1):
        value = BETA * value + (1 - BETA) * raw
        result.append(value / (1 - BETA ** count))
    return np.asarray(result)


def rolling_median(values):
    return np.asarray([np.median(values[max(0, i - MEDIAN_WINDOW + 1):i + 1]) for i in range(len(values))])


def style_axis(axis, title, ylabel, limits, boundaries, phases):
    axis.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=10)
    axis.set_xlabel("Completed optimizer updates")
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", color="#D9E0E6", linewidth=0.65, alpha=0.75)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    axis.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    if limits is None:
        axis.text(0.5, 0.5, "No overlapping updates", ha="center", transform=axis.transAxes)
        return
    low, high = limits
    axis.set_xlim(low if low < high else low - 0.5, high if low < high else high + 0.5)
    for boundary in boundaries:
        if low <= boundary + 0.5 <= high:
            axis.axvline(boundary + 0.5, color="#7D8791", linestyle=":", linewidth=1)
    for phase in phases:
        boundary = phase["completed_update"] - 0.5
        if low <= boundary <= high:
            axis.axvline(boundary, color="#16836C", linestyle="--", linewidth=1)
    if axis.lines:
        axis.legend(loc="best", fontsize=9, ncol=2, frameon=True, facecolor="white", edgecolor="none", framealpha=0.92)


def draw(axis, x, y, limits, **kwargs):
    if limits is not None:
        mask = (x >= limits[0]) & (x <= limits[1])
        if np.any(mask):
            axis.plot(x[mask], y[mask], **kwargs)


def save_figure(figure, output, name, title, subtitle, notes):
    figure.suptitle(title, x=0.065, ha="left", y=0.975, fontsize=19, fontweight="bold")
    figure.text(0.065, 0.975 - 0.52 / figure.get_figheight(), subtitle, fontsize=10)
    for index, note in enumerate(notes):
        figure.text(0.065, 0.075 - index * 0.024, note, fontsize=8.8)
    for suffix in ("png", "svg"):
        figure.savefig(output / f"{name}.{suffix}", dpi=180)
    plt.close(figure)


def render(output, rows, original, settings):
    x = np.asarray([row.update for row in rows])
    mean = ema([row.mean for row in rows])
    last = ema([row.last_batch for row in rows])
    cycles = np.column_stack([ema([row.cycles[i] for row in rows]) for i in range(len(rows[0].cycles))])
    seconds = np.asarray([row.seconds for row in rows])
    colors = plt.get_cmap("tab10").colors
    rlm_range = (int(x[0]), int(x[-1]))
    full = (min(rlm_range[0], int(original[0, 0])), max(rlm_range[1], int(original[-1, 0]))) if original is not None else rlm_range
    recent = (max(rlm_range[0], rlm_range[1] - settings["recent_updates"] + 1), rlm_range[1])
    comparison = original is not None
    second = settings["shared_range"] if comparison else recent
    second_title = "Shared available range" if comparison else "Recent RLM updates"
    boundary_note = "Dotted: resume. Dashed green: observed embedding-phase change. Update axis excludes downtime."
    loss_notes = [
        "Loss: bias-corrected EMA, beta = 0.9; original is already smoothed. Cycle curves are means over accumulation batches." if comparison else
        "Loss: bias-corrected EMA, beta = 0.9. Cycle curves are means over accumulation batches. Original baseline unavailable.",
        "Observed training metrics; objectives and initialization may differ. These curves do not establish held-out quality.", boundary_note,
    ]
    def decorate(axis, title, ylabel, limits):
        style_axis(axis, title, ylabel, limits, settings["resume_after_updates"], settings["phase_changes"])

    def loss_panel(axis, limits):
        if comparison:
            draw(axis, original[:, 0], original[:, 1], limits, color=BLUE, lw=1.7, label="NanoChat last-batch EMA")
        draw(axis, x, last, limits, color=ORANGE, lw=1.6, label="RLM last-batch EMA")
        draw(axis, x, mean, limits, color=INK, lw=1.3, ls="--", label="RLM update-mean EMA")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                         "axes.labelcolor": INK, "figure.facecolor": "white", "axes.facecolor": "white",
                         "legend.frameon": False, "svg.fonttype": "none"})
    figure, axes = plt.subplots(2, 2, figsize=(14.4, 9.6))
    figure.subplots_adjust(left=0.065, right=0.98, bottom=0.16, top=0.86, hspace=0.44, wspace=0.2)
    for axis, title, limits in ((axes[0, 0], "Full available loss history", full), (axes[0, 1], second_title, second)):
        loss_panel(axis, limits)
        decorate(axis, title, "Cross-entropy loss (nats)", limits)
    for axis, title, limits in ((axes[1, 0], "RLM loss by recurrent cycle", rlm_range), (axes[1, 1], "Recent cycle detail", recent)):
        for cycle in range(cycles.shape[1]):
            draw(axis, x, cycles[:, cycle], limits, color=colors[(cycle + 2) % len(colors)], lw=1.5, label=f"Cycle {cycle + 1}")
        draw(axis, x, mean, limits, color=INK, ls="--", lw=1.2, label="Update mean")
        decorate(axis, title, "Cycle mean loss (nats)", limits)
    save_figure(figure, output, "loss_curves", "Pretraining loss: NanoChat and RLM" if comparison else "RLM pretraining loss", settings["subtitle"], loss_notes)

    figure, axes = plt.subplots(1, 2, figsize=(14.4, 6.7))
    figure.subplots_adjust(left=0.065, right=0.98, bottom=0.22, top=0.82, wspace=0.2)
    for axis, title, limits in ((axes[0], "Full RLM history", rlm_range), (axes[1], "Recent detail", recent)):
        if comparison:
            draw(axis, original[:, 0], original[:, 1], limits, color=BLUE, lw=1.7, label="NanoChat last-batch EMA")
        for cycle in range(min(2, cycles.shape[1])):
            draw(axis, x, cycles[:, cycle], limits, color=colors[(cycle + 2) % len(colors)], lw=1.7, label=f"RLM cycle {cycle + 1} mean EMA")
        decorate(axis, title, "Cross-entropy loss (nats)", limits)
    save_figure(figure, output, "cycles_1_2", "Recurrent cycle 1" if cycles.shape[1] == 1 else "Recurrent cycles 1 and 2", settings["subtitle"], loss_notes)

    figure, axes = plt.subplots(1, 2, figsize=(14.4, 6.7))
    figure.subplots_adjust(left=0.065, right=0.98, bottom=0.22, top=0.82, wspace=0.2)
    timing = [(x, seconds, ORANGE, "RLM")]
    if comparison:
        timing.insert(0, (original[:, 0], original[:, 2], BLUE, "Original NanoChat"))
    for axis, title, limits in ((axes[0], "Full available history", full), (axes[1], second_title, second)):
        for tx, values, color, label in timing:
            draw(axis, tx, values, limits, color=color, lw=0.7, alpha=0.35)
            draw(axis, tx, rolling_median(values), limits, color=color, lw=1.8, label=label)
        decorate(axis, title, "Seconds per update", limits)
        axis.set_ylim(bottom=0)
    save_figure(figure, output, "time_per_update", "Recorded time per optimizer update", settings["subtitle"], [
        "Thin lines: every recorded duration, including startup spikes. Bold lines: trailing 31-update median.",
        "Timer scope and machine workloads may differ; these durations are not elapsed wall time or a controlled speed benchmark.", boundary_note,
    ])
    return last, mean, cycles


def write_csv(path, rows, original, last, mean, cycles):
    cycle_fields = [f"cycle_{i + 1}_mean" for i in range(cycles.shape[1])]
    ema_fields = [f"cycle_{i + 1}_ema" for i in range(cycles.shape[1])]
    fields = ["run", "completed_update", "attempt", "raw_update_mean", "raw_last_batch", "logged_last_batch_ema",
              "last_batch_ema", "update_mean_ema", *cycle_fields, *ema_fields, "duration_seconds", "input_embedding_frozen"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        if original is not None:
            for update, loss, duration in original:
                writer.writerow({"run": "original_nanochat", "completed_update": int(update),
                                 "logged_last_batch_ema": loss, "last_batch_ema": loss, "duration_seconds": duration})
        for i, row in enumerate(rows):
            writer.writerow({"run": "rlm", "completed_update": row.update, "attempt": row.attempt,
                             "raw_update_mean": row.mean, "raw_last_batch": row.last_batch,
                             "last_batch_ema": last[i], "update_mean_ema": mean[i],
                             "duration_seconds": row.seconds, "input_embedding_frozen": row.frozen,
                             **dict(zip(cycle_fields, row.cycles)), **dict(zip(ema_fields, cycles[i]))})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="RLM run containing attempts/*/updates.jsonl")
    parser.add_argument("--original-log", type=Path, help="Optional original base_train text log; omitted means RLM-only")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for figures, CSV, and provenance")
    parser.add_argument("--through-update", type=int, help="Include RLM history through this completed update")
    parser.add_argument("--recent-updates", type=int, default=200, help="Number of updates in recent-detail panels (default: 200)")
    parser.add_argument("--timezone", default="UTC", help="IANA display timezone (default: UTC)")
    args = parser.parse_args()
    if args.recent_updates < 1:
        parser.error("--recent-updates must be positive")
    stamp = datetime.now(timezone.utc)
    display_time = stamp.astimezone(ZoneInfo(args.timezone)).strftime("%d %b %Y, %H:%M %Z")
    rows, sources, contracts, ignored = read_run(args.run_dir.expanduser().resolve(), args.through_update)
    original, original_source = read_original(args.original_log.expanduser().resolve()) if args.original_log else (None, None)
    boundaries = [left.update for left, right in zip(rows, rows[1:]) if left.attempt != right.attempt]
    phases = [{"completed_update": right.update, "input_embedding_frozen": right.frozen}
              for left, right in zip(rows, rows[1:]) if left.frozen is not None and right.frozen is not None and left.frozen != right.frozen]
    shared = None
    if original is not None:
        low, high = max(rows[0].update, int(original[0, 0])), min(rows[-1].update, int(original[-1, 0]))
        shared = [low, high] if low <= high else None
    budget = contracts[-1]["training_config"].get("num_iterations") if contracts else None
    progress = f"RLM updates {rows[0].update:,}–{rows[-1].update:,}" + (f" / {budget:,}" if budget is not None else "")
    frozen = {row.frozen for row in rows}
    phase_label = "input frozen" if frozen == {True} else "input trainable" if frozen == {False} else "mixed/unknown input phase"
    subtitle = f"{progress}  |  {len(rows[0].cycles)} cycles  |  {phase_label}  |  {display_time}"
    provenance = {
        "mode": "comparison" if original is not None else "rlm-only", "snapshot_utc": stamp.isoformat(),
        "display_timezone": args.timezone, "rlm_updates": len(rows), "rlm_range": [rows[0].update, rows[-1].update],
        "original_updates": len(original) if original is not None else 0,
        "original_range": [int(original[0, 0]), int(original[-1, 0])] if original is not None else None,
        "shared_range": shared, "budget_updates": budget, "cycle_count": len(rows[0].cycles),
        "resume_after_updates": boundaries, "phase_changes": phases, "initial_input_embedding_frozen": rows[0].frozen,
        "rlm_sources": sources, "invocations": contracts, "original_source": original_source,
        "ignored_partial_trailing_bytes": ignored, "through_update": args.through_update,
        "original_printed_step_offset": 1 if original is not None else None,
        "loss_smoothing": {"kind": "bias_corrected_ema", "beta": BETA, "rlm_start_update": rows[0].update,
                           "continues_across_attempts": True, "original_already_smoothed": original is not None},
        "timing_trend": {"kind": "trailing_median", "window_updates": MEDIAN_WINDOW},
        "recent_updates": args.recent_updates, "subtitle": subtitle,
        "software": {"numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "generator": source_record(Path(__file__).resolve(), Path(__file__).read_bytes()),
        "caveats": ["Original loss is already-smoothed last-microbatch loss; RLM cycle means span accumulation batches."
                    if original is not None else "Original NanoChat baseline unavailable; no baseline values inferred.",
                    "Initialization, objectives, token exposure, hardware, and timer scope may differ between runs.",
                    "Training loss does not establish held-out performance; recorded durations exclude downtime."],
    }
    output = args.output_dir.expanduser().absolute()
    output.mkdir(parents=True, exist_ok=False)
    last, mean, cycles = render(output, rows, original, provenance)
    write_csv(output / "metrics.csv", rows, original, last, mean, cycles)
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "mode": provenance["mode"], "rlm_range": provenance["rlm_range"],
                      "original_updates": provenance["original_updates"], "shared_range": shared,
                      "ignored_partial_trailing_bytes": ignored}))


if __name__ == "__main__":
    main()
