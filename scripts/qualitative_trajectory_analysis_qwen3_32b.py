from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from shutil import get_terminal_size
from typing import Any, Iterable
import textwrap
from tqdm import tqdm 

@dataclass(frozen=True)
class TrajectoryInfo:
    instance_id: str
    input_path: Path
    comparison_path: Path | None
    input_data: dict[str, Any] | None
    comparison_data: dict[str, Any] | None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Qualitatively compare SWE-agent trajectories between an input directory and a comparison directory."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path(
            "trajectories/lindenbauer/main_experiments/"
            "local-Qwen3-32B_thinking-baseline_raw-collect_reasoning.0-verified-500"
        ),
        help="Directory to scan recursively for .traj files (source).",
    )
    parser.add_argument(
        "--comparison_dir",
        type=Path,
        default=Path(
            "trajectories/lindenbauer/main_experiments/"
            "local-qwen3-32b-baseline_raw.0-verified-500"
        ),
        help=(
            "Directory to scan recursively for .traj files (comparison)."
        ),
    )
    parser.add_argument(
        "--only_with_comparison",
        action="store_true",
        help="Only show instances that exist in both input and comparison directories.",
    )
    parser.add_argument(
        "--max_instances",
        type=int,
        default=None,
        help="Limit the number of instances printed (after sorting by instance id).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Override terminal width for formatting. Defaults to detected terminal width.",
    )
    return parser.parse_args(argv)


def find_all_instance_files(root_dir: Path, desc: str) -> dict[str, Path]:
    print(f"[setup] Scanning for .traj files in: {root_dir}")
    mapping: dict[str, Path] = {}
    # We don't know total upfront; tqdm still gives a responsive spinner
    for file_path in tqdm(root_dir.rglob("*.traj"), desc=desc):
        if not file_path.is_file():
            continue
        mapping[file_path.stem] = file_path
    print(f"[setup] Found {len(mapping)} .traj files in {root_dir}")
    return mapping


def find_comparison_file_in_index(index: dict[str, Path], instance_id: str) -> Path | None:
    return index.get(instance_id)


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        return None
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return data


def get_turns(data: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not data:
        return []
    trajectory = data.get("trajectory")
    if not isinstance(trajectory, list):
        return []
    return [t for t in trajectory if isinstance(t, dict)]


def format_turn(turn: dict[str, Any], width: int) -> str:
    def fmt_block(label: str, value: Any) -> str:
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        wrapped = textwrap.fill(text, width=width)
        return f"{label}:\n" + textwrap.indent(wrapped, prefix="  ")

    parts: list[str] = []
    parts.append(fmt_block("Thought", turn.get("thought", "")))
    internal_reasoning: Any | None = None
    ts = turn.get("turn_statistics")
    if isinstance(ts, dict):
        internal_reasoning = ts.get("internal_reasoning")
    if internal_reasoning:
        parts.append(fmt_block("Internal Reasoning", internal_reasoning))
    parts.append(fmt_block("Action", turn.get("action", "")))
    parts.append(fmt_block("Observation", turn.get("observation", "")))
    return "\n".join(parts)


def print_instance_block(info: TrajectoryInfo, width: int) -> None:
    input_turns = get_turns(info.input_data)
    cmp_turns = get_turns(info.comparison_data)

    header = f"Instance: {info.instance_id}"
    turns_line = (
        f"Turns — input: {len(input_turns)}"
        + (
            f", comparison: {len(cmp_turns)}"
            if info.comparison_data is not None
            else ", comparison: MISSING"
        )
    )
    sep = "-" * max(20, min(width, 120))

    print(sep)
    print(header)
    print(turns_line)
    print(sep)

    print("Input trajectory — last up to 2 turns:")
    if input_turns:
        for turn in input_turns[-2:]:
            print()
            print(format_turn(turn, width=width))
    else:
        print("  <empty>")

    print()
    print("Comparison trajectory — last up to 2 turns:")
    if info.comparison_data is None:
        print("  <missing>")
    elif cmp_turns:
        for turn in cmp_turns[-2:]:
            print()
            print(format_turn(turn, width=width))
    else:
        print("  <empty>")


def iter_sorted(mapping: dict[str, Path]) -> Iterable[tuple[str, Path]]:
    # Sort by instance id string; if numeric, numeric order still acceptable via zero-padding assumptions.
    for k in sorted(mapping.keys()):
        yield k, mapping[k]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.input_dir.exists():
        print(f"Input directory not found: {args.input_dir}", file=sys.stderr)
        return 2
    if not args.comparison_dir.exists():
        print(
            f"Comparison directory not found: {args.comparison_dir}",
            file=sys.stderr,
        )

    detected_width = args.width or max(60, min(140, get_terminal_size(fallback=(100, 24)).columns))
    wrap_width = detected_width - 4

    print("[setup] Starting dataset indexing...")
    input_files = find_all_instance_files(args.input_dir, desc="Indexing input")
    comparison_files = find_all_instance_files(args.comparison_dir, desc="Indexing comparison")

    # Build ordered list of instance ids to iterate (and optionally interact with)
    print("[setup] Preparing instance id list...")
    if args.only_with_comparison:
        overlapping = sorted(set(input_files.keys()) & set(comparison_files.keys()))
        ordered_ids = overlapping
    else:
        ordered_ids = list(sorted(input_files.keys()))
    print(f"[setup] Prepared {len(ordered_ids)} instance ids" + (" (overlap only)" if args.only_with_comparison else ""))

    if not ordered_ids:
        print("No instances to display.")
        return 0

    # Non-interactive batch mode if stdin not a TTY
    is_tty = sys.stdin.isatty()
    if not is_tty:
        print("[setup] Non-interactive mode detected (stdin is not a TTY). Printing in batch...")
        # Fall back to simple batch print respecting max_instances
        count_printed = 0
        for instance_id in tqdm(ordered_ids, desc="Printing instances"):
            input_path = input_files[instance_id]
            cmp_path = find_comparison_file_in_index(comparison_files, instance_id)
            input_data = load_json(input_path)
            cmp_data = load_json(cmp_path) if cmp_path else None
            info = TrajectoryInfo(
                instance_id=instance_id,
                input_path=input_path,
                comparison_path=cmp_path,
                input_data=input_data,
                comparison_data=cmp_data,
            )
            print_instance_block(info, width=wrap_width)
            count_printed += 1
            if args.max_instances is not None and count_printed >= args.max_instances:
                break
        print("-" * max(20, min(detected_width, 120)))
        return 0

    # Interactive navigation controls
    # Commands: n/next, p/prev, j <id>/jump <id>, q/quit, h/help
    idx = 0
    if args.max_instances is not None:
        # Trim the list to the requested size in interactive mode as well
        ordered_ids = ordered_ids[: args.max_instances]

    help_text = (
        "Commands: [n]ext, [p]rev, [j]ump <instance_id>, [q]uit, [h]elp"
    )
    print(f"[setup] Interactive mode ready with {len(ordered_ids)} instances. {help_text}")
    while 0 <= idx < len(ordered_ids):
        instance_id = ordered_ids[idx]
        input_path = input_files[instance_id]
        cmp_path = find_comparison_file_in_index(comparison_files, instance_id)
        input_data = load_json(input_path)
        cmp_data = load_json(cmp_path) if cmp_path else None
        info = TrajectoryInfo(
            instance_id=instance_id,
            input_path=input_path,
            comparison_path=cmp_path,
            input_data=input_data,
            comparison_data=cmp_data,
        )
        print_instance_block(info, width=wrap_width)

        print(help_text)
        try:
            command = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not command:
            idx += 1
            continue

        if command in {"n", "next"}:
            idx = min(idx + 1, len(ordered_ids))
        elif command in {"p", "prev", "previous"}:
            idx = max(idx - 1, 0)
        elif command.startswith("j ") or command.startswith("jump "):
            parts = command.split(maxsplit=1)
            if len(parts) == 2:
                target = parts[1].strip()
                if target in input_files and target in ordered_ids:
                    idx = ordered_ids.index(target)
                else:
                    print(f"Unknown instance id: {target}")
            else:
                print("Usage: j <instance_id>")
        elif command in {"h", "help"}:
            print(help_text)
        elif command in {"q", "quit", "exit"}:
            break
        else:
            print("Unrecognized command. Type 'h' for help.")

        if idx >= len(ordered_ids):
            break

    print("-" * max(20, min(detected_width, 120)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


