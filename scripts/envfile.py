from __future__ import annotations

import argparse
import json
from pathlib import Path


def _consume_quoted_value(
    lines: list[str],
    start_index: int,
    value: str,
) -> tuple[str, int]:
    quote = value[0]
    remainder = value[1:]
    if remainder.endswith(quote):
        return remainder[:-1], start_index

    collected = [remainder]
    index = start_index
    while index + 1 < len(lines):
        index += 1
        next_line = lines[index]
        if next_line.endswith(quote):
            collected.append(next_line[:-1])
            return "\n".join(collected), index
        collected.append(next_line)

    return "\n".join(collected), index


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(raw_lines):
        raw_line = raw_lines[index]
        line = raw_line.strip()
        if not line or line.startswith("#"):
            index += 1
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            index += 1
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            index += 1
            continue
        if value.startswith(("'", '"')):
            value, index = _consume_quoted_value(raw_lines, index, value)
        values[key] = value
        index += 1
    return values


def cmd_get(args: argparse.Namespace) -> int:
    values = parse_env_file(Path(args.file))
    value = values.get(args.key)
    if value is None:
        return 1
    print(value)
    return 0


def cmd_to_cloudrun_yaml(args: argparse.Namespace) -> int:
    values = parse_env_file(Path(args.file))
    excluded = set(args.exclude or [])
    output = Path(args.output)
    lines = [
        f"{key}: {json.dumps(value, ensure_ascii=False)}"
        for key, value in values.items()
        if key not in excluded
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Utilities for reading .env-style files.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    get_parser = subparsers.add_parser("get", help="Read a single key from an env file.")
    get_parser.add_argument("--file", required=True, help="Path to env file.")
    get_parser.add_argument("key", help="Key to read.")
    get_parser.set_defaults(func=cmd_get)

    yaml_parser = subparsers.add_parser(
        "to-cloudrun-yaml",
        help="Convert an env file into a Cloud Run env-vars YAML file.",
    )
    yaml_parser.add_argument("--file", required=True, help="Path to env file.")
    yaml_parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Env key to exclude from the generated YAML. Can be passed multiple times.",
    )
    yaml_parser.add_argument("output", help="Path to output YAML file.")
    yaml_parser.set_defaults(func=cmd_to_cloudrun_yaml)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
