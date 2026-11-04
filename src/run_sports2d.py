#!/usr/bin/env python3
"""Preview or run a thesis-derived Sports2D configuration."""

from __future__ import print_function

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = REPO_ROOT / "configs" / "sports2d.thesis.toml"


def toml_string(path):
    """Return a portable single-quoted TOML string for a filesystem path."""
    return path.resolve().as_posix().replace("'", "''")


def render_config(template, video, result_dir):
    text = template.read_text(encoding="utf-8-sig")
    return text.replace("__VIDEO_INPUT__", toml_string(video)).replace(
        "__RESULT_DIR__", toml_string(result_dir)
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="input study video")
    parser.add_argument("--output-dir", type=Path, required=True, help="directory for Sports2D results")
    parser.add_argument("--config-template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--camera", default="CAM_1")
    parser.add_argument("--execute", action="store_true", help="run the installed Sports2D adapter")
    parser.add_argument("--keep-config", action="store_true", help="retain the generated configuration")
    parser.add_argument("--print-config", action="store_true", help="print the rendered TOML")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not args.config_template.is_file():
        print("Configuration template not found: {}".format(args.config_template), file=sys.stderr)
        return 2
    if args.execute and not args.video.is_file():
        print("Input video does not exist: {}".format(args.video), file=sys.stderr)
        return 2

    rendered = render_config(args.config_template, args.video, args.output_dir)
    generated = args.output_dir.parent / (args.output_dir.name + ".sports2d.generated.toml")
    command = [
        "sports2d-dvideow", "pose-sports2d", "--video", str(args.video),
        "--config", str(generated),
        "--side", args.side, "--camera", args.camera, "--output", str(args.output_dir),
    ]
    print("Sports2D route: Halpe-26 body-with-feet, greatest-displacement ordering, DeepSORT")
    print("Command: {}".format(" ".join(command)))

    if not args.execute:
        print("Preview only: no video was opened and no output was written")
        if args.print_config:
            print("--- configuration ---")
            print(rendered)
        return 0

    executable = shutil.which("sports2d-dvideow")
    if executable is None:
        print("sports2d-dvideow is not installed in the active environment", file=sys.stderr)
        return 2

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    generated.write_text(rendered, encoding="utf-8")
    try:
        subprocess.run([executable] + command[1:], check=True)
    finally:
        if not args.keep_config and generated.exists():
            generated.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
