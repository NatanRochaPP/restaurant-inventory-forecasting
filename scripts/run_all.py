"""Run the whole pipeline in order, with a log, as one command.

The five batch scripts have to run in a fixed order: models are trained, the backtest
measures forecast accuracy, the simulation replays the holdout and sweeps the frontier, and
the figures are drawn from what those wrote. Running them by hand is fine for development
and wrong for anything scheduled, because a step that fails in the middle leaves the
outputs half old and half new with nothing recording it.

This runner executes the steps in order, stops at the first failure, writes everything both
to the console and to a timestamped log under ``outputs/logs/``, and exits non-zero when a
step fails so a scheduler notices. It does not schedule anything itself: a real deployment
would trigger it overnight (see the deployment section of the report).

Usage::

    python scripts/run_all.py
    python scripts/run_all.py --skip-figures
    python scripts/run_all.py --config config/config.yaml
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "outputs" / "logs"


def steps(config: str | None = None, *, skip_figures: bool = False) -> list[tuple[str, list[str]]]:
    """The pipeline in the order it has to run, as (name, command) pairs."""
    config_args = ["--config", config] if config else []
    planned: list[tuple[str, list[str]]] = [
        ("Train models and select one per product", ["scripts/train_models.py", *config_args]),
        ("Backtest forecast accuracy", ["scripts/run_backtest.py", *config_args]),
        ("Simulate both policies and sweep the frontier", ["scripts/run_simulation.py", "--frontier", *config_args]),
    ]
    if not skip_figures:
        planned.append(("Generate report figures and tables", ["scripts/generate_report_figures.py"]))
    return [(name, [sys.executable, *args]) for name, args in planned]


def _run(command: Sequence[str], log) -> int:
    """Run one command, copying its output to the console and the log."""
    process = subprocess.Popen(
        list(command), cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        sys.stdout.write(line)
        log.write(line)
    return process.wait()


def run(
    planned: Sequence[tuple[str, list[str]]],
    log,
    runner: Callable[[Sequence[str], object], int] = _run,
) -> int:
    """Run the steps in order, stopping at the first failure.

    Args:
        planned: The (name, command) pairs to run.
        log: An open text file the output is copied to.
        runner: How to run one command; replaced in tests.

    Returns:
        0 when every step succeeded, otherwise the exit code of the step that failed.
    """
    for number, (name, command) in enumerate(planned, start=1):
        header = f"\n[{number}/{len(planned)}] {name}\n    {' '.join(command)}\n"
        sys.stdout.write(header)
        log.write(header)
        code = runner(command, log)
        if code != 0:
            failure = f"\nFAILED at step {number} ({name}) with exit code {code}. Later steps were not run.\n"
            sys.stdout.write(failure)
            log.write(failure)
            return code
    done = "\nAll steps completed.\n"
    sys.stdout.write(done)
    log.write(done)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the full forecasting and simulation pipeline.")
    parser.add_argument("--config", default=None, help="Configuration file to pass to each step.")
    parser.add_argument("--skip-figures", action="store_true", help="Stop after the simulation.")
    args = parser.parse_args(argv)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"run_{dt.datetime.now():%Y%m%d_%H%M%S}.log"
    planned = steps(args.config, skip_figures=args.skip_figures)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"Pipeline run started {dt.datetime.now():%Y-%m-%d %H:%M:%S}\n")
        code = run(planned, log)
        log.write(f"Finished {dt.datetime.now():%Y-%m-%d %H:%M:%S} with exit code {code}\n")
    print(f"Log written to {log_path}")
    return code


if __name__ == "__main__":
    sys.exit(main())
