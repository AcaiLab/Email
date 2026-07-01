import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"


def pipeline_env() -> dict[str, str]:
    env = os.environ.copy()
    current = env.get("PYTHONPATH", "")
    paths = [str(SRC_DIR)]
    if current:
        paths.append(current)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def run_experiment(module_name: str, args: list[str] | None = None) -> None:
    args = args or []
    module_name = module_name.removesuffix(".py")
    command = [sys.executable, "-m", f"llmail_research.experiments.{module_name}", *args]
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, env=pipeline_env(), check=True)


def run_step(script_path: Path, args: list[str] | None = None) -> None:
    args = args or []
    command = [sys.executable, str(script_path), *args]
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, env=pipeline_env(), check=True)
