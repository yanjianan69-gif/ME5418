"""Run and verify training, checkpoint restart and evaluation in real simulation.

Each simulator runs in its own process, using this interpreter and the packaged
robot assets. A successful demo demonstrates software integration, not convergence.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("runs/learning_agent_demo"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output directory already exists; choose a new --output")
    args.output.mkdir(parents=True)
    output = args.output.resolve()
    root = Path(__file__).resolve().parent

    def run(script, *arguments):
        command = [sys.executable, str(root / script), "--sim-device", args.sim_device,
                   "--device", args.device, *map(str, arguments)]
        print("Running: " + " ".join(command), flush=True)
        subprocess.run(command, cwd=root, check=True)

    run("train.py", "--iterations", 2, "--save-interval", 1, "--output", output / "train")
    run("train.py", "--resume", output / "train/policy.pt", "--iterations", 1,
        "--save-interval", 1, "--output", output / "resumed")
    run("evaluate.py", "--checkpoint", output / "resumed/policy.pt", "--episodes", 2,
        "--max-steps", 100, "--output", output / "evaluation.json")

    # The parent does not create a simulator; importing torch here is safe.
    import torch
    initial = torch.load(output / "train/initial.pt", map_location="cpu", weights_only=True)
    trained = torch.load(output / "train/policy.pt", map_location="cpu", weights_only=True)
    resumed_initial = torch.load(output / "resumed/initial.pt", map_location="cpu", weights_only=True)
    resumed = torch.load(output / "resumed/policy.pt", map_location="cpu", weights_only=True)
    changes = {}
    for group in ("actor_body", "critic_body", "log_std"):
        names = [name for name in initial["model_state_dict"] if name.startswith(group)]
        changes[group] = any(not torch.equal(initial["model_state_dict"][name],
                                             trained["model_state_dict"][name]) for name in names)
    checks = {
        "actor_critic_noise_updated": all(changes.values()),
        "resume_restored_weights": all(torch.equal(value, resumed_initial["model_state_dict"][name])
                                       for name, value in trained["model_state_dict"].items()),
        "resume_advanced_counters": resumed["iteration"] == 3 and resumed["environment_steps"] == 72,
        "checkpoints_created": all((output / folder / "checkpoints" / f"iteration_{it:06d}.pt").is_file()
                                   for folder, it in (("train", 1), ("train", 2), ("resumed", 3))),
        "finite_weights": all(torch.isfinite(value).all().item() for value in resumed["model_state_dict"].values()),
    }
    for folder in ("train", "resumed"):
        with (output / folder / "training.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        checks[folder + "_finite_metrics"] = bool(rows) and all(
            math.isfinite(float(row[key])) for row in rows
            for key in ("loss", "grad_norm", "reward_mean", "lin_vel_rmse_mps"))
    evaluation = json.loads((output / "evaluation.json").read_text())
    checks["evaluation_completed"] = evaluation["summary"]["episodes"] == 2
    summary = {"passed": all(checks.values()), "checks": checks, "updated_parameter_groups": changes,
               "evaluation": evaluation["summary"],
               "scope": "Real simulation integration check; no claim of policy convergence."}
    (output / "verification.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    if not summary["passed"]:
        raise RuntimeError(f"Learning agent verification failed: {checks}")
    print(f"PASS: train -> save -> resume -> evaluate. Evidence: {output / 'verification.json'}")


if __name__ == "__main__":
    main()
