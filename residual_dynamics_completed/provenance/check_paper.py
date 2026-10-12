"""Static phase-two audit only: deliberately does not compile LaTeX."""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT.parent.parent / "papers/unified_tracking_control_for_drone_ball_balancing"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    before = (ROOT/"provenance/paper_before/root.tex").read_text()
    after = (PAPER/"root.tex").read_text()
    real_start = r"\subsection{Real-World Closed-Loop Evaluation}"
    conclusion = r"\section{CONCLUSIONS}"
    assert before.split(real_start)[1].split(conclusion)[0] == after.split(real_start)[1].split(conclusion)[0]
    offline = after.split(r"\subsection{Offline Residual-Dynamics Prediction}")[1].split(real_start)[0]
    for header in ("1) Dataset:", "2) Causal rollout protocol:", "3) Baselines:", "4) Results:"):
        assert header in offline
    assert r"\begin{figure*}[t]" in offline
    assert r"\includegraphics[width=\textwidth]{figures/offline_residual_rollout_rmse.pdf}" in offline
    assert r"\begin{table}" not in offline
    labels = re.findall(r"\\label\{([^}]+)\}", after)
    for label in re.findall(r"\\(?:eqref|ref)\{([^}]+)\}", offline):
        assert labels.count(label) == 1, label
    provenance = json.loads((PAPER/"figures/offline_residual_rollout_provenance.json").read_text())
    for filename, digest in provenance["copied_artifacts"].items():
        assert sha(PAPER/"figures"/filename) == digest
    assert sha(ROOT/"results/acceptance.json") == provenance["acceptance_sha256"]
    result = {"passed": True, "latex_compiled": False, "reason": "AGENTS.md explicitly excludes full-paper compilation",
        "offline_four_part_structure": True, "double_column_figure": True, "offline_result_table_absent": True,
        "real_world_section_unchanged": True, "figure_hashes_verified": True, "local_references_resolve": True,
        "paper_sha256": sha(PAPER/"root.tex"), "paper_figure_provenance": provenance}
    (ROOT/"results/paper_acceptance.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
