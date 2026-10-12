"""One-time, explicit source migration; never imported by training/evaluation."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "residual_dynamics_v2"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    source = SOURCE / "data/processed_rbf_pv"
    target = ROOT / "data/processed"
    records = []
    for src in sorted(source.rglob("*")):
        if not src.is_file():
            continue
        relative = src.relative_to(source)
        dst = target / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
        assert sha(src) == sha(dst), str(relative)
        assert not dst.is_symlink()
        records.append({"file": str(relative), "sha256": sha(src), "bytes": src.stat().st_size})
    for name in ("checkpoints/profile_plus_friction.pt", "summary.json", "detailed_rollout_metrics.json"):
        src = SOURCE / "results_profile_friction_probe" / name
        if src.exists():
            dst = ROOT / "provenance/old_structured" / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(src, dst)
            assert sha(src) == sha(dst)
    manifest = json.loads((target / "manifest.json").read_text())
    counts = {s: sum(e.get("included", False) and e.get("split") == s for e in manifest["entries"])
              for s in ("train", "validation", "test", "stress")}
    assert [counts[s] for s in ("train", "validation", "test")] == [79, 6, 6], counts
    record = {"source": str(source), "destination": str(target), "files": records,
              "split_counts": counts, "dataset_hash": manifest["dataset_hash"],
              "manifest_sha256": sha(target / "manifest.json"), "manifest_preserved": True}
    (ROOT / "provenance/migration.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"copied_files": len(records), "split_counts": counts, "all_hashes_match": True}))

if __name__ == "__main__":
    main()
