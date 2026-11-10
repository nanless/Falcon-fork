"""Ad hoc archive check; this does not alter the original pytest skip result."""
import hashlib
import importlib.util
import tempfile
from pathlib import Path

repo = Path("/root/code/github_repos/Falcon-fork")
test_file = repo / "tools/repro/tests/test_phase2_pipeline.py"
archive = repo / "docs/verification/phase2/P2D-train-metrics-1gpu.csv"
print("test_source:", test_file)
print("archive:", archive)
print("archive_sha256:", hashlib.sha256(archive.read_bytes()).hexdigest())
print("formal_status: SKIPPED, original /root/tmp/p2d_train/... path absent")
spec = importlib.util.spec_from_file_location("phase2_archive_review", test_file)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
with tempfile.TemporaryDirectory(prefix="falcon_archived_metrics_") as temporary:
    temporary_root = Path(temporary)
    version = temporary_root / "archived" / "version_0"
    version.mkdir(parents=True)
    (version / "metrics.csv").symlink_to(archive)
    module.P2D_TRAIN_ROOT = temporary_root
    print("checking: direct test function with P2D_TRAIN_ROOT redirected to temporary symlink")
    module.test_phase2_p2d_training_curve_sane()
print("archive_assertions: PASS")
print("interpretation: alternative evidence check only; original suite remains 166 passed, 2 skipped")
