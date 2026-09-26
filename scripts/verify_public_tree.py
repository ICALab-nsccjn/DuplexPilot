from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PARTS = {"__pycache__", ".pytest_cache", "artifacts", "results", "models", "checkpoints"}
FORBIDDEN_SUFFIXES = {".log", ".jsonl", ".ckpt", ".safetensors", ".pth", ".pt"}

violations = []
for path in ROOT.rglob("*"):
    if not path.is_file():
        continue
    if any(part in FORBIDDEN_PARTS for part in path.relative_to(ROOT).parts):
        violations.append(str(path.relative_to(ROOT)))
    elif path.suffix.lower() in FORBIDDEN_SUFFIXES:
        violations.append(str(path.relative_to(ROOT)))
if violations:
    raise SystemExit("forbidden public files:\n" + "\n".join(violations))
print("public tree check passed")
