"""Create relocated copies of local RealSSA trial indexes without altering evidence.

The generated tr/cv directories contain index JSON only. Their paths point to
existing media under the relocated trial root; media files are not copied.
"""
import argparse
import hashlib
import json
from pathlib import Path

SPLITS = ("tr", "cv")
INDEX_NAMES = ("mix.json", "target.json", "near.json")


def relocated_copies(source_root, old_root, new_root, output_root):
    source_root = Path(source_root).expanduser().resolve()
    old_input = Path(old_root).expanduser()
    new_root = Path(new_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if not old_input.is_absolute():
        raise ValueError("old root must be absolute")
    old_roots = {str(old_input), str(old_input.resolve())}
    if str(new_root) in old_roots:
        raise ValueError("old and new roots must differ")
    if output_root.exists():
        raise FileExistsError(f"output directory already exists: {output_root}")
    plans = []
    for split in SPLITS:
        for name in INDEX_NAMES:
            source = source_root / split / name
            raw = source.read_bytes()
            data = json.loads(raw)
            count = 0

            def replace(value):
                nonlocal count
                if isinstance(value, str):
                    for old in sorted(old_roots, key=len, reverse=True):
                        if value.startswith(old + "/"):
                            suffix = value[len(old) + 1:]
                            target = (new_root / suffix).resolve()
                            if not target.is_relative_to(new_root) or not target.is_file():
                                raise ValueError(f"relocated file is missing or outside new root: {target}")
                            count += 1
                            return str(target)
                    return value
                if isinstance(value, list):
                    return [replace(item) for item in value]
                if isinstance(value, dict):
                    return {key: replace(item) for key, item in value.items()}
                return value

            rewritten = replace(data)
            if not count:
                raise ValueError(f"no matching paths in {source}")
            payload = (json.dumps(rewritten, ensure_ascii=False, indent=2) + "\n").encode()
            plans.append((split, name, raw, payload, count))
    output_root.mkdir(parents=True, exist_ok=False)
    report = {"source_root": str(source_root), "old_roots": sorted(old_roots),
              "new_root": str(new_root), "indexes": []}
    for split, name, raw, payload, count in plans:
        target = output_root / split / name
        target.parent.mkdir(exist_ok=True)
        with target.open("xb") as file:
            file.write(payload)
        report["indexes"].append({"source": str(source_root / split / name), "output": str(target),
                                  "source_sha256": hashlib.sha256(raw).hexdigest(),
                                  "output_sha256": hashlib.sha256(payload).hexdigest(), "replaced_paths": count})
    with (output_root / "relocation-report.json").open("x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "old-root", "new-root", "output-root"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    report = relocated_copies(args.source_root, args.old_root, args.new_root, args.output_root)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
