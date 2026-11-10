"""Verify release checksums and checkpoint metadata using the standard library."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def release_path(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Path outside the repository: {relative}")
    if not path.is_file():
        raise ValueError(f"Missing release file: {relative}")
    return path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify():
    checksums = {}
    for line in (ROOT / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        if relative in checksums:
            raise ValueError(f"Duplicate checksum entry: {relative}")
        if sha256(release_path(relative)) != expected:
            raise ValueError(f"SHA256 mismatch: {relative}")
        checksums[relative] = expected

    manifest = json.loads(
        release_path("quick_start/parameter_manifest.json").read_text(encoding="utf-8")
    )
    index = json.loads(
        release_path("quick_start/parameter_tensor_index.json").read_text(encoding="utf-8")
    )
    folds = manifest["folds"]
    folders = {fold["folder"] for fold in folds}
    if len(folds) != manifest["bundles"] or len(folders) != len(folds):
        raise ValueError("Fold count differs from the manifest or contains duplicates.")
    if folders != set(index):
        raise ValueError("Tensor index folders differ from the manifest.")

    fold_files = set()
    required_files = {
        "config.json", "eeg_encoder.pt", "eog_encoder.pt", "mapper.pt",
        "qjm_head.pt", "normalizers.pt",
    }
    for fold in folds:
        folder = Path(fold["folder"])
        entries = fold["files"]
        if len(entries) != len(required_files) or {
            Path(entry["file"]).name for entry in entries
        } != required_files:
            raise ValueError(f"Incomplete fold bundle: {folder}")
        for entry in entries:
            relative = entry["file"]
            path = release_path(relative)
            if Path(relative).parent != folder:
                raise ValueError(f"File outside its fold folder: {relative}")
            if path.stat().st_size != entry["bytes"]:
                raise ValueError(f"File size mismatch: {relative}")
            if checksums.get(relative) != entry["sha256"]:
                raise ValueError(f"Manifest/checksum mismatch: {relative}")
            fold_files.add(relative)

    shared = "quick_start/weights/shared/eegpt/"
    reference = json.loads(release_path(shared + "reference.json").read_text(encoding="utf-8"))
    if checksums.get(shared + "braindecode_eegpt.safetensors") != reference["sha256"]:
        raise ValueError("Shared EEGPT reference/checksum mismatch.")
    expected_weights = fold_files | {
        shared + "reference.json", shared + "braindecode_eegpt.safetensors"
    }
    actual_weights = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "quick_start/weights").rglob("*") if path.is_file()
    }
    if actual_weights != expected_weights:
        raise ValueError("Weight directory contents differ from the release manifest.")

    print(f"OK: {len(checksums)} file hashes, {len(folds)} fold bundles, tensor index paths,")
    print("    and shared EEGPT checkpoint verified. Paths are relative to the repository root.")


if __name__ == "__main__":
    try:
        verify()
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f"Verification failed: {error}") from error
