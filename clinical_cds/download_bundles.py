"""Download and SHA-256-verify pre-built Zero-Pickle Clinical CDS bundles.

Allows reviewers, clinicians, and developers to boot the live 2-Stage Clinical CDS API
and Bedside UI in ~10 seconds without retraining the 68,589-patient OpenML cohort:

    heart-pipeline download-bundles
    uvicorn clinical_cds.api:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

from project_paths import ARTIFACTS_DIR

DEFAULT_REPO = "YahyaLendani/heart-disease-clinical-ml"
DEFAULT_RELEASE_TAG = "v2.0.0"
DEFAULT_CHECKSUM_FILE = ARTIFACTS_DIR / "SHA256SUMS.txt"

BUNDLE_FILENAMES = (
    "clinical_cds_bundle.pt",
    "clinical_workup_bundle.pt",
)


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_sha256sums(checksum_path: Path) -> dict[str, str]:
    """Parse standard ``<sha256>  <filename>`` checksum manifest."""
    if not checksum_path.exists():
        return {}
    mapping: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            mapping[parts[1].lstrip("*")] = parts[0].lower()
    return mapping


def write_sha256sums(
    artifacts_dir: Path = ARTIFACTS_DIR,
    checksum_path: Path = DEFAULT_CHECKSUM_FILE,
) -> dict[str, str]:
    """Generate ``artifacts/SHA256SUMS.txt`` for all present production bundle files."""
    checksums: dict[str, str] = {}
    lines: list[str] = []
    for fname in BUNDLE_FILENAMES:
        fpath = artifacts_dir / fname
        if fpath.exists():
            digest = compute_file_sha256(fpath)
            checksums[fname] = digest
            lines.append(f"{digest}  {fname}")
    checksum_path.parent.mkdir(parents=True, exist_ok=True)
    checksum_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return checksums


def verify_or_remove(file_path: Path, expected_sha256: str | None) -> str:
    """Verify SHA-256 of a bundle file; delete and raise ValueError if tampered."""
    actual = compute_file_sha256(file_path)
    if expected_sha256 is not None and actual.lower() != expected_sha256.lower():
        file_path.unlink(missing_ok=True)
        raise ValueError(
            f"SHA-256 mismatch for {file_path.name}: expected {expected_sha256}, got {actual}. "
            "Deleted unverified file."
        )
    return actual


def download_bundles(
    repo: str = DEFAULT_REPO,
    tag: str = DEFAULT_RELEASE_TAG,
    output_dir: str | Path = ARTIFACTS_DIR,
    checksum_file: str | Path = DEFAULT_CHECKSUM_FILE,
    force: bool = False,
) -> dict[str, Path]:
    """Download pre-built Stage-1 & Stage-2 bundles from a GitHub Release and verify SHA-256."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    expected_map = parse_sha256sums(Path(checksum_file))
    downloaded: dict[str, Path] = {}

    for fname in BUNDLE_FILENAMES:
        dest = out_dir / fname
        expected_sha = expected_map.get(fname)
        if dest.exists() and not force:
            actual = compute_file_sha256(dest)
            if expected_sha is None or actual == expected_sha:
                print(f"[OK] {fname} already present and SHA-256 verified ({actual[:16]}...)")
                downloaded[fname] = dest
                continue

        url = f"https://github.com/{repo}/releases/download/{tag}/{fname}"
        print(f"Downloading {fname} from {url} ...")
        tmp_dest = dest.with_suffix(".pt.tmp")
        urllib.request.urlretrieve(url, tmp_dest)
        actual = verify_or_remove(tmp_dest, expected_sha)
        tmp_dest.replace(dest)
        print(f"[VERIFIED] {fname} -> {dest} (SHA-256: {actual[:16]}...)")
        downloaded[fname] = dest

    return downloaded


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download & SHA-256-verify pre-built Zero-Pickle Clinical CDS bundles."
    )
    parser.add_argument("--repo", type=str, default=DEFAULT_REPO)
    parser.add_argument("--tag", type=str, default=DEFAULT_RELEASE_TAG)
    parser.add_argument("--output-dir", type=str, default=str(ARTIFACTS_DIR))
    parser.add_argument("--force", action="store_true", help="Re-download even if present.")
    parser.add_argument(
        "--write-checksums",
        action="store_true",
        help="Write artifacts/SHA256SUMS.txt from currently built local bundles instead of downloading.",
    )
    args = parser.parse_args(argv)
    if args.write_checksums:
        sums = write_sha256sums(Path(args.output_dir), Path(args.output_dir) / "SHA256SUMS.txt")
        for k, v in sums.items():
            print(f"{v}  {k}")
        return
    download_bundles(repo=args.repo, tag=args.tag, output_dir=args.output_dir, force=args.force)


if __name__ == "__main__":
    main()
