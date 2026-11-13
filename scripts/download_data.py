"""Tự viết (B): tải và kiểm tra Amazon Reviews 2014 - Beauty bằng thư viện chuẩn.

Nguồn trùng với genrec/datasets/AmazonReviews2014/dataset.py của RPG_KDD2025.
SHA-256 trong manifest là mốc tái lập của nhóm, không phải chữ ký của nhà phát hành.
"""

import argparse
import gzip
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs" / "data_sources.json"
CHUNK_SIZE = 1024 * 1024


def inspect_file(path, spec):
    """Kiểm tra kích thước, SHA-256, toàn bộ CRC gzip và số bản ghi."""
    size = path.stat().st_size
    if size != spec["size_bytes"]:
        raise ValueError(f"{path.name}: size {size}, expected {spec['size_bytes']}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(block)
    checksum = digest.hexdigest()
    if spec.get("sha256") and checksum != spec["sha256"]:
        raise ValueError(f"{path.name}: SHA-256 mismatch")
    with gzip.open(path, "rb") as stream:
        records = sum(1 for line in stream if line.strip())
    if records != spec["records"]:
        raise ValueError(f"{path.name}: {records} records, expected {spec['records']}")
    return {"sha256": checksum, "size_bytes": size, "records": records}


def download_file(spec, output_dir, *, timeout=60, retries=3):
    """Tải qua file .part; chỉ đặt tên chính thức sau khi kiểm tra thành công."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / spec["filename"]
    if target.exists():
        result = inspect_file(target, spec)
        print(f"Verified existing: {target.name} ({result['records']:,} records)", flush=True)
        return result

    partial = target.with_name(target.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                spec["url"], headers={"User-Agent": "rpg-semantic-ids/1.0"}
            )
            downloaded = 0
            last_report = 0.0
            with urllib.request.urlopen(request, timeout=timeout) as response:
                with partial.open("wb") as stream:
                    while block := response.read(CHUNK_SIZE):
                        stream.write(block)
                        downloaded += len(block)
                        now = time.monotonic()
                        if now - last_report >= 5:
                            print(
                                f"{target.name}: {downloaded / CHUNK_SIZE:.1f} / "
                                f"{spec['size_bytes'] / CHUNK_SIZE:.1f} MiB",
                                flush=True,
                            )
                            last_report = now
            result = inspect_file(partial, spec)
            partial.replace(target)
            print(f"Downloaded and verified: {target.name} ({result['records']:,} records)", flush=True)
            return result
        except ValueError:
            # Sai checksum/kích thước/số bản ghi: dừng để kiểm tra nguồn hoặc manifest.
            partial.unlink(missing_ok=True)
            raise
        except (OSError, EOFError, urllib.error.URLError) as error:
            partial.unlink(missing_ok=True)
            if attempt == retries:
                raise
            print(f"Attempt {attempt}/{retries} failed: {error}; retrying", file=sys.stderr)
            time.sleep(min(attempt * 2, 5))


def main(argv=None):
    """Mặc định tải cả reviews và metadata; --verify-only không dùng mạng."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--record-checksums", action="store_true",
        help="Record missing SHA-256 references after the first trusted download",
    )
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args(argv)
    if args.timeout <= 0 or args.retries < 1:
        parser.error("--timeout must be positive and --retries must be at least 1")
    if args.verify_only and args.record_checksums:
        parser.error("--verify-only and --record-checksums cannot be combined")

    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        files = manifest["files"]
        if not args.record_checksums and any(not spec.get("sha256") for spec in files):
            raise ValueError("Manifest is missing SHA-256 references; use --record-checksums for initial acquisition")
        for spec in files:
            if args.verify_only:
                result = inspect_file(args.output_dir / spec["filename"], spec)
                print(f"Verified: {spec['filename']} ({result['records']:,} records)", flush=True)
            else:
                result = download_file(spec, args.output_dir, timeout=args.timeout, retries=args.retries)
            if args.record_checksums and not spec.get("sha256"):
                spec["sha256"] = result["sha256"]
                args.manifest.write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
    except (OSError, EOFError, ValueError, KeyError, urllib.error.URLError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
