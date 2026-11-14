import hashlib
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_FILES = "ppe_50ep.pt,stgnn.pth,calib.npz"
CHUNK = 1 << 20


def download(url: str, dest: Path) -> None:
    part = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url, timeout=60) as resp, open(part, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        next_report = 0
        while chunk := resp.read(CHUNK):
            out.write(chunk)
            done += len(chunk)
            if total and done >= next_report:
                print(f"  {dest.name}: {done / 1e6:.0f}/{total / 1e6:.0f} MB", flush=True)
                next_report = done + max(total // 10, CHUNK)
    part.replace(dest)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def load_checksums(base_url: str) -> dict[str, str]:
    try:
        with urllib.request.urlopen(f"{base_url}/SHA256SUMS", timeout=30) as resp:
            text = resp.read().decode()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print("No SHA256SUMS asset in release; skipping verification.", flush=True)
            return {}
        raise
    sums = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2:
            sums[parts[1].lstrip("*")] = parts[0].lower()
    return sums


def main() -> int:
    repo = os.environ.get("MODEL_REPO", "ggarciabaez/liveleak")
    tag = os.environ.get("MODEL_RELEASE_TAG", "models-v1")
    models_dir = Path(os.environ.get("MODELS_DIR", "/models"))
    names = [n.strip() for n in os.environ.get("MODEL_FILES", DEFAULT_FILES).split(",") if n.strip()]
    base_url = os.environ.get("MODEL_BASE_URL") or f"https://github.com/{repo}/releases/download/{tag}"

    models_dir.mkdir(parents=True, exist_ok=True)
    missing = [n for n in names if not (models_dir / n).is_file()]
    if not missing:
        print(f"All model files present in {models_dir}.", flush=True)
        return 0

    print(f"Fetching {', '.join(missing)} from {base_url}", flush=True)
    try:
        sums = load_checksums(base_url)
        for name in missing:
            dest = models_dir / name
            download(f"{base_url}/{name}", dest)
            expected = sums.get(name)
            if expected and sha256(dest) != expected:
                dest.unlink()
                print(f"Checksum mismatch for {name}; deleted.", file=sys.stderr)
                return 1
    except urllib.error.HTTPError as e:
        print(f"Download failed ({e.code} for {e.url}). Does release '{tag}' exist "
              f"in {repo} with these assets? Or place the files in {models_dir} manually.",
              file=sys.stderr)
        return 1
    except urllib.error.URLError as e:
        print(f"Download failed: {e.reason}", file=sys.stderr)
        return 1
    print("Models ready.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
