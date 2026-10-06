"""Download only the three Clef artifacts approved on 2026-10-05.

Standard library only. No installation/execution
of downloaded artifacts. Partial or mismatching files are preserved.
"""
import hashlib
import json
from pathlib import Path
import shutil
import time
from datetime import datetime, timezone
import urllib.request


ROOT = Path.home() / "AI" / "clef-flash"
ARTIFACTS = [
    dict(name="Clef-Flash-Q4_K_M.gguf", size=6486448288,
         sha256="fd3e90605e8103307dca37cb5a8cdb036267e2fe3cb2d908d80a8ceb9ec0638c",
         url="https://huggingface.co/ggml-org/Clef-Flash-GGUF/resolve/4a192915ef971886004b5b13294f2b4c7a7fc39d/Clef-Flash-Q4_K_M.gguf"),
    dict(name="llama-b11429-bin-win-cuda-13.4-x64.zip", size=153089864,
         sha256="76ddc6eff2389570789ed608881efc6977751a722015d8e8c94f302224ff1a3a",
         url="https://github.com/ggml-org/llama.cpp/releases/download/b11429/llama-b11429-bin-win-cuda-13.4-x64.zip"),
    dict(name="cudart-llama-bin-win-cuda-13.4-x64.zip", size=423535356,
         sha256="738f8c251ac22b70c3ae6f83a10cf222725df0395246a2cf58f32bdb85fbe668",
         url="https://github.com/ggml-org/llama.cpp/releases/download/b11429/cudart-llama-bin-win-cuda-13.4-x64.zip"),
]


def emit(**record):
    print(json.dumps(record), flush=True)


def verify(path, item):
    if path.stat().st_size != item["size"]:
        raise RuntimeError(f"Size mismatch; preserved {path}")
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != item["sha256"]:
        raise RuntimeError(f"Hash mismatch; preserved {path}")


def download(partial, item):
    started = time.monotonic()
    for attempt in range(3):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == item["size"]:
            return
        request = urllib.request.Request(item["url"], headers={"Range": f"bytes={offset}-"} if offset else {})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                if not response.url.startswith("https://"):
                    raise RuntimeError("Unexpected non-HTTPS redirect")
                if offset and (response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                    raise RuntimeError("Server did not honor resume offset; partial preserved")
                length = response.headers.get("Content-Length")
                if length and int(length) != item["size"] - offset:
                    raise RuntimeError("Unexpected download length; partial preserved")
                with partial.open("ab" if partial.exists() else "xb") as output:
                    next_report = time.monotonic() + 10
                    while chunk := response.read(4 * 1024**2):
                        if offset + len(chunk) > item["size"]:
                            raise RuntimeError("Download exceeded the approved byte count")
                        output.write(chunk)
                        offset += len(chunk)
                        now = time.monotonic()
                        if now - started > 1800:
                            raise RuntimeError("Download exceeded its 30-minute time budget")
                        if now >= next_report:
                            output.flush()
                            emit(event="progress", file=item["name"], bytes=offset, elapsed_s=round(now-started))
                            next_report = now + 10
            if offset != item["size"]:
                raise OSError("Incomplete response body")
            return
        except (OSError, TimeoutError) as error:
            emit(event="transport_error", file=item["name"], attempt=attempt+1, error=str(error))
            if attempt == 2:
                raise


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(ROOT).free < 12 * 1024**3:
        raise RuntimeError("Need 12 GiB free before starting this bounded download.")
    manifest = dict(approved="2026-10-05 user approved the named files", root=str(ROOT),
                    artifacts=ARTIFACTS, verified=[])
    manifest_path = ROOT / "download-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for item in ARTIFACTS:
        target = ROOT / item["name"]
        partial = ROOT / (item["name"] + ".part")
        if not target.exists():
            emit(event="download_start", file=item["name"], expected_bytes=item["size"])
            if partial.exists() and partial.stat().st_size > item["size"]:
                raise RuntimeError(f"Oversized partial preserved: {partial}")
            download(partial, item)
            verify(partial, item)
            partial.rename(target)
        else:
            verify(target, item)
        manifest["verified"].append(dict(name=item["name"], size=item["size"], sha256=item["sha256"],
                                         verified_at=datetime.now(timezone.utc).isoformat()))
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        emit(event="verified", file=item["name"], bytes=item["size"])
    emit(event="complete", root=str(ROOT), total_bytes=sum(x["size"] for x in ARTIFACTS))


if __name__ == "__main__":
    main()
