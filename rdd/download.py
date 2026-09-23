"""Download RDD2022 India + Japan from the official figshare zip.

The official file is one 13.3 GB zip holding one inner zip per country. We read only the
byte ranges of RDD2022/India.zip and RDD2022/Japan.zip (~1.5 GB) using HTTP Range requests.
If the server refuses ranges, we fall back to downloading the whole zip (resumable).

Result: data/raw/<Country>/train/images/*.jpg and data/raw/<Country>/train/annotations/xmls/*.xml
Skips everything if the folders already hold the expected counts.
"""
from __future__ import annotations

import io
import shutil
import tempfile
import zipfile
from pathlib import Path

import requests

from .common import cfg, detect_env, log, path, step

EXPECTED = {"India": 7706, "Japan": 10506}
CHUNK = 8 * 1024 * 1024


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP Range requests, with a small block cache."""

    def __init__(self, url: str, size: int | None = None):
        self.session = requests.Session()
        self.url = self._resolve(url)
        self.size = size or int(self.session.head(self.url, allow_redirects=True, timeout=60).headers["Content-Length"])
        self.pos = 0
        self.cache_start, self.cache = -1, b""

    def _resolve(self, url):
        r = self.session.get(url, stream=True, allow_redirects=True, timeout=60, headers={"Range": "bytes=0-0"})
        r.close()
        if r.status_code != 206:
            raise OSError(f"server does not support byte ranges (HTTP {r.status_code})")
        self.orig = url
        return r.url

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def _fetch(self, start, end):
        for attempt in range(6):
            try:
                r = self.session.get(self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
                if r.status_code in (403, 410):  # signed URL expired -> re-resolve
                    self.url = self._resolve(self.orig)
                    continue
                if r.status_code != 206:
                    raise OSError(f"HTTP {r.status_code}")
                return r.content
            except (requests.RequestException, OSError) as e:
                log(f"  range request failed ({e}); retry {attempt + 1}/6")
        raise OSError("range requests keep failing")

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        c0, c1 = self.cache_start, self.cache_start + len(self.cache)
        if c0 <= self.pos and self.pos + n <= c1:
            out = self.cache[self.pos - c0 : self.pos - c0 + n]
        else:
            want = max(n, 1024 * 1024)
            end = min(self.pos + want, self.size) - 1
            self.cache_start, self.cache = self.pos, self._fetch(self.pos, end)
            out = self.cache[:n]
        self.pos += len(out)
        return out

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def counts_ok(country: str) -> bool:
    d = path("raw") / country / "train"
    imgs = list((d / "images").glob("*.jpg")) if (d / "images").exists() else []
    xmls = list((d / "annotations" / "xmls").glob("*.xml")) if (d / "annotations" / "xmls").exists() else []
    return len(imgs) >= EXPECTED[country] * 0.98 and len(xmls) >= EXPECTED[country] * 0.98


def _place(extract_dir: Path, country: str) -> None:
    """Find <something>/train/images inside the extracted inner zip and move it to data/raw/<Country>."""
    target = path("raw") / country
    cands = [p.parent for p in extract_dir.rglob("images") if p.is_dir() and p.parent.name == "train"]
    if not cands:
        raise RuntimeError(f"{country}: no train/images folder found in the inner zip")
    src_country_dir = cands[0].parent
    target.mkdir(parents=True, exist_ok=True)
    for child in src_country_dir.iterdir():
        dst = target / child.name
        if dst.exists():
            shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
        shutil.move(str(child), str(dst))


def _extract_inner(outer: zipfile.ZipFile, country: str, tmp: Path) -> None:
    name = next(n for n in outer.namelist() if n.endswith(f"/{country}.zip") or n == f"{country}.zip")
    inner_path = tmp / f"{country}.zip"
    info = outer.getinfo(name)
    log(f"{country}: copying inner zip ({info.file_size / 1e9:.2f} GB)")
    done = 0
    with outer.open(name) as src, open(inner_path, "wb") as dst:
        while True:
            buf = src.read(CHUNK)
            if not buf:
                break
            dst.write(buf)
            done += len(buf)
            if done // (200 * 1024 * 1024) != (done - len(buf)) // (200 * 1024 * 1024):
                log(f"  {country}: {done / 1e9:.2f} / {info.file_size / 1e9:.2f} GB")
    log(f"{country}: extracting")
    ex = tmp / f"{country}_x"
    with zipfile.ZipFile(inner_path) as z:
        z.testzip()
        z.extractall(ex)
    _place(ex, country)
    inner_path.unlink()
    shutil.rmtree(ex, ignore_errors=True)


def _full_download(url: str, dest: Path, size: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    if have == size:
        return
    log(f"full download to {dest} (resuming at {have / 1e9:.2f} GB)")
    with requests.get(url, stream=True, headers={"Range": f"bytes={have}-"} if have else {}, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "ab" if have else "wb") as f:
            for i, buf in enumerate(r.iter_content(CHUNK)):
                f.write(buf)
                if i % 128 == 0:
                    log(f"  {f.tell() / 1e9:.2f} / {size / 1e9:.2f} GB")
    if dest.stat().st_size != size:
        raise RuntimeError("download incomplete; re-run to resume")


def main() -> None:
    step("Dataset: RDD2022 India + Japan")
    d = cfg()["dataset"]
    todo = [c for c in d["countries"] if not counts_ok(c)]
    if not todo:
        log("already present with the expected counts; nothing to download")
        return
    path("raw").mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="rdd_", dir=str(path("raw").parent)))
    try:
        try:
            outer = zipfile.ZipFile(HttpRangeFile(d["url"], d["zip_size"]))
            log("byte-range mode: fetching only the India/Japan parts of the official zip")
        except Exception as e:  # noqa: BLE001
            log(f"byte-range mode unavailable ({e}); falling back to the full 13.3 GB zip")
            big = (path("raw") / "RDD2022.zip") if detect_env() == "local" else Path("/tmp/RDD2022.zip")
            _full_download(d["url"], big, d["zip_size"])
            outer = zipfile.ZipFile(big)
        for c in todo:
            _extract_inner(outer, c, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        if detect_env() != "local":
            Path("/tmp/RDD2022.zip").unlink(missing_ok=True)
    for c in d["countries"]:
        if not counts_ok(c):
            raise SystemExit(f"FAIL: {c} counts still wrong after download")
        log(f"{c}: OK")


if __name__ == "__main__":
    main()
