"""Read members of a zip split into raw byte volumes (name.zip.001, .002, ...) without extracting it.

AV-Deepfake1M ships each split as one zip cut into 1 GB pieces (7z / `split` style: plain byte
chunks, not a multi-disk zip). Concatenating the pieces gives the original zip, so a seekable view
over the volumes lets `zipfile` read the central directory and pull out only the sampled videos.
"""

import io
import threading
import zipfile
from bisect import bisect_right
from pathlib import Path


class Volumes(io.RawIOBase):
    """One seekable, read-only file over several byte-split volumes."""

    def __init__(self, paths):
        self.paths = [Path(p) for p in paths]
        self.sizes = [p.stat().st_size for p in self.paths]
        self.starts = [0]
        for size in self.sizes:
            self.starts.append(self.starts[-1] + size)
        self.pos = 0
        self.handles = {}

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.starts[-1]}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def handle(self, i):
        if i not in self.handles:
            self.handles[i] = open(self.paths[i], "rb")
        return self.handles[i]

    def readinto(self, buffer):
        view = memoryview(buffer)
        done = 0
        while done < len(view) and self.pos < self.starts[-1]:
            i = bisect_right(self.starts, self.pos) - 1
            f = self.handle(i)
            f.seek(self.pos - self.starts[i])
            chunk = f.readinto(view[done : done + min(len(view) - done, self.starts[i + 1] - self.pos)])
            if not chunk:
                break
            done += chunk
            self.pos += chunk
        return done

    def close(self):
        for f in self.handles.values():
            f.close()
        self.handles = {}
        super().close()


def volumes_of(first):
    """All volumes of `name.zip.001` (sorted), or [name.zip] for a plain zip."""
    first = Path(first)
    if first.suffix[1:].isdigit():
        stem = first.name[: -len(first.suffix)]
        return sorted(first.parent.glob(f"{stem}.[0-9][0-9][0-9]"))
    return [first]


def open_split_zip(first):
    return zipfile.ZipFile(io.BufferedReader(Volumes(volumes_of(first)), buffer_size=1 << 20))


_OPEN = threading.local()  # one open archive per thread: the central directory (57k entries) is read once


def extract(first, member, target):
    """Copy one member of a (split) zip to `target`; returns the path."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".part")
    cache = _OPEN.__dict__.setdefault("archives", {})
    if str(first) not in cache:
        cache[str(first)] = open_split_zip(first)
    with cache[str(first)].open(member) as src, open(tmp, "wb") as dst:
        while chunk := src.read(1 << 20):
            dst.write(chunk)
    tmp.replace(target)
    return target
