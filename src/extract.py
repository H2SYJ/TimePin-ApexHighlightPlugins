"""Host-owned frame batching, loading, and scratch cleanup for recognition."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from .frames import Frame, FrameClient


@dataclass
class FrameBatch:
    frames: list[Frame]
    scratch_dir: Path | None = None
    complete: bool = False

    def __iter__(self) -> Iterator[Frame]:
        return iter(self.frames)

    def __len__(self) -> int:
        return len(self.frames)


class FrameExtractor:
    """Extract batches of host-delivered frames, decode them, and clean up.

    The host owns decoding and file creation. This class only consumes the
    returned frame paths and removes the scratch directory once the request
    has returned, matching the existing plugin safety contract.
    """

    def __init__(
        self,
        client: FrameClient,
        video_path: Path,
        mode: str,
        sample_fps: float,
        output_dir: Path,
        *,
        batch_seconds: float = 8.0,
        max_samples: int = 128,
    ) -> None:
        self.client = client
        self.video_path = video_path
        self.mode = mode
        self.sample_fps = sample_fps
        self.output_dir = output_dir
        self.batch_seconds = batch_seconds
        self.batch_samples = max(
            1, min(max_samples, int(sample_fps * batch_seconds))
        )
        self.scratch_dirs: list[Path] = []

    def iter_batches(
        self,
        time_range: tuple[float, float],
        *,
        on_batch: Callable[[int, int], None] | None = None,
    ) -> Iterator[FrameBatch]:
        """Yield host-delivered batches for one extraction window."""
        start, end = time_range
        total = max(0, int((end - start) * self.sample_fps))
        position = start
        done = 0
        while position < end:
            batch_end = min(end, position + self.batch_samples / self.sample_fps)
            scratch = Path(
                tempfile.mkdtemp(prefix="apex-frames-", dir=self.output_dir)
            )
            self.scratch_dirs.append(scratch)
            extraction_complete = False
            try:
                frames = self.client.extract(
                    str(self.video_path),
                    time_range=(position, batch_end),
                    scratch_dir=str(scratch),
                    mode=self.mode,
                    sample_fps=self.sample_fps,
                )
                extraction_complete = True
                batch = FrameBatch(frames=frames, scratch_dir=scratch)
                done += len(frames)
                if on_batch is not None:
                    on_batch(done, total)
                yield batch
            finally:
                if (
                    extraction_complete
                    and scratch.resolve() == scratch
                    and not scratch.is_symlink()
                ):
                    shutil.rmtree(scratch)
            position = batch_end

    @staticmethod
    def load_images(frames) -> list:
        """Decode extracted frame files into BGR arrays without modifying them.

        Reads raw bytes and decodes via ``cv2.imdecode`` instead of
        ``cv2.imread`` so Unicode/ANSI-mangled paths (中文目录、临时目录)
        work exactly like AHC's ``image_io.read_image``.
        """
        from timepin_vision.frames import decode_frame

        images = []
        for frame in frames:
            images.append(decode_frame(frame.path.read_bytes()))
        return images
