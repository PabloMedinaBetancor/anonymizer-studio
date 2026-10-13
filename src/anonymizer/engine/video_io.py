"""Reading and writing video with PyAV (FFmpeg comes inside the wheel: nothing to install on Windows).

Ported from the earlier application:
- a clip is read twice, once to analyse it and once to write it, and never held whole in memory;
- the work happens on the clip's own YUV planes, so unblurred pixels go through no colour conversion (a round trip
  through RGB shifts the colours of the whole video); detectors get a BGR copy;
- the phone rotation (display matrix) is applied on reading: frames are analysed and written upright and the
  output carries no rotation;
- audio is copied without re-encoding when MP4 can hold it; container metadata (GPS, date, camera) is not copied.
"""

from __future__ import annotations

import fractions
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np

MP4_AUDIO = {"aac", "mp3", "alac", "ac3", "eac3", "opus", "flac"}  # what MP4 takes without re-encoding


def _av():
    import av  # late import: the app starts, and reports it, even when PyAV is missing

    av.logging.set_level(av.logging.ERROR)
    return av


@dataclass
class VideoInfo:
    width: int  # after rotation, as displayed
    height: int
    rotation: int  # 0, 90, 180 or 270: degrees counter-clockwise that show the frame upright
    fps: float
    frames: int  # as the container estimates it (0 when it does not say)
    duration: float  # seconds
    pix_fmt: str  # yuv420p, yuv420p10le...
    has_audio: bool
    codec: str
    odd: bool = False  # an odd side: 1 px is cropped (H.264 4:2:0 needs even sides)
    audio_tracks: int = 0  # all of them go to the output (an earlier audit, A12)
    video_tracks: int = 1  # only the first is anonymised and written; the job warns about the rest
    track_duration: float = 0.0  # as the video track says (0 when it does not), to spot a truncated source


def _single_thread(stream) -> None:
    """Decode without frame threads. With them, closing a video half-way (Stop, a test that wants one frame) hung
    forever once onnxruntime had started its own threads (the earlier application, seen with a stack dump). Reading is under 3 % of
    the total time."""
    stream.thread_type = "NONE"
    stream.codec_context.thread_count = 1


def _rotation_of(frame) -> int:
    r = int(round(float(getattr(frame, "rotation", 0) or 0))) % 360
    return r if r in (0, 90, 180, 270) else 0


def probe(path: str) -> VideoInfo:
    av = _av()
    with av.open(path) as container:
        if not container.streams.video:
            raise ValueError("no video stream")
        vs = container.streams.video[0]
        _single_thread(vs)
        rotation = 0
        for frame in container.decode(vs):
            rotation = _rotation_of(frame)
            break
        w, h = vs.codec_context.width, vs.codec_context.height
        odd = bool(w % 2 or h % 2)
        w, h = w - w % 2, h - h % 2
        if rotation in (90, 270):
            w, h = h, w
        fps = float(vs.average_rate or vs.guessed_rate or 30)
        track = float(vs.duration * vs.time_base) if vs.duration else 0.0
        duration = track or (float(container.duration) / 1e6 if container.duration else 0.0)
        frames = int(vs.frames or 0) or int(round(duration * fps))
        return VideoInfo(
            w,
            h,
            rotation,
            fps,
            frames,
            duration,
            vs.codec_context.pix_fmt or "?",
            bool(container.streams.audio),
            vs.codec_context.name,
            odd,
            len(container.streams.audio),
            len(container.streams.video),
            track,
        )


def rotate(plane: np.ndarray, rotation: int) -> np.ndarray:
    """Turn a plane (or an HxWxC image) upright; `rotation` is FFmpeg's display matrix angle, counter-clockwise."""
    if rotation == 0:
        return plane
    return np.ascontiguousarray(np.rot90(plane, k={90: 1, 180: 2, 270: 3}[rotation]))


def _plane_array(p) -> np.ndarray:
    """A PyAV plane as an array without the line padding (a copy: the decoder reuses its memory)."""
    a = np.frombuffer(p, np.uint8).reshape(-1, p.line_size)
    return np.array(a[: p.height, : p.width])


def yuv420_planes(frame, rotation: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Y, U and V (8-bit 4:2:0) of a frame, upright and with even sides. Anything else is converted first (the job
    warns: an HDR video comes out in SDR)."""
    if frame.format.name != "yuv420p":
        # Converted to limited range explicitly: otherwise a full-range MJPEG (yuvj420p) changed format but not
        # values and the output came out 5 levels darker (an earlier check, V15).
        from av.video.reformatter import ColorRange

        frame = frame.reformat(format="yuv420p", dst_color_range=ColorRange.MPEG)
    h, w = frame.height - frame.height % 2, frame.width - frame.width % 2
    y, u, v = (_plane_array(p) for p in frame.planes[:3])
    y, u, v = y[:h, :w], u[: h // 2, : w // 2], v[: h // 2, : w // 2]
    return rotate(y, rotation), rotate(u, rotation), rotate(v, rotation)


def bgr_of(frame, rotation: int) -> np.ndarray:
    """An upright BGR copy for the detectors, cropped like yuv420_planes so boxes are valid in both."""
    h, w = frame.height - frame.height % 2, frame.width - frame.width % 2
    return rotate(np.ascontiguousarray(frame.to_ndarray(format="bgr24")[:h, :w]), rotation)


def read_frame(path: str, index: int) -> Optional[np.ndarray]:
    """The upright BGR frame number `index` (as frames() numbers them), or None past the end. Seeks to the key frame
    before it instead of decoding from the start; frame numbers come from the timestamps, so with variable frame rate the
    frame returned is the nearest one."""
    av = _av()
    with av.open(path) as container:
        vs = container.streams.video[0]
        _single_thread(vs)
        fps = float(vs.average_rate or vs.guessed_rate or 30)
        start = vs.start_time or 0
        seekable = index > 0 and vs.time_base is not None
        if seekable:
            container.seek(start + int(index / fps / vs.time_base), stream=vs, backward=True)
        for k, frame in enumerate(container.decode(vs)):
            if seekable:
                if frame.pts is None:
                    return _read_sequentially(path, index)
                number = int(round((frame.pts - start) * vs.time_base * fps))
            else:
                number = k
            if number >= index:
                return bgr_of(frame, _rotation_of(frame))
    return None


def _read_sequentially(path: str, index: int) -> Optional[np.ndarray]:
    for i, frame, rotation in frames(path):
        if i == index:
            return bgr_of(frame, rotation)
    return None


def frames(path: str) -> Iterator[Tuple[int, object, int]]:
    """(index, PyAV frame, rotation) in presentation order. The container closes when the iteration ends."""
    av = _av()
    with av.open(path) as container:
        vs = container.streams.video[0]
        _single_thread(vs)
        for i, frame in enumerate(container.decode(vs)):
            yield i, frame, _rotation_of(frame)


class Writer:
    """Rewrites a video frame by frame: H.264 (libx264, crf 20) and the audio copied.

        with Writer(source, partial_path, info) as w:
            for t, frame, rotation in w.frames():
                y, u, v = yuv420_planes(frame, rotation)
                ...
                w.write(frame, y, u, v)

    Audio packets are copied (or re-encoded to AAC when MP4 cannot hold them) while frames() runs.
    """

    def __init__(
        self, source: str, target: str, info: VideoInfo, crf: int = 20, preset: str = "veryfast", threads: int = 0
    ):
        self.av = _av()
        self.source, self.target, self.info = source, target, info
        self.crf, self.preset, self.threads = crf, preset, threads
        self.ic = self.oc = self.ov = None
        self.written = 0
        self.audio_reencoded = False
        self.audio_lost = 0
        self._first_pts = None
        self._last_pts = None
        self._time_base = None
        self._audio: dict = {}

    def __enter__(self) -> "Writer":
        av = self.av
        self.ic = av.open(self.source)
        self.vs = self.ic.streams.video[0]
        _single_thread(self.vs)
        self.oc = av.open(self.target, "w", format="mp4", options={"movflags": "+faststart"})
        # The video stream goes in before anything else: if the first packet reaching the muxer is audio, the header
        # is written without the video and the encoder crashes ("Floating point exception").
        self._open_video(self.info.width, self.info.height)
        for aus in self.ic.streams.audio:  # every audio track, not only the first (an earlier audit, A12)
            # The codec's name, not the decoder's: "mp3float" is not in MP4_AUDIO and an MP3 was re-encoded to AAC
            # with a false warning (A11).
            codec = aus.codec_context.codec
            name = getattr(codec, "canonical_name", None) or aus.codec_context.name
            try:
                if name in MP4_AUDIO:
                    out = self.oc.add_stream_from_template(aus)
                    self._audio[aus.index] = {"in": aus, "out": out, "resampler": None}
                else:
                    self._add_aac(aus)
            except (av.error.FFmpegError, ValueError):
                self.audio_lost += 1  # that track is not in the output; the job warns
        return self

    def _add_aac(self, aus) -> None:
        """PCM from a .mov/.avi, WMA from a .wmv: MP4 cannot hold them as they are, so they become AAC."""
        rate = aus.codec_context.sample_rate or 48000
        out = self.oc.add_stream("aac", rate=rate)
        # The layout comes from the channel count: an AVI's PCM says "1 channels", which AAC does not accept.
        channels = getattr(aus.codec_context.layout, "nb_channels", 2) or 2
        layout = "mono" if channels == 1 else "stereo"
        out.layout = layout
        resampler = self.av.AudioResampler(format="fltp", layout=layout, rate=rate)
        self._audio[aus.index] = {"in": aus, "out": out, "rate": rate, "pts": None, "resampler": resampler}
        self.audio_reencoded = True

    def _open_video(self, width: int, height: int) -> None:
        cc_in = self.vs.codec_context
        rate = self.vs.average_rate or self.vs.guessed_rate or fractions.Fraction(30, 1)
        # Kept as plain numbers: the stream object must not be touched once the input is closed (reading
        # average_rate after __exit__ read freed memory: wrong durations and segfaults).
        self._frame_seconds = 1.0 / float(rate)
        self._average_rate = float(self.vs.average_rate or 30)
        ov = self.oc.add_stream("libx264", rate=rate)
        ov.width, ov.height = width, height
        ov.pix_fmt = "yuv420p"
        # Non-square pixels (PAL 720x576 shown as 16:9, HDV 1440x1080): the shape on the screen is the size times this ratio, and a video
        # written without it plays squeezed. A quarter turn made the picture upright, so the ratio turns over with it. Unknown (0) stays 1:1.
        sar = cc_in.sample_aspect_ratio
        if sar is not None and sar > 0 and sar != 1:
            ov.codec_context.sample_aspect_ratio = 1 / sar if self.info.rotation in (90, 270) else sar
        ov.time_base = self.vs.time_base
        ov.codec_context.time_base = self.vs.time_base
        ov.options = {"crf": str(self.crf), "preset": self.preset}
        if self.threads:
            ov.codec_context.thread_count = self.threads
        # The source's colour tags only when its pixels pass through untouched (8-bit yuv420p). Converted pixels
        # (full-range yuvj420p, 10-bit HDR) are already 8-bit limited range: copying "full range" or the HLG curve
        # would make them look wrong.
        self.copy_colour = cc_in.pix_fmt == "yuv420p"
        if self.copy_colour:
            for key in ("color_primaries", "color_trc", "colorspace", "color_range"):
                value = getattr(cc_in, key, None)
                if value is not None:
                    setattr(ov.codec_context, key, value)
        self.ov = ov

    def frames(self) -> Iterator[Tuple[int, object, int]]:
        """Demux the source: copy (or re-encode) the audio into the output and yield the video frames in order."""
        streams = [self.vs] + [a["in"] for a in self._audio.values()]
        i = 0
        for packet in self.ic.demux(*streams):
            track = self._audio.get(packet.stream.index) if packet.stream is not self.vs else None
            if track is not None:
                if track["resampler"] is not None:
                    for af in packet.decode():
                        self._reencode(track, af)
                elif packet.dts is not None:
                    packet.stream = track["out"]
                    self.oc.mux(packet)
                continue
            if packet.stream is not self.vs:
                continue
            for frame in packet.decode():
                yield i, frame, _rotation_of(frame)
                i += 1

    def _reencode(self, track: dict, af) -> None:
        """Re-encoded audio keeps its start time. Starting it at 0 put it 11 s off when the video starts later, as
        in a camera or broadcast .m2ts (an earlier audit, A5)."""
        if track["pts"] is None:
            starts = af is not None and af.pts is not None and af.time_base
            track["pts"] = int(round(float(af.pts * af.time_base) * track["rate"])) if starts else 0
        for rf in track["resampler"].resample(af):
            rf.pts = track["pts"]
            rf.time_base = fractions.Fraction(1, track["rate"])
            track["pts"] += rf.samples
            for p in track["out"].encode(rf):
                self.oc.mux(p)

    def write(self, original, y: np.ndarray, u: np.ndarray, v: np.ndarray) -> None:
        h, w = y.shape
        if (w, h) != (self.ov.width, self.ov.height):
            raise ValueError(f"frame of {w}x{h} in a video of {self.ov.width}x{self.ov.height}")
        nf = self.av.VideoFrame(w, h, "yuv420p")
        for plane, data in zip(nf.planes, (y, u, v), strict=True):
            line = plane.line_size
            ph, pw = data.shape
            if line == pw:
                plane.update(np.ascontiguousarray(data))
            else:
                buf = np.zeros((ph, line), np.uint8)
                buf[:, :pw] = data
                plane.update(buf)
        # Timestamps are the source's; when it has none (some AVI/MKV) or they do not advance, they follow on from
        # the previous frame.
        tb = original.time_base or self.vs.time_base
        step = max(1, int(round(1 / (self._average_rate * float(tb))))) if tb else 1
        pts = original.pts
        if self._time_base is None:
            self._time_base = tb
        if pts is None:
            pts = (self._last_pts + step) if self._last_pts is not None else 0
        elif self._last_pts is not None and pts <= self._last_pts:
            pts = self._last_pts + 1
        self._last_pts = pts
        if self._first_pts is None:
            self._first_pts = pts
        nf.pts = pts
        nf.time_base = tb
        if self.copy_colour:
            nf.color_range = original.color_range
            nf.colorspace = original.colorspace
        for p in self.ov.encode(nf):
            self.oc.mux(p)
        self.written += 1

    @property
    def written_duration(self) -> float:
        """Seconds of video written (first to last frame, plus one). The output is checked against this (an earlier audit, A6),
        not against the source container, whose duration may be the audio's (MKV/WebM), a size-based estimate
        (.mpg) or that of a truncated file."""
        if not self.written or self._time_base is None or self._first_pts is None:
            return 0.0
        return float((self._last_pts - self._first_pts) * self._time_base) + self._frame_seconds

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            if exc_type is None and self.ov is not None:
                for p in self.ov.encode(None):
                    self.oc.mux(p)
                for track in self._audio.values():
                    if track["resampler"] is not None:
                        if track["pts"] is not None:
                            self._reencode(track, None)  # what is left in the resampler
                        for p in track["out"].encode(None):
                            self.oc.mux(p)
        finally:
            try:
                if self.oc is not None:
                    self.oc.close()
            finally:
                if self.ic is not None:
                    self.ic.close()
        return False


def check_output(path: str, source: VideoInfo, written: int, written_duration: float = 0.0) -> List[str]:
    """What is wrong with a written video (empty when nothing is). It is decoded whole, since a file cut half-way
    has a plausible size, and frames, size, duration, audio and location metadata are compared."""
    av = _av()
    problems = []
    try:
        with av.open(path) as container:
            if not container.streams.video:
                return ["the output has no video"]
            vs = container.streams.video[0]
            _single_thread(vs)
            n = 0
            w = h = 0
            for frame in container.decode(vs):
                n += 1
                w, h = frame.width, frame.height
            if n != written:
                problems.append(f"the output has {n} frames and {written} were written")
            if n and (w, h) != (source.width, source.height):
                problems.append(f"size {w}x{h} differs from the source's {source.width}x{source.height}")
            if source.has_audio and not container.streams.audio:
                problems.append("the source has audio and the output has none")
            duration = float(vs.duration * vs.time_base) if vs.duration else 0.0
            expected = written_duration or source.duration
            if expected and duration and abs(duration - expected) > max(0.25, 2 / max(1.0, source.fps)):
                problems.append(f"duration {duration:.2f} s differs from the {expected:.2f} s written")
            meta = {k.lower() for k in (container.metadata or {})}
            if meta & {"location", "com.apple.quicktime.location.iso6709", "location-eng"}:
                problems.append("the output keeps the location (GPS)")
    except Exception as e:
        # Whatever stops a full decode means the file is not delivered: the caller turns this into the clip's error.
        problems.append(f"the output cannot be read whole: {type(e).__name__}: {e}")
    return problems
