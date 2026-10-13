"""What a photo file holds, read from its bytes and not through Pillow (review 2, N2).

Pillow shows a decoded picture and the metadata it knows about. A file can hold more: a COM segment between the scans of a
progressive JPEG, a thumbnail inside the JFIF header, a private PNG chunk, an extra RIFF chunk in a WebP, a thumbnail item in an
AVIF. Each analyser here walks one container and lists what is not on the short list the output keeps; photos.py uses them both
to rewrite a JPEG by segments and to check every file it writes. The lists are white lists: what is not named is a leak.
"""

from __future__ import annotations

import struct
from typing import Iterator, List, Optional, Tuple

JFIF_PAYLOAD = 14  # "JFIF\0", version, units, two densities, thumbnail width and height
ADOBE_PAYLOAD = 12  # "Adobe", version, two flags, transform
NO_LENGTH = frozenset([0x01, 0xD8, 0xD9, *range(0xD0, 0xD8)])


def jpeg_units(data: bytes) -> Iterator[Tuple[Optional[int], int, int]]:
    """(marker, start, end) for each part of a JPEG, in order: every marker segment (marker + length + payload), the entropy-coded
    data after a scan header (marker None), and the end of image. After the end of image what is left of the file is one more
    unit with marker -1. ValueError when the file does not follow the format."""
    n = len(data)
    if data[:2] != b"\xff\xd8":
        raise ValueError("not a JPEG")
    yield 0xD8, 0, 2
    i = 2
    while i < n:
        if data[i] != 0xFF:
            raise ValueError(f"no marker at {i}")
        if i + 1 >= n:
            raise ValueError("cut at a marker")
        marker = data[i + 1]
        if marker == 0xFF:  # fill byte
            i += 1
            continue
        if marker in NO_LENGTH:
            yield marker, i, i + 2
            i += 2
            if marker == 0xD9:
                if i < n:
                    yield -1, i, n
                return
            continue
        if i + 4 > n:
            raise ValueError("cut segment")
        length = int.from_bytes(data[i + 2 : i + 4], "big")
        if length < 2 or i + 2 + length > n:
            raise ValueError("bad segment length")
        end = i + 2 + length
        yield marker, i, end
        i = end
        if marker == 0xDA:
            j = i
            while True:
                j = data.find(b"\xff", j)
                if j < 0 or j + 1 >= n:
                    raise ValueError("scan without end")
                follower = data[j + 1]
                if follower == 0x00 or 0xD0 <= follower <= 0xD7 or follower == 0xFF:
                    j += 2 if follower != 0xFF else 1
                    continue
                break
            if j > i:
                yield None, i, j
            i = j
    raise ValueError("no end of image")


SOF_MARKERS = frozenset([0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF])


def _exact_length(marker: int, payload: bytes) -> Optional[int]:
    """How long the payload of a structure segment is, from what it says about itself; None when it cannot be made out (the file is
    then not rebuilt by segments). Anything beyond that length is bytes the decoder never reads."""
    n = len(payload)
    if marker in SOF_MARKERS:
        return 6 + 3 * payload[5] if n >= 6 else None
    if marker == 0xDD or marker == 0xDC:  # DRI, DNL
        return 2 if n >= 2 else None
    if marker == 0xDA:  # SOS: components, two bytes each, then spectral selection and approximation
        return 1 + 2 * payload[0] + 3 if n >= 1 else None
    if marker == 0xDB:  # DQT: tables of 64 values of one or two bytes each, after one byte that says which
        pos = 0
        while pos < n and pos + 1 + 64 * ((payload[pos] >> 4) + 1) <= n:
            pos += 1 + 64 * ((payload[pos] >> 4) + 1)
        return pos or None
    if (
        marker == 0xC4
    ):  # DHT: for each table one byte, 16 counts and that many symbols; bytes that are not a whole table are not read
        pos = 0
        while pos + 17 <= n and pos + 17 + sum(payload[pos + 1 : pos + 17]) <= n:
            pos += 17 + sum(payload[pos + 1 : pos + 17])
        return pos or None
    if marker == 0xCC:  # DAC: pairs; an arithmetic-coded file has a few
        return min(n - n % 2, 64)
    return None


def jpeg_kept_payload(marker: int, payload: bytes) -> Optional[bytes]:
    """What a segment keeps in a clean JPEG, by a white list (review 3, R3-3): the structure markers (frame, tables, restart interval,
    scan header, line count) with exactly their own length, the JFIF header without its thumbnail, the ICC profile and Adobe's colour
    flag. EXIF is rebuilt by the caller. None for every other segment: comments, XMP, IPTC, any other application segment, MPF, the
    reserved markers. ValueError when a structure segment is not what its marker says."""
    if marker == 0xE0:
        if payload.startswith(b"JFIF\x00") and len(payload) >= JFIF_PAYLOAD:
            return payload[: JFIF_PAYLOAD - 2] + b"\x00\x00"
        return None
    if marker == 0xE2:
        return payload if payload.startswith(b"ICC_PROFILE\x00") else None
    if marker == 0xEE:
        return payload[:ADOBE_PAYLOAD] if payload.startswith(b"Adobe") and len(payload) >= ADOBE_PAYLOAD else None
    if marker in SOF_MARKERS or marker in (0xC4, 0xCC, 0xDA, 0xDB, 0xDC, 0xDD):
        exact = _exact_length(marker, payload)
        if exact is None:
            raise ValueError(f"segment {marker:#04x} is not what its marker says")
        return payload[:exact]
    return None


def jpeg_leaks(data: bytes) -> List[str]:
    """What a JPEG holds that the clean one does not: any segment not kept, a JFIF thumbnail, an image hidden in a header segment,
    and anything after the end of image. Looks at every segment, also those between the scans of a progressive file."""
    leaks: List[str] = []
    try:
        for marker, start, end in jpeg_units(data):
            if marker is None or marker in NO_LENGTH and marker != -1:
                continue
            if marker == -1:
                leaks.append("data after the end of the image")
                continue
            payload = data[start + 4 : end]
            if b"\xff\xd8\xff" in payload:
                leaks.append(f"an image inside segment {marker:#04x}")
            if marker == 0xE1 and payload.startswith(b"Exif\x00\x00"):
                continue  # checked as EXIF by the caller
            kept = jpeg_kept_payload(marker, payload)
            if marker == 0xE0 and kept is not None and kept != payload:
                leaks.append("JFIF thumbnail or extra bytes")
            elif kept is not None and kept != payload:
                leaks.append(f"extra bytes in the JPEG segment {marker:#04x}")
            elif kept is None:
                leaks.append(f"JPEG segment {marker:#04x}")
    except ValueError as e:
        leaks.append(f"JPEG not understood: {e}")
    leaks += icc_leaks(jpeg_icc_profile(data))
    return leaks


def jpeg_icc_profile(data: bytes) -> Optional[bytes]:
    """The ICC profile of a JPEG, put together from its APP2 chunks (numbered from 1), or None if it has none."""
    chunks = {}
    try:
        for marker, start, end in jpeg_units(data):
            payload = data[start + 4 : end]
            if marker == 0xE2 and payload.startswith(b"ICC_PROFILE\x00") and len(payload) > 14:
                chunks[payload[12]] = payload[14:]
    except ValueError:
        return None
    return b"".join(chunks[k] for k in sorted(chunks)) if chunks else None


def exif_segments(data: bytes) -> List[bytes]:
    """The payload of every EXIF segment (APP1 "Exif") of a JPEG, in whatever part of the file it is."""
    found: List[bytes] = []
    try:
        for marker, start, end in jpeg_units(data):
            if marker == 0xE1 and data[start + 4 : start + 10] == b"Exif\x00\x00":
                found.append(data[start + 4 : end])
    except ValueError:
        pass
    return found


ICC_TAGS = frozenset(
    (
        "A2B0 A2B1 A2B2 B2A0 B2A1 B2A2 B2D0 B2D1 B2D2 B2D3 D2B0 D2B1 D2B2 D2B3 bXYZ bTRC bkpt calt chad chrm cicp ciis clro clrt cprt "
        "desc devs dmdd dmnd dscm gXYZ gTRC gamt kTRC lumi meas mmod ncl2 ncol pre0 pre1 pre2 pseq psd0 psd1 psd2 psd3 ps2i ps2s "
        "rXYZ rTRC resp scrd scrn targ tech vcgt view vued wtpt"
    )
    .encode()
    .split()
)
ICC_HEADER = 128


def icc_leaks(profile: Optional[bytes]) -> List[str]:
    """Why an ICC profile is not carried over: it does not measure what its header says, its tag table points outside it or at
    nothing it names, it holds a tag no standard names, or it has bytes that no tag and no header field owns (review 3, R3-4: a profile
    is the one metadata block that is kept whole, and PNG stores it compressed, so only reading it can tell)."""
    if not profile:
        return []
    n = len(profile)
    if n < ICC_HEADER + 4 or struct.unpack(">I", profile[:4])[0] != n or profile[36:40] != b"acsp":
        return ["ICC profile that does not measure what its header says"]
    if any(profile[100:ICC_HEADER]):
        return ["ICC profile with bytes in the reserved part of its header"]
    count = struct.unpack(">I", profile[ICC_HEADER : ICC_HEADER + 4])[0]
    table_end = ICC_HEADER + 4 + 12 * count
    if count > 256 or table_end > n:
        return ["ICC profile with a tag table that does not fit"]
    covered = bytearray(n)
    covered[:table_end] = b"\x01" * table_end
    for k in range(count):
        sig, offset, size = struct.unpack(">4sII", profile[ICC_HEADER + 4 + 12 * k : ICC_HEADER + 16 + 12 * k])
        if sig not in ICC_TAGS:
            return [f"ICC profile with the tag {sig.decode('latin-1')!r}, which no standard names"]
        if offset < table_end or offset + size > n:
            return ["ICC profile with a tag outside the profile"]
        covered[offset : offset + size] = b"\x01" * size
    if any(profile[i] for i in range(n) if not covered[i]):
        return ["ICC profile with bytes that no tag owns"]
    return []


# The EXIF blob, which PNG, WebP and AVIF carry as one piece.

TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 13: 4, 16: 8, 17: 8, 18: 8}
SUB_IFDS = frozenset([0x8769, 0x8825, 0xA005, 0x014A])  # Exif, GPS, interoperability, SubIFDs


def exif_blob_leaks(blob: bytes) -> List[str]:
    """The structure of an EXIF blob (a TIFF header and directories, with or without the leading "Exif\0\0"): there must be no next
    directory after the first (IFD1 holds the camera's thumbnail), no JPEG start anywhere, and every non-zero byte must belong to the
    header, a directory or a value that a directory points at (review 3, R3-1: what Pillow does not show is read here)."""
    if blob[:6] == b"Exif\x00\x00":
        blob = blob[6:]
    n = len(blob)
    if n < 8 or blob[:4] not in (b"II*\x00", b"MM\x00*"):
        return ["EXIF that is not a TIFF structure"]
    endian = "<" if blob[:2] == b"II" else ">"
    leaks: List[str] = []
    if b"\xff\xd8\xff" in blob:
        leaks.append("an image inside the EXIF")
    covered = bytearray(n)
    covered[:8] = b"\x01" * 8
    seen = set()

    def walk(offset: int, depth: int) -> None:
        if offset in seen or depth > 4:
            return
        seen.add(offset)
        if offset < 8 or offset + 2 > n:
            leaks.append("EXIF directory outside the data")
            return
        entries = struct.unpack(endian + "H", blob[offset : offset + 2])[0]
        end = offset + 2 + 12 * entries + 4
        if end > n:
            leaks.append("EXIF directory outside the data")
            return
        covered[offset:end] = b"\x01" * (end - offset)
        for k in range(entries):
            at = offset + 2 + 12 * k
            tag, kind, count = struct.unpack(endian + "HHI", blob[at : at + 8])
            size = TYPE_SIZE.get(kind, 1) * count
            if size > 4:
                value = struct.unpack(endian + "I", blob[at + 8 : at + 12])[0]
                if value + size > n:
                    leaks.append("EXIF value outside the data")
                    continue
                covered[value : value + size] = b"\x01" * size
            if tag in SUB_IFDS and kind in (4, 13) and count == 1:
                walk(struct.unpack(endian + "I", blob[at + 8 : at + 12])[0], depth + 1)
        following = struct.unpack(endian + "I", blob[end - 4 : end])[0]
        if following:
            leaks.append("EXIF with a second directory (a thumbnail)")

    walk(struct.unpack(endian + "I", blob[4:8])[0], 0)
    if any(blob[i] for i in range(n) if not covered[i]):
        leaks.append("bytes in the EXIF that no directory points at")
    return leaks


def tiff_leaks(data: bytes) -> List[str]:
    """The structure of a TIFF file (classic, one directory): the header, the directory, the values it points at and the image data
    its strips or tiles point at are the whole file; a non-zero byte anywhere else, a second directory or an image inside a tag is
    listed (review 3, R3-2). Tags are judged by photos.py, which sees the first directory through Pillow."""
    if data[:4] not in (b"II*\x00", b"MM\x00*"):
        return ["TIFF that is not a classic TIFF"]
    endian = "<" if data[:2] == b"II" else ">"
    n = len(data)
    leaks: List[str] = []
    covered = bytearray(n)
    covered[:8] = b"\x01" * 8
    meta = bytearray(n)  # the header, the directory and its values: not the image data
    meta[:8] = b"\x01" * 8

    def span(offset: int, size: int, image: bool = False) -> bool:
        if offset < 0 or size < 0 or offset + size > n:
            return False
        covered[offset : offset + size] = b"\x01" * size
        if not image:
            meta[offset : offset + size] = b"\x01" * size
        return True

    first = struct.unpack(endian + "I", data[4:8])[0]
    if first < 8 or first + 2 > n:
        return ["TIFF directory outside the file"]
    entries = struct.unpack(endian + "H", data[first : first + 2])[0]
    end = first + 2 + 12 * entries + 4
    if end > n:
        return ["TIFF directory outside the file"]
    span(first, end - first)
    if struct.unpack(endian + "I", data[end - 4 : end])[0]:
        leaks.append("TIFF with a second directory")
    values = {}
    for k in range(entries):
        at = first + 2 + 12 * k
        tag, kind, count = struct.unpack(endian + "HHI", data[at : at + 8])
        size = TYPE_SIZE.get(kind, 1) * count
        raw = data[at + 8 : at + 12]
        if size > 4:
            offset = struct.unpack(endian + "I", raw)[0]
            if not span(offset, size):
                leaks.append("TIFF value outside the file")
                continue
            raw = data[offset : offset + size]
        if tag in (273, 279, 324, 325) and kind in (3, 4):
            width = 2 if kind == 3 else 4
            values[tag] = list(struct.unpack(endian + ("H" if kind == 3 else "I") * count, raw[: count * width]))
    for offsets, counts in ((273, 279), (324, 325)):
        for offset, size in zip(values.get(offsets, []), values.get(counts, []), strict=False):
            if not span(offset, size, image=True):
                leaks.append("TIFF image data outside the file")
    if any(data[i] for i in range(n) if not covered[i]):
        leaks.append("bytes in the TIFF that no directory points at")
    if b"\xff\xd8\xff" in bytes(b for b, m in zip(data, meta, strict=True) if m):
        leaks.append("an image inside a TIFF tag")
    return leaks


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# Structure, colour and the EXIF chunk (which photos.py checks as EXIF). No text, no time, no private or unknown chunk.
PNG_ALLOWED = frozenset(
    [
        b"IHDR",
        b"PLTE",
        b"IDAT",
        b"IEND",
        b"tRNS",
        b"gAMA",
        b"cHRM",
        b"sRGB",
        b"iCCP",
        b"sBIT",
        b"pHYs",
        b"bKGD",
        b"eXIf",
        b"cICP",
    ]
)


def png_leaks(data: bytes) -> List[str]:
    leaks: List[str] = []
    i, n, ended = 8, len(data), False
    while i + 8 <= n:
        size = struct.unpack(">I", data[i : i + 4])[0]
        kind = data[i + 4 : i + 8]
        if i + 12 + size > n:
            leaks.append("PNG chunk cut short")
            return leaks
        if ended:
            leaks.append("data after the end of the PNG")
            return leaks
        if kind not in PNG_ALLOWED:
            leaks.append(f"PNG chunk {kind.decode('latin-1')}")
        elif kind == b"eXIf":
            leaks += exif_blob_leaks(data[i + 8 : i + 8 + size])
        ended = kind == b"IEND"
        i += 12 + size
    if i < n:
        leaks.append("data after the end of the PNG")
    elif not ended:
        leaks.append("PNG without its end chunk")
    return leaks


WEBP_ALLOWED = frozenset([b"VP8 ", b"VP8L", b"VP8X", b"ALPH", b"ICCP", b"EXIF"])


def webp_leaks(data: bytes) -> List[str]:
    leaks: List[str] = []
    if len(data) < 12:
        return ["WebP too short"]
    riff_size = struct.unpack("<I", data[4:8])[0]
    end = min(len(data), 8 + riff_size)
    if len(data) > 8 + riff_size + (riff_size & 1):
        leaks.append("data after the end of the WebP")
    i = 12
    while i + 8 <= end:
        kind = data[i : i + 4]
        size = struct.unpack("<I", data[i + 4 : i + 8])[0]
        if kind not in WEBP_ALLOWED:
            leaks.append(f"WebP chunk {kind.decode('latin-1').strip()}")
        elif kind == b"EXIF":
            leaks += exif_blob_leaks(data[i + 8 : i + 8 + size])
        elif kind == b"ICCP":
            leaks += icc_leaks(data[i + 8 : i + 8 + size])
        i += 8 + size + (size & 1)
    return leaks


ALPHA_URNS = (b"urn:mpeg:mpegB:cicp:systems:auxiliary:alpha", b"urn:mpeg:hevc:2015:auxid:1")


def _boxes(data: bytes, start: int, end: int) -> Iterator[Tuple[bytes, int, int]]:
    """(type, payload start, box end) of the boxes between start and end."""
    i = start
    while i + 8 <= end:
        size, kind = struct.unpack(">I4s", data[i : i + 8])
        head = 8
        if size == 1:
            if i + 16 > end:
                raise ValueError("cut box")
            size, head = struct.unpack(">Q", data[i + 8 : i + 16])[0], 16
        elif size == 0:
            size = end - i
        if size < head or i + size > end:
            raise ValueError("bad box size")
        yield kind, i + head, i + size
        i += size
    if i != end:
        raise ValueError("bytes between boxes")


def _iloc_extents(data: bytes, where: Tuple[int, int]) -> dict:
    """{item id: [(file offset, length), ...]} from an item location box, for the items kept in the file itself (construction method 0)."""
    start, end = where
    version = data[start]
    sizes = struct.unpack(">BB", data[start + 4 : start + 6])
    offset_size, length_size = sizes[0] >> 4, sizes[0] & 15
    base_size, index_size = sizes[1] >> 4, (sizes[1] & 15) if version in (1, 2) else 0
    pos = start + 6
    wide = 4 if version == 2 else 2
    count = int.from_bytes(data[pos : pos + wide], "big")
    pos += wide
    found = {}
    for _ in range(count):
        item_id = int.from_bytes(data[pos : pos + wide], "big")
        pos += wide
        method = 0
        if version in (1, 2):
            method = data[pos + 1] & 15
            pos += 2
        pos += 2  # data reference index
        base = int.from_bytes(data[pos : pos + base_size], "big")
        pos += base_size
        extents = int.from_bytes(data[pos : pos + 2], "big")
        pos += 2
        spans = []
        for _ in range(extents):
            pos += index_size
            offset = int.from_bytes(data[pos : pos + offset_size], "big")
            pos += offset_size
            length = int.from_bytes(data[pos : pos + length_size], "big")
            pos += length_size
            spans.append((base + offset, length))
        if method == 0:
            found[item_id] = spans
        if pos > end:
            raise ValueError("item location box cut short")
    return found


def avif_leaks(data: bytes) -> List[str]:
    """An AVIF may hold: the picture (av01 items), the EXIF item and the alpha plane. A thumbnail (a thmb reference), a second
    auxiliary image, an XMP (mime) item, any other item or box is listed."""
    leaks: List[str] = []
    try:
        item_types = {}
        aux_urns = {}
        thumbs = set()
        auxl = set()
        iloc = None
        for kind, start, end in _boxes(data, 0, len(data)):
            if kind not in (b"ftyp", b"meta", b"mdat"):
                leaks.append(f"AVIF box {kind.decode('latin-1')}")
            if kind != b"meta":
                continue
            for inner, istart, iend in _boxes(data, start + 4, end):  # full box: version and flags first
                if inner == b"iinf":
                    version = data[istart]
                    pos = istart + 4 + (2 if version == 0 else 4)
                    for sub, sstart, _ in _boxes(data, pos, iend):
                        if sub != b"infe":
                            leaks.append(f"AVIF item box {sub.decode('latin-1')}")
                            continue
                        infe_version = data[sstart]
                        if infe_version < 2:
                            leaks.append("AVIF item of an old kind")
                            continue
                        cursor = sstart + 4
                        item_id = int.from_bytes(data[cursor : cursor + (2 if infe_version == 2 else 4)], "big")
                        cursor += (2 if infe_version == 2 else 4) + 2
                        item_types[item_id] = data[cursor : cursor + 4]
                elif inner == b"iref":
                    version = data[istart]
                    width = 2 if version == 0 else 4
                    for ref, rstart, _ in _boxes(data, istart + 4, iend):
                        source = int.from_bytes(data[rstart : rstart + width], "big")
                        if ref == b"thmb":
                            thumbs.add(source)
                        elif ref == b"auxl":
                            auxl.add(source)
                        elif ref not in (b"cdsc", b"dimg", b"prem"):
                            leaks.append(f"AVIF reference {ref.decode('latin-1')}")
                elif inner == b"iprp":
                    for sub, sstart, send in _boxes(data, istart, iend):
                        if sub != b"ipco":
                            continue
                        for prop, pstart, pend in _boxes(data, sstart, send):
                            if prop == b"auxC":
                                aux_urns[len(aux_urns) + 1] = data[pstart + 4 : pend].split(b"\x00")[0]
                elif inner == b"iloc":
                    iloc = (istart, iend)
                elif inner not in (b"hdlr", b"pitm", b"idat"):
                    leaks.append(f"AVIF meta box {inner.decode('latin-1')}")
        locations = _iloc_extents(data, iloc) if iloc is not None else {}
        for item_id, kind in item_types.items():
            if kind == b"Exif" and item_id in locations:
                blob = b"".join(data[o : o + n] for o, n in locations[item_id])
                leaks += exif_blob_leaks(blob[4 + struct.unpack(">I", blob[:4])[0] :] if len(blob) >= 4 else b"")
        for kind in item_types.values():
            if kind not in (b"av01", b"Exif", b"grid"):
                leaks.append(f"AVIF item {kind.decode('latin-1').strip()}")
        for _ in thumbs:
            leaks.append("AVIF thumbnail")
        if auxl and not all(urn in ALPHA_URNS for urn in aux_urns.values()):
            leaks.append("AVIF auxiliary image that is not alpha")
        if len(auxl) > 1:
            leaks.append("AVIF with several auxiliary images")
    except (ValueError, struct.error, IndexError) as e:
        leaks.append(f"AVIF not understood: {e}")
    return leaks


def container_leaks(data: bytes) -> List[str]:
    """What the file holds beyond the short list, by its container (chosen by its first bytes). EXIF, XMP-in-EXIF and the tags of a
    TIFF are not here: photos.py reads those."""
    if data[:2] == b"\xff\xd8":
        return jpeg_leaks(data)
    if data[:8] == PNG_SIGNATURE:
        return png_leaks(data)
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return webp_leaks(data)
    if data[4:8] == b"ftyp":
        return avif_leaks(data)
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return tiff_leaks(data)
    return []
