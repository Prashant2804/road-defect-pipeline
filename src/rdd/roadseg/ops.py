"""Mask and feature primitives shared by the road-segmentation backends."""
from __future__ import annotations

from dataclasses import dataclass

# 1.4826 * MAD is a consistent estimator of sigma for normally distributed data.
_MAD_TO_SIGMA = 1.4826


def polygon_mask(polygon, w: int, h: int):
    """Rasterise an (N,2) polygon into a bool mask."""
    import cv2
    import numpy as np

    m = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(m, [polygon.astype(np.int32)], 1)
    return m.astype(bool)


def _kernel(px: int):
    import cv2

    px = max(1, int(px))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))


def erode(mask, px: int):
    import cv2

    if px <= 0:
        return mask
    return cv2.erode(mask.astype("uint8"), _kernel(px)).astype(bool)


def dilate(mask, px: int):
    import cv2

    if px <= 0:
        return mask
    return cv2.dilate(mask.astype("uint8"), _kernel(px)).astype(bool)


def morph_clean(mask, open_px: int = 2, close_px: int = 4):
    """Drop speckle (open), then bridge small gaps (close)."""
    import cv2

    m = mask.astype("uint8")
    if open_px > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _kernel(open_px))
    if close_px > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _kernel(close_px))
    return m.astype(bool)


def fill_holes(mask):
    """Fill enclosed holes in a mask.

    This is load-bearing, not cosmetic. The road is found by appearance
    similarity, but potholes, puddles and mud patches are exactly the pixels
    that *differ* from the road baseline — so they get carved out as holes. The
    road mask must contain the defects that sit on it, otherwise the gating step
    would reject every defect it is supposed to keep.

    Implemented by padding one pixel of guaranteed background all round and
    flood-filling inward: whatever the flood cannot reach is enclosed. Seeding
    from a corner of the unpadded image would be wrong whenever the road touches
    that corner.
    """
    import cv2
    import numpy as np

    m = mask.astype(np.uint8)
    h, w = m.shape
    padded = np.zeros((h + 2, w + 2), np.uint8)
    padded[1:-1, 1:-1] = m
    ff_mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(padded, ff_mask, (0, 0), 255)
    outside = padded[1:-1, 1:-1] == 255
    return (m.astype(bool) | ~outside)


def keep_largest_component(mask, seed=None):
    """Keep one connected component: the one overlapping `seed` most, else the largest.

    Road is a single contiguous surface, so isolated look-alike blobs (a dirt
    verge, a rooftop) can be discarded outright.
    """
    import cv2
    import numpy as np

    m = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return np.zeros_like(mask, dtype=bool)

    best, best_score = 0, -1.0
    for i in range(1, n):
        comp = labels == i
        if seed is not None and seed.any():
            score = float((comp & seed).sum())
            if score == 0:
                continue
        else:
            score = float(stats[i, cv2.CC_STAT_AREA])
        if score > best_score:
            best, best_score = i, score

    if best == 0:
        # Nothing touched the seed — fall back to the largest component overall.
        areas = stats[1:, cv2.CC_STAT_AREA]
        best = int(np.argmax(areas)) + 1
    return labels == best


def keep_components_touching(mask, seed):
    """Every connected component of `mask` containing at least one `seed` pixel.

    The growth half of a hysteresis threshold: `seed` marks confident cores and
    `mask` a permissive extent, so a region is kept in full if any part of it was
    confidently detected, and dropped entirely otherwise.
    """
    import cv2
    import numpy as np

    m = mask.astype(np.uint8)
    if not m.any() or not seed.any():
        return np.zeros_like(mask, dtype=bool)

    n, labels, _, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return np.zeros_like(mask, dtype=bool)

    # Labels present under the seed are exactly the components to keep.
    keep = np.unique(labels[seed & mask])
    keep = keep[keep != 0]
    if keep.size == 0:
        return np.zeros_like(mask, dtype=bool)
    return np.isin(labels, keep)


@dataclass
class Features:
    """Per-pixel appearance channels used to describe the road surface."""

    l: "object"        # LAB lightness
    a: "object"        # LAB green-red
    b: "object"        # LAB blue-yellow
    s: "object"        # HSV saturation
    tex: "object"      # local standard deviation of L (absolute texture energy)
    rtex: "object"     # texture relative to brightness (illumination-invariant)
    cr: "object"       # linear-RGB red chromaticity (illumination-invariant)
    cg: "object"       # linear-RGB green chromaticity
    hue: "object"      # HSV hue, degrees 0..179 (OpenCV convention)
    v: "object"        # HSV value

    def channels(self) -> dict:
        return {"l": self.l, "a": self.a, "b": self.b, "s": self.s,
                "tex": self.tex, "rtex": self.rtex,
                "cr": self.cr, "cg": self.cg, "v": self.v}


# Sigma floors stop a near-uniform seed region from producing a microscopic sigma
# and therefore enormous z-scores from trivial differences. Real footage often has
# *less* surface variation than expected — video compression smooths fine gravel
# texture away entirely — so these floors are set to roughly the smallest change
# in each channel that is physically meaningful, not to the smallest measurable.
SIGMA_FLOORS = {"l": 3.0, "a": 2.0, "b": 2.0, "s": 5.0,
                "tex": 1.0, "rtex": 0.8,
                # Chromaticity is a ratio in [0,1], so a floor here has to encode
                # "how big a colour shift actually means a different material".
                # ~0.015 is a clearly visible shift; wet laterite moves cr by ~0.3,
                # so real mud still clears the bar by an order of magnitude while
                # ordinary surface variation does not.
                "cr": 0.015, "cg": 0.015, "v": 4.0}


def compute_features(bgr, texture_ksize: int = 7) -> Features:
    """Colour + texture channels for a BGR image, all float32.

    Two channels exist specifically to be *illumination-invariant*, because the
    hardest false positive in road inspection is shade being read as
    contamination.

    `rtex` — texture divided by brightness. A shadow is a multiplicative change:
    it scales surface detail and mean brightness together, so absolute texture
    drops while the ratio holds. Water is specular and destroys the detail, so the
    ratio collapses. Keying smoothness on absolute texture reads deep shade as
    standing water.

    `cr`/`cg` — chromaticity in *linearised* RGB. Under a shadow the reflected
    spectrum is unchanged and the linear RGB triple scales by one factor, so
    chromaticity is exactly invariant. Mud is a different material, so it shifts.
    LAB a/b cannot do this job: they are computed from gamma-encoded values, so
    simply darkening a neutral grey moves them several units — measured at 8 units
    of `b` on shadowed road here, which is larger than a real mud signal. That is
    a colour-space artifact masquerading as evidence.
    """
    import cv2
    import numpy as np

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    l, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    k = (max(3, int(texture_ksize) | 1),) * 2
    mean = cv2.boxFilter(l, -1, k, normalize=True)
    mean_sq = cv2.boxFilter(l * l, -1, k, normalize=True)
    tex = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
    # Scaled by 100 so values sit in a comparable range to the other channels.
    rtex = 100.0 * tex / np.maximum(mean, 1.0)

    # Approximate sRGB -> linear with a 2.2 gamma; exact enough, and far cheaper
    # than the piecewise transfer function.
    lin = np.power(bgr.astype(np.float32) / 255.0, 2.2)
    total = lin.sum(axis=2) + 1e-6
    cr = lin[..., 2] / total
    cg = lin[..., 1] / total

    return Features(l=l, a=a, b=b, s=hsv[..., 1], tex=tex, rtex=rtex,
                    cr=cr, cg=cg, hue=hsv[..., 0], v=hsv[..., 2])


def robust_stats(values, sigma_floor: float = 1.0) -> tuple[float, float]:
    """(median, robust sigma) via MAD.

    Median/MAD rather than mean/std because the seed region deliberately
    contains outliers — that is where the potholes are.
    """
    import numpy as np

    if values.size == 0:
        return 0.0, sigma_floor
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return med, max(mad * _MAD_TO_SIGMA, sigma_floor)


def channel_stats(feats: Features, region, channels) -> dict:
    """{channel: (median, sigma)} over `region`, using per-channel sigma floors."""
    chans = feats.channels()
    out = {}
    for ch in channels:
        arr = chans.get(ch)
        if arr is None:
            arr = getattr(feats, ch)
        out[ch] = robust_stats(arr[region], SIGMA_FLOORS.get(ch, 1.0))
    return out


def resize_mask(mask, w: int, h: int):
    """Nearest-neighbour resize of a bool mask."""
    import cv2

    if mask.shape[:2] == (h, w):
        return mask
    return cv2.resize(mask.astype("uint8"), (w, h),
                      interpolation=cv2.INTER_NEAREST).astype(bool)


def overlap_fraction(a, b) -> float:
    """|a AND b| / |a| — 0.0 when `a` is empty."""
    total = float(a.sum())
    if total <= 0:
        return 0.0
    return float((a & b).sum()) / total


def principal_axis(mask) -> str:
    """'vertical' or 'horizontal' — which way an elongated mask runs.

    Second central moments of the mask; the road is the long dimension.
    """
    import numpy as np

    ys, xs = np.nonzero(mask)
    if xs.size < 16:
        return "vertical"
    return "horizontal" if float(xs.var()) > float(ys.var()) else "vertical"
