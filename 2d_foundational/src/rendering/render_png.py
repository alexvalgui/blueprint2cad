from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import ezdxf
from ezdxf import bbox
from ezdxf.addons.drawing import Frontend, RenderContext
from ezdxf.addons.drawing.config import (
    BackgroundPolicy,
    ColorPolicy,
    Configuration,
)
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MARGIN = 0.08
DPI = 100  
BASE_ID_RE = re.compile(r"^(sample_\d+)")  


def base_id(stem: str) -> str:
    m = BASE_ID_RE.match(stem)
    return m.group(1) if m else stem


def add_auto_dimensions(doc) -> None:
    """Overall width/height + one diameter dim per distinct circle radius.

    Deterministic and style-identical across parts, so 'dims present' is the
    only varying factor in the ablation.
    """
    msp = doc.modelspace()
    extents = bbox.extents(msp)
    xmin, ymin = extents.extmin.x, extents.extmin.y
    xmax, ymax = extents.extmax.x, extents.extmax.y
    w, h = xmax - xmin, ymax - ymin
    ref = max(w, h)
    offset = ref * 0.12

    if "AUTO" not in doc.dimstyles:
        style = doc.dimstyles.new("AUTO")
        style.dxf.dimtxt = ref * 0.03
        style.dxf.dimasz = ref * 0.02
        style.dxf.dimexo = ref * 0.01
        style.dxf.dimdec = 2
        style.dxf.dimdsep = ord(".")

    msp.add_linear_dim(
        base=(xmin, ymin - offset), p1=(xmin, ymin), p2=(xmax, ymin),
        dimstyle="AUTO",
    ).render()
    msp.add_linear_dim(
        base=(xmax + offset, ymin), p1=(xmax, ymin), p2=(xmax, ymax),
        angle=90, dimstyle="AUTO",
    ).render()

    seen: set[float] = set()
    for circle in msp.query("CIRCLE"):
        r = round(circle.dxf.radius, 2)
        if r in seen:
            continue
        seen.add(r)
        c = circle.dxf.center
        msp.add_diameter_dim(
            center=(c.x, c.y), radius=circle.dxf.radius, angle=45,
            dimstyle="AUTO",
        ).render()


def render(doc, out_path: Path, size_px: int, show_hidden: bool) -> None:
    msp = doc.modelspace()

    if not show_hidden and "HIDDEN" in doc.layers:
        doc.layers.get("HIDDEN").off()

    extents = bbox.extents(msp)  # after dims, so nothing gets clipped
    cx = (extents.extmin.x + extents.extmax.x) / 2
    cy = (extents.extmin.y + extents.extmax.y) / 2
    half = max(extents.size.x, extents.size.y) / 2 * (1 + MARGIN)
    if half <= 0:
        raise ValueError("empty drawing extents")

    config = Configuration(
        background_policy=BackgroundPolicy.WHITE,
        color_policy=ColorPolicy.BLACK,
    )

    fig = plt.figure(figsize=(size_px / DPI, size_px / DPI))
    ax = fig.add_axes([0, 0, 1, 1])
    Frontend(RenderContext(doc), MatplotlibBackend(ax), config=config
             ).draw_layout(msp, finalize=True)

    # ezdxf's finalize resizes the figure to the content aspect ratio —
    # this was the root cause of the recurring non-square outputs.
    # Force the square canvas back, then apply framing LAST.
    fig.set_size_inches(size_px / DPI, size_px / DPI, forward=False)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    fig.savefig(out_path, dpi=DPI)  # no bbox_inches: keeps exact square size
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dxf_dir", type=Path)
    ap.add_argument("png_dir", type=Path)
    ap.add_argument("--size", type=int, default=1024,
                    help="output PNG side in px (default 1024)")
    ap.add_argument("--with-dims", action="store_true",
                    help="also emit *_dim.png dimensioned variant")
    ap.add_argument("--no-hidden", action="store_true",
                    help="omit hidden (dashed) lines")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip DXFs whose output PNGs already exist")
    args = ap.parse_args()

    args.png_dir.mkdir(parents=True, exist_ok=True)
    dxfs = sorted(args.dxf_dir.glob("*.dxf"))
    print(f"{len(dxfs)} DXF files -> {args.png_dir} at {args.size}px")

    n_ok = n_fail = n_skip = 0
    for path in dxfs:
        obj_dir = args.png_dir / base_id(path.stem)  # group views per sample
        out_png = obj_dir / f"{path.stem}.png"
        dim_png = obj_dir / f"{path.stem}_dim.png"
        if (args.skip_existing and out_png.exists()
                and (not args.with_dims or dim_png.exists())):
            n_skip += 1
            continue
        try:
            obj_dir.mkdir(parents=True, exist_ok=True)
            doc = ezdxf.readfile(path)
            render(doc, out_png,
                   args.size, show_hidden=not args.no_hidden)
            if args.with_dims:
                doc_d = ezdxf.readfile(path)  # fresh copy: base stays pristine
                add_auto_dimensions(doc_d)
                render(doc_d, dim_png,
                       args.size, show_hidden=not args.no_hidden)
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            print(f"FAIL {path.name}: {type(e).__name__}: {e}")

    print(f"done: {n_ok} ok, {n_fail} failed, {n_skip} skipped")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())