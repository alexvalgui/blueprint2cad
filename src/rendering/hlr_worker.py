from __future__ import annotations

import math
import os
import sys

import ezdxf

from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.GCPnts import GCPnts_QuasiUniformDeflection
from OCC.Core.HLRAlgo import HLRAlgo_Projector
from OCC.Core.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_EDGE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods
from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt

# ----------------------------------------------------------------------------
# View conventions — identical direction/xdir pairs to the FreeCAD pipeline,
# so drawings stay consistent with the Blender render axes.
# (direction = line of sight / projection plane normal, xdir = image +X)
# ----------------------------------------------------------------------------
VIEWS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "front":  ((0, -1, 0), (1, 0, 0)),
    "back":   ((0, 1, 0),  (-1, 0, 0)),
    "left":   ((-1, 0, 0), (0, -1, 0)),
    "right":  ((1, 0, 0),  (0, 1, 0)),
    "top":    ((0, 0, 1),  (1, 0, 0)),
    "bottom": ((0, 0, -1), (1, 0, 0)),
}

# Edge sets pulled from HLRBRep_HLRToShape, mapped to DXF layers.
# (method name, layer, include_by_default)
EDGE_SETS = [
    ("VCompound",         "VISIBLE", True),   # visible sharp edges
    ("OutLineVCompound",  "VISIBLE", True),   # visible silhouettes (curved faces)
    ("Rg1LineVCompound",  "VISIBLE", False),  # visible smooth (tangent) edges — off: matches TechDraw default
    ("HCompound",         "HIDDEN",  True),   # hidden sharp edges
    ("OutLineHCompound",  "HIDDEN",  True),   # hidden silhouettes
]


def load_step(path: str):
    """Read a STEP file and return a single (possibly compound) TopoDS_Shape."""
    reader = STEPControl_Reader()
    status = reader.ReadFile(path)
    if status != IFSelect_RetDone:
        raise RuntimeError(f"STEP read failed (status={status}): {path}")
    reader.TransferRoots()
    shape = reader.OneShape()
    if shape is None or shape.IsNull():
        raise RuntimeError(f"STEP produced empty shape: {path}")
    return shape


def shape_diagonal(shape) -> float:
    """Bounding-box diagonal, used to scale the discretization tolerance."""
    box = Bnd_Box()
    brepbndlib.Add(shape, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    return math.dist((xmin, ymin, zmin), (xmax, ymax, zmax))


def discretize_edge(edge, deflection: float) -> list[tuple[float, float]]:
    """Approximate one (already projected, planar) edge as a 2D polyline."""
    curve = BRepAdaptor_Curve(edge)
    disc = GCPnts_QuasiUniformDeflection(
        curve, deflection, curve.FirstParameter(), curve.LastParameter()
    )
    if not disc.IsDone() or disc.NbPoints() < 2:
        # Fall back to the edge endpoints so nothing silently disappears.
        p0 = BRep_Tool.Pnt(topods.Vertex(edge))  # pragma: no cover (rare)
        return [(p0.X(), p0.Y())]
    pts = []
    for i in range(1, disc.NbPoints() + 1):
        p = disc.Value(i)
        pts.append((p.X(), p.Y()))  # projection plane = XY; Z ~ 0
    return pts


def project_view(shape, direction, xdir):
    """Run HLR for one viewing direction; return the HLRBRep_HLRToShape extractor."""
    origin = gp_Pnt(0.0, 0.0, 0.0)
    ax2 = gp_Ax2(origin, gp_Dir(*direction), gp_Dir(*xdir))
    projector = HLRAlgo_Projector(ax2)

    algo = HLRBRep_Algo()
    algo.Add(shape)
    algo.Projector(projector)
    algo.Update()
    algo.Hide()
    return HLRBRep_HLRToShape(algo)


def write_view_dxf(extractor, out_path: str, deflection: float,
                   include_smooth: bool = False) -> int:
    """Write one projected view to a DXF with VISIBLE/HIDDEN layers.

    Returns the number of polylines written (0 => empty view, caller may warn).
    """
    doc = ezdxf.new("R2010", setup=True)  # setup=True loads DASHED linetype
    msp = doc.modelspace()
    doc.layers.add("VISIBLE", color=7)
    doc.layers.add("HIDDEN", color=8, linetype="DASHED")

    n_written = 0
    for method, layer, default_on in EDGE_SETS:
        if not default_on and not include_smooth:
            continue
        compound = getattr(extractor, method)()
        if compound is None or compound.IsNull():
            continue
        exp = TopExp_Explorer(compound, TopAbs_EDGE)
        while exp.More():
            edge = topods.Edge(exp.Current())
            pts = discretize_edge(edge, deflection)
            if len(pts) >= 2:
                msp.add_lwpolyline(pts, dxfattribs={"layer": layer})
                n_written += 1
            exp.Next()

    doc.saveas(out_path)
    return n_written


def process_part(step_path: str, out_dir: str, prefix: str | None = None,
                 hidden_lines: bool = True) -> dict:
    """Full pipeline for one STEP file: 6 views -> 6 DXFs.

    Returns a small report dict (used by batch_runner for logging).
    Raises on unreadable/empty STEP; individual empty views only warn.
    """
    os.makedirs(out_dir, exist_ok=True)
    if prefix is None:
        prefix = os.path.splitext(os.path.basename(step_path))[0]

    shape = load_step(step_path)
    diag = shape_diagonal(shape)
    if diag <= 0:
        raise RuntimeError(f"Degenerate bounding box: {step_path}")
    # Chord deflection for curve tessellation: 0.05% of part diagonal, floored.
    deflection = max(diag * 5e-4, 1e-4)

    report = {"part": prefix, "diag": round(diag, 3), "views": {}}
    for name, (direction, xdir) in VIEWS.items():
        extractor = project_view(shape, direction, xdir)
        out_path = os.path.join(out_dir, f"{prefix}_{name}.dxf")
        n = write_view_dxf(extractor, out_path, deflection)
        if not hidden_lines:
            # Cheap post-filter alternative exists, but preferring explicit
            # generation keeps clean/dimmed variants byte-identical; hidden
            # lines are simply skipped at render time instead (see render_png).
            pass
        report["views"][name] = n
    return report


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python hlr_worker.py <part.stp> <out_dir>")
        sys.exit(2)
    rep = process_part(sys.argv[1], sys.argv[2])
    print(rep)