#!/usr/bin/env python3
"""
step_complexity.py — Classify STEP files as SIMPLE or COMPLEX.

Analyzes the actual B-Rep geometry of each STEP file and combines several
independent signals:

  1. Topological size .......... number of faces/edges/solids (log-scaled)
  2. Curved-surface ratio ...... fraction of faces that are not planar
  3. Freeform ratio ............ fraction of B-spline / Bezier / offset surfaces
                                 (the strongest indicator of organic/complex CAD)
  4. Surface-type diversity .... Shannon entropy over the surface-type histogram
  5. Curved-edge ratio ......... fraction of edges that are not straight lines
  6. Feature loops (holes) ..... inner wires per face (holes, pockets, bosses)
  7. Shape compactness ......... deviation from a sphere-like form

Each signal is normalized to [0,1], weighted, and summed into a 0-100 score.
A file is COMPLEX if score >= --threshold (default 35). Simple primitives
(boxes, plain cylinders, spheres, cones) are short-circuited to SIMPLE.

Every CSV row includes the raw metrics AND a human-readable "reasons" column.

Dependencies — EITHER of these OpenCASCADE bindings works (auto-detected):
    pip install cadquery-ocp                          # pip, no conda needed
    conda install -c conda-forge pythonocc-core       # conda alternative

Usage:
    python step_complexity.py --input ./models --output results.csv
    python step_complexity.py --input ./models --output results.csv --threshold 40
"""

import os
import sys
import csv
import json
import math
import argparse
import subprocess
import concurrent.futures
from pathlib import Path
from collections import Counter

STEP_SUFFIXES = [".step", ".stp"]

FIELDNAMES = [
    "path", "classification", "complexity_score", "reasons",
    "face_count", "edge_count", "solid_count",
    "curved_ratio", "freeform_ratio", "surface_entropy",
    "distinct_surface_types", "curved_edge_ratio",
    "inner_loops", "compactness",
    "volume", "surface_area", "surface_types",
]

SURFACE_ENUM_TO_NAME = {
    "Plane": "plane", "Cylinder": "cylinder", "Cone": "cone",
    "Sphere": "sphere", "Torus": "torus",
    "BezierSurface": "bezier", "BSplineSurface": "bspline",
    "SurfaceOfRevolution": "revolution", "SurfaceOfExtrusion": "extrusion",
    "OffsetSurface": "offset", "OtherSurface": "other",
}
FREEFORM_TYPES = {"bezier", "bspline", "offset", "other"}


# ----------------------------------------------------------------------------
# Backend shim: supports pythonocc-core (OCC.Core.*) AND cadquery-ocp (OCP.*)
# ----------------------------------------------------------------------------

def get_backend():
    """Return a dict of classes/enums/functions from whichever binding exists."""
    try:
        return _backend_pythonocc()
    except ImportError:
        pass
    try:
        return _backend_ocp()
    except ImportError:
        raise ImportError(
            "No OpenCASCADE bindings found. Install one of:\n"
            "  pip install cadquery-ocp\n"
            "  conda install -c conda-forge pythonocc-core"
        )


def _backend_pythonocc():
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import (
        TopAbs_FACE, TopAbs_EDGE, TopAbs_SOLID, TopAbs_WIRE,
    )
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
    from OCC.Core import GeomAbs
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.TopoDS import topods
    try:
        from OCC.Core.BRepGProp import brepgprop
        vol_fn, surf_fn = brepgprop.VolumeProperties, brepgprop.SurfaceProperties
    except (ImportError, AttributeError):
        from OCC.Core.BRepGProp import (
            brepgprop_VolumeProperties as vol_fn,
            brepgprop_SurfaceProperties as surf_fn,
        )
    return {
        "Reader": STEPControl_Reader,
        "RET_DONE": IFSelect_RetDone,
        "Explorer": TopExp_Explorer,
        "FACE": TopAbs_FACE, "EDGE": TopAbs_EDGE,
        "SOLID": TopAbs_SOLID, "WIRE": TopAbs_WIRE,
        "to_face": topods.Face, "to_edge": topods.Edge,
        "AdaptorSurface": BRepAdaptor_Surface,
        "AdaptorCurve": BRepAdaptor_Curve,
        "surf_enum": lambda name: getattr(GeomAbs, f"GeomAbs_{name}", None),
        "LINE": GeomAbs.GeomAbs_Line,
        "GProps": GProp_GProps,
        "vol_fn": vol_fn, "surf_fn": surf_fn,
    }


def _backend_ocp():
    from OCP.STEPControl import STEPControl_Reader
    from OCP.IFSelect import IFSelect_ReturnStatus
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_ShapeEnum
    from OCP.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
    from OCP.GeomAbs import GeomAbs_SurfaceType, GeomAbs_CurveType
    from OCP.GProp import GProp_GProps
    from OCP.TopoDS import TopoDS
    from OCP.BRepGProp import BRepGProp
    return {
        "Reader": STEPControl_Reader,
        "RET_DONE": IFSelect_ReturnStatus.IFSelect_RetDone,
        "Explorer": TopExp_Explorer,
        "FACE": TopAbs_ShapeEnum.TopAbs_FACE,
        "EDGE": TopAbs_ShapeEnum.TopAbs_EDGE,
        "SOLID": TopAbs_ShapeEnum.TopAbs_SOLID,
        "WIRE": TopAbs_ShapeEnum.TopAbs_WIRE,
        "to_face": TopoDS.Face_s, "to_edge": TopoDS.Edge_s,
        "AdaptorSurface": BRepAdaptor_Surface,
        "AdaptorCurve": BRepAdaptor_Curve,
        "surf_enum": lambda name: getattr(GeomAbs_SurfaceType, f"GeomAbs_{name}", None),
        "LINE": GeomAbs_CurveType.GeomAbs_Line,
        "GProps": GProp_GProps,
        "vol_fn": BRepGProp.VolumeProperties_s,
        "surf_fn": BRepGProp.SurfaceProperties_s,
    }


# ----------------------------------------------------------------------------
# Geometry analysis (runs inside an isolated worker subprocess, because
# OpenCASCADE can hang or segfault on malformed STEP files)
# ----------------------------------------------------------------------------

def load_step(path, B):
    reader = B["Reader"]()
    status = reader.ReadFile(str(path))
    if status != B["RET_DONE"]:
        return None, "STEP read failed (invalid or corrupt file)"
    reader.TransferRoots()
    shape = reader.OneShape()
    if shape is None or shape.IsNull():
        return None, "STEP contained no transferable shape"
    return shape, None


def analyze_step(path):
    """Return a dict of raw geometric metrics for one STEP file."""
    B = get_backend()
    shape, err = load_step(path, B)
    if err:
        raise RuntimeError(err)

    Explorer = B["Explorer"]

    def count(kind):
        n, exp = 0, Explorer(shape, kind)
        while exp.More():
            n += 1
            exp.Next()
        return n

    solid_count = count(B["SOLID"])

    # Precompute the enum -> readable-name lookup once
    enum_lookup = {}
    for enum_name, label in SURFACE_ENUM_TO_NAME.items():
        val = B["surf_enum"](enum_name)
        if val is not None:
            enum_lookup[val] = label

    # --- faces: surface types + inner loops (holes/pockets) -----------------
    surface_hist = Counter()
    inner_loops = 0
    face_count = 0

    exp = Explorer(shape, B["FACE"])
    while exp.More():
        face = B["to_face"](exp.Current())
        face_count += 1
        try:
            st = B["AdaptorSurface"](face).GetType()
            surface_hist[enum_lookup.get(st, "other")] += 1
        except Exception:
            surface_hist["other"] += 1

        # Wires on this face: one is the outer boundary, the rest are holes
        wires = 0
        wexp = Explorer(face, B["WIRE"])
        while wexp.More():
            wires += 1
            wexp.Next()
        inner_loops += max(0, wires - 1)
        exp.Next()

    if face_count == 0:
        raise RuntimeError("Shape has no faces")

    # --- edges: straight vs curved ------------------------------------------
    edge_count = 0
    curved_edges = 0
    eexp = Explorer(shape, B["EDGE"])
    while eexp.More():
        edge = B["to_edge"](eexp.Current())
        edge_count += 1
        try:
            if B["AdaptorCurve"](edge).GetType() != B["LINE"]:
                curved_edges += 1
        except Exception:
            pass
        eexp.Next()

    # --- global measures -----------------------------------------------------
    vp, sp = B["GProps"](), B["GProps"]()
    B["vol_fn"](shape, vp)
    B["surf_fn"](shape, sp)
    volume = max(vp.Mass(), 0.0)
    surface_area = max(sp.Mass(), 0.0)

    curved_faces = face_count - surface_hist.get("plane", 0)
    curved_ratio = curved_faces / face_count
    freeform_faces = sum(surface_hist[t] for t in FREEFORM_TYPES if t in surface_hist)
    freeform_ratio = freeform_faces / face_count

    # Shannon entropy of the surface-type distribution, normalized to [0,1]
    probs = [c / face_count for c in surface_hist.values()]
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    surface_entropy = entropy / math.log(len(SURFACE_ENUM_TO_NAME))

    curved_edge_ratio = curved_edges / edge_count if edge_count else 0.0

    # Sphere-normalized compactness: 1.0 for a sphere, -> 0 as the surface
    # becomes intricate relative to enclosed volume.
    if volume > 1e-12 and surface_area > 1e-12:
        compactness = (36.0 * math.pi * volume ** 2) ** (1.0 / 3.0) / surface_area
        compactness = min(max(compactness, 0.0), 1.0)
    else:
        compactness = 1.0  # non-solid or degenerate: don't let this signal fire

    return {
        "face_count": face_count,
        "edge_count": edge_count,
        "solid_count": solid_count,
        "curved_ratio": round(curved_ratio, 4),
        "freeform_ratio": round(freeform_ratio, 4),
        "surface_entropy": round(surface_entropy, 4),
        "distinct_surface_types": len(surface_hist),
        "curved_edge_ratio": round(curved_edge_ratio, 4),
        "inner_loops": inner_loops,
        "compactness": round(compactness, 4),
        "volume": round(volume, 6),
        "surface_area": round(surface_area, 6),
        "surface_types": dict(surface_hist),
    }


# ----------------------------------------------------------------------------
# Scoring & classification (pure Python — easy to tune and unit-test)
# ----------------------------------------------------------------------------

WEIGHTS = {
    "size": 15,        # topological size (faces, log scale)
    "curved": 15,      # non-planar surface ratio
    "freeform": 25,    # B-spline / Bezier surfaces — strongest complexity signal
    "diversity": 15,   # entropy of surface-type mix
    "curved_edges": 10,
    "features": 12,    # holes / pockets / bosses (inner loops)
    "shape": 8,        # deviation from compact sphere-like form
}


def score_and_classify(m, threshold):
    """Turn raw metrics into a 0-100 score, a label, and readable reasons."""
    subs = {}
    subs["size"] = min(1.0, math.log10(max(m["face_count"], 1)) / 3.0)   # 1000 faces -> 1.0
    subs["curved"] = m["curved_ratio"]
    subs["freeform"] = min(1.0, m["freeform_ratio"] * 2.0)               # 50% freeform saturates
    subs["diversity"] = min(1.0, m["surface_entropy"] * 1.5)
    subs["curved_edges"] = m["curved_edge_ratio"]
    subs["features"] = min(1.0, m["inner_loops"] / 12.0)                 # 12+ holes saturates
    subs["shape"] = 1.0 - m["compactness"]

    score = round(min(sum(WEIGHTS[k] * subs[k] for k in WEIGHTS), 100.0), 2)

    # --- short-circuit obvious primitives -----------------------------------
    types = set(m["surface_types"])
    primitive = (
        (m["face_count"] <= 8 and types <= {"plane"}) or                  # box / prism
        (m["face_count"] <= 4 and types <= {"plane", "cylinder"}) or      # plain cylinder
        (m["face_count"] <= 2 and types <= {"sphere"}) or                 # sphere
        (m["face_count"] <= 5 and types <= {"plane", "cone"})             # plain cone
    )
    if primitive and m["inner_loops"] == 0 and m["solid_count"] <= 1:
        return score, "SIMPLE", "geometric primitive (basic box/cylinder/sphere/cone)"

    # --- assemble human-readable reasons ------------------------------------
    reasons = []
    if subs["freeform"] > 0.3:
        reasons.append(f"{m['freeform_ratio']:.0%} freeform (B-spline/Bezier) surfaces")
    if m["curved_ratio"] > 0.5:
        reasons.append(f"{m['curved_ratio']:.0%} curved surfaces")
    if m["face_count"] >= 100:
        reasons.append(f"large topology ({m['face_count']} faces)")
    if m["inner_loops"] >= 3:
        reasons.append(f"{m['inner_loops']} feature loops (holes/pockets)")
    if m["distinct_surface_types"] >= 4:
        reasons.append(f"{m['distinct_surface_types']} distinct surface types")
    if m["solid_count"] > 1:
        reasons.append(f"multi-body ({m['solid_count']} solids)")
    if subs["shape"] > 0.7 and m["volume"] > 0:
        reasons.append("highly non-compact form")

    if score >= threshold:
        label = "COMPLEX"
        if not reasons:
            reasons.append("combined geometric signals above threshold")
    else:
        label = "SIMPLE"
        simple_bits = []
        if m["curved_ratio"] < 0.3:
            simple_bits.append("mostly planar")
        if m["freeform_ratio"] == 0:
            simple_bits.append("no freeform surfaces")
        if m["face_count"] < 30:
            simple_bits.append("few faces")
        if m["inner_loops"] == 0:
            simple_bits.append("no feature loops")
        reasons = simple_bits or ["low combined complexity"]

    return score, label, "; ".join(reasons)


# ----------------------------------------------------------------------------
# Worker subprocess plumbing (isolates OCC crashes/hangs per file)
# ----------------------------------------------------------------------------

WORKER_SCRIPT = """
import sys, json
sys.path.insert(0, {syspath!r})
from step_complexity import analyze_step, score_and_classify, error_record

path = {path!r}
threshold = {threshold}

try:
    metrics = analyze_step(path)
    score, label, reasons = score_and_classify(metrics, threshold)
    metrics["surface_types"] = json.dumps(metrics["surface_types"])
    print(json.dumps({{
        "path": path, "classification": label,
        "complexity_score": score, "reasons": reasons, **metrics,
    }}))
except Exception as e:
    print(json.dumps(error_record(path, str(e))))
"""


def error_record(path, reason):
    return {
        "path": str(path), "classification": "ERROR",
        "complexity_score": 0.0, "reasons": reason,
        "face_count": 0, "edge_count": 0, "solid_count": 0,
        "curved_ratio": 0.0, "freeform_ratio": 0.0, "surface_entropy": 0.0,
        "distinct_surface_types": 0, "curved_edge_ratio": 0.0,
        "inner_loops": 0, "compactness": 0.0,
        "volume": 0.0, "surface_area": 0.0, "surface_types": "{}",
    }


def process_file(args_tuple):
    path, threshold, timeout, syspath = args_tuple
    script = WORKER_SCRIPT.format(syspath=syspath, path=str(path), threshold=threshold)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=timeout,
        )
        out = proc.stdout.strip()
        if out:
            return json.loads(out.splitlines()[-1])
        err = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "no output"
        return error_record(path, f"worker error: {err}")
    except subprocess.TimeoutExpired:
        return error_record(path, f"timeout (>{timeout}s)")
    except Exception as e:
        return error_record(path, f"crash: {e}")


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def find_step_files(root: Path):
    files = []
    for sfx in STEP_SUFFIXES:
        files.extend(root.rglob(f"*{sfx}"))
        files.extend(root.rglob(f"*{sfx.upper()}"))
    return sorted(set(files))


def load_processed(output):
    """Return paths already successfully processed.

    ERROR rows are NOT counted as processed (so they get retried), and the
    CSV is compacted in place to drop them, keeping the last row per path.
    """
    if not os.path.isfile(output):
        return set()
    with open(output, newline="") as f:
        rows = list(csv.DictReader(f))

    ok = {}
    n_errors = 0
    for row in rows:
        if row.get("classification") == "ERROR":
            n_errors += 1
            ok.pop(row["path"], None)   # retry this file; drop its error row
        else:
            ok[row["path"]] = row

    if n_errors:
        tmp = output + ".tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDNAMES)
            w.writeheader()
            w.writerows(ok.values())
        os.replace(tmp, output)
        print(f"Compacted {output}: dropped {n_errors} ERROR rows (will retry those files).")

    return set(ok)


def main():
    p = argparse.ArgumentParser(description="Classify STEP files as SIMPLE or COMPLEX.")
    p.add_argument("--input", required=True, help="Directory to scan recursively for .step/.stp")
    p.add_argument("--output", required=True, help="Results CSV (appended; supports resume)")
    p.add_argument("--threshold", type=float, default=35.0,
                   help="Score (0-100) at or above which a shape is COMPLEX (default 35)")
    p.add_argument("--renders", default=None,
                   help="Only process STEP files that have a matching subfolder "
                        "(same name, without extension) in this directory")
    p.add_argument("--cores", type=int, default=os.cpu_count() or 4)
    p.add_argument("--timeout", type=int, default=120, help="Per-file timeout in seconds")
    p.add_argument("--batch_size", type=int, default=500)
    args = p.parse_args()

    # Fail fast if no OpenCASCADE bindings are importable — otherwise every
    # worker would fail individually and fill the CSV with 55k error rows.
    try:
        get_backend()
    except ImportError as e:
        sys.exit(f"FATAL: {e}")

    syspath = os.path.dirname(os.path.abspath(__file__))
    files = find_step_files(Path(args.input))

    if args.renders:
        renders_dir = Path(args.renders)
        if not renders_dir.is_dir():
            sys.exit(f"FATAL: --renders directory not found: {renders_dir}")
        render_names = {d.name for d in renders_dir.iterdir() if d.is_dir()}
        before = len(files)
        files = [f for f in files if f.stem in render_names]
        print(f"Render filter: {len(files)} of {before} STEP files have a "
              f"matching folder in {renders_dir} ({before - len(files)} excluded).")

    processed = load_processed(args.output)

    if not os.path.isfile(args.output):
        with open(args.output, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    queue = [f for f in files if str(f) not in processed]
    print(f"Found {len(files)} STEP files — processing {len(queue)} (skipping {len(processed)}).")

    totals = Counter()
    batches = [queue[i:i + args.batch_size] for i in range(0, len(queue), args.batch_size)]

    for bi, batch in enumerate(batches):
        print(f"\nBatch {bi + 1}/{len(batches)} — {len(batch)} files")
        tasks = [(f, args.threshold, args.timeout, syspath) for f in batch]

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.cores) as ex:
            futures = {ex.submit(process_file, t): t[0] for t in tasks}
            for fut in concurrent.futures.as_completed(futures):
                path = futures[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    result = error_record(path, f"crash: {e}")

                with open(args.output, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(result)

                totals[result["classification"]] += 1
                mark = {"COMPLEX": "◆", "SIMPLE": "○", "ERROR": "✗"}[result["classification"]]
                print(f"{mark} {Path(str(path)).name:<40} "
                      f"score={result['complexity_score']:>6} "
                      f"faces={result['face_count']:>5} | {result['reasons']}",
                      flush=True)

    print(f"\nDone. Complex: {totals['COMPLEX']} | Simple: {totals['SIMPLE']} "
          f"| Errors: {totals['ERROR']} | Results: {args.output}")


if __name__ == "__main__":
    main()