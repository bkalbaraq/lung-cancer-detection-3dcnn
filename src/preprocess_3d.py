

from __future__ import annotations

import argparse
import random
from typing import List, Tuple

import numpy as np

# fix for the issue it prints about int after running
if not hasattr(np, "int"):
    np.int = int

from tqdm import tqdm
import pylidc as pl

from config import (
    VOX_MM, CUBE, VAL_SPLIT, SEED, OUT_DIR,
    NEG_PER_POS, NEG_FROM_EMPTY_SCAN,
    MIN_BODY_INTENSITY, MAX_EMPTY_TRIES_PER_SCAN,
)
from utils_3d import resample_isotropic, crop_cube, window_and_norm



def rng_choice_coords(mask: np.ndarray, num: int, rng: np.random.Generator):
    """Pick random (i,j,k) indices where mask==True."""
    inds = np.argwhere(mask)
    if len(inds) == 0:
        return np.empty((0, 3), dtype=np.int32)
    sel = rng.choice(len(inds), size=min(num, len(inds)), replace=False)
    return inds[sel]


def sample_body_mask(vol01: np.ndarray, min_intensity=MIN_BODY_INTENSITY) -> np.ndarray:
    """Body mask ~ not-air region after window+norm in [0,1]."""
    return (vol01 >= float(min_intensity))


def extract_cube_safe(vol_iso: np.ndarray, center_ijk: np.ndarray, cube_zyx) -> np.ndarray | None:
    """
    Center-based crop on resampled volume.
    NOTE: naming uses (z,y,x), but in practice this is (i,j,k) = vol_iso.shape.
    """
    D, H, W = cube_zyx
    Z, Y, X = vol_iso.shape

    cz, cy, cx = np.round(center_ijk).astype(int)
    z0, y0, x0 = cz - D // 2, cy - H // 2, cx - W // 2
    z1, y1, x1 = z0 + D, y0 + H, x0 + W

    if z1 <= 0 or y1 <= 0 or x1 <= 0 or z0 >= Z or y0 >= Y or x0 >= X:
        return None

    cube = crop_cube(vol_iso, (cz, cy, cx), (D, H, W))
    if cube.shape != (D, H, W):
        return None
    return cube




def centers_from_annotation(ann) -> List[Tuple[float, float, float]]:
   
    centers: List[Tuple[float, float, float]] = []

    
    try:
        c = np.asarray(ann.centroid)
        if c.shape == (3,):
            ci, cj, ck = map(float, c)
            centers.append((ci, cj, ck))
            return centers
    except Exception:
        pass

    
    try:
        bbox = ann.bbox()  # tuple of 3 slices
        if isinstance(bbox, tuple) and len(bbox) == 3:
            s0, s1, s2 = bbox  # (i_slice, j_slice, k_slice)
            ci = 0.5 * (float(s0.start) + float(s0.stop))
            cj = 0.5 * (float(s1.start) + float(s1.stop))
            ck = 0.5 * (float(s2.start) + float(s2.stop))
            centers.append((ci, cj, ck))
            return centers
    except Exception:
        pass

    # Nothing worked
    return centers


def positive_centers_from_annotations(scan) -> List[Tuple[float, float, float]]:
    """Collect (i,j,k) centers for all annotations in a scan."""
    centers: List[Tuple[float, float, float]] = []
    for ann in scan.annotations:
        centers.extend(centers_from_annotation(ann))
    return centers


def stratified_indices(y, val_split: float, seed: int):
    
    rng = np.random.default_rng(seed)
    y = np.asarray(y)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    n_val_pos = int(len(pos_idx) * val_split)
    n_val_neg = int(len(neg_idx) * val_split)

    val_idx = np.concatenate([pos_idx[:n_val_pos], neg_idx[:n_val_neg]])
    train_idx = np.concatenate([pos_idx[n_val_pos:], neg_idx[n_val_neg:]])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx




def build_dataset(limit_scans: int | None = None):
    random.seed(SEED)
    rng = np.random.default_rng(SEED)

    scans = pl.query(pl.Scan).all()
    if limit_scans is not None:
        scans = scans[:int(limit_scans)]

    X_cubes: list[np.ndarray] = []
    y_labels: list[int] = []

    n_scans_processed = 0
    n_annotated_scans = 0
    n_empty_scans = 0
    n_empty_scans_contributed = 0

    for scan in tqdm(scans, desc="Scans"):
        anns_count = len(list(scan.annotations))  # from pylidc dataset

        try:
            vol_hu = scan.to_volume()
        except Exception as e:
            print("Could not load scan:", getattr(scan, "patient_id", "?"), e)
            continue

        Z, Y, X = vol_hu.shape  

        
        px = scan.pixel_spacing
        if isinstance(px, float):
            px = (px, px)
        spacing_zyx = (
            float(getattr(scan, "slice_thickness", 1.0)),  
            float(px[0]),
            float(px[1]),
        )

        
        try:
            vol_iso, _, zoom = resample_isotropic(vol_hu, spacing_zyx, VOX_MM)
        except Exception as e:
            print("Resample failed for", getattr(scan, "patient_id", "?"), e)
            continue

        vol01 = window_and_norm(vol_iso)  # [0,1]
        body_mask = sample_body_mask(vol01)

        n_scans_processed += 1

        if anns_count > 0:
            n_annotated_scans += 1

            centers_ijk = positive_centers_from_annotations(scan)

            if len(centers_ijk) == 0 and n_annotated_scans <= 8:
                #debugging for pylidc
                print(f"DEBUG: {scan.patient_id} has {anns_count} annotations but 0 centers extracted.")
                try:
                    ann0 = list(scan.annotations)[0]
                    print("       centroid (raw):", getattr(ann0, "centroid", "<no centroid attr>"))
                    print("       bbox   (raw):", ann0.bbox())
                except Exception as e:
                    print("       (Failed to print ann[0] details:", e, ")")

            
            for ci, cj, ck in centers_ijk:
                
                cz = ci * zoom[0]
                cy = cj * zoom[1]
                cx = ck * zoom[2]

                cube = extract_cube_safe(vol01, np.array([cz, cy, cx]), CUBE)
                if cube is None:
                    continue
                X_cubes.append(cube)
                y_labels.append(1)

                # Negatives per positive from body
                if NEG_PER_POS > 0:
                    coords = rng_choice_coords(body_mask, NEG_PER_POS * 2, rng)
                    got = 0
                    for p in coords:
                        cube_n = extract_cube_safe(vol01, p, CUBE)
                        if cube_n is None:
                            continue
                        X_cubes.append(cube_n)
                        y_labels.append(0)
                        got += 1
                        if got >= NEG_PER_POS:
                            break

        else:
            # getting negatives from non-annotated scans
            n_empty_scans += 1
            contributed = 0
            tries = 0
            needed = int(NEG_FROM_EMPTY_SCAN)
            while contributed < needed and tries < int(MAX_EMPTY_TRIES_PER_SCAN):
                tries += 1
                ijk = rng_choice_coords(body_mask, 1, rng)
                if ijk.shape[0] == 0:
                    break
                p = ijk[0]
                cube_n = extract_cube_safe(vol01, p, CUBE)
                if cube_n is None:
                    continue
                X_cubes.append(cube_n)
                y_labels.append(0)
                contributed += 1
            if contributed > 0:
                n_empty_scans_contributed += 1

    
    X = np.array(X_cubes, dtype=np.float32)
    y = np.array(y_labels, dtype=np.int32)
    if X.ndim == 4:
        X = X[..., None]  

    # this will stop the program if theres no positives
    if (y == 1).sum() == 0:
        print("\n=== Preprocessing Summary ===")
        print(f"Scans processed:            {n_scans_processed}")
        print(f"  • Annotated scans:        {n_annotated_scans}")
        print(f"  • Empty (no-ann) scans:   {n_empty_scans}")
        print(f"  • Empty scans contributed negatives: {n_empty_scans_contributed}")
        print(f"Samples total:              {len(y)}")
        print(f"  • Positives (1):          0")
        print(f"  • Negatives (0):          {int((y == 0).sum())}")
        raise RuntimeError("No positive cubes found. Check annotation centroid/bbox parsing.")

    # S split
    train_idx, val_idx = stratified_indices(y, VAL_SPLIT, SEED)
    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]

    
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    print("\n=== Preprocessing Summary ===")
    print(f"Scans processed:            {n_scans_processed}")
    print(f"  • Annotated scans:        {n_annotated_scans}")
    print(f"  • Empty (no-ann) scans:   {n_empty_scans}")
    print(f"  • Empty scans contributed negatives: {n_empty_scans_contributed}")
    print(f"Samples total:              {len(y)}")
    print(f"  • Positives (1):          {n_pos}")
    print(f"  • Negatives (0):          {n_neg}")
    print("  • Train positives:", int((y_train == 1).sum()),
          "negatives:", int((y_train == 0).sum()))
    print("  • Val   positives:", int((y_val   == 1).sum()),
          "negatives:", int((y_val   == 0).sum()))
    print("Train: X=", X_train.shape, "y=", y_train.shape)
    print("Val:   X=", X_val.shape,   "y=", y_val.shape)

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "X_train.npy", X_train)
    np.save(OUT_DIR / "y_train.npy", y_train)
    np.save(OUT_DIR / "X_val.npy",   X_val)
    np.save(OUT_DIR / "y_val.npy",   y_val)


def main():
    parser = argparse.ArgumentParser(description="Preprocess LIDC-IDRI into 3D cubes.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optionally limit number of scans for quick runs.")
    args = parser.parse_args()
    build_dataset(limit_scans=args.limit)


if __name__ == "__main__":
    main()
