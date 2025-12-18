import math
import numpy as np
from scipy.ndimage import zoom

def resample_isotropic(vol, spacing_zyx, target_zyx):
    sz, sy, sx = spacing_zyx
    tz, ty, tx = target_zyx
    zoom_factors = (sz/tz, sy/ty, sx/tx)
    vol_iso = zoom(vol, zoom_factors, order=1)
    return vol_iso, target_zyx, zoom_factors

def resample_mask(mask, zoom_factors):
    return zoom(mask.astype(np.float32), zoom_factors, order=0) > 0.5

def crop_cube(vol, center_zyx, size_zyx):
    D,H,W = size_zyx
    Z,Y,X = vol.shape
    cz, cy, cx = [int(round(v)) for v in center_zyx]
    z0, z1 = cz - D//2, cz + math.ceil(D/2)
    y0, y1 = cy - H//2, cy + math.ceil(H/2)
    x0, x1 = cx - W//2, cx + math.ceil(W/2)

    pad_z0, pad_y0, pad_x0 = max(0,-z0), max(0,-y0), max(0,-x0)
    pad_z1, pad_y1, pad_x1 = max(0, z1-Z), max(0, y1-Y), max(0, x1-X)

    z0c, y0c, x0c = max(0,z0), max(0,y0), max(0,x0)
    z1c, y1c, x1c = min(Z,z1), min(Y,y1), min(X,x1)

    cube = np.zeros((D,H,W), dtype=vol.dtype)
    cube[pad_z0:D-pad_z1, pad_y0:H-pad_y1, pad_x0:W-pad_x1] = vol[z0c:z1c, y0c:y1c, x0c:x1c]
    return cube

def window_and_norm(vol, wl=-600, ww=1500):
    lo, hi = wl - ww//2, wl + ww//2
    v = np.clip(vol, lo, hi)
    v = (v - lo) / (hi - lo + 1e-6)
    return v.astype(np.float32)
