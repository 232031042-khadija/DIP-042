import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


here = os.path.dirname(os.path.abspath(__file__))

def in_here(name):
    return os.path.join(here, name)


pic = np.asarray(Image.open(in_here("input.jpg")).convert("RGB"))
H, W, _ = pic.shape
cy, cx = H / 2, W / 2          

# ---- 3x3 matrix 
def shift_m(tx, ty):
    return np.array([[1, 0, tx],
                     [0, 1, ty],
                     [0, 0, 1]], dtype=float)

def zoom_m(sx, sy):
    return np.array([[sx, 0, 0],
                     [0, sy, 0],
                     [0, 0, 1]], dtype=float)

def spin_m(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0],
                     [np.sin(a),  np.cos(a), 0],
                     [0, 0, 1]], dtype=float)

def slant_m(kx):
    return np.array([[1, kx, 0],
                     [0, 1, 0],
                     [0, 0, 1]], dtype=float)


def around_center(m):
    return shift_m(cx, cy) @ m @ shift_m(-cx, -cy)


def apply_affine(img, m):
    inv = np.linalg.inv(m)
    ys, xs = np.mgrid[0:H, 0:W]
    ones = np.ones_like(xs)
    coords = np.stack([xs.ravel(), ys.ravel(), ones.ravel()])   # 3 x N
    src = inv @ coords
    sx = np.round(src[0]).astype(int)
    sy = np.round(src[1]).astype(int)

    ok = (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)
    out = np.zeros_like(img)
    out.reshape(-1, 3)[ok] = img[sy[ok], sx[ok]]
    return out

# ---- 4-ta transformation ----
moved   = apply_affine(pic, shift_m(60, 40))
bigger  = apply_affine(pic, around_center(zoom_m(1.4, 1.4)))
turned  = apply_affine(pic, around_center(spin_m(30)))
slanted = apply_affine(pic, around_center(slant_m(0.4)))


results = [("Original", pic),
           ("Translate (60, 40)", moved),
           ("Scale 1.4x", bigger),
           ("Rotate 30 deg", turned),
           ("Shear 0.4", slanted)]

fig, grid = plt.subplots(1, 5, figsize=(18, 4))
for a, (name, im) in zip(grid, results):
    a.imshow(im)
    a.set_title(name)
    a.axis("off")
plt.tight_layout()
plt.savefig(in_here("affine_overview.png"), dpi=120)


Image.fromarray(moved).save(in_here("my_translated.jpg"))
Image.fromarray(bigger).save(in_here("my_scaled.jpg"))
Image.fromarray(turned).save(in_here("my_rotated.jpg"))
Image.fromarray(slanted).save(in_here("my_sheared.jpg"))

print("4-ta transformation shesh, file save hoyeche:", here)
plt.show()