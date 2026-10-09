import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


here = os.path.dirname(os.path.abspath(__file__))
def in_here(name):
    return os.path.join(here, name)


raw = Image.open(in_here("input.jpg")).convert("RGB")
raw.thumbnail((800, 800))
full = np.asarray(raw, dtype=float)

K = 4                                   
H4, W4 = (full.shape[0] // K) * K, (full.shape[1] // K) * K
full = full[:H4, :W4]                   
small = full[::K, ::K]                  
h, w, _ = small.shape


def where_from(out_len, in_len):
    return (np.arange(out_len) + 0.5) * in_len / out_len - 0.5


def nearest_up(a, nh, nw):
    ys = np.clip(np.floor(where_from(nh, a.shape[0]) + 0.5).astype(int), 0, a.shape[0] - 1)
    xs = np.clip(np.floor(where_from(nw, a.shape[1]) + 0.5).astype(int), 0, a.shape[1] - 1)
    return a[ys][:, xs]


def bilinear_up(a, nh, nw):
    y = where_from(nh, a.shape[0])
    x = where_from(nw, a.shape[1])
    y0, x0 = np.floor(y).astype(int), np.floor(x).astype(int)
    dy = (y - y0)[:, None, None]
    dx = (x - x0)[None, :, None]
    y0c, y1c = np.clip(y0, 0, a.shape[0] - 1), np.clip(y0 + 1, 0, a.shape[0] - 1)
    x0c, x1c = np.clip(x0, 0, a.shape[1] - 1), np.clip(x0 + 1, 0, a.shape[1] - 1)

    upper = a[y0c][:, x0c] * (1 - dx) + a[y0c][:, x1c] * dx
    lower = a[y1c][:, x0c] * (1 - dx) + a[y1c][:, x1c] * dx
    return upper * (1 - dy) + lower * dy


def cubic_w(t, s=-0.5):
    t = np.abs(t)
    near = (s + 2) * t**3 - (s + 3) * t**2 + 1
    far = s * t**3 - 5 * s * t**2 + 8 * s * t - 4 * s
    return np.where(t <= 1, near, np.where(t < 2, far, 0.0))

def bicubic_up(a, nh, nw):
    y = where_from(nh, a.shape[0])
    x = where_from(nw, a.shape[1])
    y0, x0 = np.floor(y).astype(int), np.floor(x).astype(int)

    
    mid = np.zeros((a.shape[0], nw, 3))
    for k in (-1, 0, 1, 2):
        idx = np.clip(x0 + k, 0, a.shape[1] - 1)
        mid += a[:, idx] * cubic_w(x - (x0 + k))[None, :, None]

    out = np.zeros((nh, nw, 3))
    for k in (-1, 0, 1, 2):
        idx = np.clip(y0 + k, 0, a.shape[0] - 1)
        out += mid[idx] * cubic_w(y - (y0 + k))[:, None, None]
    return out

def to_u8(a):
    return np.clip(np.round(a), 0, 255).astype(np.uint8)


def psnr(a, b):
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)

res_near = to_u8(nearest_up(small, H4, W4))
res_bil = to_u8(bilinear_up(small, H4, W4))
res_cub = to_u8(bicubic_up(small, H4, W4))
orig = to_u8(full)


cy, cx, z = H4 // 2, W4 // 2, 40
panels = [("Mul chobi", orig),
          (f"Nearest  {psnr(orig, res_near):.1f} dB", res_near),
          (f"Bilinear  {psnr(orig, res_bil):.1f} dB", res_bil),
          (f"Bicubic  {psnr(orig, res_cub):.1f} dB", res_cub)]

fig, cells = plt.subplots(2, 4, figsize=(16, 8))
for i, (name, im) in enumerate(panels):
    cells[0, i].imshow(im)
    cells[0, i].set_title(name)
    cells[0, i].axis("off")

    cells[1, i].imshow(im[cy - z:cy + z, cx - z:cx + z], interpolation="nearest")
    cells[1, i].set_title("zoom")
    cells[1, i].axis("off")

plt.tight_layout()
plt.savefig(in_here("interp_overview.png"), dpi=120)

Image.fromarray(res_near).save(in_here("my_nearest.jpg"))
Image.fromarray(res_bil).save(in_here("my_bilinear.jpg"))
Image.fromarray(res_cub).save(in_here("my_bicubic.jpg"))

print(f"chhoto size: {w}x{h}, boro korar por: {W4}x{H4}")
print(f"PSNR  nearest : {psnr(orig, res_near):.2f} dB")
print(f"PSNR  bilinear: {psnr(orig, res_bil):.2f} dB")
print(f"PSNR  bicubic : {psnr(orig, res_cub):.2f} dB")
print("kaj shesh, file save hoyeche:", here)
plt.show()