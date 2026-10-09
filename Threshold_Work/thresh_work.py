import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


here = os.path.dirname(os.path.abspath(__file__))
def in_here(name):
    return os.path.join(here, name)

# chobi niye gray banai
gray = np.asarray(Image.open(in_here("input.jpg")).convert("L"), dtype=np.uint8)

# ---- 1. fixed threshold: nijer chosen value ----
def cut_at(arr, t):
    return np.where(arr > t, 255, 0).astype(np.uint8)

# ---- 2. Otsu: dui bhager moddhe spread shobcheye beshi hoy jei t-te ----
def find_otsu(arr):
    counts = np.bincount(arr.ravel(), minlength=256).astype(float)
    prob = counts / counts.sum()
    levels = np.arange(256)

    best_t, best_var = 0, -1.0
    for t in range(1, 255):
        w0 = prob[:t].sum()          
        w1 = prob[t:].sum()          
        if w0 == 0 or w1 == 0:
            continue
        m0 = (levels[:t] * prob[:t]).sum() / w0
        m1 = (levels[t:] * prob[t:]).sum() / w1
        between = w0 * w1 * (m0 - m1) ** 2
        if between > best_var:
            best_var, best_t = between, t
    return best_t

# ---- 3. local (adaptive): prottek pixel nijer ashe-pashe-r gor-er shathe tulona ----
def local_cut(arr, size=31, offset=5):
    pad = size // 2
    padded = np.pad(arr.astype(float), pad, mode="edge")

    # integral image diye ashe-pashe-r jog druto ber kora
    integ = padded.cumsum(axis=0).cumsum(axis=1)
    integ = np.pad(integ, ((1, 0), (1, 0)))
    h, w = arr.shape
    s = (integ[size:size + h, size:size + w]
         - integ[:h, size:size + w]
         - integ[size:size + h, :w]
         + integ[:h, :w])
    mean_map = s / (size * size)
    return np.where(arr > mean_map - offset, 255, 0).astype(np.uint8)

my_t = 110
otsu_t = find_otsu(gray)

fixed_out = cut_at(gray, my_t)
otsu_out = cut_at(gray, otsu_t)
local_out = local_cut(gray)

# ---- dekhano ----
panels = [("Mul gray", gray),
          (f"Fixed t={my_t}", fixed_out),
          (f"Otsu t={otsu_t}", otsu_out),
          ("Local (adaptive)", local_out)]

fig, cells = plt.subplots(1, 4, figsize=(16, 4.5))
for c, (name, im) in zip(cells, panels):
    c.imshow(im, cmap="gray", vmin=0, vmax=255)
    c.set_title(name)
    c.axis("off")
plt.tight_layout()
plt.savefig(in_here("thresh_overview.png"), dpi=120)

Image.fromarray(fixed_out).save(in_here("my_fixed.jpg"))
Image.fromarray(otsu_out).save(in_here("my_otsu.jpg"))
Image.fromarray(local_out).save(in_here("my_local.jpg"))

print("Otsu-r ber kora threshold:", otsu_t)
print("kaj shesh, file save hoyeche:", here)
plt.show()