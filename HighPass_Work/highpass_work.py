import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

here = os.path.dirname(os.path.abspath(__file__))
def in_here(name):
    return os.path.join(here, name)

g = np.asarray(Image.open(in_here("input.jpg")).convert("L"), dtype=float)

def gauss_kernel(size, sigma):
    r = np.arange(size) - size // 2
    k1 = np.exp(-(r ** 2) / (2 * sigma ** 2))
    k1 /= k1.sum()
    return k1

# blur: age row-e, tarpor column-e
def blur(arr, size=9, sigma=2.0):
    k = gauss_kernel(size, sigma)
    pad = size // 2
    p = np.pad(arr, pad, mode="edge")
    tmp = np.zeros_like(p)
    for i, w in enumerate(k):
        tmp += w * np.roll(p, i - pad, axis=1)
    out = np.zeros_like(p)
    for i, w in enumerate(k):
        out += w * np.roll(tmp, i - pad, axis=0)
    return out[pad:-pad, pad:-pad]

soft = blur(g)
edges = g - soft

def sharpen(k):
    return np.clip(g + k * edges, 0, 255).astype(np.uint8)

amounts = [0.8, 1.5, 3.0]

fig, cells = plt.subplots(2, 3, figsize=(14, 9))
cells[0, 0].imshow(g, cmap="gray", vmin=0, vmax=255)
cells[0, 0].set_title("Mul gray")
cells[0, 1].imshow(soft, cmap="gray", vmin=0, vmax=255)
cells[0, 1].set_title("Blur (low-pass)")
cells[0, 2].imshow(edges, cmap="gray")
cells[0, 2].set_title("Mul - Blur (dhar)")

for c, k in zip(cells[1], amounts):
    c.imshow(sharpen(k), cmap="gray", vmin=0, vmax=255)
    c.set_title(f"Sharp, k = {k}")
for c in cells.ravel():
    c.axis("off")

plt.tight_layout()
plt.savefig(in_here("highpass_overview.png"), dpi=120)

for k in amounts:
    Image.fromarray(sharpen(k)).save(in_here(f"my_sharp_{k}.jpg"))

print("kaj shesh, file save hoyeche:", here)
plt.show()