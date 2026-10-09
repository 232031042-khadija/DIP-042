import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# code-er folder-er path
here = os.path.dirname(os.path.abspath(__file__))
def in_here(name):
    return os.path.join(here, name)

# chobi niye gray banai
g = np.asarray(Image.open(in_here("input.jpg")).convert("L"), dtype=float)

# ---- 1. s = a*r + b (contrast ar brightness) ----
def line_map(arr, a, b):
    return np.clip(a * arr + b, 0, 255).astype(np.uint8)

# ---- 2. negative: s = 255 - r ----
def flip_dark_light(arr):
    return (255 - arr).astype(np.uint8)

# ---- 3. contrast stretching: min-max ke 0-255 e tana ----
def stretch(arr):
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return arr.astype(np.uint8)
    return ((arr - lo) / (hi - lo) * 255).astype(np.uint8)

a_val, b_val = 1.8, 20           

adjusted = line_map(g, a_val, b_val)
negative = flip_dark_light(g)
stretched = stretch(g)

# ---- dekhano: upore chobi, niche histogram ----
panels = [("Mul gray", g.astype(np.uint8)),
          (f"a={a_val}, b={b_val}", adjusted),
          ("Negative", negative),
          ("Stretched", stretched)]

fig, cells = plt.subplots(2, 4, figsize=(16, 7))
for i, (name, im) in enumerate(panels):
    cells[0, i].imshow(im, cmap="gray", vmin=0, vmax=255)
    cells[0, i].set_title(name)
    cells[0, i].axis("off")

    cells[1, i].hist(im.ravel(), bins=64, range=(0, 255), color="steelblue")
    cells[1, i].set_title("histogram")

plt.tight_layout()
plt.savefig(in_here("linear_overview.png"), dpi=120)

Image.fromarray(adjusted).save(in_here("my_adjusted.jpg"))
Image.fromarray(negative).save(in_here("my_negative.jpg"))
Image.fromarray(stretched).save(in_here("my_stretched.jpg"))

print("min/max age:", int(g.min()), int(g.max()))
print("min/max stretch-er pore:", int(stretched.min()), int(stretched.max()))
print("kaj shesh, file save hoyeche:", here)
plt.show()