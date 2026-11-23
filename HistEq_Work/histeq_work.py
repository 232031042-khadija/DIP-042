import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

here = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(here, "my_output")
os.makedirs(out_dir, exist_ok=True)

img = Image.open(os.path.join(here, "input.jpg")).convert("RGB")
rgb = np.asarray(img)

# ek channel-er equalization (nijer formula)
def eq_one(a):
    cnt = np.bincount(a.ravel(), minlength=256)
    cdf = cnt.cumsum()
    first = cdf[cdf > 0][0]
    denom = max(a.size - first, 1)
    lut = np.round((cdf - first) / denom * 255).clip(0, 255).astype(np.uint8)
    return lut[a]

# ---- way 1: R, G, B alada alada ----
per_channel = np.dstack([eq_one(rgb[..., i]) for i in range(3)])

# ---- way 2: shudhu brightness (Y), rong (Cb, Cr) ager moto ----
ycc = np.asarray(img.convert("YCbCr"))
y_new = eq_one(ycc[..., 0])
y_only = np.asarray(Image.merge("YCbCr", (Image.fromarray(y_new),
                                          Image.fromarray(ycc[..., 1]),
                                          Image.fromarray(ycc[..., 2]))).convert("RGB"))

def brightness(a):
    return np.asarray(Image.fromarray(a).convert("L"))

sets = [("Mul", rgb), ("R,G,B alada", per_channel), ("Shudhu Y channel", y_only)]

fig, c = plt.subplots(2, 3, figsize=(15, 8))
for i, (name, im) in enumerate(sets):
    c[0, i].imshow(im)
    c[0, i].set_title(name)
    c[0, i].axis("off")
    c[1, i].hist(brightness(im).ravel(), bins=64, range=(0, 255), color="seagreen")
    c[1, i].set_title("brightness histogram")

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "histeq_overview.png"), dpi=120)
Image.fromarray(per_channel).save(os.path.join(out_dir, "my_rgb_eq.jpg"))
Image.fromarray(y_only).save(os.path.join(out_dir, "my_y_eq.jpg"))

for name, im in sets:
    print(f"{name}: brightness std = {brightness(im).std():.1f}")
print("kaj shesh, file save hoyeche:", out_dir)
plt.show()