import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


here = os.path.dirname(os.path.abspath(__file__))
def in_here(name):
    return os.path.join(here, name)


rgb = np.asarray(Image.open(in_here("input.jpg")).convert("RGB"), dtype=float) / 255.0
R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
tiny = 1e-8     

gray_v = 0.299 * R + 0.587 * G + 0.114 * B


inten = (R + G + B) / 3.0


total = R + G + B
satur = 1.0 - 3.0 * np.minimum(np.minimum(R, G), B) / (total + tiny)


top = 0.5 * ((R - G) + (R - B))
bottom = np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + tiny
theta = np.degrees(np.arccos(np.clip(top / bottom, -1.0, 1.0)))
hue_deg = np.where(B <= G, theta, 360.0 - theta)


def to_u8(a, top_val):
    return np.clip(a / top_val * 255, 0, 255).astype(np.uint8)

gray_img = to_u8(gray_v, 1.0)
int_img = to_u8(inten, 1.0)
sat_img = to_u8(satur, 1.0)
hue_img = to_u8(hue_deg, 360.0)

# ---- dekhano ----
panels = [("Mul (RGB)", rgb),
          ("Grayscale", gray_img),
          ("Hue", hue_img),
          ("Saturation", sat_img),
          ("Intensity", int_img)]

fig, cells = plt.subplots(1, 5, figsize=(18, 4))
for c, (name, im) in zip(cells, panels):
    if im.ndim == 2:
        c.imshow(im, cmap="gray", vmin=0, vmax=255)
    else:
        c.imshow(im)
    c.set_title(name)
    c.axis("off")
plt.tight_layout()
plt.savefig(in_here("color_overview.png"), dpi=120)


Image.fromarray(gray_img).save(in_here("my_gray.jpg"))
Image.fromarray(hue_img).save(in_here("my_hue.jpg"))
Image.fromarray(sat_img).save(in_here("my_sat.jpg"))
Image.fromarray(int_img).save(in_here("my_int.jpg"))

print("kaj shesh, file save hoyeche:", here)
plt.show()