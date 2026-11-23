import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

here = os.path.dirname(os.path.abspath(__file__))
def in_here(name):
    return os.path.join(here, name)

g = np.asarray(Image.open(in_here("input.jpg")).convert("L"), dtype=float)
h, w = g.shape

u = np.arange(w) - w // 2
v = np.arange(h) - h // 2
U, V = np.meshgrid(u, v)
D = np.sqrt(U ** 2 + V ** 2)

cut = 20
order = 2

# low-pass er ulto = high-pass
hp_ideal = (D > cut).astype(float)
hp_gauss = 1.0 - np.exp(-(D ** 2) / (2 * cut ** 2))
hp_butter = 1.0 - 1.0 / (1.0 + (D / cut) ** (2 * order))

spec = np.fft.fftshift(np.fft.fft2(g))

def run_filter(mask):
    return np.real(np.fft.ifft2(np.fft.ifftshift(spec * mask)))

masks = [hp_ideal, hp_gauss, hp_butter]
names = ["Ideal", "Gaussian", f"Butterworth (n={order})"]
edges = [run_filter(m) for m in masks]

# dhar-er sathe mul jog korle sharp chobi
boost = 1.0
sharp = [np.clip(g + boost * e, 0, 255) for e in edges]

def stretch(a):
    return (a - a.min()) / (a.max() - a.min() + 1e-8)

fig, c = plt.subplots(3, 4, figsize=(16, 11))
c[0, 0].imshow(np.log1p(np.abs(spec)), cmap="magma")
c[0, 0].set_title("Spectrum")
c[1, 0].imshow(g, cmap="gray", vmin=0, vmax=255)
c[1, 0].set_title("Mul gray")
c[2, 0].imshow(g, cmap="gray", vmin=0, vmax=255)
c[2, 0].set_title("Mul gray")

for i in range(3):
    c[0, i + 1].imshow(masks[i], cmap="gray")
    c[0, i + 1].set_title(f"{names[i]} mask")
    c[1, i + 1].imshow(stretch(edges[i]), cmap="gray")
    c[1, i + 1].set_title(f"{names[i]} (shudhu dhar)")
    c[2, i + 1].imshow(sharp[i], cmap="gray", vmin=0, vmax=255)
    c[2, i + 1].set_title(f"Mul + dhar ({names[i]})")
for a in c.ravel():
    a.axis("off")

plt.tight_layout()
plt.savefig(in_here("hp_freq_overview.png"), dpi=120)

for nm, e in zip(("ideal", "gauss", "butter"), sharp):
    Image.fromarray(e.astype(np.uint8)).save(in_here(f"my_hp_{nm}.jpg"))

print("kaj shesh, file save hoyeche:", here)
plt.show()