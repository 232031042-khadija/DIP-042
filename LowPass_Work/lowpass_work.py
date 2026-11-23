import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

here = os.path.dirname(os.path.abspath(__file__))
def in_here(name):
    return os.path.join(here, name)

g = np.asarray(Image.open(in_here("input.jpg")).convert("L"), dtype=float)
h, w = g.shape

# frequency grid: majhkhan theke dooratto
u = np.arange(w) - w // 2
v = np.arange(h) - h // 2
U, V = np.meshgrid(u, v)
D = np.sqrt(U ** 2 + V ** 2)

cut = 30          # nijer moto bodlao
order = 2

# ---- 3-ta low-pass mask ----
mask_ideal = (D <= cut).astype(float)
mask_gauss = np.exp(-(D ** 2) / (2 * cut ** 2))
mask_butter = 1.0 / (1.0 + (D / cut) ** (2 * order))

spec = np.fft.fftshift(np.fft.fft2(g))

def run_filter(mask):
    return np.real(np.fft.ifft2(np.fft.ifftshift(spec * mask)))

outs = [run_filter(m) for m in (mask_ideal, mask_gauss, mask_butter)]
names = ["Ideal", "Gaussian", f"Butterworth (n={order})"]
masks = [mask_ideal, mask_gauss, mask_butter]

# ---- dekhano ----
fig, c = plt.subplots(2, 4, figsize=(16, 8))
c[0, 0].imshow(np.log1p(np.abs(spec)), cmap="magma")
c[0, 0].set_title("Spectrum")
c[1, 0].imshow(g, cmap="gray", vmin=0, vmax=255)
c[1, 0].set_title("Mul gray")

for i in range(3):
    c[0, i + 1].imshow(masks[i], cmap="gray")
    c[0, i + 1].set_title(f"{names[i]} mask")
    c[1, i + 1].imshow(np.clip(outs[i], 0, 255), cmap="gray", vmin=0, vmax=255)
    c[1, i + 1].set_title(f"{names[i]}, D0={cut}")
for a in c.ravel():
    a.axis("off")

plt.tight_layout()
plt.savefig(in_here("lowpass_overview.png"), dpi=120)

for nm, o in zip(("ideal", "gauss", "butter"), outs):
    Image.fromarray(np.clip(o, 0, 255).astype(np.uint8)).save(in_here(f"my_lp_{nm}.jpg"))

print("kaj shesh, file save hoyeche:", here)
plt.show()