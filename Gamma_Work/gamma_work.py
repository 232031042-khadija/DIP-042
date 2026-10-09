import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


here = os.path.dirname(os.path.abspath(__file__))
def in_here(name):
    return os.path.join(here, name)


src = Image.open(in_here("input.jpg")).convert("L")
px = np.asarray(src, dtype=np.uint8)

# 0-255 prottek brightness-er notun value age-i hishab kore rakhi
def make_lut(g):
    levels = np.arange(256) / 255.0
    return np.uint8(np.clip((levels ** g) * 255, 0, 255))

# gamma < 1 hole ujjol, gamma > 1 hole dark
trials = [0.3, 0.5, 1.0, 2.2]

fig, axes = plt.subplots(2, len(trials), figsize=(15, 7))
fig.suptitle("Gamma trial - amar chobi", fontsize=13)

for col, g in enumerate(trials):
    lut = make_lut(g)
    out = lut[px]                      

    axes[0, col].imshow(out, cmap="gray", vmin=0, vmax=255)
    axes[0, col].set_title(f"gamma = {g}")
    axes[0, col].axis("off")

    axes[1, col].plot(lut, color="crimson", lw=2)
    axes[1, col].set_title("transfer curve")
    axes[1, col].set_xlim(0, 255)
    axes[1, col].set_ylim(0, 255)
    axes[1, col].grid(alpha=0.3)

    
    Image.fromarray(out).save(in_here(f"my_gamma_{g}.jpg"))

plt.tight_layout()
plt.savefig(in_here("my_gamma_result.png"), dpi=120)


for g in trials:
    print(f"gamma {g}: mean brightness = {make_lut(g)[px].mean():.1f}")

print("kaj shesh, file save hoyeche:", here)
plt.show()