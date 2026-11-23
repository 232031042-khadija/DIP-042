import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

here = os.path.dirname(os.path.abspath(__file__))
def in_here(name):
    return os.path.join(here, name)

pic = Image.open(in_here("input.jpg")).convert("L")
pic = pic.resize((128, 128))
g = np.asarray(pic, dtype=float)

spec = np.fft.fftshift(np.fft.fft2(g))
mag = np.log1p(np.abs(spec))

h, w = mag.shape
u = np.arange(w) - w // 2
v = np.arange(h) - h // 2
U, V = np.meshgrid(u, v)

fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(1, 2, 1)
ax1.imshow(mag, cmap="magma")
ax1.set_title("Spectrum (2D)")
ax1.axis("off")

ax2 = fig.add_subplot(1, 2, 2, projection="3d")
ax2.plot_surface(U, V, mag, cmap="viridis", linewidth=0, antialiased=True)
ax2.set_title("Spectrum (3D)")
ax2.set_xlabel("u (freq)")
ax2.set_ylabel("v (freq)")
ax2.set_zlabel("log magnitude")
ax2.view_init(elev=35, azim=-60)

plt.tight_layout()
plt.savefig(in_here("my_freq3d.png"), dpi=120)

# bonus: frequency domain-e high-pass
dist = np.sqrt(U ** 2 + V ** 2)
cutoff = 12
hp_mask = (dist > cutoff).astype(float)
hp_img = np.real(np.fft.ifft2(np.fft.ifftshift(spec * hp_mask)))

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(g, cmap="gray")
plt.title("Mul")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(np.abs(hp_img), cmap="gray")
plt.title(f"High-pass (cutoff {cutoff})")
plt.axis("off")
plt.tight_layout()
plt.savefig(in_here("my_freq_highpass.png"), dpi=120)

print("kaj shesh, file save hoyeche:", here)
plt.show()