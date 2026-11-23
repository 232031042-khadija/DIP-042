import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

here = os.path.dirname(os.path.abspath(__file__))
def in_here(name):
    return os.path.join(here, name)

# dui-ta alada chobi lagbe: input_a.jpg ar input_b.jpg
pa = Image.open(in_here("input_a.jpg")).convert("L")
pb = Image.open(in_here("input_b.jpg")).convert("L").resize(pa.size)
A = np.asarray(pa, dtype=float)
B = np.asarray(pb, dtype=float)

FA, FB = np.fft.fft2(A), np.fft.fft2(B)
mag_a, ph_a = np.abs(FA), np.angle(FA)
mag_b, ph_b = np.abs(FB), np.angle(FB)

# magnitude ar phase jore chobi banano
def build(mag, ph):
    return np.real(np.fft.ifft2(mag * np.exp(1j * ph)))

a_mag_b_ph = build(mag_a, ph_b)      # A-r magnitude + B-r phase
b_mag_a_ph = build(mag_b, ph_a)      # B-r magnitude + A-r phase
only_phase_a = build(np.ones_like(mag_a), ph_a)   # magnitude muche dilam
only_mag_a = build(mag_a, np.zeros_like(ph_a))    # phase muche dilam

def norm(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

panels = [("Chobi A", A / 255), ("Chobi B", B / 255),
          ("A-mag + B-phase", norm(a_mag_b_ph)),
          ("B-mag + A-phase", norm(b_mag_a_ph)),
          ("Shudhu A-phase", norm(only_phase_a)),
          ("Shudhu A-mag", norm(np.abs(only_mag_a) ** 0.3))]

fig, cells = plt.subplots(2, 3, figsize=(14, 9))
for c, (name, im) in zip(cells.ravel(), panels):
    c.imshow(im, cmap="gray")
    c.set_title(name)
    c.axis("off")
plt.tight_layout()
plt.savefig(in_here("magphase_overview.png"), dpi=120)

for nm, im in [("a_mag_b_phase", a_mag_b_ph), ("b_mag_a_phase", b_mag_a_ph),
               ("only_phase_a", only_phase_a)]:
    Image.fromarray((norm(im) * 255).astype(np.uint8)).save(in_here(f"my_{nm}.jpg"))

print("kaj shesh, file save hoyeche:", here)
plt.show()