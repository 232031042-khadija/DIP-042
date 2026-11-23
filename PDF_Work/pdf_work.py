import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

here = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(here, "my_output")
os.makedirs(out_dir, exist_ok=True)

gray = np.asarray(Image.open(os.path.join(here, "input.jpg")).convert("L"))

levels = np.arange(256)
counts = np.bincount(gray.ravel(), minlength=256)
pdf = counts / gray.size             # normalised histogram, jog = 1
cdf = np.cumsum(pdf)

mean_val = (levels * pdf).sum()
var_val = (((levels - mean_val) ** 2) * pdf).sum()
nz = pdf[pdf > 0]
entropy = -(nz * np.log2(nz)).sum()  # chobir "information"

# coarse version: 256 ke 32 bin-e bhag kora
pdf32 = pdf.reshape(32, 8).sum(axis=1)

fig, c = plt.subplots(2, 2, figsize=(13, 8))
c[0, 0].imshow(gray, cmap="gray")
c[0, 0].set_title("Chobi")
c[0, 0].axis("off")

c[0, 1].bar(levels, pdf, color="indigo", width=1.0)
c[0, 1].axvline(mean_val, color="red", lw=1.5, label=f"mean = {mean_val:.1f}")
c[0, 1].set_title("Normalised histogram (PDF)")
c[0, 1].set_xlabel("intensity")
c[0, 1].set_ylabel("probability")
c[0, 1].legend()

c[1, 0].plot(levels, cdf, color="darkgreen", lw=2)
c[1, 0].set_title("CDF")
c[1, 0].grid(alpha=0.3)

c[1, 1].bar(np.arange(32) * 8 + 4, pdf32, width=7, color="darkorange")
c[1, 1].set_title("32-bin PDF (coarse)")

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "pdf_overview.png"), dpi=120)
np.savetxt(os.path.join(out_dir, "my_pdf.txt"), pdf, fmt="%.8f")

print("PDF-er jog:", round(pdf.sum(), 6))
print(f"mean = {mean_val:.2f}, std = {np.sqrt(var_val):.2f}, entropy = {entropy:.3f} bit")
print("shobcheye beshi pixel-wala intensity:", int(levels[pdf.argmax()]))
print("kaj shesh, file save hoyeche:", out_dir)
plt.show()