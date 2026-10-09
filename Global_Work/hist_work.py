import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


raw = Image.open("pic_a.jpg").convert("L")
gray = np.asarray(raw)


counts = np.bincount(gray.ravel(), minlength=256)

# probability -> cumulative -> notun brightness
cum = counts.cumsum() / gray.size
new_level = np.round(cum * 255).astype(np.uint8)

# purono value diye notun value tule anlam
boosted = new_level[gray]
after_counts = np.bincount(boosted.ravel(), minlength=256)


fig, box = plt.subplots(2, 3, figsize=(13, 7))

box[0, 0].imshow(gray, cmap="gray")
box[0, 0].set_title("Age")
box[0, 0].axis("off")

box[0, 1].imshow(boosted, cmap="gray")
box[0, 1].set_title("Pore")
box[0, 1].axis("off")

box[0, 2].plot(cum, color="teal", lw=2)
box[0, 2].set_title("CDF (mapping curve)")

box[1, 0].fill_between(range(256), counts, color="slategray")
box[1, 0].set_title("Age-er histogram")

box[1, 1].fill_between(range(256), after_counts, color="darkorange")
box[1, 1].set_title("Pore-r histogram")

box[1, 2].axis("off")
box[1, 2].text(0.05, 0.5,
               f"min/max age: {gray.min()}/{gray.max()}\n"
               f"min/max pore: {boosted.min()}/{boosted.max()}\n"
               f"std age: {gray.std():.1f}\n"
               f"std pore: {boosted.std():.1f}",
               fontsize=12, va="center")

plt.tight_layout()
plt.savefig("result_a.png", dpi=120)
plt.show()

print("kaj shesh, result_a.png save hoyeche")