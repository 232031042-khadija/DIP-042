import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


here = os.path.dirname(os.path.abspath(__file__))
def in_here(name):
    return os.path.join(here, name)


gray = np.asarray(Image.open(in_here("input.jpg")).convert("L"), dtype=np.uint8)


def get_layer(arr, k):
    return (arr >> k) & 1

layers = [get_layer(gray, k) for k in range(8)]


def rebuild(arr, top_n):
    total = np.zeros_like(arr, dtype=np.uint8)
    for k in range(8 - top_n, 8):
        total += (get_layer(arr, k) << k).astype(np.uint8)
    return total


fig = plt.figure(figsize=(16, 8))

plt.subplot(3, 4, 1)
plt.imshow(gray, cmap="gray")
plt.title("Mul chobi")
plt.axis("off")

for k in range(7, -1, -1):
    pos = 8 - k + 1            # bit 7 -> 2 number box, bit 0 -> 9 number box
    plt.subplot(3, 4, pos)
    plt.imshow(layers[k], cmap="gray")
    plt.title(f"Bit {k}")
    plt.axis("off")

for i, n in enumerate([1, 2, 3]):
    plt.subplot(3, 4, 10 + i)
    plt.imshow(rebuild(gray, n), cmap="gray", vmin=0, vmax=255)
    plt.title(f"Shudhu top {n} bit")
    plt.axis("off")

plt.tight_layout()
plt.savefig(in_here("bits_overview.png"), dpi=120)


Image.fromarray((layers[7] * 255).astype(np.uint8)).save(in_here("my_bit7.png"))
Image.fromarray((layers[0] * 255).astype(np.uint8)).save(in_here("my_bit0.png"))


for k in range(7, -1, -1):
    print(f"bit {k}: 1-er shonkha = {layers[k].sum()}")

print("kaj shesh, file save hoyeche:", here)
plt.show()