import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

here = os.path.dirname(os.path.abspath(__file__))
def in_here(name):
    return os.path.join(here, name)

g = np.asarray(Image.open(in_here("input.jpg")).convert("L"), dtype=float)

# correlation: kernel ke chobir upor shift kore gun-jog (kernel ulta kori na)
def correlate(arr, k):
    kh, kw = k.shape
    ph, pw = kh // 2, kw // 2
    p = np.pad(arr, ((ph, ph), (pw, pw)), mode="edge")
    h, w = arr.shape
    out = np.zeros_like(arr)
    for i in range(kh):
        for j in range(kw):
            out += k[i, j] * p[i:i + h, j:j + w]
    return out

# convolution = kernel 180 degree ghuriye correlation
def convolve(arr, k):
    return correlate(arr, k[::-1, ::-1])

box = np.ones((5, 5)) / 25
sharpen = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]], dtype=float)
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=float)
# ekta ulto-palta (asymmetric) kernel, jate correlation/convolution parthokko dekha jay
slant = np.array([[0, 0, 0],
                  [0, 1, 2],
                  [0, 0, 0]], dtype=float) / 3

def show_u8(a):
    return np.clip(a, 0, 255)

r_box = correlate(g, box)
r_sharp = correlate(g, sharpen)
r_sobel = correlate(g, sobel_x)
c_sobel = convolve(g, sobel_x)

fig, c = plt.subplots(2, 3, figsize=(15, 9))
items = [("Mul", g, True), ("Box 5x5 (blur)", r_box, True),
         ("Sharpen", r_sharp, True), ("Sobel-x correlation", r_sobel, False),
         ("Sobel-x convolution", c_sobel, False),
         ("Correlation - Convolution", r_sobel - c_sobel, False)]
for a, (name, im, is_img) in zip(c.ravel(), items):
    if is_img:
        a.imshow(show_u8(im), cmap="gray", vmin=0, vmax=255)
    else:
        a.imshow(im, cmap="gray")
    a.set_title(name)
    a.axis("off")
plt.tight_layout()
plt.savefig(in_here("corr_overview.png"), dpi=120)

Image.fromarray(show_u8(r_box).astype(np.uint8)).save(in_here("my_box.jpg"))
Image.fromarray(show_u8(r_sharp).astype(np.uint8)).save(in_here("my_sharpen.jpg"))
edge = np.clip(np.abs(r_sobel), 0, 255).astype(np.uint8)
Image.fromarray(edge).save(in_here("my_sobel_corr.jpg"))

# symmetric kernel-e dui-i ek, asymmetric-e alada
d_sym = np.abs(correlate(g, box) - convolve(g, box)).max()
d_asym = np.abs(correlate(g, slant) - convolve(g, slant)).max()
print(f"box (symmetric): corr vs conv max parthokko = {d_sym:.4f}")
print(f"slant (asymmetric): corr vs conv max parthokko = {d_asym:.4f}")
print("kaj shesh, file save hoyeche:", here)
plt.show()