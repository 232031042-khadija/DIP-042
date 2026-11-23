import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

here = os.path.dirname(os.path.abspath(__file__))
def in_here(name):
    return os.path.join(here, name)

N = 256
yy, xx = np.mgrid[0:N, 0:N]

# kritrim (synthetic) pattern: koto bar dheu thakle spectrum-e kothay dag pore
wave_x = np.sin(2 * np.pi * 8 * xx / N)                       # left-right 8 dheu
wave_y = np.sin(2 * np.pi * 20 * yy / N)                      # up-down 20 dheu
wave_mix = wave_x + 0.5 * np.sin(2 * np.pi * (15 * xx + 15 * yy) / N)

real = Image.open(in_here("input.jpg")).convert("L").resize((N, N))
real = np.asarray(real, dtype=float)

def spectrum(a):
    return np.fft.fftshift(np.fft.fft2(a))

def log_mag(s):
    return np.log1p(np.abs(s))

items = [("sin x, 8 dheu", wave_x), ("sin y, 20 dheu", wave_y),
         ("duita mishano", wave_mix), ("Asol chobi", real)]

fig, c = plt.subplots(2, 4, figsize=(17, 8.5))
for i, (name, a) in enumerate(items):
    s = spectrum(a)
    c[0, i].imshow(a, cmap="gray")
    c[0, i].set_title(name)
    c[0, i].axis("off")
    c[1, i].imshow(log_mag(s), cmap="inferno")
    c[1, i].set_title("spectrum (majh = DC)")
    c[1, i].axis("off")
plt.tight_layout()
plt.savefig(in_here("s2f_overview.png"), dpi=120)

# sin x-er spectrum-e shobcheye ujjol dag kothay?
s = np.abs(spectrum(wave_x))
s[N // 2, N // 2] = 0                       # DC bad
py, px = np.unravel_index(s.argmax(), s.shape)
print(f"sin x-er peak: kendro theke {abs(px - N // 2)} ghar dure (ashole 8 dheu chhilo)")

# asol chobir spectrum, phase, ar ulte (inverse) chobi ferot ana
S = spectrum(real)
back = np.real(np.fft.ifft2(np.fft.ifftshift(S)))
print("inverse-er por max bhul:", float(np.abs(back - real).max()))

# kendro-r ekta chhoto ongsho-e koto % energy
yc, xc = np.mgrid[0:N, 0:N]
near = np.sqrt((xc - N // 2) ** 2 + (yc - N // 2) ** 2) <= 10
share = (np.abs(S[near]) ** 2).sum() / (np.abs(S) ** 2).sum() * 100
print(f"kendro-r 10 radius-e energy: {share:.1f}%")

fig2, d = plt.subplots(1, 3, figsize=(14, 4.5))
d[0].imshow(real, cmap="gray"); d[0].set_title("Mul"); d[0].axis("off")
d[1].imshow(np.angle(S), cmap="twilight"); d[1].set_title("Phase"); d[1].axis("off")
d[2].imshow(back, cmap="gray"); d[2].set_title("Inverse FFT-er por"); d[2].axis("off")
plt.tight_layout()
plt.savefig(in_here("s2f_phase_back.png"), dpi=120)

print("kaj shesh, file save hoyeche:", here)
plt.show()