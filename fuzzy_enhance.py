"""
Fuzzy contrast enhancement — after Gonzalez & Woods, "Digital Image Processing",
section "Using Fuzzy Sets for Intensity Transformations" (Einstein example).

Rules:
    R1: IF a pixel is dark   THEN make it darker
    R2: IF a pixel is gray   THEN make it gray
    R3: IF a pixel is bright THEN make it brighter

Inputs are fuzzified with three membership functions (dark, gray, bright).
Outputs are singletons (v_d, v_g, v_b), so implication + aggregation +
centroid defuzzification reduce to:

    v0 = (mu_dark*v_d + mu_gray*v_g + mu_bright*v_b) / (mu_dark + mu_gray + mu_bright)

Usage:
    python fuzzy_enhance.py IMAGE [--out DIR]
    python fuzzy_enhance.py --demo          # synthetic low-contrast test image
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# ---------------------------------------------------------------- membership
def ramp_down(z, a, b):
    """1 for z<=a, falls linearly to 0 at z=b."""
    return np.clip((b - z) / (b - a), 0.0, 1.0)


def ramp_up(z, a, b):
    """0 for z<=a, rises linearly to 1 at z=b."""
    return np.clip((z - a) / (b - a), 0.0, 1.0)


def triangle(z, a, m, b):
    return np.minimum(ramp_up(z, a, m), ramp_down(z, m, b))


def memberships(z, p):
    """Input fuzzy sets. p = dict(dark_end, gray_peak, bright_start)."""
    d, g, b = p["dark_end"], p["gray_peak"], p["bright_start"]
    mu_dark = ramp_down(z, d, g)
    mu_gray = triangle(z, d, g, b)
    mu_bright = ramp_up(z, g, b)
    return mu_dark, mu_gray, mu_bright


def params_from_histogram(img):
    """Place the fuzzy sets on the image's own intensity range, as the book
    does (the sets are chosen by looking at the narrow histogram):
    'dark' is fully true at the darkest pixel, 'bright' at the brightest,
    and 'gray' peaks at the mean intensity. Centering 'gray' on the mean
    (not the middle of the range) keeps detail in the dominant mode instead
    of crushing it to black."""
    return dict(dark_end=float(img.min()), gray_peak=float(img.mean()),
                bright_start=float(img.max()))


# ---------------------------------------------------------------- inference
def fuzzy_enhance(img, p, v_d=0.0, v_g=127.0, v_b=255.0):
    """Apply the 3-rule system via a 256-entry lookup table."""
    z = np.arange(256, dtype=float)
    mu_d, mu_g, mu_b = memberships(z, p)
    lut = (mu_d * v_d + mu_g * v_g + mu_b * v_b) / (mu_d + mu_g + mu_b + 1e-12)
    lut = np.clip(np.round(lut), 0, 255).astype(np.uint8)
    return lut[img], lut


def hist_equalize(img):
    hist = np.bincount(img.ravel(), minlength=256)
    cdf = hist.cumsum() / img.size
    lut = np.round(255 * cdf).astype(np.uint8)
    return lut[img]


# ---------------------------------------------------------------- I/O + plots
def load_gray(path):
    return np.array(Image.open(path).convert("L"))


def demo_image():
    """Low-contrast version of a standard test photo (range squeezed to ~[90,170])."""
    from skimage import data
    cam = data.camera().astype(float)
    return np.round(90 + cam / 255 * 80).astype(np.uint8)


def save_figures(img, fuzzy, he, lut, p, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(fuzzy).save(out_dir / "fuzzy.png")
    Image.fromarray(he).save(out_dir / "histeq.png")

    z = np.arange(256)
    mu_d, mu_g, mu_b = memberships(z.astype(float), p)

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    ax[0].plot(z, mu_d, label="dark"); ax[0].plot(z, mu_g, label="gray"); ax[0].plot(z, mu_b, label="bright")
    ax[0].set_title("Input membership functions"); ax[0].set_xlabel("z"); ax[0].legend()
    for v, name in [(0, "darker"), (127, "gray"), (255, "brighter")]:
        ax[1].stem([v], [1], basefmt=" ", label=name)
    ax[1].set_xlim(-5, 260); ax[1].set_title("Output singletons"); ax[1].set_xlabel("v"); ax[1].legend()
    ax[2].plot(z, lut, label="fuzzy"); ax[2].plot(z, z, "k--", lw=0.8, label="identity")
    ax[2].set_title("Resulting transformation"); ax[2].set_xlabel("input z"); ax[2].set_ylabel("output v0"); ax[2].legend()
    fig.tight_layout(); fig.savefig(out_dir / "membership.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(2, 3, figsize=(14, 8))
    for col, (im, title) in enumerate([(img, "Original"), (he, "Histogram equalization"), (fuzzy, "Fuzzy rule-based")]):
        ax[0, col].imshow(im, cmap="gray", vmin=0, vmax=255); ax[0, col].set_title(title); ax[0, col].axis("off")
        ax[1, col].bar(np.arange(256), np.bincount(im.ravel(), minlength=256), width=1.0, color="0.3")
        ax[1, col].set_xlim(0, 255)
    fig.tight_layout(); fig.savefig(out_dir / "comparison.png", dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--out", default="results")
    ap.add_argument("--dark-end", type=float); ap.add_argument("--gray-peak", type=float)
    ap.add_argument("--bright-start", type=float)
    args = ap.parse_args()

    if args.demo or not args.image:
        img = demo_image()
        Path(args.out).mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(Path(args.out) / "input_demo.png")
    else:
        img = load_gray(args.image)

    p = params_from_histogram(img)
    for k in ("dark_end", "gray_peak", "bright_start"):
        if getattr(args, k) is not None:
            p[k] = getattr(args, k)
    print(f"input range [{img.min()}, {img.max()}], fuzzy sets: {p}")

    fuzzy, lut = fuzzy_enhance(img, p)
    he = hist_equalize(img)
    save_figures(img, fuzzy, he, lut, p, Path(args.out))
    print(f"wrote results to {args.out}/  (fuzzy range [{fuzzy.min()}, {fuzzy.max()}])")


if __name__ == "__main__":
    main()
