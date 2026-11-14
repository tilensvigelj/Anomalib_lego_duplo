import math
import random
from pathlib import Path
import numpy as np
from PIL import Image

# Nastavitve map
INPUT_FOLDER = "./data/processed/mat/test/good"
OUTPUT_FOLDER = "./data/processed/mat/test/good"
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

# Nastavitve augmentacije
RANDOM_VARIANTS_PER_IMAGE = 5        # Število dodatnih naključnih različic na sliko
HUE_SHIFT_RANGE = (-128, 127)        # Celoten spekter barvnega kroga (-180° do +180° v 8-bit HSV)
SAT_SHIFT_RANGE = (-25, 25)          # Manjša naključna sprememba nasičenosti (Saturation)
VAL_SHIFT_RANGE = (-25, 25)          # Manjša naključna sprememba svetlosti (Value)
GRADIENT_STRENGTH_RANGE = (0.15, 0.4)  # Jakost osvetlitvenega gradienta (15 % do 40 % razlike med svetlo in temno stranjo)
ANGLE_RANGE = (0, 360)               # Nestandardni koti rotacije (0° - 360°)
SCALE_RANGE = (0.8, 1.2)             # Nestandardne velikosti / faktor povečave (80 % - 120 %)


def apply_random_lighting_gradient(img: Image.Image) -> Image.Image:
    """Doda linearni osvetlitveni gradient v naključni smeri (0°–360°)."""
    has_alpha = img.mode == "RGBA"
    if has_alpha:
        alpha = img.getchannel("A")

    rgb_arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w, _ = rgb_arr.shape

    # Naključna smer gradienta (kot v radianih) in jakost
    angle_rad = random.uniform(0, 2 * math.pi)
    strength = random.uniform(*GRADIENT_STRENGTH_RANGE)

    # Normalizirani koordinatni mreži v razponu [-1, 1]
    y_norm, x_norm = np.mgrid[0:h, 0:w].astype(np.float32)
    x_norm = (x_norm / max(1, w - 1)) * 2.0 - 1.0
    y_norm = (y_norm / max(1, h - 1)) * 2.0 - 1.0

    # Projekcija koordinat na naključni smerni vektor (vrednosti približno v [-1, 1])
    projection = x_norm * math.cos(angle_rad) + y_norm * math.sin(angle_rad)
    max_abs = np.max(np.abs(projection)) or 1.0
    projection = projection / max_abs

    # Multiplikativni faktor osvetlitve (npr. od 0.70 na senčni strani do 1.30 na osvetljeni)
    illumination_map = 1.0 + projection * strength
    lit_arr = np.clip(rgb_arr * illumination_map[..., np.newaxis], 0, 255).astype(np.uint8)

    result = Image.fromarray(lit_arr, mode="RGB")
    if has_alpha:
        result.putalpha(alpha)
    return result


def adjust_hsv(img: Image.Image, hue_delta: int, sat_delta: int, val_delta: int) -> Image.Image:
    """Premakne Hue (ciklično) ter prilagodi Saturation in Value (omejeno na 0–255)."""
    has_alpha = img.mode == "RGBA"
    if has_alpha:
        alpha = img.getchannel("A")

    hsv = img.convert("RGB").convert("HSV")
    h, s, v = hsv.split()

    h = h.point(lambda p: (p + hue_delta) % 256)
    s = s.point(lambda p: max(0, min(255, p + sat_delta)))
    v = v.point(lambda p: max(0, min(255, p + val_delta)))

    result = Image.merge("HSV", (h, s, v)).convert("RGB")
    if has_alpha:
        result.putalpha(alpha)
    return result


def random_nonstandard_transform(img: Image.Image) -> Image.Image:
    """Uporabi zrcaljenje, nestandardno rotacijo, skaliranje, HSV premik in osvetlitveni gradient."""
    out = img.copy()

    # 1. Naključno zrcaljenje
    if random.choice([True, False]):
        out = out.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if random.choice([True, False]):
        out = out.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    # 2. Nestandardna rotacija (poljuben kot med 0° in 360°)
    angle = random.uniform(*ANGLE_RANGE)
    out = out.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)

    # 3. Nestandardna velikost (naključno skaliranje)
    scale = random.uniform(*SCALE_RANGE)
    new_w = max(1, int(out.width * scale))
    new_h = max(1, int(out.height * scale))
    out = out.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)

    # 4. Naključni Hue, Saturation in Value
    hue_delta = random.randint(*HUE_SHIFT_RANGE)
    sat_delta = random.randint(*SAT_SHIFT_RANGE)
    val_delta = random.randint(*VAL_SHIFT_RANGE)
    out = adjust_hsv(out, hue_delta, sat_delta, val_delta)

    # 5. Osvetlitveni gradient v naključno smer
    out = apply_random_lighting_gradient(out)

    return out


def augment_images(input_dir: str, output_dir: str) -> None:
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"Mapa '{input_path.resolve()}' ne obstaja.")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    image_files = [
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
    ]

    if not image_files:
        print(f"V mapi '{input_path.resolve()}' ni najdenih slik.")
        return

    total_created = 0

    for img_file in image_files:
        with Image.open(img_file) as img:
            stem = img_file.stem
            ext = img_file.suffix

            # Standardne transformacije + naključni HSV + osvetlitveni gradient
            standard_transforms = {
                "orig": img,
                "rot90": img.transpose(Image.Transpose.ROTATE_90),
                "rot180": img.transpose(Image.Transpose.ROTATE_180),
                "rot270": img.transpose(Image.Transpose.ROTATE_270),
                "flip_h": img.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
                "flip_v": img.transpose(Image.Transpose.FLIP_TOP_BOTTOM),
            }

            for suffix, t_img in standard_transforms.items():
                hue_delta = random.randint(*HUE_SHIFT_RANGE)
                sat_delta = random.randint(*SAT_SHIFT_RANGE)
                val_delta = random.randint(*VAL_SHIFT_RANGE)

                aug_img = adjust_hsv(t_img, hue_delta, sat_delta, val_delta)
                aug_img = apply_random_lighting_gradient(aug_img)
                aug_img.save(output_path / f"{stem}_{suffix}_h{hue_delta}_s{sat_delta}_v{val_delta}{ext}")
                total_created += 1

            # Dodatne različice z nestandardnimi koti, velikostmi, HSV in gradientom
            for i in range(1, RANDOM_VARIANTS_PER_IMAGE + 1):
                custom_img = random_nonstandard_transform(img)
                custom_img.save(output_path / f"{stem}_custom_{i}{ext}")
                total_created += 1

        print(f"Obdelana slika: {img_file.name}")

    print(f"\nKončano! Ustvarjenih je bilo {total_created} slik v mapi '{output_path.resolve()}'.")


if __name__ == "__main__":
    augment_images(INPUT_FOLDER, OUTPUT_FOLDER)