# =========================================================
# 1) LIBRARIES
# =========================================================
import os
import re

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    raise ImportError("Missing tqdm. Install it with: pip install tqdm")


# =========================================================
# 2) PATHS
# =========================================================
lytro_root_dir = os.path.expanduser("~/Documentos/Lytro/Lytro/sourceimages/color")

# Outputs
output_dir = os.path.join(lytro_root_dir, "Tenengrad_Fusion_Results_Lytro")
os.makedirs(output_dir, exist_ok=True)

# Save fused images and intermediate maps.
save_visualizations = True
visualization_dir = os.path.join(output_dir, "vis")
if save_visualizations:
    os.makedirs(visualization_dir, exist_ok=True)

print("==============================================")
print("PATHS (Lytro)")
print("lytro_root_dir       :", lytro_root_dir)
print("output_dir           :", output_dir)
print("save_visualizations  :", save_visualizations)
print("==============================================\n")


# =========================================================
# 3) CONFIGURATION / PARAMETERS
# =========================================================
sobel_kernel_size = 3
window_size = 7

valid_extensions = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")


# =========================================================
# 4) READING AND PREPROCESSING FUNCTIONS
# =========================================================
def read_color_image_any_depth(path: str):
    """Read an image preserving its original bit depth and return a BGR image."""
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if image is None:
        return None

    # Convert grayscale images to BGR to keep the pipeline consistent.
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # If an alpha channel is present (BGRA), remove it.
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    return image


def bgr_to_gray01(bgr_image: np.ndarray) -> np.ndarray:
    """Convert a BGR uint8/uint16 image to grayscale float32 in [0, 1]."""
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    if gray.dtype == np.uint16:
        return gray.astype(np.float32) / 65535.0

    return gray.astype(np.float32) / 255.0


def bgr_to_rgb01(bgr_image: np.ndarray) -> np.ndarray:
    """Convert a BGR uint8/uint16 image to RGB float32 in [0, 1]."""
    if bgr_image.dtype == np.uint16:
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 65535.0
    else:
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    return np.clip(rgb, 0.0, 1.0)


def float01_to_uint8(x01: np.ndarray) -> np.ndarray:
    """Convert a float image in [0, 1] to uint8."""
    x = np.clip(x01, 0.0, 1.0)
    return (x * 255.0 + 0.5).astype(np.uint8)


def save_rgb01_png(path: str, rgb01: np.ndarray):
    """Save an RGB float image in [0, 1] as a PNG file."""
    bgr_u8 = cv2.cvtColor(float01_to_uint8(rgb01), cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr_u8)


def save_gray01_png(path: str, gray01: np.ndarray):
    """Save a grayscale float image in [0, 1] as a PNG file."""
    cv2.imwrite(path, float01_to_uint8(gray01))


def normalize01_for_visualization(x: np.ndarray) -> np.ndarray:
    """Normalize an array to [0, 1] for visualization only."""
    x = x.astype(np.float32)
    minimum = float(x.min())
    maximum = float(x.max())
    return (x - minimum) / (maximum - minimum + 1e-12)


# =========================================================
# 5) TENENGRAD FOCUS MEASURE
# =========================================================
def tenengrad_focus_measure(
    gray01: np.ndarray,
    local_sobel_kernel_size: int = 3,
    local_window_size: int = 7,
) -> np.ndarray:
    """Compute a local Tenengrad focus map using Sobel gradient energy."""
    gx = cv2.Sobel(gray01, cv2.CV_32F, 1, 0, ksize=local_sobel_kernel_size)
    gy = cv2.Sobel(gray01, cv2.CV_32F, 0, 1, ksize=local_sobel_kernel_size)

    energy = gx**2 + gy**2

    focus_map = cv2.boxFilter(
        energy,
        ddepth=-1,
        ksize=(local_window_size, local_window_size),
        normalize=True,
        borderType=cv2.BORDER_REFLECT,
    )

    return focus_map


# =========================================================
# 6) FUSION FUNCTIONS
# =========================================================
def additive_rgb_fusion(
    image_a_rgb01: np.ndarray,
    image_b_rgb01: np.ndarray,
    alpha_2d: np.ndarray,
) -> np.ndarray:
    """Pixel-wise additive fusion using the alpha focus map."""
    alpha_3d = alpha_2d[..., None]
    fused = alpha_3d * image_a_rgb01 + (1.0 - alpha_3d) * image_b_rgb01
    return np.clip(fused, 0.0, 1.0)


def exponential_rgb_fusion(
    image_a_rgb01: np.ndarray,
    image_b_rgb01: np.ndarray,
    alpha_2d: np.ndarray,
) -> np.ndarray:
    """Pixel-wise exponential fusion using the alpha focus map."""
    alpha_3d = alpha_2d[..., None]

    safe_image_a = np.clip(image_a_rgb01, 1e-6, 1.0)
    safe_image_b = np.clip(image_b_rgb01, 1e-6, 1.0)

    fused = (safe_image_a ** alpha_3d) * (safe_image_b ** (1.0 - alpha_3d))

    return np.clip(fused, 0.0, 1.0)


# =========================================================
# 7) DETECT LYTRO PAIRS
# =========================================================
pattern = re.compile(r"^(c_\d+)_([12])\.(tif|tiff|png|jpg|jpeg|bmp)$", re.IGNORECASE)

pairs = {}  # {scene_id: {'1': path, '2': path}}

for filename in os.listdir(lytro_root_dir):
    if not filename.lower().endswith(valid_extensions):
        continue

    match = pattern.match(filename)

    if not match:
        continue

    scene_id = match.group(1)
    source_index = match.group(2)  # '1' or '2'

    pairs.setdefault(scene_id, {})[source_index] = os.path.join(lytro_root_dir, filename)

valid_pairs = {scene_id: paths for scene_id, paths in pairs.items() if "1" in paths and "2" in paths}
scene_ids = sorted(valid_pairs.keys())

print(f"[INFO] Detected valid Lytro pairs: {len(scene_ids)}")

if len(scene_ids) == 0:
    raise RuntimeError(
        "No valid pairs were detected. Expected names such as c_01_1.tif and c_01_2.tif.\n"
        "Check that the dataset folder is correct and that the filenames match the expected pattern."
    )


# =========================================================
# 8) MAIN LOOP: TENENGRAD FUSION
# =========================================================
processed_count = 0
failed_count = 0

for scene_id in tqdm(scene_ids, desc="Fusing Lytro", unit="pair"):
    path_1 = valid_pairs[scene_id]["1"]
    path_2 = valid_pairs[scene_id]["2"]

    image_a_bgr = read_color_image_any_depth(path_1)
    image_b_bgr = read_color_image_any_depth(path_2)

    if image_a_bgr is None or image_b_bgr is None:
        failed_count += 1
        continue

    # Ensure both source images have the same size.
    if image_a_bgr.shape[:2] != image_b_bgr.shape[:2]:
        height, width = image_a_bgr.shape[:2]
        image_b_bgr = cv2.resize(image_b_bgr, (width, height), interpolation=cv2.INTER_AREA)

    # Convert images to [0, 1].
    image_a_rgb01 = bgr_to_rgb01(image_a_bgr)
    image_b_rgb01 = bgr_to_rgb01(image_b_bgr)

    image_a_gray01 = bgr_to_gray01(image_a_bgr)
    image_b_gray01 = bgr_to_gray01(image_b_bgr)

    # Tenengrad focus maps.
    focus_map_a = tenengrad_focus_measure(
        image_a_gray01,
        local_sobel_kernel_size=sobel_kernel_size,
        local_window_size=window_size,
    )
    focus_map_b = tenengrad_focus_measure(
        image_b_gray01,
        local_sobel_kernel_size=sobel_kernel_size,
        local_window_size=window_size,
    )

    # Pixel-wise alpha map.
    eps = 1e-6
    alpha_map = focus_map_a / (focus_map_a + focus_map_b + eps)

    # RGB fusion.
    fused_add_rgb = additive_rgb_fusion(image_a_rgb01, image_b_rgb01, alpha_map)
    fused_exp_rgb = exponential_rgb_fusion(image_a_rgb01, image_b_rgb01, alpha_map)

    # Save outputs.
    if save_visualizations:
        scene_visualization_dir = os.path.join(visualization_dir, scene_id)
        os.makedirs(scene_visualization_dir, exist_ok=True)

        save_rgb01_png(os.path.join(scene_visualization_dir, f"{scene_id}_FADD.png"), fused_add_rgb)
        save_rgb01_png(os.path.join(scene_visualization_dir, f"{scene_id}_FEXP.png"), fused_exp_rgb)

        save_gray01_png(
            os.path.join(scene_visualization_dir, f"{scene_id}_Ten_A.png"),
            normalize01_for_visualization(focus_map_a),
        )
        save_gray01_png(
            os.path.join(scene_visualization_dir, f"{scene_id}_Ten_B.png"),
            normalize01_for_visualization(focus_map_b),
        )
        save_gray01_png(
            os.path.join(scene_visualization_dir, f"{scene_id}_Alpha.png"),
            np.clip(alpha_map, 0.0, 1.0),
        )

    processed_count += 1


print("\n==================== SUMMARY ====================")
print(f"Detected pairs : {len(scene_ids)}")
print(f"Processed OK   : {processed_count}")
print(f"Failed         : {failed_count}")
print(f"Output folder  : {visualization_dir if save_visualizations else output_dir}")
print("=================================================\n")
