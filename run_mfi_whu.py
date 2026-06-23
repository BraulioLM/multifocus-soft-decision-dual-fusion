# =========================================================
# 1) LIBRARIES
# =========================================================
import os

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    raise ImportError("Missing tqdm. Install it with: pip install tqdm")


# =========================================================
# 2) PATHS
# =========================================================
# Expected MFI-WHU structure:
# MFI-WHU/
# ├── source_1/
# ├── source_2/
# └── full_clear/   # not required for fusion, only part of the original dataset
root_dir = os.path.join(os.path.dirname(__file__), "data", "mfi_whu")

source_1_dir = os.path.join(root_dir, "source_1")
source_2_dir = os.path.join(root_dir, "source_2")

output_dir = os.path.join(os.path.dirname(__file__), "outputs", "mfi_whu")
os.makedirs(output_dir, exist_ok=True)

# Output folders
fused_add_dir = os.path.join(output_dir, "fused_add")
fused_exp_dir = os.path.join(output_dir, "fused_exp")
alpha_dir = os.path.join(output_dir, "alpha_maps")
focus_map_dir = os.path.join(output_dir, "focus_maps")

os.makedirs(fused_add_dir, exist_ok=True)
os.makedirs(fused_exp_dir, exist_ok=True)
os.makedirs(alpha_dir, exist_ok=True)
os.makedirs(focus_map_dir, exist_ok=True)

print("==============================================")
print("PATHS (MFI-WHU)")
print("root_dir     :", root_dir)
print("source_1_dir :", source_1_dir)
print("source_2_dir :", source_2_dir)
print("output_dir   :", output_dir)
print("==============================================\n")


# =========================================================
# 3) CONFIGURATION / PARAMETERS
# =========================================================
sobel_kernel_size = 3
window_size = 7

valid_extensions = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


# =========================================================
# 4) READING AND PREPROCESSING FUNCTIONS
# =========================================================
def read_color_image_any_depth(path: str):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    return image


def bgr_to_gray01(image_bgr):
    # BGR uint8/uint16 -> gray float32 [0, 1], without NORM_MINMAX.
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if gray.dtype == np.uint16:
        return gray.astype(np.float32) / 65535.0
    return gray.astype(np.float32) / 255.0


def bgr_to_rgb01(image_bgr):
    # BGR uint8/uint16 -> RGB float32 [0, 1], without NORM_MINMAX.
    if image_bgr.dtype == np.uint16:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 65535.0
    else:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.clip(rgb, 0.0, 1.0)


def float01_to_uint8(x01: np.ndarray) -> np.ndarray:
    x = np.clip(x01, 0.0, 1.0)
    return (x * 255.0 + 0.5).astype(np.uint8)


def save_rgb01_png(path: str, rgb01: np.ndarray):
    bgr_u8 = cv2.cvtColor(float01_to_uint8(rgb01), cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr_u8)


def save_gray01_png(path: str, gray01: np.ndarray):
    cv2.imwrite(path, float01_to_uint8(gray01))


def normalize01_for_visualization(x: np.ndarray) -> np.ndarray:
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
    alpha_3d = alpha_2d[..., None]
    fused = alpha_3d * image_a_rgb01 + (1.0 - alpha_3d) * image_b_rgb01
    return np.clip(fused, 0.0, 1.0)


def exponential_rgb_fusion(
    image_a_rgb01: np.ndarray,
    image_b_rgb01: np.ndarray,
    alpha_2d: np.ndarray,
) -> np.ndarray:
    alpha_3d = alpha_2d[..., None]
    safe_image_a = np.clip(image_a_rgb01, 1e-6, 1.0)
    safe_image_b = np.clip(image_b_rgb01, 1e-6, 1.0)
    fused = (safe_image_a ** alpha_3d) * (safe_image_b ** (1.0 - alpha_3d))
    return np.clip(fused, 0.0, 1.0)


# =========================================================
# 7) INDEX FILES BY BASENAME
# =========================================================
def index_files_by_basename(folder: str, allowed_extensions=valid_extensions):
    index = {}

    for filename in os.listdir(folder):
        if not filename.lower().endswith(allowed_extensions):
            continue

        basename = os.path.splitext(filename)[0]

        # If duplicates exist with the same basename and different extensions, keep the first one.
        index.setdefault(basename, os.path.join(folder, filename))

    return index


source_1_index = index_files_by_basename(source_1_dir)
source_2_index = index_files_by_basename(source_2_dir)

print(f"[INFO] Indexed source_1 files : {len(source_1_index)}")
print(f"[INFO] Indexed source_2 files : {len(source_2_index)}")

scene_ids = sorted(set(source_1_index.keys()) & set(source_2_index.keys()))

print(f"[INFO] Valid source pairs (A, B): {len(scene_ids)}")

if len(scene_ids) == 0:
    raise RuntimeError(
        "No source_1 and source_2 pairs were detected.\n"
        "Check that the dataset folder is correct and that the filenames match."
    )


# =========================================================
# 8) MAIN LOOP: TENENGRAD + ALPHA MAP + FUSION
# =========================================================
num_ok = 0
num_failed = 0

for scene_id in tqdm(scene_ids, desc="Fusing MFI-WHU", unit="img"):
    path_a = source_1_index[scene_id]
    path_b = source_2_index[scene_id]

    image_a_bgr = read_color_image_any_depth(path_a)
    image_b_bgr = read_color_image_any_depth(path_b)

    if image_a_bgr is None or image_b_bgr is None:
        num_failed += 1
        print(f"[WARN] Could not read image pair: {scene_id}")
        continue

    # Ensure both source images have the same spatial size.
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

    # Pixel-wise adaptive alpha map.
    eps = 1e-6
    alpha_map = focus_map_a / (focus_map_a + focus_map_b + eps)

    # RGB fusion.
    fused_add_rgb = additive_rgb_fusion(image_a_rgb01, image_b_rgb01, alpha_map)
    fused_exp_rgb = exponential_rgb_fusion(image_a_rgb01, image_b_rgb01, alpha_map)

    # Save outputs.
    save_rgb01_png(os.path.join(fused_add_dir, f"{scene_id}_FADD.png"), fused_add_rgb)
    save_rgb01_png(os.path.join(fused_exp_dir, f"{scene_id}_FEXP.png"), fused_exp_rgb)
    save_gray01_png(os.path.join(alpha_dir, f"{scene_id}_Alpha.png"), np.clip(alpha_map, 0.0, 1.0))
    save_gray01_png(os.path.join(focus_map_dir, f"{scene_id}_Ten_A.png"), normalize01_for_visualization(focus_map_a))
    save_gray01_png(os.path.join(focus_map_dir, f"{scene_id}_Ten_B.png"), normalize01_for_visualization(focus_map_b))

    num_ok += 1


print("\n==================== SUMMARY ====================")
print(f"Total source pairs : {len(scene_ids)}")
print(f"Processed OK       : {num_ok}")
print(f"Failed             : {num_failed}")
print(f"Outputs saved to   : {output_dir}")
print("=================================================\n")
