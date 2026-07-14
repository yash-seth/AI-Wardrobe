import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from fashn_human_parser import FashnHumanParser, IDS_TO_LABELS
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

if not hasattr(np, "asscalar"):
    def _asscalar(a):
        # a is a 0-dim numpy array; .item() returns the Python scalar
        return a.item()
    np.asscalar = _asscalar  # type: ignore


# ---------- Color palette and helpers ----------
PALETTE_HEX = {
    # neutrals
    "black":        "#000000",
    "charcoal":     "#36454F",  # dark gray-blue
    "dark gray":    "#4F4F4F",
    "gray":         "#808080",
    "light gray":   "#D3D3D3",
    "off-white":    "#F5F5F0",  # slightly warm white
    "white":        "#FFFFFF",

    # blues / greens
    "navy":         "#1A2340",
    "blue":         "#1E88E5",
    "light blue":   "#90CAF9",
    "teal":         "#00897B",
    "green":        "#43A047",
    "olive":        "#556B2F",

    # beiges / browns / khakis
    "beige":        "#F5DEB3",
    "tan":          "#D2B48C",
    "khaki":        "#C3B091",  # classic khaki trouser color
    "camel":        "#C19A6B",
    "brown":        "#8B4513",

    # yellows / oranges / reds
    "yellow":       "#FFEB3B",
    "mustard":      "#D4AF37",
    "orange":       "#FB8C00",
    "rust":         "#B7410E",
    "red":          "#E53935",
    "burgundy":     "#800020",

    # pinks / purples
    "dusty pink":   "#D8A7B1",  # muted pink often in fashion datasets
    "pink":         "#EC407A",
    "mauve":        "#AF8DA5",
    "purple":       "#8E24AA",
}

def hex_to_lab_color(hex_str: str) -> LabColor:
    """Convert hex string (#RRGGBB) to colormath LabColor."""
    h = hex_str.lstrip("#")
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    rgb = sRGBColor(r, g, b, is_upscaled=True)
    return convert_color(rgb, LabColor)

PALETTE_LAB = {name: hex_to_lab_color(hx) for name, hx in PALETTE_HEX.items()}

def opencv_lab_to_colormath_lab(lab_center: np.ndarray) -> LabColor:
    """
    OpenCV LAB → colormath LabColor.

    OpenCV: L in [0,255], a,b in [0,255] with 128 offset.
    Colormath: L* in [0,100], a*,b* ~ [-128,127].
    """
    L_cv, a_cv, b_cv = lab_center
    L_star = (L_cv / 255.0) * 100.0
    a_star = a_cv - 128.0
    b_star = b_cv - 128.0
    return LabColor(lab_l=L_star, lab_a=a_star, lab_b=b_star)


def classify_color_from_lab(lab_center: np.ndarray) -> str:
    """
    Robust color naming with explicit handling of neutrals (black/charcoal/gray/off-white),
    then ΔE2000 to a fashion palette for chromatic colors.
    """
    L_cv, a_cv, b_cv = lab_center

    # Compute approximate chroma in OpenCV Lab space
    a_off = a_cv - 128.0
    b_off = b_cv - 128.0
    chroma_cv = np.sqrt(a_off * a_off + b_off * b_off)

    # Convert to CIE Lab scale for better-lightness reasoning
    L_star = (L_cv / 255.0) * 100.0

    # ----- Neutral handling: black / charcoal / grays / off-white -----
    # Very low chroma: treat as neutral (no strong hue)
    if chroma_cv < 10:
        # Very dark neutral: black
        if L_star < 18:
            return "black"
        # Dark neutral: charcoal
        if 18 <= L_star < 35:
            return "charcoal"
        # Mid-dark neutral: dark gray
        if 35 <= L_star < 55:
            return "dark gray"
        # Mid-light neutral: gray
        if 55 <= L_star < 72:
            return "gray"
        # Light neutral: light gray / off-white
        if 72 <= L_star < 88:
            return "light gray"
        # Very light neutral: off-white
        if L_star >= 88:
            return "off-white"

    # Slightly more chroma but still near-neutral: merge into gray scale
    if chroma_cv < 16:
        if L_star < 20:
            return "black"
        if 20 <= L_star < 40:
            return "charcoal"
        if 40 <= L_star < 65:
            return "gray"
        if 65 <= L_star < 85:
            return "light gray"
        return "off-white"

    # ----- Chromatic colors: use ΔE2000 to fashion palette -----
    color_lab = opencv_lab_to_colormath_lab(lab_center)

    best_name = None
    best_de = float("inf")
    for name, pal_lab in PALETTE_LAB.items():
        de = delta_e_cie2000(color_lab, pal_lab)
        if de < best_de:
            best_de = de
            best_name = name

    return best_name


def extract_dominant_colors(original_bgr: np.ndarray,
                            class_mask: np.ndarray,
                            max_clusters: int = 3):
    """
    Given the original BGR image and a binary mask for one class,
    return a list of dominant colors: [{name, percentage, lab_center}, ...]
    """
    ys, xs = np.where(class_mask == 1)
    if len(ys) < 50:
        # Too few pixels, result would be noisy
        return []

    garment_pixels_bgr = original_bgr[ys, xs]

    # Convert to LAB for perceptual clustering
    garment_pixels_lab = cv2.cvtColor(
        garment_pixels_bgr.reshape(-1, 1, 3),
        cv2.COLOR_BGR2LAB
    ).reshape(-1, 3)

    # Filter out extreme highlights/shadows
    L = garment_pixels_lab[:, 0]
    valid_idx = (L > 10) & (L < 245)
    garment_pixels_lab = garment_pixels_lab[valid_idx]
    if len(garment_pixels_lab) < 50:
        return []

    # Decide number of clusters
    # n_clusters = min(max_clusters, max(1, len(garment_pixels_lab) // 500))
    n_clusters = min(max_clusters, max(1, len(garment_pixels_lab) // 1000))

    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    labels = kmeans.fit_predict(garment_pixels_lab)
    centers = kmeans.cluster_centers_

    counts = np.bincount(labels)
    total = counts.sum()

    color_info = []
    for i, center in enumerate(centers):
        pct = counts[i] / total
        name = classify_color_from_lab(center)
        color_info.append({
            "lab_center": center,
            "name": name,
            "percentage": pct,
        })

    # Sort by dominance
    color_info.sort(key=lambda c: c["percentage"], reverse=True)
    return color_info


# ---------- Main script: parsing + color attributes + wardrobe extraction ----------

parser = FashnHumanParser()
image_path = "./Images/women_pants_2.jpg"

original_bgr = cv2.imread(image_path)
if original_bgr is None:
    raise RuntimeError(f"Could not read image at {image_path}")

h, w, _ = original_bgr.shape

# 2D segmentation map (H, W)
mask_2d = parser.predict(image_path)

# Ensure segmentation map matches image size
if mask_2d.shape[:2] != (h, w):
    mask_2d = cv2.resize(mask_2d, (w, h), interpolation=cv2.INTER_NEAREST)

# Visualization palette
np.random.seed(42)
colors = np.random.randint(0, 255, size=(18, 3), dtype=np.uint8)
colors[0] = [0, 0, 0]  # background

colored_mask = colors[mask_2d]

alpha = 0.6
beta = 0.4
overlay_result = cv2.addWeighted(original_bgr, alpha, colored_mask, beta, 0)

attributes_per_piece = []

CLOTHING_LABELS = {
    "top", "dress", "skirt", "pants", "belt",
    "scarf", "bag", "hat", "feet"  # feet covers shoes
}

# Loop over all possible class IDs (1..17)
for class_id in range(1, 18):
    class_mask = (mask_2d == class_id).astype(np.uint8)
    if not np.any(class_mask):
        continue

    moments = cv2.moments(class_mask)
    if moments["m00"] <= 0:
        continue

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    label_text = IDS_TO_LABELS.get(class_id, f"ID {class_id}")

    # Skip color attributes for non-clothing regions (skin, torso, etc.)
    if label_text not in CLOTHING_LABELS:
        color_label = label_text
        # draw text as before...
        continue

    # --- color extraction only for clothing pieces ---
    color_info = extract_dominant_colors(original_bgr, class_mask, max_clusters=3)

    if color_info:
        dominant = color_info[0]
        primary_color_name = dominant["name"]
        primary_pct = dominant["percentage"]

        secondary_colors = [
            {"name": c["name"], "percentage": c["percentage"]}
            for c in color_info[1:]
            if c["percentage"] > 0.05
        ]

        attributes_per_piece.append({
            "class_id": class_id,
            "label": label_text,
            "primary_color": primary_color_name,
            "primary_color_pct": primary_pct,
            "secondary_colors": secondary_colors,
        })

        color_label = f"{label_text} ({primary_color_name})"
    else:
        color_label = label_text

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    cv2.putText(
        overlay_result, color_label, (cx + 1, cy + 1),
        font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA
    )
    cv2.putText(
        overlay_result, color_label, (cx, cy),
        font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
    )

# Debug print: attributes for all pieces
for piece in attributes_per_piece:
    print(piece)


# ---------- Wardrobe extraction: isolate tops and pants ----------

def save_items(mask_2d, class_ids, original_bgr, out_dir, base_name,
               min_area=500, prefix="item"):
    os.makedirs(out_dir, exist_ok=True)

    if not class_ids:
        print(f"No matching class IDs found for {prefix} in IDS_TO_LABELS.")
        return 0

    combined_mask = np.zeros_like(mask_2d, dtype=np.uint8)
    for cid in class_ids:
        combined_mask[mask_2d == cid] = 1

    if not np.any(combined_mask):
        print(f"No pixels found for {prefix} in this image.")
        return 0

    num_labels, labels_cc, stats, centroids = cv2.connectedComponentsWithStats(
        combined_mask, connectivity=8
    )

    saved_count = 0
    for label_id in range(1, num_labels):  # 0 is background
        x, y, w_box, h_box, area = stats[label_id]
        if area < min_area:
            continue

        crop_bgr = original_bgr[y:y + h_box, x:x + w_box]
        component_mask = (labels_cc[y:y + h_box, x:x + w_box] == label_id).astype(np.uint8)

        crop_bgr_masked = cv2.bitwise_and(crop_bgr, crop_bgr, mask=component_mask)

        saved_count += 1
        out_path = os.path.join(out_dir, f"{base_name}_{prefix}_{saved_count}.png")
        cv2.imwrite(out_path, crop_bgr_masked)

    print(f"Saved {saved_count} {prefix}(s) to {out_dir}")
    return saved_count


base_name = os.path.splitext(os.path.basename(image_path))[0]

# Top-related labels: top, dress, scarf
top_class_ids = []
for cid, label in IDS_TO_LABELS.items():
    if label in {"top", "dress", "scarf"}:
        top_class_ids.append(cid)

# Pants-related labels: pants, skirt, belt
pants_class_ids = []
for cid, label in IDS_TO_LABELS.items():
    if label in {"pants", "skirt", "belt"}:
        pants_class_ids.append(cid)

wardrobe_root = "./wardrobe"
tops_dir = os.path.join(wardrobe_root, "tops")
pants_dir = os.path.join(wardrobe_root, "pants")

save_items(mask_2d, top_class_ids, original_bgr, tops_dir, base_name,
           min_area=500, prefix="top")
save_items(mask_2d, pants_class_ids, original_bgr, pants_dir, base_name,
           min_area=500, prefix="pants")

# ---------- Display overlay ----------

labelled_dir = "./labelled_images"
os.makedirs(labelled_dir, exist_ok=True)  # create if missing [web:143]

base_name = os.path.basename(image_path)          # e.g. rachkonda_sparsh.JPG
out_path = os.path.join(labelled_dir, "labelled_" + base_name)

cv2.imshow("Human Parser with Color Attributes", overlay_result)
cv2.imwrite(out_path, overlay_result)            # writes into labelled_images/ [web:132]
cv2.waitKey(0)
cv2.destroyAllWindows()

# add percentage checks - top identified percentage of torso - if less than threshold, dont save since it will be a small cut out of the whole top