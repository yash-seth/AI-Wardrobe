import os
import json
import cv2
import argparse
import numpy as np
from sklearn.cluster import KMeans

# ---- Patch np.asscalar for colormath + modern NumPy ----
if not hasattr(np, "asscalar"):
    def _asscalar(a):
        # a is a 0-dim numpy array; .item() returns the Python scalar
        return a.item()
    np.asscalar = _asscalar  # type: ignore


from fashn_human_parser import FashnHumanParser, IDS_TO_LABELS
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000


# set the image path here
image_path = "./Images/cargo-shorts-1-man.jpg"

# ---- CLI flags: what to save from the image ----
parser = argparse.ArgumentParser(description="AI Wardrobe extractor")

# Tops flags: --save-tops / --no-save-tops
parser.add_argument(
    "--save-tops",
    dest="save_tops",
    action="store_true",
    help="Save top garments into wardrobe/tops",
)
parser.add_argument(
    "--no-save-tops",
    dest="save_tops",
    action="store_false",
    help="Do NOT save top garments",
)
parser.set_defaults(save_tops=True)  # default: save tops

# Pants flags: --save-pants / --no-save-pants
parser.add_argument(
    "--save-pants",
    dest="save_pants",
    action="store_true",
    help="Save bottom garments into wardrobe/pants",
)
parser.add_argument(
    "--no-save-pants",
    dest="save_pants",
    action="store_false",
    help="Do NOT save bottom garments",
)
parser.set_defaults(save_pants=True)  # default: save pants

args = parser.parse_args()


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
# image_path = "./Images/women_pants_2.jpg"

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

def append_metadata(metadata_path: str, record: dict) -> None:
    """
    Upsert one garment metadata record into metadata.json.
    If a record with the same filename already exists, override it.
    """
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    # Use filename as the unique key; you can also use (source_image, kind) if you prefer
    new_filename = record.get("filename")

    # Keep all records whose filename differs, then append the new one
    data = [r for r in data if r.get("filename") != new_filename]
    data.append(record)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_items(mask_2d, class_ids, original_bgr, out_dir, base_name,
               piece_attributes,
               min_area=500, min_area_fraction=0.05,
               prefix="item"):
    
    # Tune these if needed
    MIN_TOP_AREA_FRACTION_IMAGE = 0.05   # 5% of full image area
    MIN_TOP_AREA_FRACTION_MASK  = 0.30   # 30% of all top pixels in this image
    os.makedirs(out_dir, exist_ok=True)

    if not class_ids:
        print(f"No matching class IDs found for {prefix} in IDS_TO_LABELS.")
        return 0

    H, W = mask_2d.shape[:2]

    combined_mask = np.zeros_like(mask_2d, dtype=np.uint8)
    for cid in class_ids:
        combined_mask[mask_2d == cid] = 1

    if not np.any(combined_mask):
        print(f"No pixels found for {prefix} in this image.")
        return 0

    num_labels, labels_cc, stats, centroids = cv2.connectedComponentsWithStats(
        combined_mask, connectivity=8
    )

    total_pixels = combined_mask.sum()

    valid_components = []
    for label_id in range(1, num_labels):  # 0 is background
        x, y, w_box, h_box, area = stats[label_id]
        if area < min_area:
            continue

        # Fraction of combined garment mask (all top pixels or all pants pixels)
        area_fraction_mask = area / float(total_pixels)

        # Fraction of the whole image area
        area_fraction_image = area / float(H * W)

        touches_top = (y == 0)
        touches_bottom = (y + h_box >= H)
        touches_left = (x == 0)
        touches_right = (x + w_box >= W)

        if prefix == "top":
            # 1) Discard top fragments that are too small relative to full image
            if area_fraction_image < MIN_TOP_AREA_FRACTION_IMAGE:
                continue

            # 2) Discard tiny fragments of the top mask itself
            if area_fraction_mask < MIN_TOP_AREA_FRACTION_MASK:
                continue

            # 3) (Optional, keep if you like) skip tops clearly chopped by the top edge
            if touches_top and not touches_bottom:
                continue
        else:
            # For pants or other garments, keep your existing mask-based threshold
            if area_fraction_mask < min_area_fraction:
                continue

    valid_components.append((label_id, x, y, w_box, h_box, area))

    if not valid_components:
        print(f"No sufficiently large {prefix} components to save.")
        return 0

    # Determine color metadata for this garment type from attributes_per_piece
    matching_attrs = [p for p in piece_attributes if p["class_id"] in class_ids]
    if matching_attrs:
        primary_color = matching_attrs[0]["primary_color"]
        secondary_colors = matching_attrs[0]["secondary_colors"]
    else:
        primary_color = None
        secondary_colors = []

    # Metadata file at wardrobe root (parent of out_dir)
    wardrobe_root = os.path.dirname(out_dir)  # "./wardrobe"
    metadata_path = os.path.join(wardrobe_root, "metadata.json")

    saved_count = 0

    # Merge multiple top components into one, as per previous logic
    if prefix in {"top", "pants"} and len(valid_components) > 1:
        xs = [x for (_, x, _, w_box, _, _) in valid_components]
        ys = [y for (_, _, y, _, h_box, _) in valid_components]
        x_maxs = [x + w_box for (_, x, _, w_box, _, _) in valid_components]
        y_maxs = [y + h_box for (_, _, y, _, h_box, _) in valid_components]

        x_min = min(xs)
        y_min = min(ys)
        x_max = max(x_maxs)
        y_max = max(y_maxs)

        crop_bgr = original_bgr[y_min:y_max, x_min:x_max]

        union_mask = np.zeros((y_max - y_min, x_max - x_min), dtype=np.uint8)
        for (label_id, _, _, _, _, _) in valid_components:
            component_mask = (labels_cc[y_min:y_max, x_min:x_max] == label_id).astype(np.uint8)
            union_mask = np.maximum(union_mask, component_mask)

        crop_bgr_masked = cv2.bitwise_and(crop_bgr, crop_bgr, mask=union_mask)

        saved_count = 1
        out_filename = f"{base_name}_{prefix}_{saved_count}.png"
        out_path = os.path.join(out_dir, out_filename)
        cv2.imwrite(out_path, crop_bgr_masked)

        # ---- write metadata record ----
        record = {
            "filename": out_filename,
            "path": out_path,
            "kind": prefix,
            "source_image": image_path,
            "class_ids": list(class_ids),
            "labels": [IDS_TO_LABELS[cid] for cid in class_ids],
            "primary_color": primary_color,
            "secondary_colors": secondary_colors,
        }
        append_metadata(metadata_path, record)

        print(f"Saved merged {prefix} to {out_dir} (union of {len(valid_components)} parts)")
        return saved_count

    # Default behaviour: one record per component (pants, or single top)
    for (label_id, x, y, w_box, h_box, area) in valid_components:
        crop_bgr = original_bgr[y:y + h_box, x:x + w_box]
        component_mask = (labels_cc[y:y + h_box, x:x + w_box] == label_id).astype(np.uint8)
        crop_bgr_masked = cv2.bitwise_and(crop_bgr, crop_bgr, mask=component_mask)

        saved_count += 1
        out_filename = f"{base_name}_{prefix}_{saved_count}.png"
        out_path = os.path.join(out_dir, out_filename)
        cv2.imwrite(out_path, crop_bgr_masked)

        record = {
            "filename": out_filename,
            "path": out_path,
            "kind": prefix,
            "source_image": image_path,
            "class_ids": list(class_ids),
            "labels": [IDS_TO_LABELS[cid] for cid in class_ids],
            "primary_color": primary_color,
            "secondary_colors": secondary_colors,
        }
        append_metadata(metadata_path, record)

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

if args.save_tops:
    save_items(mask_2d, top_class_ids, original_bgr, tops_dir, base_name,
               attributes_per_piece,
               min_area=500, min_area_fraction=0.05, prefix="top")

if args.save_pants:
    save_items(mask_2d, pants_class_ids, original_bgr, pants_dir, base_name,
               attributes_per_piece,
               min_area=500, min_area_fraction=0.05, prefix="pants")

# ---------- Display overlay ----------

labelled_dir = "./labelled_images"
os.makedirs(labelled_dir, exist_ok=True)  # create if missing [web:143]

base_name = os.path.basename(image_path)          # e.g. rachkonda_sparsh.JPG
out_path = os.path.join(labelled_dir, "labelled_" + base_name)

cv2.imshow("Human Parser with Color Attributes", overlay_result)
cv2.imwrite(out_path, overlay_result)            # writes into labelled_images/ [web:132]
cv2.waitKey(0)
cv2.destroyAllWindows()

# CLI flags to control detection

# python main.py → saves tops and pants.

# python main.py --no-save-tops → pants only.

# python main.py --no-save-pants → tops only.