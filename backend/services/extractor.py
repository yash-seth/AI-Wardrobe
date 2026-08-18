import json
import re
import threading
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import KMeans

if not hasattr(np, "asscalar"):
    def _asscalar(a):
        return a.item()
    np.asscalar = _asscalar  # type: ignore[attr-defined]

from fashn_human_parser import FashnHumanParser, IDS_TO_LABELS
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
STATIC_DIR = PROJECT_ROOT / "static"
DATA_DIR = PROJECT_ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
LABELLED_DIR = DATA_DIR / "labelled_images"
WARDROBE_DIR = DATA_DIR / "wardrobe"
TOPS_DIR = WARDROBE_DIR / "tops"
PANTS_DIR = WARDROBE_DIR / "pants"
METADATA_PATH = WARDROBE_DIR / "metadata.json"

PARSER_LOCK = threading.Lock()
PARSER_INSTANCE = None

PALETTE_HEX = {
    "black": "#000000",
    "charcoal": "#36454F",
    "dark gray": "#4F4F4F",
    "gray": "#808080",
    "light gray": "#D3D3D3",
    "off-white": "#F5F5F0",
    "white": "#FFFFFF",
    "navy": "#1A2340",
    "blue": "#1E88E5",
    "light blue": "#90CAF9",
    "teal": "#00897B",
    "green": "#43A047",
    "olive": "#556B2F",
    "beige": "#F5DEB3",
    "tan": "#D2B48C",
    "khaki": "#C3B091",
    "camel": "#C19A6B",
    "brown": "#8B4513",
    "yellow": "#FFEB3B",
    "mustard": "#D4AF37",
    "orange": "#FB8C00",
    "rust": "#B7410E",
    "red": "#E53935",
    "burgundy": "#800020",
    "dusty pink": "#D8A7B1",
    "pink": "#EC407A",
    "mauve": "#AF8DA5",
    "purple": "#8E24AA",
}
PALETTE_LAB = {}
CLOTHING_LABELS = {"top", "dress", "skirt", "pants", "belt", "scarf", "bag", "hat", "feet"}
TOP_LABELS = {"top", "dress", "scarf"}
PANTS_LABELS = {"pants", "skirt", "belt"}


def ensure_directories() -> None:
    for path in [DATA_DIR, UPLOADS_DIR, LABELLED_DIR, WARDROBE_DIR, TOPS_DIR, PANTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    if not METADATA_PATH.exists():
        METADATA_PATH.write_text("[]", encoding="utf-8")


def slugify_filename(name: str) -> str:
    stem = Path(name).stem.lower()
    stem = re.sub(r"[^a-z0-9]+", "-", stem).strip("-")
    return stem or "upload"


def get_parser() -> FashnHumanParser:
    global PARSER_INSTANCE
    with PARSER_LOCK:
        if PARSER_INSTANCE is None:
            PARSER_INSTANCE = FashnHumanParser()
    return PARSER_INSTANCE


def hex_to_lab_color(hex_str: str) -> LabColor:
    h = hex_str.lstrip("#")
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    rgb = sRGBColor(r, g, b, is_upscaled=True)
    return convert_color(rgb, LabColor)


def build_palette_lab() -> None:
    global PALETTE_LAB
    if not PALETTE_LAB:
        PALETTE_LAB = {name: hex_to_lab_color(value) for name, value in PALETTE_HEX.items()}


def opencv_lab_to_colormath_lab(lab_center: np.ndarray) -> LabColor:
    l_cv, a_cv, b_cv = lab_center
    l_star = (l_cv / 255.0) * 100.0
    return LabColor(lab_l=l_star, lab_a=a_cv - 128.0, lab_b=b_cv - 128.0)


def classify_color_from_lab(lab_center: np.ndarray) -> str:
    l_cv, a_cv, b_cv = lab_center
    chroma_cv = np.sqrt((a_cv - 128.0) ** 2 + (b_cv - 128.0) ** 2)
    l_star = (l_cv / 255.0) * 100.0

    if chroma_cv < 10:
        if l_star < 18:
            return "black"
        if l_star < 35:
            return "charcoal"
        if l_star < 55:
            return "dark gray"
        if l_star < 72:
            return "gray"
        if l_star < 88:
            return "light gray"
        return "off-white"

    if chroma_cv < 16:
        if l_star < 20:
            return "black"
        if l_star < 40:
            return "charcoal"
        if l_star < 65:
            return "gray"
        if l_star < 85:
            return "light gray"
        return "off-white"

    color_lab = opencv_lab_to_colormath_lab(lab_center)
    best_name = None
    best_de = float("inf")
    for name, pal_lab in PALETTE_LAB.items():
        de = delta_e_cie2000(color_lab, pal_lab)
        if de < best_de:
            best_de = de
            best_name = name
    return best_name or "unknown"


def extract_dominant_colors(original_bgr: np.ndarray, class_mask: np.ndarray, max_clusters: int = 3) -> list:
    ys, xs = np.where(class_mask == 1)
    if len(ys) < 50:
        return []

    garment_pixels_bgr = original_bgr[ys, xs]
    garment_pixels_lab = cv2.cvtColor(garment_pixels_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    valid_idx = (garment_pixels_lab[:, 0] > 10) & (garment_pixels_lab[:, 0] < 245)
    garment_pixels_lab = garment_pixels_lab[valid_idx]
    if len(garment_pixels_lab) < 50:
        return []

    n_clusters = min(max_clusters, max(1, len(garment_pixels_lab) // 1000))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    labels = kmeans.fit_predict(garment_pixels_lab)
    centers = kmeans.cluster_centers_
    counts = np.bincount(labels)
    total = counts.sum()

    results = []
    for idx, center in enumerate(centers):
        results.append({
            "lab_center": center,
            "name": classify_color_from_lab(center),
            "percentage": float(counts[idx] / total),
        })
    results.sort(key=lambda item: item["percentage"], reverse=True)
    return results


def load_metadata() -> list:
    ensure_directories()
    try:
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_metadata(records: list) -> None:
    METADATA_PATH.write_text(json.dumps(records, indent=2), encoding="utf-8")


def upsert_metadata(record: dict) -> None:
    records = load_metadata()
    records = [item for item in records if item.get("filename") != record.get("filename")]
    records.append(record)
    save_metadata(records)


def to_public_path(path: Path) -> str:
    return f"/{path.relative_to(PROJECT_ROOT).as_posix()}"

def to_legacy_wardrobe_path(out_dir: Path, filename: str) -> str:
    kind = out_dir.name  # tops or pants
    return f"./wardrobe/{kind}/{filename}"

def to_legacy_source_path(image_path: Path) -> str:
    return f"./Images/{image_path.name}"

def to_legacy_labelled_path(image_name: str) -> str:
    return f"./labelled_images/labelled_{image_name}"


def save_items(mask_2d, class_ids, original_bgr, out_dir: Path, base_name: str, piece_attributes: list, image_path: Path,
               prefix: str, min_area: int = 500, min_area_fraction: float = 0.05) -> int:
    min_top_area_fraction_image = 0.05
    min_top_area_fraction_mask = 0.30

    if not class_ids:
        return 0

    h, w = mask_2d.shape[:2]
    combined_mask = np.zeros_like(mask_2d, dtype=np.uint8)
    for cid in class_ids:
        combined_mask[mask_2d == cid] = 1

    if not np.any(combined_mask):
        return 0

    num_labels, labels_cc, stats, _ = cv2.connectedComponentsWithStats(combined_mask, connectivity=8)
    total_pixels = combined_mask.sum()
    valid_components = []

    for label_id in range(1, num_labels):
        x, y, w_box, h_box, area = stats[label_id]
        if area < min_area:
            continue

        area_fraction_mask = area / float(total_pixels)
        area_fraction_image = area / float(h * w)
        touches_top = y == 0
        touches_bottom = y + h_box >= h

        if prefix == "top":
            if area_fraction_image < min_top_area_fraction_image:
                continue
            if area_fraction_mask < min_top_area_fraction_mask:
                continue
            if touches_top and not touches_bottom:
                continue
        elif area_fraction_mask < min_area_fraction:
            continue

        valid_components.append((label_id, x, y, w_box, h_box))

    if not valid_components:
        return 0

    matching_attrs = [item for item in piece_attributes if item["class_id"] in class_ids]
    primary_color = matching_attrs[0]["primary_color"] if matching_attrs else None
    secondary_colors = matching_attrs[0]["secondary_colors"] if matching_attrs else []

    saved_count = 0
    if prefix in {"top", "pants"} and len(valid_components) > 1:
        xs = [x for (_, x, _, w_box, _) in valid_components]
        ys = [y for (_, _, y, _, h_box) in valid_components]
        x_maxs = [x + w_box for (_, x, _, w_box, _) in valid_components]
        y_maxs = [y + h_box for (_, _, y, _, h_box) in valid_components]
        x_min, y_min, x_max, y_max = min(xs), min(ys), max(x_maxs), max(y_maxs)

        crop_bgr = original_bgr[y_min:y_max, x_min:x_max]
        union_mask = np.zeros((y_max - y_min, x_max - x_min), dtype=np.uint8)
        for label_id, *_ in valid_components:
            component_mask = (labels_cc[y_min:y_max, x_min:x_max] == label_id).astype(np.uint8)
            union_mask = np.maximum(union_mask, component_mask)

        crop_bgra = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2BGRA)
        crop_bgra[:, :, 3] = (union_mask * 255).astype(np.uint8)
        saved_count = 1
        out_filename = f"{base_name}_{prefix}_{saved_count}.png"
        out_path = out_dir / out_filename
        cv2.imwrite(str(out_path), crop_bgra)
        upsert_metadata({
            "filename": out_filename,
            # "path": to_public_path(out_path),
            "path": to_legacy_wardrobe_path(out_dir, out_filename),
            "kind": prefix,
            # "source_image": to_public_path(image_path),
            "source_image": to_legacy_source_path(image_path),
            "class_ids": list(class_ids),
            "labels": [IDS_TO_LABELS[cid] for cid in class_ids],
            "primarycolor": primary_color,
            "secondarycolors": secondary_colors,
        })
        return saved_count

    for idx, (label_id, x, y, w_box, h_box) in enumerate(valid_components, start=1):
        crop_bgr = original_bgr[y:y + h_box, x:x + w_box]
        component_mask = (labels_cc[y:y + h_box, x:x + w_box] == label_id).astype(np.uint8)
        crop_bgra = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2BGRA)
        crop_bgra[:, :, 3] = (component_mask * 255).astype(np.uint8)
        out_filename = f"{base_name}_{prefix}_{idx}.png"
        out_path = out_dir / out_filename
        cv2.imwrite(str(out_path), crop_bgra)
        upsert_metadata({
            "filename": out_filename,
            # "path": to_public_path(out_path),
            "path": to_legacy_wardrobe_path(out_dir, out_filename),
            "kind": prefix,
            # "source_image": to_public_path(image_path),
            "source_image": to_legacy_source_path(image_path),
            "class_ids": list(class_ids),
            "labels": [IDS_TO_LABELS[cid] for cid in class_ids],
            "primarycolor": primary_color,
            "secondarycolors": secondary_colors,
        })
        saved_count += 1

    return saved_count


def create_overlay(original_bgr: np.ndarray, mask_2d: np.ndarray, attributes_per_piece: list) -> np.ndarray:
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(18, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]
    overlay_result = cv2.addWeighted(original_bgr, 0.6, colors[mask_2d], 0.4, 0)
    attribute_map = {item["class_id"]: item for item in attributes_per_piece}

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
        attr = attribute_map.get(class_id)
        color_label = f"{label_text} ({attr['primary_color']})" if attr else label_text
        cv2.putText(overlay_result, color_label, (cx + 1, cy + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(overlay_result, color_label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return overlay_result


def process_image(image_path: str, save_tops: bool = True, save_pants: bool = True) -> dict:
    ensure_directories()
    build_palette_lab()

    image_file = Path(image_path)
    original_bgr = cv2.imread(str(image_file))
    if original_bgr is None:
        raise RuntimeError(f"Could not read image at {image_path}")

    parser = get_parser()
    with PARSER_LOCK:
        mask_2d = parser.predict(str(image_file))

    h, w, _ = original_bgr.shape
    if mask_2d.shape[:2] != (h, w):
        mask_2d = cv2.resize(mask_2d, (w, h), interpolation=cv2.INTER_NEAREST)

    attributes_per_piece = []
    for class_id in range(1, 18):
        class_mask = (mask_2d == class_id).astype(np.uint8)
        if not np.any(class_mask):
            continue
        label_text = IDS_TO_LABELS.get(class_id, f"ID {class_id}")
        if label_text not in CLOTHING_LABELS:
            continue

        color_info = extract_dominant_colors(original_bgr, class_mask, max_clusters=3)
        if not color_info:
            continue

        dominant = color_info[0]
        secondary = [
            {"name": item["name"], "percentage": float(item["percentage"])}
            for item in color_info[1:]
            if item["percentage"] > 0.05
        ]
        attributes_per_piece.append({
            "class_id": class_id,
            "label": label_text,
            "primary_color": dominant["name"],
            "primary_color_pct": float(dominant["percentage"]),
            "secondary_colors": secondary,
        })

    top_class_ids = [cid for cid, label in IDS_TO_LABELS.items() if label in TOP_LABELS]
    pants_class_ids = [cid for cid, label in IDS_TO_LABELS.items() if label in PANTS_LABELS]
    base_name = slugify_filename(image_file.name)

    tops_saved = save_items(mask_2d, top_class_ids, original_bgr, TOPS_DIR, base_name, attributes_per_piece, image_file, "top") if save_tops else 0
    pants_saved = save_items(mask_2d, pants_class_ids, original_bgr, PANTS_DIR, base_name, attributes_per_piece, image_file, "pants") if save_pants else 0

    labelled_path = LABELLED_DIR / f"labelled_{image_file.name}"
    cv2.imwrite(str(labelled_path), create_overlay(original_bgr, mask_2d, attributes_per_piece))

    return {
        "status": "success",
        # "source_image": to_public_path(image_file),
        # "labelled_image": to_public_path(labelled_path),
        "source_image": to_legacy_source_path(image_file),
        "labelled_image": to_legacy_labelled_path(image_file.name),
        "tops_saved": tops_saved,
        "pants_saved": pants_saved,
        "pieces_detected": len(attributes_per_piece),
        "items": load_metadata(),
    }
