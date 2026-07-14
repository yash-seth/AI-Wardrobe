# v1 - segmenting the various pieces of clothing

# import cv2
# import numpy as np
# from fashn_human_parser import FashnHumanParser, IDS_TO_LABELS

# # tasks:
# # - how to single out each identified entity in the image such as t-shirt, glasses etc

# # 1. Initialize parser and load image
# parser = FashnHumanParser()
# image_path = "./Images/rachkonda_sparsh.JPG"

# original_bgr = cv2.imread(image_path)
# h, w, _ = original_bgr.shape

# # 2. Get the 2D segmentation map (shape: H, W)
# mask_2d = parser.predict(image_path)
# print(mask_2d)

# # Ensure dimensions match perfectly
# if mask_2d.shape[:2] != (h, w):
#     mask_2d = cv2.resize(mask_2d, (w, h), interpolation=cv2.INTER_NEAREST)

# # 3. Create a unique, distinct color palette for the 18 classes
# np.random.seed(42)  # Fixed seed for consistent mask colors
# colors = np.random.randint(0, 255, size=(18, 3), dtype=np.uint8)
# colors[0] = [0, 0, 0]  # Force background (ID 0) to stay black

# # 4. Map the 2D segmentation IDs to the 3D BGR color palette
# colored_mask = colors[mask_2d]

# # 5. Blend the original image and the colored mask (Alpha Blending)
# alpha = 0.6
# beta = 0.4
# overlay_result = cv2.addWeighted(original_bgr, alpha, colored_mask, beta, 0)

# # 6. Find masks and dynamically draw text labels onto the segments
# # Loop over every possible label ID except 0 (Background)
# for class_id in range(1, 18):
#     # Create a binary mask specifically for this class
#     class_mask = (mask_2d == class_id).astype(np.uint8)
    
#     # Check if the class is actually present in the image
#     if np.any(class_mask):
#         # Find spatial moments of the class mask to locate its center
#         moments = cv2.moments(class_mask)
        
#         if moments["m00"] > 0:  # Avoid division by zero
#             # Calculate coordinates for the center of the segment
#             cx = int(moments["m10"] / moments["m00"])
#             cy = int(moments["m01"] / moments["m00"])
            
#             # Retrieve string name from fashn mapping utilities
#             label_text = IDS_TO_LABELS.get(class_id, f"ID {class_id}")
            
#             # Setup text parameters
#             font = cv2.FONT_HERSHEY_SIMPLEX
#             font_scale = 0.5
#             thickness = 1
            
#             # Draw a black text shadow/outline for visibility against any color
#             cv2.putText(overlay_result, label_text, (cx + 1, cy + 1), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
#             # Draw the clean white label text in the center
#             cv2.putText(overlay_result, label_text, (cx, cy), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

# # 7. Display and save the labeled result
# cv2.imshow("Human Parser with Labels", overlay_result)
# cv2.imwrite("labelled_" + image_path.split('/')[-1], overlay_result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# v2 - identifying attributes - colour

import cv2
import numpy as np
from sklearn.cluster import KMeans
from fashn_human_parser import FashnHumanParser, IDS_TO_LABELS


# ---------- Color palette and helpers ----------

# Simple LAB palette for mapping cluster centers to human-readable names.
# You can refine these values later by sampling real colors.
NAMED_COLORS = {
    "black":   np.array([0,   0,   0]),
    "white":   np.array([255, 0,   0]),
    "gray":    np.array([128, 0,   0]),
    "red":     np.array([136, 208, 195]),
    "orange":  np.array([191, 190, 120]),
    "yellow":  np.array([220,  20, 200]),
    "green":   np.array([182,  84, 182]),
    "blue":    np.array([136, 208,  20]),
    "navy":    np.array([80,   20,  20]),
    "purple":  np.array([108, 200,  20]),
    "pink":    np.array([200, 180, 140]),
    "brown":   np.array([100, 140, 150]),
    "beige":   np.array([220, 110, 130]),
}

def lab_distance(c1: np.ndarray, c2: np.ndarray) -> float:
    return np.linalg.norm(c1 - c2)

def nearest_named_color(lab_center: np.ndarray) -> str:
    best_name = None
    best_dist = float("inf")
    for name, lab_ref in NAMED_COLORS.items():
        d = lab_distance(lab_center, lab_ref)
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_name

# def classify_color_from_lab(lab_center):
#     """
#     lab_center: np.array([L, a, b]) cluster center in LAB
#     Returns a human-readable color name.
#     """
#     # Convert single LAB color back to BGR
#     lab_img = lab_center.reshape(1, 1, 3).astype(np.uint8)
#     bgr_img = cv2.cvtColor(lab_img, cv2.COLOR_Lab2BGR)
#     hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

#     H, S, V = hsv_img[0, 0]  # hue, saturation, value (0–179, 0–255, 0–255)

#     # ----- Basic brightness/saturation rules -----
#     # Black: very low value, any saturation
#     if V < 40:
#         return "black"

#     # White: very high value, low saturation
#     if V > 220 and S < 40:
#         return "white"

#     # Gray: mid/high value, very low saturation
#     if S < 40:
#         return "gray"

#     # ----- Hue-based color ranges -----
#     # Hue ranges depend on OpenCV’s 0–179 scale [web:99][web:96]
#     if (H < 10) or (H >= 170):
#         return "red"
#     elif 10 <= H < 25:
#         return "orange"
#     elif 25 <= H < 40:
#         return "yellow"
#     elif 40 <= H < 80:
#         return "green"
#     elif 80 <= H < 130:
#         return "blue"
#     elif 130 <= H < 155:
#         return "purple"
#     elif 155 <= H < 170:
#         # Pink / magenta region
#         return "pink"

#     # Distinguish brown vs beige by value/saturation
#     # (brown is darker, beige is lighter with moderate V)
#     if V < 140:
#         return "brown"
#     else:
#         return "beige"

def classify_color_from_lab(lab_center):
    """
    lab_center: np.array([L, a, b]) cluster center in LAB
    Returns a human-readable color name.
    """
    # Convert single LAB color back to BGR
    lab_img = lab_center.reshape(1, 1, 3).astype(np.uint8)
    bgr_img = cv2.cvtColor(lab_img, cv2.COLOR_Lab2BGR)
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

    H, S, V = hsv_img[0, 0]  # Hue (0–179), Sat (0–255), Val (0–255)

    # ----- Basic brightness/saturation rules: black / white / gray -----
    if V < 40:          # very dark
        return "black"
    if V > 220 and S < 40:   # very bright, almost no color
        return "white"
    if S < 40:          # low saturation mid/high value
        return "gray"

    # ----- Brown vs orange split (same hue band, different V) -----
    # Brown in OpenCV HSV typically has H ~ 10–20, high S, mid V [web:123]
    if 10 <= H < 25 and S > 80:
        if V < 140:     # darker → brown
            return "brown"
        else:           # brighter → orange/tan
            return "orange"

    # ----- Hue-based ranges for other saturated colors -----
    if (H < 10) or (H >= 170):
        return "red"
    elif 25 <= H < 40:
        return "yellow"
    elif 40 <= H < 80:
        return "green"
    elif 80 <= H < 130:
        return "blue"
    elif 130 <= H < 155:
        return "purple"
    elif 155 <= H < 170:
        return "pink"

    # For remaining hues near orange/yellow but with mid V, treat as beige
    if V >= 140:
        return "beige"
    else:
        return "brown"

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

    # Convert to LAB for perceptual clustering [web:99][web:86]
    garment_pixels_lab = cv2.cvtColor(
        garment_pixels_bgr.reshape(-1, 1, 3),
        cv2.COLOR_BGR2LAB
    ).reshape(-1, 3)

    # Filter out extreme highlights/shadows (optional but helps robustness)
    L = garment_pixels_lab[:, 0]
    valid_idx = (L > 10) & (L < 245)
    garment_pixels_lab = garment_pixels_lab[valid_idx]
    if len(garment_pixels_lab) < 50:
        return []

    # Decide number of clusters
    n_clusters = min(max_clusters, max(1, len(garment_pixels_lab) // 500))

    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    labels = kmeans.fit_predict(garment_pixels_lab)
    centers = kmeans.cluster_centers_

    counts = np.bincount(labels)
    total = counts.sum()

    color_info = []
    for i, center in enumerate(centers):
        pct = counts[i] / total
        # name = nearest_named_color(center)
        name = classify_color_from_lab(center)
        color_info.append({
            "lab_center": center,
            "name": name,
            "percentage": pct,
        })

    # Sort by dominance
    color_info.sort(key=lambda c: c["percentage"], reverse=True)
    return color_info


# ---------- Main script: parsing + color attributes ----------

parser = FashnHumanParser()
image_path = "./Images/rachkonda_sparsh.JPG"

original_bgr = cv2.imread(image_path)
if original_bgr is None:
    raise RuntimeError(f"Could not read image at {image_path}")

h, w, _ = original_bgr.shape

# 2D segmentation map (H, W)
mask_2d = parser.predict(image_path)

# Ensure segmentation map matches image size
if mask_2d.shape[:2] != (h, w):
    mask_2d = cv2.resize(mask_2d, (w, h), interpolation=cv2.INTER_NEAREST)

# Color palette for visualization (not used for attributes)
np.random.seed(42)
colors = np.random.randint(0, 255, size=(18, 3), dtype=np.uint8)
colors[0] = [0, 0, 0]  # background

colored_mask = colors[mask_2d]

alpha = 0.6
beta = 0.4
overlay_result = cv2.addWeighted(original_bgr, alpha, colored_mask, beta, 0)

attributes_per_piece = []  # list of dicts with attributes per clothing part

# Loop over all possible class IDs (1..17)
for class_id in range(1, 18):
    class_mask = (mask_2d == class_id).astype(np.uint8)

    if not np.any(class_mask):
        continue

    # Center of this mask for label placement
    moments = cv2.moments(class_mask)
    if moments["m00"] <= 0:
        continue

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    label_text = IDS_TO_LABELS.get(class_id, f"ID {class_id}")

    # --- color extraction for this clothing piece ---
    color_info = extract_dominant_colors(original_bgr, class_mask, max_clusters=3)

    if color_info:
        dominant = color_info[0]
        primary_color_name = dominant["name"]
        primary_pct = dominant["percentage"]

        secondary_colors = [
            {"name": c["name"], "percentage": c["percentage"]}
            for c in color_info[1:]
            if c["percentage"] > 0.05  # ignore tiny stripes/noise
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
        # Fallback if color extraction failed
        color_label = label_text

    # Draw label with outline for visibility
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

cv2.imshow("Human Parser with Color Attributes", overlay_result)
cv2.imwrite("labelled_" + image_path.split("/")[-1], overlay_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
