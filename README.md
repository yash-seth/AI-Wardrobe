# Fashion Wardrobe Parser

This project uses the **FASHN Human Parser** to segment people in images into semantic regions (top, pants, dress, accessories, body parts, etc.), then derives **per-garment color attributes** and saves cropped garments (tops and pants) to a local “wardrobe”. It is intended as a foundation for personal wardrobe datasets, fashion search, or virtual try-on pre-processing.

## Features

- Per-pixel **human parsing** into 18 fashion-oriented classes using FASHN Human Parser (SegFormer-B4 backbone).
- Overlay visualization of segmentation + labels for quick inspection.
- **Color attribute extraction** per clothing piece using:
  - segmentation mask → garment pixels only
  - LAB color space + KMeans clustering
  - HSV-based rules for robust color naming (black/white/gray/brown/orange/etc.)
- Automatic cropping and saving of:
  - **tops** (top / dress / scarf-like regions)
  - **pants/bottoms** (pants, skirts, belts)
- Clean output folders for:
  - Cropped garments (`wardrobe/tops`, `wardrobe/pants`)
  - Labelled visualization images (`labelled_images`).

## Tech Stack

- **Language:** Python 3.10+
- **Core libraries:**
  - `fashn-human-parser` – human parsing model + ID→label mappings
  - `opencv-python` – image IO, resizing, overlay rendering
  - `numpy` – numerical operations
  - `scikit-learn` – KMeans clustering for dominant colors

## Folder Structure

A typical project layout:

```text
.
├── Images/                 # Raw input images (ignored in git)
├── wardrobe/
│   ├── tops/               # Cropped top regions (masked)
│   └── pants/              # Cropped pants/bottom regions (masked)
├── labelled_images/        # Overlay images with labels + color annotations
├── env/                    # Virtual environment (ignored in git)
├── main.py                 # Main script (segmentation + color + wardrobe)
└── README.md
```

Recommended `.gitignore`:

```gitignore
env/
Images/
wardrobe/
labelled_images/
```

## Installation

Create and activate a virtual environment (optional but recommended), then install dependencies:

```bash
pip install fashn-human-parser
pip install opencv-python
pip install scikit-learn
pip install numpy
```

If you are on a headless server (no GUI), use `opencv-python-headless` instead of `opencv-python`.

## Usage

1. Place input images in the `Images/` folder.
2. Set `image_path` in `main.py` to point to an image (or loop over files).
3. Run:

```bash
python main.py
```

4. Check outputs:
   - `labelled_images/labelled_<image>.png` – segmentation overlay with `(label + primary color)` text.
   - `wardrobe/tops/*.png` – cropped, masked tops.
   - `wardrobe/pants/*.png` – cropped, masked pants.

## FASHN Human Parser Details

The project uses the **FASHN Human Parser** model, a SegFormer-B4 vision transformer fine-tuned for human parsing in fashion contexts.

Key properties:

- **Architecture:** SegFormer-B4 encoder + MLP decoder.
- **Task:** 18-class human parsing focused on fashion and virtual try-on.
- **I/O:**
  - Input images in standard formats (file path, PIL, or NumPy array).
  - Output: integer mask of shape `(H, W)` with class IDs in `[0, 17]`.
- **Python utilities:**
  - `FashnHumanParser` class for model inference.
  - `IDS_TO_LABELS` and `LABELS_TO_IDS` mappings for readable labels.
  - `IDENTITY_LABELS` list for identity-preserving regions (face, hair, jewelry, etc.).

### Class IDs and Labels

FASHN Human Parser uses the following semantic classes:

| ID | Label      |
|----|-----------|
| 0  | background |
| 1  | face       |
| 2  | hair       |
| 3  | top        |
| 4  | dress      |
| 5  | skirt      |
| 6  | pants      |
| 7  | belt       |
| 8  | bag        |
| 9  | hat        |
| 10 | scarf      |
| 11 | glasses    |
| 12 | arms       |
| 13 | hands      |
| 14 | legs       |
| 15 | feet       |
| 16 | torso      |
| 17 | jewelry    |

In code, the installed package exposes this mapping as `IDS_TO_LABELS`. The project uses these labels as the single source of truth, rather than hard-coding the strings.

### Category Grouping Used Here

For wardrobe extraction, this project groups classes as follows:

- **Tops / Upper garments**
  - `top` (ID 3)
  - `dress` (ID 4) – treated as a one-piece that belongs with tops
  - `scarf` (ID 10)

- **Bottoms / Pants**
  - `pants` (ID 6)
  - `skirt` (ID 5)
  - `belt` (ID 7)

- **Accessories / Identity regions (not cropped into wardrobe)**
  - `bag`, `hat`, `glasses`, `jewelry` (IDs 8, 9, 11, 17)
  - `face`, `hair`, `arms`, `hands`, `legs`, `feet`, `torso` (IDs 1, 2, 12–16)

You can easily adjust these groupings depending on whether you want separate folders for dresses, skirts, accessories, etc.

## Implementation Details

### 1. Human Parsing

The core entry point is:

```python
parser = FashnHumanParser()
mask_2d = parser.predict(image_path)   # shape: (H_mask, W_mask)
```

We then ensure the mask matches the image resolution:

```python
original_bgr = cv2.imread(image_path)
h, w, _ = original_bgr.shape

if mask_2d.shape[:2] != (h, w):
    mask_2d = cv2.resize(mask_2d, (w, h), interpolation=cv2.INTER_NEAREST)
```

For visualization, a random color is assigned to each class and alpha-blended with the original image:

```python
np.random.seed(42)
colors = np.random.randint(0, 255, size=(18, 3), dtype=np.uint8)
colors[0] = [0, 0, 0]  # background

colored_mask = colors[mask_2d]
overlay_result = cv2.addWeighted(original_bgr, 0.6, colored_mask, 0.4, 0)
```

Label text is placed at the centroid of each class mask using spatial moments (`cv2.moments`).

### 2. Color Extraction per Garment

For each class ID, we:

1. Build a **binary mask** for that class:

   ```python
   class_mask = (mask_2d == class_id).astype(np.uint8)
   ```

2. Sample only garment pixels from the original image:

   ```python
   ys, xs = np.where(class_mask == 1)
   garment_pixels_bgr = original_bgr[ys, xs]
   ```

3. Convert to LAB for perceptual clustering:

   ```python
   garment_pixels_lab = cv2.cvtColor(
       garment_pixels_bgr.reshape(-1, 1, 3),
       cv2.COLOR_BGR2LAB
   ).reshape(-1, 3)
   ```

4. Filter out extreme highlights/shadows (based on the L channel).

5. Run **KMeans** to find up to 3 dominant clusters:

   ```python
   kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
   labels = kmeans.fit_predict(garment_pixels_lab)
   centers = kmeans.cluster_centers_
   ```

6. For each cluster center, convert LAB → BGR → HSV and classify color based on the HSV thresholds:

   - `V < 40` → `black`
   - `V > 220 and S < 40` → `white`
   - `S < 40` → `gray`
   - `10 <= H < 25 and S > 80 and V < 140` → `brown`
   - `10 <= H < 25 and S > 80 and V >= 140` → `orange`
   - Other hue bands → `red`, `yellow`, `green`, `blue`, `purple`, `pink`
   - Remaining near-orange/yellow hues split into `beige`/`brown` by value.

7. The most populous cluster becomes the **primary color** and smaller clusters above a small area threshold become **secondary colors**.

The result is a structure like:

```python
{
    "class_id": 3,
    "label": "top",
    "primary_color": "navy",
    "primary_color_pct": 0.81,
    "secondary_colors": [
        {"name": "white", "percentage": 0.19}
    ],
}
```

### 3. Wardrobe Extraction (Tops and Pants)

Using the official IDs, we group classes into tops and pants and then:

1. Build a combined mask per group:

   ```python
   combined_mask = np.zeros_like(mask_2d, dtype=np.uint8)
   for cid in top_class_ids:
       combined_mask[mask_2d == cid] = 1
   ```

2. Run connected components to split different garments/people:

   ```python
   num_labels, labels_cc, stats, centroids = cv2.connectedComponentsWithStats(
       combined_mask, connectivity=8
   )
   ```

3. For each component above a pixel-area threshold, crop the bounding box from the original image and apply the local component mask:

   ```python
   crop_bgr = original_bgr[y:y + h_box, x:x + w_box]
   component_mask = (labels_cc[y:y + h_box, x:x + w_box] == label_id).astype(np.uint8)
   crop_bgr_masked = cv2.bitwise_and(crop_bgr, crop_bgr, mask=component_mask)
   ```

4. Save the masked crop into either `wardrobe/tops/` or `wardrobe/pants/` with a unique filename.

## Extending the Project

Ideas for future work:

- **Fine-grained top classification** (shirt vs T-shirt vs hoodie vs crop top) on masked top crops using a ViT/ConvNeXt or CLIP-style encoder.
- Additional attributes: sleeve length, neckline, fit, pattern, garment length.
- Batch processing over entire photo libraries to build a personal wardrobe index.
- REST API (FastAPI) or UI (Gradio/Streamlit) for interactive upload and exploration.

## License

This repo is a project wrapper around the FASHN Human Parser package and model. See the upstream FASHN Human Parser repository and model card for licensing and usage terms of the underlying model and weights.