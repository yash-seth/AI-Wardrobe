# Fashion Wardrobe Parser

An AI-powered wardrobe extraction app built around the **FASHN Human Parser**. It segments people in fashion images into semantic regions, extracts per-garment color attributes, saves transparent garment cutouts, and presents the results through a FastAPI-powered web interface.

Repository: [yash-seth/AI-Wardrobe](https://github.com/yash-seth/AI-Wardrobe)

The project is intended as a foundation for personal wardrobe datasets, fashion search, wardrobe cataloguing, and virtual try-on preprocessing.

## Features

- Per-pixel human parsing into 18 fashion-oriented classes using FASHN Human Parser.
- Labelled segmentation overlays for visual inspection.
- Dominant color extraction from garment pixels using LAB color space and K-Means clustering.
- Perceptual color naming with CIE Lab and Delta E 2000, including neutral-color handling.
- Automatic extraction of transparent PNG cutouts for tops and pants/bottoms.
- Local metadata storage in `data/wardrobe/metadata.json`.
- FastAPI endpoints for uploading images, processing garments, and loading wardrobe metadata.
- Browser UI with search, top/pants tabs, color filters, lazy-loaded garment cards, theme switching, and upload previews.
- Clean separation between API code, extraction services, static frontend files, and generated runtime data.

## Technology stack

- **Python 3.10+**
- **FastAPI** and Uvicorn for the web API.
- **FASHN Human Parser** for human parsing and label mappings.
- **OpenCV** for image I/O, resizing, masks, connected components, and overlays.
- **NumPy** for array operations.
- **scikit-learn** for K-Means clustering.
- **colormath** for CIE Lab conversion and Delta E 2000 comparisons.
- **HTML, CSS, and vanilla JavaScript** for the frontend.

## Architecture

```text
Browser UI
   │
   ├── GET /                         → static/wardrobe-hanger-ui.html
   ├── POST /api/process             → backend/app.py
   └── GET /api/wardrobe             → data/wardrobe/metadata.json
                                      │
                                      ▼
                         backend/services/extractor.py
                                      │
              FASHN parsing + color analysis + crop saving
                                      │
                                      ▼
                            generated data/ assets
```

## Directory structure

```text
AI-Wardrobe/
├── backend/
│   ├── __init__.py
│   ├── app.py                         # FastAPI app and API routes
│   └── services/
│       ├── __init__.py
│       └── extractor.py                # Parsing, colors, crops, metadata
├── static/
│   └── wardrobe-hanger-ui.html         # Frontend interface
├── data/                               # Generated runtime data
│   ├── uploads/                        # Uploaded source images
│   ├── labelled_images/                # Segmentation overlay images
│   └── wardrobe/
│       ├── tops/                       # Extracted top/dress/scarf PNGs
│       ├── pants/                      # Extracted pants/skirt/belt PNGs
│       └── metadata.json                # Saved wardrobe records
├── requirements.txt
├── .gitignore
└── README.md
```

The `data/` directories are created automatically by `ensure_directories()` when the application starts or processes an image.

## Browser paths and storage paths

The backend writes generated files inside `data/`, then exposes them through FastAPI static mounts. Metadata should contain browser-accessible paths, not Windows filesystem paths such as `D:\\...`.

| Asset | Local storage | Browser/API access |
|---|---|---|
| Uploaded images | `data/uploads/` | `/Images/<filename>` |
| Labelled overlays | `data/labelled_images/` | `/labelled_images/<filename>` |
| Top cutouts | `data/wardrobe/tops/` | `/wardrobe/tops/<filename>` |
| Pants cutouts | `data/wardrobe/pants/` | `/wardrobe/pants/<filename>` |
| Metadata | `data/wardrobe/metadata.json` | `/api/wardrobe` |

Keep these two concepts separate: Python uses local `Path` objects for reading and writing; the frontend uses URLs served by FastAPI.

## Installation

Create and activate a virtual environment from the project root:

### Windows PowerShell

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\env\Scripts\Activate.ps1
```

### macOS/Linux

```bash
python3 -m venv env
source env/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For a headless server, use `opencv-python-headless` instead of `opencv-python` if the dependency set permits it.

## Run the application

Run Uvicorn from the directory containing `backend/`:

```bash
uvicorn backend.app:app --reload
```

Open the application at:

```text
http://127.0.0.1:8000/
```

FastAPI documentation is available at:

```text
http://127.0.0.1:8000/docs
```

Because `backend/app.py` uses a relative import such as `from .services.extractor import ...`, keep `backend/__init__.py` and `backend/services/__init__.py` in place and do not run the command from inside `backend/`.

## API endpoints

### `GET /`

Serves the frontend from `static/wardrobe-hanger-ui.html`.

### `GET /api/wardrobe`

Returns the items stored in `data/wardrobe/metadata.json`:

```json
{
  "items": [
    {
      "filename": "example_top_1.png",
      "path": "/wardrobe/tops/example_top_1.png",
      "kind": "top",
      "source_image": "/Images/example.jpg",
      "primarycolor": "blue",
      "secondarycolors": []
    }
  ]
}
```

### `POST /api/process`

Accepts multipart form data:

- `file`: JPG, JPEG, PNG, or WEBP image.
- `save_tops`: boolean flag.
- `save_pants`: boolean flag.

Example:

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/process `
  -F "file=@Images/example.jpg" `
  -F "save_tops=true" `
  -F "save_pants=true"
```

The response includes the source image URL, labelled preview URL, number of saved tops and pants, detected pieces, and the updated metadata list.

## Processing flow

1. The UI creates a multipart request and sends the image to `/api/process`.
2. `backend/app.py` assigns a UUID-based filename and saves the upload in `data/uploads/`.
3. `extractor.py` loads the image and runs FASHN human parsing.
4. The segmentation mask is resized to the source image dimensions if necessary.
5. Clothing labels are grouped into upper garments and bottoms.
6. Garment pixels are clustered in LAB space to estimate dominant colors.
7. Connected components identify crop regions.
8. Transparent PNG garment cutouts are written to `data/wardrobe/tops/` or `data/wardrobe/pants/`.
9. A labelled overlay is written to `data/labelled_images/`.
10. `data/wardrobe/metadata.json` is updated.
11. The UI reloads the metadata and renders wardrobe cards.

## FASHN Human Parser details

The parser provides an 18-class semantic mask with class IDs from 0 to 17. The installed package exposes the mapping through `IDS_TO_LABELS`, which the project uses rather than duplicating label strings in the extraction logic.

| ID | Label |
|---:|---|
| 0 | background |
| 1 | face |
| 2 | hair |
| 3 | top |
| 4 | dress |
| 5 | skirt |
| 6 | pants |
| 7 | belt |
| 8 | bag |
| 9 | hat |
| 10 | scarf |
| 11 | glasses |
| 12 | arms |
| 13 | hands |
| 14 | legs |
| 15 | feet |
| 16 | torso |
| 17 | jewelry |

### Wardrobe grouping

- **Tops:** `top`, `dress`, `scarf`.
- **Pants/bottoms:** `pants`, `skirt`, `belt`.
- **Not cropped:** accessories and identity/body regions such as bags, hats, glasses, jewelry, face, hair, arms, hands, legs, feet, and torso.

These groupings can be changed in `backend/services/extractor.py` if dresses, skirts, or accessories should receive separate folders.

## Color extraction

For each clothing class, the extractor:

1. Builds a binary class mask.
2. Samples only pixels inside that mask.
3. Converts the sampled pixels to OpenCV LAB space.
4. Removes extreme highlights and shadows using the L channel.
5. Runs K-Means with up to three clusters.
6. Converts each cluster center to CIE Lab.
7. Compares it with a fashion-oriented palette using Delta E 2000.
8. Stores the dominant cluster as `primarycolor` and qualifying smaller clusters as `secondarycolors`.

Example metadata:

```json
{
  "class_id": 3,
  "label": "top",
  "primarycolor": "navy",
  "primary_color_pct": 0.81,
  "secondarycolors": [
    {"name": "white", "percentage": 0.19}
  ]
}
```

## Troubleshooting

### `ModuleNotFoundError`

From the project root, run:

```bash
uvicorn backend.app:app --reload
```

Use relative imports in `backend/app.py`:

```python
from .services.extractor import process_image
```

### `NameError: ROOT_DIR is not defined`

Use the current path variables exposed by `extractor.py`, such as `STATIC_DIR`, `DATA_DIR`, and `UPLOADS_DIR`. Do not mix them with old names such as `ROOT_DIR`, `IMAGES_DIR`, or `WARDROBE_DIR` unless those names are explicitly defined.

### Images return 404

Verify that:

1. The physical file exists under `data/`.
2. The metadata URL matches the FastAPI mount.
3. The URL uses forward slashes.
4. The UI is not prepending an extra `/data` segment.
5. The server has been restarted after changing path logic.

### Source preview is blank

The browser preview should use `URL.createObjectURL(file)`. Keep the object URL alive until the preview image loads, and revoke it only after `img.onload` or when replacing it with another upload.

### No garment cutouts are saved

Check the server logs for detected class labels and verify the `save_tops` and `save_pants` form values. The extractor applies minimum-area thresholds to ignore very small segmentation fragments.

## Extending the project

Potential next steps include:

- Fine-grained classification such as shirt, T-shirt, hoodie, or crop top.
- Sleeve length, neckline, fit, pattern, and garment-length attributes.
- Batch processing of an entire image library.
- Separate folders for dresses, skirts, and accessories.
- Outfit recommendation and compatibility search.
- A vector or database-backed wardrobe index.
- Virtual try-on preprocessing and downstream generation.

## Git recommendations

Generated images and model artifacts can be large. A suitable `.gitignore` can include:

```gitignore
__pycache__/
*.py[cod]
.env
.venv/
env/

# Runtime data
data/uploads/*
data/labelled_images/*
data/wardrobe/tops/*
data/wardrobe/pants/*

# Keep metadata if desired
data/wardrobe/metadata.json
```

If the repository includes demo images, keep a small curated set in an `examples/` directory rather than committing every runtime upload.

## License

This repository is a project wrapper around the FASHN Human Parser package and model. Follow the licensing and usage terms of the upstream FASHN Human Parser repository and model weights. Add a project-specific license file if you intend to distribute this application.
