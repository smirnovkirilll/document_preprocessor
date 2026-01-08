[![CI](https://github.com/smirnovkirilll/document_preprocessor/actions/workflows/python-publish.yml/badge.svg)](https://github.com/smirnovkirilll/document_preprocessor/actions/workflows/python-publish.yml)
[![codecov](https://codecov.io/github/smirnovkirilll/document_preprocessor/graph/badge.svg)](https://codecov.io/github/smirnovkirilll/document_preprocessor)
[![PyPI version](https://img.shields.io/pypi/v/document_preprocessor.svg)](https://pypi.org/project/document_preprocessor/)
[![License](https://img.shields.io/github/license/smirnovkirilll/document_preprocessor)](./LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/smirnovkirilll/document_preprocessor?style=social)](https://github.com/smirnovkirilll/document_preprocessor/stargazers)


> [!CAUTION]
> AI-created/vibe coded
>
> AI-model: ChatGPT 5.1 / Sonnet 4.5
>
> AI-participation degree: 100%


# Document Preprocessor (PIL + OpenCV)

A configurable image preprocessor for scanned/photographed documents (e.g. A4 pages shot on an iPhone), designed to produce images that are easy for OCR engines to read (Tesseract, ABBYY, etc.).

The project focuses on:

- correcting page perspective (trapezoid → rectangle),
- normalizing contrast and noise,
- robust binarization (global and adaptive),
- producing clean, sharp, high‑DPI black‑and‑white documents.

It does **not** run OCR itself – it prepares images to be fed into your OCR engine of choice.

---

## Features

### Geometry & page detection

- EXIF orientation handling (important for iPhone photos).
- Automatic page detection:
  - downsamples image to speed up contour search,
  - finds large 4‑point contours,
  - filters by:
    - minimum page area ratio (e.g. ≥ 20% of image),
    - “rectangularity” (contour area / bounding box area).
- Perspective correction:
  - warps the detected page polygon into a proper rectangle.
- Optional size normalization:
  - resizes result to a configurable *target long side* (e.g. 3500 px).

### Image enhancement for OCR

- Grayscale conversion.
- Contrast enhancement (configurable factor).
- Median filtering for noise suppression (configurable kernel size).
- Two binarization methods:
  - **Otsu** global thresholding (good for even lighting).
  - **Adaptive** Gaussian thresholding (better for shadows / uneven lighting):
    - configurable block size (window),
    - configurable `C` (constant subtracted from local threshold).
- Morphological post‑processing:
  - optional **opening** (removes isolated noise dots),
  - optional **closing** (fills small holes in letters/lines).
- Sharpening via **UnsharpMask**:
  - configurable radius, amount (percent), and threshold.
- Explicit DPI in file metadata (default 300 dpi), which some OCR engines expect.

### Configuration & integration

- All core logic exposed as a reusable Python class:
  - `DocumentPreprocessor`
  - `PreprocessorConfig`
- Clear step‑wise methods:
  - `load_and_fix_exif`, `detect_and_warp_document`, `to_grayscale`,
    `resize_long_side`, `enhance_contrast_and_denoise`,
    `binarize_otsu` / `binarize_adaptive`, `postprocess_binary`,
    `sharpen`, `save_for_ocr`.
- High‑level methods:
  - `process_image(img)` for an in‑memory PIL image,
  - `preprocess_file(input_path, output_path)`,
  - `preprocess_directory(...)` for batch processing.

### CLI, profiles, and environments

- Rich command‑line interface:
  - single file mode (`--input-file` / `--output-file`),
  - directory mode (`--input-dir` / `--output-dir`),
  - suffix for in‑place output (`--suffix _ocr`),
  - recursive processing (`--recursive`),
  - filtering by glob patterns (`--include "*.jpg" "*.png"`),
  - skip already processed files (`--skip-existing`),
  - verbose logging (`--verbose`),
  - debug mode with intermediate images per file (`--debug`).
- Built‑in **profiles** (recipes) for typical scenarios:
  - `default` — general‑purpose OCR preprocessing.
  - `dark` — underexposed, dark photos.
  - `shadows` — strong local shadows and uneven lighting.
  - `small_text` — very small text or distant pages.
- Multi‑layer configuration:
  1. **Profile defaults** (`default`, `dark`, `shadows`, `small_text`).
  2. **Environment variables** (`DOC_PREPROC_*`).
  3. **CLI arguments** (highest priority).

---

## Processing pipeline

The full pipeline applied by `DocumentPreprocessor.process_image`:

1. **EXIF orientation fix**  
   Uses `PIL.ImageOps.exif_transpose` so images from phones are correctly oriented.

2. **Page detection & perspective correction**  
   - Downscale for contour search.  
   - Canny edge detection.  
   - Find large 4‑point contour with area and rectangularity filters.  
   - Warp the page to a rectangle (`cv2.getPerspectiveTransform` + `cv2.warpPerspective`).

3. **Grayscale conversion**  
   `RGB → L` (8‑bit grayscale).

4. **Size normalization**  
   If image is larger than `target_long_side_px`, it is downscaled to that long side.

5. **Contrast & denoising**  
   - Apply contrast factor (e.g. 1.5–1.8).  
   - Optional median filter (e.g. 3×3) to reduce noise while preserving edges.

6. **Binarization**  
   - **Otsu**: global threshold based on histogram.  
   - or **Adaptive**: local thresholds for each window (block).

7. **Morphological cleanup**  
   - optional opening (`MORPH_OPEN`) to remove isolated noise,
   - optional closing (`MORPH_CLOSE`) to fill gaps within letters/lines.

8. **Sharpening**  
   UnsharpMask to make character edges crisper without oversharpening.

9. **Save with DPI**  
   Output image saved (typically PNG/TIFF) with metadata DPI (default 300).

Each step is a separate method on `DocumentPreprocessor`, so you can reuse any subset in your own code.

---

## Profiles (recipes)

Profiles are predefined configurations for common scenarios. You can select them with `--profile` or via environment variables.

### `default`

General‑purpose settings for typical office documents:
- global Otsu binarization,
- moderate contrast enhancement,
- moderate noise reduction,
- light closing to solidify letters.

### `dark`

For underexposed or very dark photos:
- stronger contrast,
- **adaptive** binarization with a larger block size,
- light opening + closing to regularize text.

### `shadows`

For documents with strong local shadows/uneven lighting:
- adaptive binarization with a somewhat smaller window (handles local variations),
- moderate contrast,
- light morphology.

### `small_text`

For very small text (e.g. distant page photos):
- higher target resolution (larger `target_long_side_px`),
- gentle smoothing (or almost none),
- slightly stronger, but controlled, sharpening,
- minimal morphology to avoid eating thin strokes.

### `small_text_hard`

For very small text, strong shadows, and low‑resolution source images (e.g. distant photos, old phone cameras):
- keeps smoothing almost disabled to avoid blurring thin strokes;
- uses stronger contrast and more aggressive sharpening to make tiny glyphs stand out;
- uses adaptive binarization with a smaller window to react to local shadows and illumination changes;
- applies only light closing (no opening) to avoid erasing fine details while still fixing small gaps inside letters.

Use this when:
- text is barely legible on the original image,
- there are pronounced shadows or uneven lighting,
- upscaling the image doesn’t really add detail (original is already low‑res).


You can still override any parameter in a profile via environment variables or CLI.

---

## Configuration via environment variables

Every key parameter can be overridden via `DOC_PREPROC_*` environment variables, for example:

```bash
export DOC_PREPROC_CONTRAST_FACTOR=1.8
export DOC_PREPROC_BINARIZATION_METHOD=adaptive
export DOC_PREPROC_ADAPTIVE_BLOCK_SIZE=45
export DOC_PREPROC_ADAPTIVE_C=8
export DOC_PREPROC_DPI=300
```


Supported variables include (all optional):

```
-- Geometry & page detection:
DOC_PREPROC_TARGET_LONG_SIDE_PX
DOC_PREPROC_MAX_PROC_DIM
DOC_PREPROC_CANNY_THRESHOLD1
DOC_PREPROC_CANNY_THRESHOLD2
DOC_PREPROC_CONTOUR_EPSILON_COEF
DOC_PREPROC_MIN_PAGE_AREA_RATIO
DOC_PREPROC_MIN_RECTANGULARITY

-- Contrast / noise / sharpening:
DOC_PREPROC_CONTRAST_FACTOR
DOC_PREPROC_MEDIAN_FILTER_SIZE
DOC_PREPROC_SHARPEN_RADIUS
DOC_PREPROC_SHARPEN_PERCENT
DOC_PREPROC_SHARPEN_THRESHOLD

-- Binarization:
DOC_PREPROC_BINARIZATION_METHOD (otsu or adaptive)
DOC_PREPROC_ADAPTIVE_BLOCK_SIZE
DOC_PREPROC_ADAPTIVE_C

-- Morphology:
DOC_PREPROC_MORPH_OPEN_KSIZE
DOC_PREPROC_MORPH_CLOSE_KSIZE

-- DPI:
DOC_PREPROC_DPI
```

Invalid values are ignored with a warning.

---

## CLI usage


### Installation

**from PyPi**
```bash
pip install document_preprocessor
```

**from GitHub**
```bash
pip install git+https://github.com/smirnovkirilll/document_preprocessor.git
```

**Required packages** (will be installed automatically if not present):
```bash
pip install pillow opencv-python numpy
```


### Single file

Process a single image:
```bash
python document_preprocessor.py \
  --input-file photo_from_iphone.jpg \
  --output-file doc_for_ocr.png \
  --profile default \
  --verbose
```

Use adaptive binarization for tricky lighting:
```bash
python document_preprocessor.py \
  --input-file photo.jpg \
  --output-file doc_for_ocr.png \
  --profile shadows \
  --binarization adaptive \
  --verbose
```

Enable debug mode to save intermediate steps:
```bash
python document_preprocessor.py \
  --input-file photo.jpg \
  --output-file doc_for_ocr.png \
  --profile default \
  --debug \
  --verbose
```

Debug mode will create a folder like doc_for_ocr_debug/ with intermediate images (edges, contours, warped page, grayscale, binary, etc.).


### Batch processing (directory)

Process all images in a directory, placing results next to originals with suffix `_ocr`:
```bash
python document_preprocessor.py \
  --input-dir ./scans \
  --suffix _ocr \
  --profile default \
  --recursive \
  --skip-existing \
  --verbose
```

Process into a separate directory, without suffix:
```bash
python document_preprocessor.py \
  --input-dir ./scans \
  --output-dir ./processed \
  --profile dark \
  --recursive \
  --verbose
```

Limit to specific file types:
```bash
python document_preprocessor.py \
  --input-dir ./scans \
  --output-dir ./processed \
  --include "*.jpg" "*.png" \
  --profile small_text \
  --verbose
```

Fine‑tune contrast, noise filter, morphology:
```bash
python document_preprocessor.py \
  --input-dir ./scans \
  --output-dir ./processed \
  --profile default \
  --contrast 1.7 \
  --median-filter-size 3 \
  --morph-open 2 \
  --morph-close 3 \
  --binarization adaptive \
  --adaptive-block-size 41 \
  --adaptive-C 10 \
  --verbose
```


### Using as a library

Basic example:
```python
from document_preprocessor import DocumentPreprocessor, PreprocessorConfig

cfg = PreprocessorConfig.from_profile_and_env(profile="default")
pre = DocumentPreprocessor(config=cfg, debug=False)
pre.preprocess_file("input.jpg", "output.png")
```

Custom pipeline, step by step:
```python
from document_preprocessor import DocumentPreprocessor, PreprocessorConfig
from pathlib import Path

cfg = PreprocessorConfig.from_profile_and_env("small_text")
pre = DocumentPreprocessor(config=cfg, debug=True)

input_path = Path("photo.jpg")
output_path = Path("photo_ocr.png")
debug_dir = output_path.parent / f"{output_path.stem}_debug"

img = pre.load_and_fix_exif(input_path, debug_dir=debug_dir)
img = pre.detect_and_warp_document(img, debug_dir=debug_dir)
img = pre.to_grayscale(img, debug_dir=debug_dir)
img = pre.resize_long_side(img, debug_dir=debug_dir)
img = pre.enhance_contrast_and_denoise(img, debug_dir=debug_dir)
img = pre.binarize_otsu(img, debug_dir=debug_dir)
img = pre.postprocess_binary(img, debug_dir=debug_dir)
img = pre.sharpen(img, debug_dir=debug_dir)

pre.save_for_ocr(img, output_path)
```

You can replace `binarize_otsu` with `binarize_adaptive` depending on your needs.

---

## How it compares to typical GitHub scripts

Compared to many small “document scanner” or “OCR pre‑processing” scripts you’ll find on GitHub, this project aims to be:

**More complete** in preprocessing:
- EXIF handling,
- robust page detection and perspective correction,
- configurable binarization, morphology, sharpening, DPI.

**More configurable:**
- profiles for common scenarios,
- environment variable overrides,
- rich CLI parameters,
- reusable Python API.

**Still lightweight:**
- no external binaries required beyond `Python + Pillow + OpenCV + NumPy`,
- no heavy ML models.

What it does **not** do (by design):
- run OCR itself (e.g. Tesseract, ABBYY, EasyOCR),
- handle multi‑page PDFs or generate text‑layer PDFs,
- perform complex layout analysis (multi‑column, tables),
- apply deep‑learning based dewarping or enhancement.

Use this project when you want:
- a solid, configurable image preprocessing pipeline for documents,
- a drop‑in component you can plug into your own OCR workflows,
- good defaults but also fine‑grained control when needed.
