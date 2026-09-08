# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`document_preprocessor` is a Python library for preprocessing scanned/photographed documents (especially A4 pages from smartphones) to prepare them for OCR engines. It handles perspective correction, contrast enhancement, binarization, and noise reduction. The library does **not** perform OCR itself—it only prepares images.

**Key characteristics:**
- Python 3.13+ required
- Dependencies: Pillow, OpenCV (opencv-python), NumPy
- Single-file core implementation in `src/document_preprocessor/core.py`
- CLI interface in `src/document_preprocessor/cli.py`
- Published to PyPI with automatic CI/CD on every push to main

## Development Commands

### Environment Setup
```bash
# Install in development mode
python -m pip install -e .

# Install test dependencies
python -m pip install pytest pytest-cov
```

### Testing
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=document_preprocessor --cov-report=xml

# Run specific test file
pytest tests/test_config.py

# Run specific test function
pytest tests/test_config.py::test_config_profile_and_env_override
```

### Building
```bash
# Build distribution packages
python -m pip install build
python -m build

# Output: dist/document_preprocessor-*.tar.gz and dist/document_preprocessor-*.whl
```

### CLI Usage (for testing)
```bash
# Single file processing
python -m document_preprocessor.cli --input-file input.jpg --output-file output.png --profile default --verbose

# Directory processing
python -m document_preprocessor.cli --input-dir ./scans --output-dir ./processed --profile shadows --recursive --verbose

# After installation, use the CLI directly
document_preprocessor --input-file input.jpg --output-file output.png
```

## Architecture

### Core Classes (`src/document_preprocessor/core.py`)

**`PreprocessorConfig`** - Dataclass holding all preprocessing parameters:
- Geometry settings (target size, page detection thresholds)
- **Deskew-first detection settings** (NEW, recommended):
  - `use_deskew_first` (default: True) - enables deskew-first algorithm
  - `detect_folds` (default: True) - detect document creases/folds
  - `simple_crop_if_folded` (default: True) - use simple crop for folded docs
  - `deskew_min_line_length_ratio` (default: 0.15) - minimum line length for skew detection
  - `deskew_angle_threshold` (default: 0.5°) - minimum angle to apply rotation
- **Enhanced detection settings**:
  - `use_enhanced_detection` (default: True) - enables multi-stage algorithm
  - `edge_dilation_ksize` (default: 3) - morphological edge enhancement
  - `max_contours_to_check` (default: 30) - vs 10 in old algorithm
  - `min_contour_score` (default: 0.5) - minimum quality threshold
  - `use_hough_fallback` (default: True) - line-based detection fallback
  - `refine_corners` (default: True) - sub-pixel corner accuracy
  - `crop_margin_px` (default: 0) - inward margin from detected edges
- Image enhancement (contrast, noise filtering, sharpening)
- Binarization settings (Otsu vs adaptive)
- Morphological post-processing
- Supports profiles (`default`, `dark`, `shadows`, `small_text`, `small_text_hard`, `cardiogram`, `ultrasound`)
- Three-layer configuration: profile defaults → environment variables (`DOC_PREPROC_*`) → CLI args

**Algorithm Selection Priority** (first matching condition wins):
1. If `use_deskew_first=True` → Deskew-First Algorithm (default, best for real-world docs)
2. Else if `use_enhanced_detection=True` → Enhanced Algorithm
3. Else → Simple Algorithm (legacy)

**`DocumentPreprocessor`** - Main processing class with step-by-step methods:
- `load_and_fix_exif()` - Handles EXIF orientation from phone photos
- `detect_and_warp_document()` - Finds page contour and applies perspective correction
- `to_grayscale()` - RGB → grayscale conversion
- `resize_long_side()` - Normalizes image size
- `enhance_contrast_and_denoise()` - Applies contrast and median filtering
- `binarize_otsu()` / `binarize_adaptive()` - Two binarization strategies
- `postprocess_binary()` - Morphological operations (opening/closing)
- `sharpen()` - UnsharpMask for edge enhancement
- `save_for_ocr()` - Saves with DPI metadata

High-level methods:
- `process_image(img: Image.Image)` - Full pipeline on PIL Image
- `preprocess_file(input_path, output_path)` - Process single file
- `preprocess_directory(...)` - Batch processing with filtering

**`DocumentProcessingError`** - Custom exception for processing failures

### Processing Pipeline

The complete pipeline executed by `process_image()`:
1. EXIF orientation fix (for phone photos)
2. Page detection via Canny edges + contour approximation
3. Perspective correction (warp to rectangle)
4. Grayscale conversion
5. Size normalization (downscale to target_long_side_px if needed)
6. Contrast enhancement + median noise filtering
7. Binarization (Otsu or adaptive)
8. Morphological cleanup (optional opening/closing)
9. Sharpening (UnsharpMask)
10. Save with DPI metadata

Each step can be called individually or skipped for custom pipelines.

### Configuration System

Three-layer priority (lowest to highest):
1. **Profile defaults** - Built-in presets in `PreprocessorConfig.from_profile_and_env()`
2. **Environment variables** - `DOC_PREPROC_*` (e.g., `DOC_PREPROC_CONTRAST_FACTOR=1.8`)
3. **CLI arguments** - Command-line flags override everything

Profiles are implemented as hardcoded parameter overrides in `from_profile_and_env()`. Each profile targets specific document conditions (dark photos, shadows, small text, etc.).

### CLI Interface (`src/document_preprocessor/cli.py`)

- Entry point: `run_from_cli()` registered as `document_preprocessor` console script
- Supports single-file and batch directory modes
- Rich argument set: profiles, recursive processing, file filtering, debug mode
- Environment variable integration via `PreprocessorConfig.from_profile_and_env()`

## Testing Strategy

Tests are in `tests/` directory:
- `test_config.py` - Configuration layer, profile, environment variable overrides
- `test_geometry.py` - Page detection and perspective correction
- `test_pipeline_single.py` - Single image processing
- `test_pipeline_directory.py` - Batch directory processing
- `test_otsu.py` - Binarization methods
- `test_cli.py` - CLI interface (30 tests): argument parsing, env vars, configuration priority, error handling, exit codes. Uses mocking, no real images required.

Tests use `monkeypatch` for environment variable testing. Most tests require test images except CLI tests which use mocking.

## CI/CD

**Workflow:** `.github/workflows/python-publish.yml`
- Triggers on every push to `main` branch
- Runs tests with coverage on Python 3.13
- Uploads coverage to Codecov
- Builds distribution packages
- Automatically publishes to PyPI using trusted publishing (no manual tokens)

**Important:** Version is managed in `src/document_preprocessor/__init__.py` (`__version__`). Update version there before merging to main if you want a new PyPI release.

## Key Implementation Details

### Page Detection Algorithm

The library now has **3 detection strategies** that can be selected via configuration:

**1. Deskew-First Algorithm** (NEW, default, `use_deskew_first=True`):

Best for: Documents with folds, creases, or significant skew. Recommended for real-world scenarios.

1. **Deskew Stage** - Corrects document rotation before detection:
   - Detects dominant lines via Hough transform
   - Calculates median angle from all detected lines
   - Rotates image to align document with axes
   - Skips rotation if angle < 0.5° threshold

2. **Fold Detection** - Identifies document creases:
   - Looks for long straight lines (vertical/horizontal) in center regions
   - Uses morphological operations to enhance crease visibility
   - Detects folds that span at least 40% of document dimension
   - Marks document as "folded" if creases found

3. **Boundary Detection** - Finds outer document edges:
   - Uses adaptive thresholding to separate document from background
   - Morphological closing to fill gaps and connect breaks
   - Finds largest contour (assumes it's the document)
   - If folded: uses convex hull to encompass entire boundary
   - Approximates to 4 corner points

4. **Adaptive Cropping** - Strategy depends on fold detection:
   - **If folded** (and `simple_crop_if_folded=True`): Simple rectangular crop by bounding box (avoids distortion from perspective transform on non-planar surface)
   - **If not folded**: Full perspective transform with corner refinement

Handles documents that were folded and unfolded (appear as 4 combined rectangles).

**2. Enhanced Multi-Stage Algorithm** (`use_enhanced_detection=True`):

1. **Enhanced Edge Detection** - Combines multiple techniques:
   - Bilateral filtering to preserve edges while removing noise
   - Adaptive thresholding for better document/background separation
   - Multiple Canny threshold combinations merged together
   - Morphological closing to connect edge gaps

2. **Contour Scoring** - Evaluates candidates by multiple criteria (0.0-1.0 score):
   - Area coverage (weight 0.3) - prefers 80% of image area
   - Rectangularity (weight 0.3) - contour area / bounding box area
   - Aspect ratio (weight 0.2) - favors A4/Letter proportions (1.2-1.6)
   - Edge strength (weight 0.2) - measures edge intensity along contour
   - Checks up to 30 contours (vs 10 in old algorithm)

3. **Hough Lines Fallback** - If no good contour found:
   - Detects dominant straight lines via Hough transform
   - Groups into horizontal and vertical lines
   - Finds corner intersections
   - Validates against paper-like proportions

4. **Corner Refinement** - Uses `cv2.cornerSubPix()` to sub-pixel accuracy
5. **Optional Margins** - Can apply inward offset via `crop_margin_px`

**3. Simple Algorithm** (legacy):

Enabled by: `use_deskew_first=False` AND `use_enhanced_detection=False`

- Basic Canny edge detection on Gaussian-blurred grayscale
- Finds 4-point contours, checks top 10 by area
- Filters by area ratio and rectangularity only
- No deskewing, no fold detection, no fallback, no corner refinement

**Recommendation:** Use Deskew-First (default) for best results with real-world documents, especially those with folds, creases, or rotation.

### Binarization Methods
- **Otsu:** Global threshold via `cv2.threshold(cv2.THRESH_OTSU)` - good for even lighting
- **Adaptive:** Local thresholds via `cv2.adaptiveThreshold(cv2.ADAPTIVE_THRESH_GAUSSIAN_C)` - better for shadows/uneven lighting

### Debug Mode
When `debug=True`, saves intermediate images to `{output_path}_debug/`:
- `01_exif_fixed.png` - After EXIF orientation
- `02_contours.png` - Detected page contour visualization
- `03_warped.png` - After perspective correction
- `04_grayscale.png`, `05_resized.png`, etc.

Useful for troubleshooting page detection and parameter tuning.

## Common Patterns

### Creating a custom processor
```python
from document_preprocessor import DocumentPreprocessor, PreprocessorConfig

# Use profile with environment overrides
cfg = PreprocessorConfig.from_profile_and_env(profile="shadows")
processor = DocumentPreprocessor(config=cfg, debug=False)
processor.preprocess_file("input.jpg", "output.png")
```

### Custom pipeline (selective steps)
```python
cfg = PreprocessorConfig.from_profile_and_env("default")
proc = DocumentPreprocessor(config=cfg)

img = proc.load_and_fix_exif(Path("input.jpg"))
img = proc.detect_and_warp_document(img)
img = proc.to_grayscale(img)
# Skip resize, go straight to binarization
img = proc.binarize_adaptive(img)
proc.save_for_ocr(img, Path("output.png"))
```

### Override specific parameters
```python
cfg = PreprocessorConfig.from_profile_and_env("default")
cfg.contrast_factor = 1.8
cfg.binarization_method = "adaptive"
cfg.adaptive_block_size = 45
processor = DocumentPreprocessor(config=cfg)
```

## Notes

- Code comments are in Russian (Cyrillic), but all public API is clear from context
- The project is heavily AI-generated (see README warning)
- OpenCV is used for geometric operations, PIL for image enhancement
- All processing is in-memory (PIL Image objects), converted to/from NumPy arrays for OpenCV operations
