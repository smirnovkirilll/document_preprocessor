from PIL import Image
import numpy as np
from document_preprocessor.core import DocumentPreprocessor, PreprocessorConfig


def test_preprocess_file_creates_output(tmp_path):
    # Готовим простой "документ"
    input_path = tmp_path / "doc.png"
    img_np = np.full((600, 400), 255, dtype=np.uint8)
    img_np[100:120, 50:350] = 0  # "строка текста"
    Image.fromarray(img_np, mode="L").save(input_path)

    output_path = tmp_path / "doc_ocr.png"

    cfg = PreprocessorConfig.from_profile_and_env("default")
    pre = DocumentPreprocessor(config=cfg, debug=False)

    pre.preprocess_file(input_path, output_path)

    assert output_path.exists()

    out = Image.open(output_path)
    assert out.mode == "L"  # ч/б
    assert max(out.size) <= cfg.target_long_side_px
