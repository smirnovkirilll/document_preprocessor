from PIL import Image
import numpy as np
from document_preprocessor.core import DocumentPreprocessor, PreprocessorConfig


def test_preprocess_directory_with_suffix(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    # Два простых серых "дока"
    for i in range(2):
        img_np = np.full((200, 100), 200, dtype=np.uint8)
        Image.fromarray(img_np, mode="L").save(input_dir / f"doc{i}.png")

    cfg = PreprocessorConfig.from_profile_and_env("default")
    pre = DocumentPreprocessor(config=cfg, debug=False)

    success, failed = pre.preprocess_directory(
        input_dir=input_dir,
        output_dir=None,
        suffix="_ocr",
        recursive=False,
    )

    assert success == 2
    assert failed == 0

    for i in range(2):
        out = input_dir / f"doc{i}_ocr.png"
        assert out.exists()
