from PIL import Image
import numpy as np
from document_preprocessor.core import DocumentPreprocessor


def test_detect_and_warp_on_flat_page():
    pre = DocumentPreprocessor()
    # Белый прямоугольник с чёрной рамкой
    img_np = np.full((500, 400, 3), 255, dtype=np.uint8)
    img_np[20:480, 20:380] = 255  # "страница" фактически заполняет почти весь кадр
    pil_img = Image.fromarray(img_np, mode="RGB")

    warped = pre.detect_and_warp_document(pil_img)

    # Размеры должны быть не нулевые и разумно близкие к исходным
    assert warped.size[0] > 0 and warped.size[1] > 0
    assert abs(warped.size[0] - pil_img.size[0]) < 100
    assert abs(warped.size[1] - pil_img.size[1]) < 100
