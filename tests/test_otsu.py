import numpy as np
from document_preprocessor.core import DocumentPreprocessor


def test_otsu_on_two_level_image():
    # Левая половина тёмная, правая светлая
    dark_value = 30
    light_value = 220
    h, w = 100, 100
    dark = np.full((h, w // 2), dark_value, dtype=np.uint8)
    light = np.full((h, w // 2), light_value, dtype=np.uint8)
    img = np.concatenate([dark, light], axis=1)

    t = DocumentPreprocessor.otsu_threshold(img)

    # Порог должен где-то отделять два кластера, но не важно, строго ли внутри интервала
    assert 0 <= t <= 255

    # Проверяем, что бинаризация по этому порогу разделяет левую и правую части
    binary = (img > t).astype(np.uint8) * 255  # 0 или 255

    left = binary[:, : w // 2]
    right = binary[:, w // 2 :]

    # Левая часть должна быть полностью "чёрной"
    assert left.max() == 0

    # Правая часть должна быть полностью "белой"
    assert right.min() == 255
