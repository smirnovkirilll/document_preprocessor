#!/usr/bin/env python3
# Требуемая версия Python: 3.13+

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple, Optional

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


logger = logging.getLogger(__name__)


class DocumentProcessingError(Exception):
    """Ошибка при обработке отдельного документа."""


@dataclass
class PreprocessorConfig:
    """
    Настройки препроцессинга документа.

    Источники значений (по возрастанию приоритета):
      1) профиль (default / dark / shadows / small_text),
      2) переменные окружения DOC_PREPROC_*,
      3) аргументы командной строки.

    Если вы не уверены, что менять — можно использовать профиль `default`
    и ничего не задавать вручную.
    """

    # ---------- Геометрия / размер ----------

    # Целевая «длинная сторона» итогового изображения (в пикселях).
    # Если фото очень большое, мы уменьшаем его так, чтобы длинная сторона
    # была примерно равна этому числу.
    target_long_side_px: int = 3500

    # Размер уменьшенного изображения (длинная сторона) при поиске контура.
    # Это влияет только на скорость и устойчивость поиска страницы.
    max_proc_dim: int = 1000

    # ---------- Поиск контура листа ----------

    # Пороговые значения для оператора Canny (поиск границ).
    canny_threshold1: int = 50
    canny_threshold2: int = 150

    # Коэффициент точности аппроксимации контура.
    contour_epsilon_coef: float = 0.02

    # Минимальная доля площади кадра, которую должен занимать лист.
    # 0.2 означает, что контур должен покрывать не менее 20% кадра.
    min_page_area_ratio: float = 0.2

    # «Прямоугольность» контура: 1.0 — идеальный прямоугольник.
    # Если значение низкое, это, вероятно, не лист, а что-то кривое.
    min_rectangularity: float = 0.7

    # --- Новые параметры для улучшенного обнаружения ---

    # Использовать улучшенный алгоритм обнаружения (многоуровневый с фоллбэками)
    use_enhanced_detection: bool = True

    # Размер ядра для морфологического закрытия границ (соединяет разрывы)
    edge_dilation_ksize: int = 3

    # Максимальное количество проверяемых контуров (вместо 10)
    max_contours_to_check: int = 30

    # Минимальный балл для принятия контура (от 0.0 до 1.0)
    min_contour_score: float = 0.5

    # Использовать Hough lines как фоллбэк если контуры не найдены
    use_hough_fallback: bool = True

    # Применять уточнение углов (corner refinement)
    refine_corners: bool = True

    # Отступ внутрь от найденных границ (в пикселях, 0 = без отступа)
    crop_margin_px: int = 0

    # --- Параметры для deskew-first подхода ---

    # Использовать deskew-first подход (сначала выравнивание, потом обнаружение)
    use_deskew_first: bool = True

    # Обнаруживать складки на документе (folded documents)
    detect_folds: bool = True

    # Для документов со складками: использовать простую обрезку по границам
    # вместо perspective transform (избегает искажений на складках)
    simple_crop_if_folded: bool = True

    # Минимальная длина линии Hough для определения угла перекоса
    deskew_min_line_length_ratio: float = 0.15  # от размера изображения

    # Угол перекоса считается значимым если больше этого порога (градусы)
    deskew_angle_threshold: float = 0.5

    # ---------- Контраст / шум / резкость ----------

    # Коэффициент контраста:
    # 1.0 — без изменений, 1.2–2.0 — разумное усиление для документов.
    contrast_factor: float = 1.5

    # Размер медианного фильтра для подавления шума.
    # Нечётное число (3, 5...), 0 или 1 — фильтр отключён.
    median_filter_size: int = 3

    # Параметры нерезкой маски (UnsharpMask):
    # radius   — радиус размытия (ширина границ),
    # percent  — сила усиления,
    # threshold — порог чувствительности.
    sharpen_radius: float = 1.0
    sharpen_percent: int = 150
    sharpen_threshold: int = 0

    # ---------- Бинаризация ----------

    # Метод бинаризации:
    # "otsu"     — один глобальный порог (хорошо при ровном освещении),
    # "adaptive" — локальные пороги (лучше при тенях и неравномерном свете).
    binarization_method: str = "otsu"

    # Размер окна для адаптивной бинаризации (локальный анализ яркости).
    # Нечётное число > 1. Типично 25–75.
    adaptive_block_size: int = 35

    # Константа C, вычитаемая из локального порога в адаптивной бинаризации.
    # Больше C — фон светлее, текст темнее.
    adaptive_C: int = 10

    # ---------- Морфология после бинаризации ----------

    # Размер ядра для MORPH_OPEN (открытие) — удаление мелкого мусора.
    # None или < 2 — не применять.
    morph_open_ksize: Optional[int] = None

    # Размер ядра для MORPH_CLOSE (закрытие) — залатывание дыр в буквах.
    # None или < 2 — не применять.
    morph_close_ksize: Optional[int] = 3

    # ---------- DPI сохранения ----------

    # DPI — плотность пикселей в метаданных. Многие OCR ожидают около 300 dpi.
    dpi: int = 300

    # ---------- Профили и окружение ----------

    @classmethod
    def from_profile_and_env(cls, profile: str = "default") -> "PreprocessorConfig":
        """
        Создать конфигурацию на основе профиля и переменных окружения DOC_PREPROC_*.

        Порядок:
          1. Берётся базовая конфигурация (значения по умолчанию).
          2. Поверх накладывается профиль (dark / shadows / small_text).
          3. Поверх профиля накладываются значения из переменных окружения.
        """
        profile = profile.lower()
        cfg = cls()

        # --- Профили (рецепты) ---

        if profile == "default":
            # Ничего не меняем — используются дефолтные значения.
            pass

        elif profile == "dark":
            # Тёмные / недоэкспонированные фото:
            # немного сильнее контраст, адаптивная бинаризация,
            # чуть более «мягкая» морфология.
            cfg.contrast_factor = 1.8
            cfg.binarization_method = "adaptive"
            cfg.adaptive_block_size = 45
            cfg.adaptive_C = 8
            cfg.median_filter_size = 3
            cfg.morph_open_ksize = 2
            cfg.morph_close_ksize = 3

        elif profile == "shadows":
            # Сильные локальные тени, неровное освещение:
            # адаптивная бинаризация с меньшим окном,
            # умеренный контраст, немного морфологии.
            cfg.contrast_factor = 1.6
            cfg.binarization_method = "adaptive"
            cfg.adaptive_block_size = 31
            cfg.adaptive_C = 12
            cfg.median_filter_size = 3
            cfg.morph_open_ksize = 2
            cfg.morph_close_ksize = 3

        elif profile == "small_text":
            # Очень мелкий текст / далеко снятые страницы:
            # сохраняем побольше пикселей, аккуратнее со сглаживанием,
            # мягкий шарпинг, минимальная морфология.
            cfg.target_long_side_px = 4200
            cfg.contrast_factor = 1.4
            cfg.median_filter_size = 1  # почти без сглаживания
            cfg.sharpen_radius = 1.2
            cfg.sharpen_percent = 170
            cfg.sharpen_threshold = 0
            cfg.binarization_method = "otsu"
            cfg.morph_open_ksize = None
            cfg.morph_close_ksize = 2

        elif profile == "small_text_hard":
            # Мелкий текст + сильные тени + невысокое исходное разрешение.
            #
            # Идея профиля:
            #   - не размывать (чтобы не потерять тонкие штрихи),
            #   - усилить контраст и шарпинг,
            #   - использовать адаптивную бинаризацию с не слишком большим окном
            #     (чтобы лучше отделять текст от локальных теней),
            #   - очень аккуратная морфология (только лёгкое закрытие).
            #
            # Замечание: из-за низкого разрешения мы НЕ сможем «добавить»
            # деталей, поэтому важно минимально скрадывать то, что есть.
            cfg.target_long_side_px = 3800          # на низком разрешении, как правило, не сработает (мы не увеличиваем), но не мешает
            cfg.contrast_factor = 1.7               # чуть сильнее контраст
            cfg.median_filter_size = 1              # фактически отключаем сглаживание

            # Шарпинг: немного сильнее, но с аккуратным радиусом
            cfg.sharpen_radius = 1.0
            cfg.sharpen_percent = 200
            cfg.sharpen_threshold = 0

            # Адаптивная бинаризация для борьбы с тенями
            cfg.binarization_method = "adaptive"
            cfg.adaptive_block_size = 25            # небольшое окно, чтобы реагировать на локальные тени
            cfg.adaptive_C = 8                      # не слишком большой сдвиг порога

            # Морфология: открытие не включаем, чтобы не съесть тонкие штрихи;
            # лишь лёгкое закрытие для заделывания небольших разрывов в буквах.
            cfg.morph_open_ksize = None
            cfg.morph_close_ksize = 2

        elif profile == "cardiogram":
            # Кардиограммы — очень тонкие линии графиков, сетка фона:
            #   - высокое разрешение для сохранения деталей тонких линий,
            #   - минимальное размытие (тонкие линии легко потерять),
            #   - умеренный контраст,
            #   - адаптивная бинаризация (может быть неравномерный фон),
            #   - НЕТ морфологии opening (съест тонкие линии),
            #   - очень легкое closing или вообще без него,
            #   - аккуратный шарпинг для подчеркивания линий.
            cfg.target_long_side_px = 4500              # высокое разрешение для тонких линий
            cfg.contrast_factor = 1.5                   # умеренное усиление контраста
            cfg.median_filter_size = 0                  # отключаем медианный фильтр (не размывать линии)

            # Шарпинг: помогает подчеркнуть тонкие линии графика
            cfg.sharpen_radius = 1.0
            cfg.sharpen_percent = 180
            cfg.sharpen_threshold = 0

            # Адаптивная бинаризация для работы с неравномерным фоном
            cfg.binarization_method = "adaptive"
            cfg.adaptive_block_size = 35
            cfg.adaptive_C = 10

            # Морфология: НЕТ opening (удалит тонкие линии), минимальное closing
            cfg.morph_open_ksize = None
            cfg.morph_close_ksize = None                # даже closing отключаем для сохранения тонких линий

        elif profile == "ultrasound":
            # Ультразвуковые медицинские изображения (УЗИ):
            #   - изображения в оттенках серого с низким контрастом,
            #   - характерный спекл-шум (speckle noise),
            #   - тёмные фоновые области,
            #   - необходимо сохранить мелкие анатомические детали,
            #   - часто присутствует текст с данными пациента и измерениями.
            cfg.target_long_side_px = 4500              # высокое разрешение для деталей
            cfg.contrast_factor = 1.7                   # усиленный контраст для проявления структур
            cfg.median_filter_size = 3                  # умеренное подавление спекл-шума

            # Шарпинг: помогает подчеркнуть границы анатомических структур
            cfg.sharpen_radius = 1.2
            cfg.sharpen_percent = 160
            cfg.sharpen_threshold = 0

            # Адаптивная бинаризация для работы с неравномерным контрастом
            cfg.binarization_method = "adaptive"
            cfg.adaptive_block_size = 41                # среднее окно для локальной адаптации
            cfg.adaptive_C = 8

            # Морфология: лёгкое opening для удаления спекл-шума, лёгкое closing для структур
            cfg.morph_open_ksize = 2
            cfg.morph_close_ksize = 3

        else:
            logger.warning(f"Неизвестный профиль '{profile}', используется 'default'.")

        # --- Переопределения из переменных окружения ---

        cfg._apply_env_overrides()
        return cfg

    def _apply_env_overrides(self) -> None:
        """
        Перезаписать значения из переменных окружения DOC_PREPROC_*,
        если они заданы. Ошибочные значения игнорируются с предупреждением.
        """
        env_map = {
            # имя_поля: (ENV_NAME, функция_преобразования)
            "target_long_side_px": ("DOC_PREPROC_TARGET_LONG_SIDE_PX", int),
            "max_proc_dim": ("DOC_PREPROC_MAX_PROC_DIM", int),
            "canny_threshold1": ("DOC_PREPROC_CANNY_THRESHOLD1", int),
            "canny_threshold2": ("DOC_PREPROC_CANNY_THRESHOLD2", int),
            "contour_epsilon_coef": ("DOC_PREPROC_CONTOUR_EPSILON_COEF", float),
            "min_page_area_ratio": ("DOC_PREPROC_MIN_PAGE_AREA_RATIO", float),
            "min_rectangularity": ("DOC_PREPROC_MIN_RECTANGULARITY", float),
            "contrast_factor": ("DOC_PREPROC_CONTRAST_FACTOR", float),
            "median_filter_size": ("DOC_PREPROC_MEDIAN_FILTER_SIZE", int),
            "sharpen_radius": ("DOC_PREPROC_SHARPEN_RADIUS", float),
            "sharpen_percent": ("DOC_PREPROC_SHARPEN_PERCENT", int),
            "sharpen_threshold": ("DOC_PREPROC_SHARPEN_THRESHOLD", int),
            "binarization_method": ("DOC_PREPROC_BINARIZATION_METHOD", str),
            "adaptive_block_size": ("DOC_PREPROC_ADAPTIVE_BLOCK_SIZE", int),
            "adaptive_C": ("DOC_PREPROC_ADAPTIVE_C", int),
            "morph_open_ksize": ("DOC_PREPROC_MORPH_OPEN_KSIZE", int),
            "morph_close_ksize": ("DOC_PREPROC_MORPH_CLOSE_KSIZE", int),
            "dpi": ("DOC_PREPROC_DPI", int),
        }

        for attr, (env_name, caster) in env_map.items():
            val = os.getenv(env_name)
            if val is None:
                continue
            try:
                casted = caster(val)
            except ValueError:
                logger.warning(
                    f"Некорректное значение переменной окружения {env_name}={val!r}, игнорируем."
                )
                continue
            setattr(self, attr, casted)

        # Нормализация некоторых параметров
        if isinstance(self.binarization_method, str):
            self.binarization_method = self.binarization_method.lower()

        # Значения <= 1 для морфологии считаем «выключенными»
        if self.morph_open_ksize is not None and self.morph_open_ksize <= 1:
            self.morph_open_ksize = None
        if self.morph_close_ksize is not None and self.morph_close_ksize <= 1:
            self.morph_close_ksize = None


class DocumentPreprocessor:
    """
    Класс для препроцессинга фото документов под OCR.
    """

    def __init__(self, config: Optional[PreprocessorConfig] = None, debug: bool = False):
        self.config = config or PreprocessorConfig()
        self.debug = debug

    # ---------- Вспомогательные функции отладки ----------

    def _debug_save_pil(self, debug_dir: Optional[Path], name: str, img: Image.Image) -> None:
        if not self.debug or debug_dir is None:
            return
        debug_dir.mkdir(parents=True, exist_ok=True)
        img.save(debug_dir / name)

    def _debug_save_cv2(self, debug_dir: Optional[Path], name: str, img: np.ndarray) -> None:
        if not self.debug or debug_dir is None:
            return
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / name), img)

    # ---------- Отдельные шаги конвейера ----------

    def load_and_fix_exif(
        self,
        input_path: str | Path,
        debug_dir: Optional[Path] = None,
    ) -> Image.Image:
        input_path = Path(input_path)
        try:
            img = Image.open(input_path)
            img = ImageOps.exif_transpose(img)
        except Exception as e:
            raise DocumentProcessingError(f"Не удалось открыть изображение {input_path}: {e}") from e

        self._debug_save_pil(debug_dir, "00_input_exif_fixed.jpg", img)
        return img

    # ---------- Вспомогательные методы для улучшенного обнаружения ----------

    def _enhanced_edge_detection(
        self,
        img_gray: np.ndarray,
        debug_dir: Optional[Path] = None,
    ) -> np.ndarray:
        """
        Улучшенное обнаружение границ с адаптивной бинаризацией и морфологией.
        """
        cfg = self.config

        # Bilateral filter для сохранения границ при удалении шума
        denoised = cv2.bilateralFilter(img_gray, 9, 75, 75)

        # Адаптивная бинаризация для лучшего разделения документа и фона
        adaptive_thresh = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        self._debug_save_cv2(debug_dir, "02a_adaptive_thresh.png", adaptive_thresh)

        # Пробуем несколько комбинаций Canny для лучшего обнаружения
        edges1 = cv2.Canny(denoised, cfg.canny_threshold1, cfg.canny_threshold2)
        edges2 = cv2.Canny(denoised, cfg.canny_threshold1 // 2, cfg.canny_threshold2 // 2)
        edges3 = cv2.Canny(adaptive_thresh, 50, 150)

        # Объединяем результаты
        edges_combined = cv2.bitwise_or(edges1, cv2.bitwise_or(edges2, edges3))
        self._debug_save_cv2(debug_dir, "02b_edges_combined.png", edges_combined)

        # Морфологическое закрытие для соединения разрывов в границах
        if cfg.edge_dilation_ksize > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (cfg.edge_dilation_ksize, cfg.edge_dilation_ksize)
            )
            edges_closed = cv2.morphologyEx(edges_combined, cv2.MORPH_CLOSE, kernel)
            edges_dilated = cv2.dilate(edges_closed, kernel, iterations=1)
            self._debug_save_cv2(debug_dir, "02c_edges_enhanced.png", edges_dilated)
            return edges_dilated

        return edges_combined

    def _score_contour(
        self,
        approx: np.ndarray,
        img_gray: np.ndarray,
        img_area: float,
    ) -> float:
        """
        Оценить качество контура как кандидата на границу документа.
        Возвращает балл от 0.0 до 1.0.
        """
        cfg = self.config

        # 1. Площадь (вес 0.3)
        area = cv2.contourArea(approx)
        if area <= 0:
            return 0.0

        area_ratio = area / img_area
        if area_ratio < cfg.min_page_area_ratio:
            return 0.0

        area_score = min(area_ratio / 0.8, 1.0)  # 80% площади = идеально

        # 2. Прямоугольность (вес 0.3)
        x, y, w, h = cv2.boundingRect(approx)
        rect_area = w * h if w > 0 and h > 0 else 1
        rectangularity = float(area) / rect_area

        if rectangularity < cfg.min_rectangularity:
            return 0.0

        rectangularity_score = rectangularity

        # 3. Соотношение сторон близко к бумаге A4/Letter (вес 0.2)
        # A4: 1.414, Letter: 1.294, допускаем 1.2-1.6
        aspect_ratio = max(w, h) / max(min(w, h), 1)
        aspect_score = 0.0
        if 1.2 <= aspect_ratio <= 1.6:
            aspect_score = 1.0
        elif 1.0 <= aspect_ratio <= 2.0:
            aspect_score = 0.5

        # 4. Сила границ вдоль контура (вес 0.2)
        # Проверяем, насколько ярко выражены границы вдоль найденного контура
        mask = np.zeros(img_gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [approx], -1, 255, 2)

        edges_along_contour = cv2.Canny(img_gray, 50, 150)
        edge_strength = np.sum(cv2.bitwise_and(edges_along_contour, mask)) / (np.sum(mask) + 1)
        edge_score = min(edge_strength / 50.0, 1.0)

        # Взвешенная сумма
        total_score = (
            0.3 * area_score +
            0.3 * rectangularity_score +
            0.2 * aspect_score +
            0.2 * edge_score
        )

        return total_score

    def _find_best_contour(
        self,
        edges: np.ndarray,
        img_gray: np.ndarray,
        img_area: float,
        debug_dir: Optional[Path] = None,
    ) -> Optional[np.ndarray]:
        """
        Найти лучший контур документа с оценкой кандидатов.
        """
        cfg = self.config

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:cfg.max_contours_to_check]

        best_contour = None
        best_score = cfg.min_contour_score

        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, cfg.contour_epsilon_coef * peri, True)

            # Принимаем 4+ точки, потом сократим до 4
            if len(approx) < 4:
                continue

            # Если больше 4 точек, упрощаем сильнее
            if len(approx) > 4:
                approx = cv2.approxPolyDP(cnt, cfg.contour_epsilon_coef * peri * 1.5, True)

            if len(approx) != 4:
                continue

            score = self._score_contour(approx, img_gray, img_area)

            if score > best_score:
                best_score = score
                best_contour = approx
                logger.debug(f"Найден контур с баллом {score:.3f}")

        if best_contour is not None:
            logger.debug(f"Лучший контур: балл={best_score:.3f}")

        return best_contour

    def _detect_document_by_hough(
        self,
        edges: np.ndarray,
        img_shape: Tuple[int, int],
        debug_dir: Optional[Path] = None,
    ) -> Optional[np.ndarray]:
        """
        Фоллбэк: обнаружение документа через Hough lines.
        Находим доминирующие прямые линии и их пересечения.
        """
        h, w = img_shape[:2]
        img_area = w * h

        # Hough transform для поиска линий
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=min(w, h) // 4,
            maxLineGap=20
        )

        if lines is None or len(lines) < 4:
            logger.debug("Hough lines: недостаточно линий")
            return None

        # Группируем линии по углу (горизонтальные и вертикальные)
        horizontal_lines = []
        vertical_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            if angle < 30 or angle > 150:  # Горизонтальные
                horizontal_lines.append((x1, y1, x2, y2))
            elif 60 < angle < 120:  # Вертикальные
                vertical_lines.append((x1, y1, x2, y2))

        # Нужно минимум 2 горизонтальных и 2 вертикальных
        if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
            logger.debug(f"Hough lines: недостаточно линий (h={len(horizontal_lines)}, v={len(vertical_lines)})")
            return None

        # Визуализация для отладки
        if self.debug and debug_dir:
            lines_vis = np.zeros((h, w, 3), dtype=np.uint8)
            for x1, y1, x2, y2 in horizontal_lines:
                cv2.line(lines_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for x1, y1, x2, y2 in vertical_lines:
                cv2.line(lines_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
            self._debug_save_cv2(debug_dir, "03b_hough_lines.png", lines_vis)

        # Простая эвристика: берем крайние линии
        # Сортируем горизонтальные по Y, вертикальные по X
        horizontal_lines.sort(key=lambda l: (l[1] + l[3]) / 2)
        vertical_lines.sort(key=lambda l: (l[0] + l[2]) / 2)

        top_line = horizontal_lines[0]
        bottom_line = horizontal_lines[-1]
        left_line = vertical_lines[0]
        right_line = vertical_lines[-1]

        # Находим углы как пересечения линий
        def line_intersection(line1, line2):
            x1, y1, x2, y2 = line1
            x3, y3, x4, y4 = line2

            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-6:
                return None

            px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
            py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

            return (int(px), int(py))

        tl = line_intersection(top_line, left_line)
        tr = line_intersection(top_line, right_line)
        br = line_intersection(bottom_line, right_line)
        bl = line_intersection(bottom_line, left_line)

        if None in [tl, tr, br, bl]:
            logger.debug("Hough lines: не удалось найти все пересечения")
            return None

        # Проверяем, что углы формируют разумный четырехугольник
        corners = np.array([tl, tr, br, bl], dtype=np.float32).reshape(4, 2)
        area = cv2.contourArea(corners)
        area_ratio = area / img_area

        if area_ratio < self.config.min_page_area_ratio:
            logger.debug(f"Hough lines: слишком маленькая площадь ({area_ratio:.2f})")
            return None

        logger.debug(f"Hough lines: найден документ с площадью {area_ratio:.2f}")
        return corners.reshape(4, 1, 2).astype(np.int32)

    def _refine_corners(
        self,
        corners: np.ndarray,
        img_gray: np.ndarray,
    ) -> np.ndarray:
        """
        Уточнить положение углов используя cornerSubPix.
        """
        if not self.config.refine_corners:
            return corners

        corners_float = corners.reshape(4, 2).astype(np.float32)

        # Критерии для cornerSubPix
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        winSize = (5, 5)
        zeroZone = (-1, -1)

        try:
            refined = cv2.cornerSubPix(img_gray, corners_float, winSize, zeroZone, criteria)
            logger.debug("Углы уточнены с помощью cornerSubPix")
            return refined
        except Exception as e:
            logger.debug(f"Не удалось уточнить углы: {e}")
            return corners_float

    def _apply_crop_margin(
        self,
        corners: np.ndarray,
        margin_px: int,
    ) -> np.ndarray:
        """
        Применить отступ внутрь от границ (чтобы избежать артефактов по краям).
        """
        if margin_px <= 0:
            return corners

        # Находим центр четырехугольника
        center = np.mean(corners, axis=0)

        # Сдвигаем каждый угол к центру на margin_px пикселей
        corners_adjusted = corners.copy()
        for i in range(4):
            vec = center - corners[i]
            vec_len = np.linalg.norm(vec)
            if vec_len > 0:
                vec_normalized = vec / vec_len
                corners_adjusted[i] = corners[i] + vec_normalized * margin_px

        return corners_adjusted

    # ---------- Методы для deskew-first подхода ----------

    def _detect_skew_angle(
        self,
        img_gray: np.ndarray,
        debug_dir: Optional[Path] = None,
    ) -> float:
        """
        Определить угол перекоса документа используя Hough lines.
        Возвращает угол в градусах (положительный = против часовой стрелки).
        """
        cfg = self.config
        h, w = img_gray.shape[:2]

        # Обнаружение границ
        edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)

        # Hough lines
        min_line_length = int(min(w, h) * cfg.deskew_min_line_length_ratio)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=80,
            minLineLength=min_line_length,
            maxLineGap=10
        )

        if lines is None or len(lines) == 0:
            logger.debug("Deskew: линии не найдены")
            return 0.0

        # Вычисляем углы всех линий
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

            # Нормализуем к диапазону [-45, 45] (предполагаем небольшой перекос)
            if angle < -45:
                angle += 90
            elif angle > 45:
                angle -= 90

            angles.append(angle)

        if not angles:
            return 0.0

        # Используем медиану для устойчивости к выбросам
        median_angle = np.median(angles)

        logger.debug(f"Deskew: обнаружен угол перекоса {median_angle:.2f}°")

        # Визуализация для отладки
        if self.debug and debug_dir:
            lines_vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(lines_vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
            self._debug_save_cv2(debug_dir, "01a_deskew_lines.png", lines_vis)

        return median_angle

    def _rotate_image(
        self,
        img: np.ndarray,
        angle: float,
        debug_dir: Optional[Path] = None,
    ) -> np.ndarray:
        """
        Повернуть изображение на заданный угол.
        """
        if abs(angle) < self.config.deskew_angle_threshold:
            logger.debug(f"Deskew: угол {angle:.2f}° слишком мал, пропускаем поворот")
            return img

        h, w = img.shape[:2]
        center = (w // 2, h // 2)

        # Матрица поворота
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Вычисляем новый размер чтобы вместить повернутое изображение
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)

        # Корректируем матрицу для нового размера
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2

        # Поворачиваем с белым фоном
        rotated = cv2.warpAffine(img, M, (new_w, new_h),
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(255, 255, 255))

        logger.debug(f"Deskew: повернуто на {angle:.2f}°")
        return rotated

    def _detect_fold_lines(
        self,
        img_gray: np.ndarray,
        debug_dir: Optional[Path] = None,
    ) -> list:
        """
        Обнаружить линии складок на документе.
        Возвращает список складок как [(x1,y1,x2,y2), ...].
        """
        h, w = img_gray.shape[:2]

        # Усиленное обнаружение границ для поиска складок
        blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 90)

        # Морфология для выделения вертикальных/горизонтальных структур
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))

        edges_h = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_h)
        edges_v = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_v)
        edges_combined = cv2.bitwise_or(edges_h, edges_v)

        # Hough lines для поиска длинных прямых линий (потенциальные складки)
        lines = cv2.HoughLinesP(
            edges_combined,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=min(w, h) // 3,  # Минимум треть размера
            maxLineGap=50
        )

        if lines is None:
            return []

        # Фильтруем линии: оставляем только почти вертикальные или горизонтальные
        fold_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Вычисляем угол
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            # Горизонтальные (около 0° или 180°) или вертикальные (около 90°)
            is_horizontal = angle < 15 or angle > 165
            is_vertical = 75 < angle < 105

            # Проверяем что линия проходит через значительную часть изображения
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            min_length = min(w, h) * 0.4

            if (is_horizontal or is_vertical) and length > min_length:
                # Проверяем что линия проходит ближе к центру (не по краям)
                if is_vertical:
                    x_pos = (x1 + x2) / 2
                    if 0.3 * w < x_pos < 0.7 * w:
                        fold_lines.append((x1, y1, x2, y2))
                elif is_horizontal:
                    y_pos = (y1 + y2) / 2
                    if 0.3 * h < y_pos < 0.7 * h:
                        fold_lines.append((x1, y1, x2, y2))

        if fold_lines:
            logger.debug(f"Обнаружено складок: {len(fold_lines)}")

        # Визуализация
        if self.debug and debug_dir and fold_lines:
            fold_vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
            for x1, y1, x2, y2 in fold_lines:
                cv2.line(fold_vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
            self._debug_save_cv2(debug_dir, "01b_fold_lines.png", fold_vis)

        return fold_lines

    def _find_document_boundary(
        self,
        img_gray: np.ndarray,
        has_folds: bool,
        debug_dir: Optional[Path] = None,
    ) -> Optional[np.ndarray]:
        """
        Найти внешнюю границу документа, учитывая возможные складки.
        Возвращает 4 угловых точки или None.
        """
        h, w = img_gray.shape[:2]
        img_area = w * h

        # Адаптивная бинаризация для отделения документа от фона
        binary = cv2.adaptiveThreshold(
            img_gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,
            10
        )

        # Инвертируем если фон светлый
        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)

        self._debug_save_cv2(debug_dir, "01c_binary.png", binary)

        # Морфология для закрытия дыр и соединения разрывов
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Заполнение дыр
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            logger.debug("Boundary: контуры не найдены")
            return None

        # Находим самый большой контур (предполагаем это документ)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        area_ratio = area / img_area

        logger.debug(f"Boundary: largest contour area ratio = {area_ratio:.3f}")

        if area_ratio < 0.1:
            logger.debug("Boundary: контур слишком маленький")
            return None

        # Визуализация
        if self.debug and debug_dir:
            contour_vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(contour_vis, [largest_contour], -1, (0, 255, 0), 3)
            self._debug_save_cv2(debug_dir, "01d_largest_contour.png", contour_vis)

        # Если есть складки, используем выпуклую оболочку для внешней границы
        if has_folds:
            hull = cv2.convexHull(largest_contour)
            largest_contour = hull
            logger.debug("Boundary: используется convex hull из-за складок")

        # Аппроксимируем контур до 4 точек
        peri = cv2.arcLength(largest_contour, True)

        # Пробуем разные epsilon для получения 4 точек
        for epsilon_coef in [0.01, 0.02, 0.03, 0.04, 0.05]:
            approx = cv2.approxPolyDP(largest_contour, epsilon_coef * peri, True)
            if len(approx) == 4:
                logger.debug(f"Boundary: найдено 4 точки с epsilon={epsilon_coef}")
                return approx

        # Если не удалось получить 4 точки, используем bounding rectangle
        logger.debug("Boundary: не удалось аппроксимировать до 4 точек, используем bounding rect")
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        return box.astype(np.int32).reshape(4, 1, 2)

    def _simple_crop_by_boundary(
        self,
        img: np.ndarray,
        corners: np.ndarray,
    ) -> np.ndarray:
        """
        Простая обрезка по границам без perspective transform.
        Используется для документов со складками.
        """
        # Находим bounding rectangle
        x, y, w, h = cv2.boundingRect(corners)

        # Добавляем небольшие отступы если есть место
        margin = 10
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(img.shape[1] - x, w + 2 * margin)
        h = min(img.shape[0] - y, h + 2 * margin)

        cropped = img[y:y+h, x:x+w]
        logger.debug(f"Simple crop: {w}x{h}")
        return cropped

    def detect_and_warp_document(
        self,
        pil_img: Image.Image,
        debug_dir: Optional[Path] = None,
    ) -> Image.Image:
        """
        Улучшенное обнаружение и выпрямление документа с многоуровневым подходом.

        Стратегия (в зависимости от настроек):

        Deskew-first подход (use_deskew_first=True, рекомендуется):
        1. Deskew - выравнивание документа по доминирующему углу
        2. Fold detection - обнаружение складок
        3. Boundary detection - поиск внешней границы (с учетом складок)
        4. Adaptive cropping - perspective transform или simple crop в зависимости от складок

        Enhanced подход (use_enhanced_detection=True):
        1. Enhanced edge detection (адаптивная бинаризация + морфология)
        2. Contour selection with scoring (оценка по множеству критериев)
        3. Hough lines fallback (если контуры не найдены)
        4. Corner refinement (уточнение углов)
        5. Optional margin (отступ от краёв)

        Simple подход (legacy):
        Базовое обнаружение контуров без улучшений
        """
        cfg = self.config

        # === DESKEW-FIRST APPROACH ===
        if cfg.use_deskew_first:
            return self._detect_and_warp_deskew_first(pil_img, debug_dir)

        # === ENHANCED APPROACH ===
        if cfg.use_enhanced_detection:
            return self._detect_and_warp_enhanced(pil_img, debug_dir)

        # === SIMPLE APPROACH (legacy) ===
        return self._detect_and_warp_simple(pil_img, debug_dir)

    def _detect_and_warp_deskew_first(
        self,
        pil_img: Image.Image,
        debug_dir: Optional[Path] = None,
    ) -> Image.Image:
        """
        Deskew-first подход: сначала выравнивание, потом обнаружение границ.
        Хорошо работает с документами со складками.
        """
        cfg = self.config

        img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        orig_h, orig_w = img_cv.shape[:2]

        logger.debug(f"detect_and_warp (deskew-first): input size {orig_w}x{orig_h}")
        self._debug_save_cv2(debug_dir, "00_input.jpg", img_cv)

        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # STAGE 1: Detect and correct skew
        skew_angle = self._detect_skew_angle(gray, debug_dir)
        img_deskewed = self._rotate_image(img_cv, skew_angle, debug_dir)
        gray_deskewed = cv2.cvtColor(img_deskewed, cv2.COLOR_BGR2GRAY)

        self._debug_save_cv2(debug_dir, "01_deskewed.jpg", img_deskewed)

        # STAGE 2: Detect folds
        fold_lines = []
        if cfg.detect_folds:
            fold_lines = self._detect_fold_lines(gray_deskewed, debug_dir)

        has_folds = len(fold_lines) > 0
        if has_folds:
            logger.info(f"Обнаружен документ со складками ({len(fold_lines)} складок)")

        # STAGE 3: Find document boundary
        boundary = self._find_document_boundary(gray_deskewed, has_folds, debug_dir)

        if boundary is None:
            logger.debug("Deskew-first: boundary not found, returning original")
            return pil_img

        # Визуализация границы
        if self.debug and debug_dir:
            boundary_vis = img_deskewed.copy()
            cv2.drawContours(boundary_vis, [boundary], -1, (0, 255, 0), 5)
            for pt in boundary.reshape(4, 2):
                cv2.circle(boundary_vis, tuple(pt.astype(int)), 10, (0, 0, 255), -1)
            self._debug_save_cv2(debug_dir, "01e_detected_boundary.png", boundary_vis)

        # STAGE 4: Adaptive cropping strategy
        if has_folds and cfg.simple_crop_if_folded:
            # Простая обрезка для документов со складками
            logger.debug("Using simple crop (document has folds)")
            cropped = self._simple_crop_by_boundary(img_deskewed, boundary)
        else:
            # Perspective transform для ровных документов
            logger.debug("Using perspective transform")

            pts = boundary.reshape(4, 2).astype("float32")

            # Corner refinement
            if cfg.refine_corners:
                pts = self._refine_corners(pts, gray_deskewed)

            # Apply margin
            if cfg.crop_margin_px > 0:
                pts = self._apply_crop_margin(pts, cfg.crop_margin_px)

            # Order points and calculate target size
            rect = self.order_points(pts)
            (tl, tr, br, bl) = rect

            widthA = np.linalg.norm(br - bl)
            widthB = np.linalg.norm(tr - tl)
            maxWidth = int(max(widthA, widthB))

            heightA = np.linalg.norm(tr - br)
            heightB = np.linalg.norm(tl - bl)
            maxHeight = int(max(heightA, heightB))

            logger.debug(f"Perspective transform target size: {maxWidth}x{maxHeight}")

            dst = np.array(
                [
                    [0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1],
                ],
                dtype="float32",
            )

            try:
                M = cv2.getPerspectiveTransform(rect, dst)
                cropped = cv2.warpPerspective(img_deskewed, M, (maxWidth, maxHeight))
            except Exception as e:
                logger.warning(f"Perspective transform failed: {e}")
                return pil_img

        self._debug_save_cv2(debug_dir, "02_final_cropped.jpg", cropped)

        # Convert back to PIL
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        pil_result = Image.fromarray(cropped_rgb)

        return pil_result

    def _detect_and_warp_enhanced(
        self,
        pil_img: Image.Image,
        debug_dir: Optional[Path] = None,
    ) -> Image.Image:
        """
        Enhanced подход: улучшенное обнаружение контуров с множественными стратегиями.
        """
        cfg = self.config

        # Выбор алгоритма: старый (простой) или новый (улучшенный)
        if not cfg.use_enhanced_detection:
            return self._detect_and_warp_simple(pil_img, debug_dir)

        img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        orig_h, orig_w = img_cv.shape[:2]

        # Downscale для ускорения обработки
        long_side = max(orig_w, orig_h)
        scale = 1.0
        if long_side > cfg.max_proc_dim:
            scale = cfg.max_proc_dim / long_side
            proc_w = int(orig_w * scale)
            proc_h = int(orig_h * scale)
            img_small = cv2.resize(img_cv, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        else:
            img_small = img_cv.copy()
            proc_w, proc_h = orig_w, orig_h

        logger.debug(
            f"detect_and_warp_document (enhanced): orig=({orig_w}x{orig_h}), "
            f"proc=({proc_w}x{proc_h}), scale={scale:.3f}"
        )
        self._debug_save_cv2(debug_dir, "01_input_resized.jpg", img_small)

        gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
        img_area = proc_w * proc_h

        # STAGE 1: Enhanced edge detection
        edges = self._enhanced_edge_detection(gray, debug_dir)

        # STAGE 2: Find best contour with scoring
        page_contour = self._find_best_contour(edges, gray, img_area, debug_dir)

        # STAGE 3: Hough lines fallback
        if page_contour is None and cfg.use_hough_fallback:
            logger.debug("Contour detection failed, trying Hough lines...")
            page_contour = self._detect_document_by_hough(edges, (proc_h, proc_w), debug_dir)

        # Если ничего не найдено, возвращаем оригинал
        if page_contour is None:
            logger.debug("Document detection failed, returning original image.")
            return pil_img

        # Визуализация найденного контура
        contours_vis = img_small.copy()
        cv2.drawContours(contours_vis, [page_contour], -1, (0, 255, 0), 3)
        for pt in page_contour.reshape(4, 2):
            cv2.circle(contours_vis, tuple(pt.astype(int)), 5, (0, 0, 255), -1)
        self._debug_save_cv2(debug_dir, "03_page_contour_detected.png", contours_vis)

        # Масштабируем контур обратно к исходному размеру
        pts_small = page_contour.reshape(4, 2).astype("float32")
        pts = pts_small / scale

        # STAGE 4: Corner refinement
        # Уточняем на исходном изображении (не на уменьшенном)
        gray_orig = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        pts = self._refine_corners(pts, gray_orig)

        # STAGE 5: Apply crop margin
        pts = self._apply_crop_margin(pts, cfg.crop_margin_px)

        # Упорядочиваем точки (TL, TR, BR, BL)
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect

        # Вычисляем размер целевого прямоугольника
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = int(max(widthA, widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = int(max(heightA, heightB))

        logger.debug(f"Target rect size: {maxWidth}x{maxHeight}")

        # Целевой прямоугольник
        dst = np.array(
            [
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1],
            ],
            dtype="float32",
        )

        # STAGE 6: Warp perspective
        try:
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(img_cv, M, (maxWidth, maxHeight))
        except Exception as e:
            logger.warning(f"Perspective transform failed: {e}. Returning original image.")
            return pil_img

        self._debug_save_cv2(debug_dir, "04_warped_color.jpg", warped)

        warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        pil_warped = Image.fromarray(warped_rgb)
        self._debug_save_pil(debug_dir, "05_warped_pil_rgb.jpg", pil_warped)
        return pil_warped

    def _detect_and_warp_simple(
        self,
        pil_img: Image.Image,
        debug_dir: Optional[Path] = None,
    ) -> Image.Image:
        """
        Старый простой алгоритм обнаружения (для обратной совместимости).
        Используется если use_enhanced_detection=False.
        """
        cfg = self.config

        img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        orig_h, orig_w = img_cv.shape[:2]

        long_side = max(orig_w, orig_h)
        scale = 1.0
        if long_side > cfg.max_proc_dim:
            scale = cfg.max_proc_dim / long_side
            proc_w = int(orig_w * scale)
            proc_h = int(orig_h * scale)
            img_small = cv2.resize(img_cv, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        else:
            img_small = img_cv.copy()
            proc_w, proc_h = orig_w, orig_h

        logger.debug(
            f"detect_and_warp_document (simple): orig=({orig_w}x{orig_h}), "
            f"proc=({proc_w}x{proc_h}), scale={scale:.3f}"
        )
        self._debug_save_cv2(debug_dir, "01_input_resized.jpg", img_small)

        gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray_blur, cfg.canny_threshold1, cfg.canny_threshold2)
        self._debug_save_cv2(debug_dir, "02_edges.png", edges)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        img_area = proc_w * proc_h
        page_contour = None

        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, cfg.contour_epsilon_coef * peri, True)
            if len(approx) != 4:
                continue

            area = cv2.contourArea(approx)
            if area <= 0:
                continue

            area_ratio = area / img_area
            if area_ratio < cfg.min_page_area_ratio:
                continue

            x, y, w, h = cv2.boundingRect(approx)
            rect_area = w * h if w > 0 and h > 0 else 1
            rectangularity = float(area) / rect_area

            if rectangularity < cfg.min_rectangularity:
                continue

            page_contour = approx
            logger.debug(
                f"Found contour: area_ratio={area_ratio:.3f}, rectangularity={rectangularity:.3f}"
            )
            break

        if page_contour is None:
            logger.debug("No page contour found, returning original image.")
            return pil_img

        contours_vis = img_small.copy()
        cv2.drawContours(contours_vis, [page_contour], -1, (0, 255, 0), 2)
        self._debug_save_cv2(debug_dir, "03_page_contour.png", contours_vis)

        pts_small = page_contour.reshape(4, 2).astype("float32")
        pts = pts_small / scale

        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect

        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = int(max(widthA, widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = int(max(heightA, heightB))

        logger.debug(f"Target rect size: {maxWidth}x{maxHeight}")

        dst = np.array(
            [
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1],
            ],
            dtype="float32",
        )

        try:
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(img_cv, M, (maxWidth, maxHeight))
        except Exception as e:
            logger.warning(f"Perspective transform error: {e}. Returning original image.")
            return pil_img

        self._debug_save_cv2(debug_dir, "04_warped_color.jpg", warped)

        warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        pil_warped = Image.fromarray(warped_rgb)
        self._debug_save_pil(debug_dir, "05_warped_pil_rgb.jpg", pil_warped)
        return pil_warped

    def to_grayscale(self, img: Image.Image, debug_dir: Optional[Path] = None) -> Image.Image:
        gray = img.convert("L")
        self._debug_save_pil(debug_dir, "06_gray.jpg", gray)
        return gray

    def resize_long_side(self, img: Image.Image, debug_dir: Optional[Path] = None) -> Image.Image:
        cfg = self.config
        w, h = img.size
        long_side = max(w, h)
        if long_side > cfg.target_long_side_px:
            scale = cfg.target_long_side_px / long_side
            new_w = int(w * scale)
            new_h = int(h * scale)
            logger.debug(f"Resize: from {w}x{h} to {new_w}x{new_h}")
            img = img.resize((new_w, new_h), Image.LANCZOS)
        self._debug_save_pil(debug_dir, "07_gray_resized.jpg", img)
        return img

    def enhance_contrast_and_denoise(
        self,
        img: Image.Image,
        debug_dir: Optional[Path] = None,
    ) -> Image.Image:
        cfg = self.config
        enhancer = ImageEnhance.Contrast(img)
        img_enh = enhancer.enhance(cfg.contrast_factor)
        if cfg.median_filter_size and cfg.median_filter_size > 1:
            img_enh = img_enh.filter(ImageFilter.MedianFilter(size=cfg.median_filter_size))
        self._debug_save_pil(debug_dir, "08_contrast_denoise.jpg", img_enh)
        return img_enh

    @staticmethod
    def otsu_threshold(gray_array: np.ndarray) -> int:
        hist, _ = np.histogram(gray_array.flatten(), bins=256, range=(0, 256))
        total = gray_array.size

        bin_indices = np.arange(256)
        sum_total = np.dot(hist, bin_indices)

        sum_b = 0.0
        weight_b = 0.0
        max_between_var = 0.0
        threshold = 0

        for t in range(256):
            weight_b += hist[t]
            if weight_b == 0:
                continue

            weight_f = total - weight_b
            if weight_f == 0:
                break

            sum_b += t * hist[t]

            mean_b = sum_b / weight_b
            mean_f = (sum_total - sum_b) / weight_f

            between_var = weight_b * weight_f * (mean_b - mean_f) ** 2
            if between_var > max_between_var:
                max_between_var = between_var
                threshold = t

        return threshold

    def binarize_otsu(self, img: Image.Image, debug_dir: Optional[Path] = None) -> Image.Image:
        gray_np = np.array(img, dtype=np.uint8)
        t = self.otsu_threshold(gray_np)
        logger.debug(f"Otsu threshold = {t}")
        binary_np = (gray_np > t).astype(np.uint8) * 255
        bin_img = Image.fromarray(binary_np, mode="L")
        self._debug_save_pil(debug_dir, "09_binary_otsu.jpg", bin_img)
        return bin_img

    def binarize_adaptive(self, img: Image.Image, debug_dir: Optional[Path] = None) -> Image.Image:
        cfg = self.config
        gray_np = np.array(img, dtype=np.uint8)
        block_size = cfg.adaptive_block_size
        if block_size % 2 == 0:
            block_size += 1
        if block_size < 3:
            block_size = 3

        binary_np = cv2.adaptiveThreshold(
            gray_np,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            cfg.adaptive_C,
        )
        bin_img = Image.fromarray(binary_np, mode="L")
        self._debug_save_pil(debug_dir, "09_binary_adaptive.jpg", bin_img)
        return bin_img

    def postprocess_binary(self, img: Image.Image, debug_dir: Optional[Path] = None) -> Image.Image:
        cfg = self.config
        arr = np.array(img, dtype=np.uint8)

        if cfg.morph_open_ksize and cfg.morph_open_ksize > 1:
            k = cfg.morph_open_ksize
            kernel = np.ones((k, k), np.uint8)
            arr = cv2.morphologyEx(arr, cv2.MORPH_OPEN, kernel)

        if cfg.morph_close_ksize and cfg.morph_close_ksize > 1:
            k = cfg.morph_close_ksize
            kernel = np.ones((k, k), np.uint8)
            arr = cv2.morphologyEx(arr, cv2.MORPH_CLOSE, kernel)

        out = Image.fromarray(arr, mode="L")
        self._debug_save_pil(debug_dir, "10_binary_morph.jpg", out)
        return out

    def sharpen(self, img: Image.Image, debug_dir: Optional[Path] = None) -> Image.Image:
        cfg = self.config
        sharp = img.filter(
            ImageFilter.UnsharpMask(
                radius=cfg.sharpen_radius,
                percent=cfg.sharpen_percent,
                threshold=cfg.sharpen_threshold,
            )
        )
        self._debug_save_pil(debug_dir, "11_binary_sharpened.jpg", sharp)
        return sharp

    def save_for_ocr(self, img: Image.Image, output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, dpi=(self.config.dpi, self.config.dpi))

    # ---------- Утилиты ----------

    @staticmethod
    def order_points(pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
        return rect

    @staticmethod
    def make_output_path(
        input_path: str | Path,
        output_dir: str | Path | None = None,
        suffix: str = "_ocr",
    ) -> Path:
        input_path = Path(input_path)
        if output_dir is None:
            return input_path.with_name(f"{input_path.stem}{suffix}{input_path.suffix}")
        else:
            output_dir = Path(output_dir)
            return output_dir / input_path.name

    # ---------- Высокоуровневые методы ----------

    def process_image(self, img: Image.Image, debug_dir: Optional[Path] = None) -> Image.Image:
        img = self.detect_and_warp_document(img, debug_dir=debug_dir)
        img = self.to_grayscale(img, debug_dir=debug_dir)
        img = self.resize_long_side(img, debug_dir=debug_dir)
        img = self.enhance_contrast_and_denoise(img, debug_dir=debug_dir)

        if self.config.binarization_method.lower() == "adaptive":
            img = self.binarize_adaptive(img, debug_dir=debug_dir)
        else:
            img = self.binarize_otsu(img, debug_dir=debug_dir)

        img = self.postprocess_binary(img, debug_dir=debug_dir)
        img = self.sharpen(img, debug_dir=debug_dir)
        return img

    def preprocess_file(self, input_path: str | Path, output_path: str | Path) -> None:
        input_path = Path(input_path)
        output_path = Path(output_path)

        debug_dir: Optional[Path] = None
        if self.debug:
            debug_dir = output_path.parent / f"{output_path.stem}_debug"

        try:
            img = self.load_and_fix_exif(input_path, debug_dir=debug_dir)
            img = self.process_image(img, debug_dir=debug_dir)
            self.save_for_ocr(img, output_path)
        except DocumentProcessingError:
            raise
        except Exception as e:
            raise DocumentProcessingError(f"Ошибка при обработке {input_path}: {e}") from e

    def _iter_input_files(
        self,
        input_dir: Path,
        patterns: Iterable[str],
        recursive: bool,
    ) -> Iterable[Path]:
        if recursive:
            for pattern in patterns:
                yield from input_dir.rglob(pattern)
        else:
            for pattern in patterns:
                yield from input_dir.glob(pattern)

    def preprocess_directory(
        self,
        input_dir: str | Path,
        output_dir: str | Path | None = None,
        suffix: str = "_ocr",
        patterns: Tuple[str, ...] = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp"),
        recursive: bool = False,
        skip_existing: bool = False,
    ) -> Tuple[int, int]:
        input_dir = Path(input_dir)
        if not input_dir.is_dir():
            raise DocumentProcessingError(f"{input_dir} не является директорией")

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(self._iter_input_files(input_dir, patterns, recursive=recursive))
        if not files:
            logger.warning(f"В директории {input_dir} не найдено файлов по маскам {patterns}")
            return 0, 0

        logger.info(f"Найдено файлов для обработки: {len(files)}")

        success = 0
        failed = 0

        for in_path in files:
            out_path = self.make_output_path(in_path, output_dir=output_dir, suffix=suffix)

            if skip_existing and out_path.exists():
                logger.info(f"Пропуск существующего файла: {out_path}")
                continue

            try:
                logger.info(f"Обработка файла: {in_path} -> {out_path}")
                self.preprocess_file(in_path, out_path)
                success += 1
            except DocumentProcessingError as e:
                logger.error(f"Ошибка при обработке {in_path}: {e}")
                failed += 1

        logger.info(f"Готово. Успешно: {success}, ошибок: {failed}")
        return success, failed
