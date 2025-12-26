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

    def detect_and_warp_document(
        self,
        pil_img: Image.Image,
        debug_dir: Optional[Path] = None,
    ) -> Image.Image:
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
            f"detect_and_warp_document: orig=({orig_w}x{orig_h}), "
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
                f"Найден подходящий контур: area_ratio={area_ratio:.3f}, rectangularity={rectangularity:.3f}"
            )
            break

        if page_contour is None:
            logger.debug("Контур страницы не найден или не прошёл фильтры, возвращаем исходное изображение.")
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
            logger.warning(f"Ошибка при перспективном преобразовании: {e}. Возвращаем исходное изображение.")
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
