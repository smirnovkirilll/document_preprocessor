#!/usr/bin/env python3
# Требуемая версия Python: 3.13+

import argparse
import logging
import os
import sys
from typing import Tuple, Optional
from .core import (
    PreprocessorConfig,
    DocumentPreprocessor,
    DocumentProcessingError,
)


logger = logging.getLogger(__name__)


def run_from_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Препроцессинг фото документа для OCR (PIL + OpenCV)"
    )

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--input-file",
        help="одиночный входной файл (фото документа)",
    )
    group.add_argument(
        "--input-dir",
        help="директория с файлами для пакетной обработки",
    )

    parser.add_argument(
        "--output-file",
        help="выходной файл для одиночного режима (может браться из DOC_PREPROC_OUTPUT_FILE)",
    )
    parser.add_argument(
        "--output-dir",
        help="директория для результатов пакетной обработки; "
             "если не указана, файлы пишутся рядом с исходными с добавлением суффикса",
    )
    parser.add_argument(
        "--suffix",
        default=None,  # будем задавать по умолчанию "_ocr" после учёта окружения
        help="суффикс для имён файлов при пакетной обработке без output-dir (по умолчанию _ocr)",
    )
    parser.add_argument(
        "--include",
        nargs="+",
        default=None,
        help='маски файлов для пакетной обработки, например: --include "*.jpg" "*.png"',
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="рекурсивный обход поддиректорий при пакетной обработке",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="не перезаписывать уже существующие выходные файлы",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="включить режим отладки (доп. файлы)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="подробный лог",
    )

    # Профиль / базовый рецепт
    parser.add_argument(
        "--profile",
        choices=["default", "dark", "shadows", "small_text"],
        default=None,  # профиль может прийти из окружения DOC_PREPROC_PROFILE
        help="готовый набор настроек (default, dark, shadows, small_text)",
    )

    # Базовые настройки геометрии и бинаризации
    parser.add_argument(
        "--long-side",
        type=int,
        default=None,
        help="целевой размер длинной стороны (по умолчанию из профиля / 3500)",
    )
    parser.add_argument(
        "--binarization",
        choices=["otsu", "adaptive"],
        default=None,
        help="метод бинаризации (по умолчанию из профиля)",
    )

    # Настройки контраста и сглаживания
    parser.add_argument(
        "--contrast",
        type=float,
        default=None,
        help="коэффициент контраста (по умолчанию из профиля)",
    )
    parser.add_argument(
        "--median-filter-size",
        type=int,
        default=None,
        help="размер медианного фильтра (нечётное число, по умолчанию из профиля, 1 или 0 = выключить)",
    )

    # Настройки шарпинга
    parser.add_argument(
        "--sharpen-radius",
        type=float,
        default=None,
        help="радиус UnsharpMask (по умолчанию из профиля)",
    )
    parser.add_argument(
        "--sharpen-percent",
        type=int,
        default=None,
        help="процент UnsharpMask (по умолчанию из профиля)",
    )
    parser.add_argument(
        "--sharpen-threshold",
        type=int,
        default=None,
        help="порог UnsharpMask (по умолчанию из профиля)",
    )

    # Морфология
    parser.add_argument(
        "--morph-open",
        type=int,
        default=None,
        help="размер ядра для MORPH_OPEN (по умолчанию из профиля, None = не применять)",
    )
    parser.add_argument(
        "--morph-close",
        type=int,
        default=None,
        help="размер ядра для MORPH_CLOSE (по умолчанию из профиля, None = не применять)",
    )

    # Адаптивная бинаризация
    parser.add_argument(
        "--adaptive-block-size",
        type=int,
        default=None,
        help="размер окна для адаптивной бинаризации (нечётное число, по умолчанию из профиля)",
    )
    parser.add_argument(
        "--adaptive-C",
        type=int,
        default=None,
        help="константа C для адаптивной бинаризации (по умолчанию из профиля)",
    )

    # DPI
    parser.add_argument(
        "--dpi",
        type=int,
        default=None,
        help="DPI в метаданных сохраняемого изображения (по умолчанию из профиля / 300)",
    )

    args = parser.parse_args()

    # --------- чтение простых булевых флагов из окружения ---------

    def env_bool(name: str) -> Optional[bool]:
        val = os.getenv(name)
        if val is None:
            return None
        return val.strip().lower() in ("1", "true", "yes", "y", "on")

    env_debug = env_bool("DOC_PREPROC_DEBUG")
    if env_debug is not None and not args.debug:
        args.debug = env_debug

    env_verbose = env_bool("DOC_PREPROC_VERBOSE")
    if env_verbose is not None and not args.verbose:
        args.verbose = env_verbose

    env_recursive = env_bool("DOC_PREPROC_RECURSIVE")
    if env_recursive is not None and not args.recursive:
        args.recursive = env_recursive

    env_skip_existing = env_bool("DOC_PREPROC_SKIP_EXISTING")
    if env_skip_existing is not None and not args.skip_existing:
        args.skip_existing = env_skip_existing

    # Логирование
    logging.basicConfig(
        level=logging.DEBUG if args.verbose or args.debug else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # --------- ввод / вывод из окружения ---------

    # input-file / input-dir
    if not args.input_file and not args.input_dir:
        env_input_file = os.getenv("DOC_PREPROC_INPUT_FILE")
        env_input_dir = os.getenv("DOC_PREPROC_INPUT_DIR")

        if env_input_file and env_input_dir:
            parser.error(
                "Both DOC_PREPROC_INPUT_FILE and DOC_PREPROC_INPUT_DIR are set; use only one."
            )
        elif env_input_file:
            args.input_file = env_input_file
        elif env_input_dir:
            args.input_dir = env_input_dir
        else:
            parser.error(
                "Either --input-file/--input-dir or DOC_PREPROC_INPUT_FILE/DOC_PREPROC_INPUT_DIR must be set."
            )

    # output-file для одиночного режима
    if args.input_file and not args.output_file:
        env_output_file = os.getenv("DOC_PREPROC_OUTPUT_FILE")
        if env_output_file:
            args.output_file = env_output_file

    # output-dir для пакетного режима
    if args.input_dir and not args.output_dir:
        env_output_dir = os.getenv("DOC_PREPROC_OUTPUT_DIR")
        if env_output_dir:
            args.output_dir = env_output_dir

    # suffix
    if args.suffix is None:
        env_suffix = os.getenv("DOC_PREPROC_SUFFIX")
        args.suffix = env_suffix if env_suffix else "_ocr"

    # include
    if args.include is None:
        env_include = os.getenv("DOC_PREPROC_INCLUDE")
        if env_include:
            parts = []
            for token in env_include.replace(",", " ").split():
                token = token.strip()
                if token:
                    parts.append(token)
            args.include = parts if parts else None

    # profile (CLI > ENV > default)
    env_profile = os.getenv("DOC_PREPROC_PROFILE")
    if args.profile is not None:
        profile_name = args.profile
    elif env_profile:
        profile_name = env_profile
    else:
        profile_name = "default"

    # 1) базовый профиль + 2) переменные окружения
    cfg = PreprocessorConfig.from_profile_and_env(profile=profile_name)

    # 3) переопределения из CLI (самый высокий приоритет)
    if args.long_side is not None:
        cfg.target_long_side_px = args.long_side
    if args.binarization is not None:
        cfg.binarization_method = args.binarization

    if args.contrast is not None:
        cfg.contrast_factor = args.contrast
    if args.median_filter_size is not None:
        cfg.median_filter_size = args.median_filter_size

    if args.sharpen_radius is not None:
        cfg.sharpen_radius = args.sharpen_radius
    if args.sharpen_percent is not None:
        cfg.sharpen_percent = args.sharpen_percent
    if args.sharpen_threshold is not None:
        cfg.sharpen_threshold = args.sharpen_threshold

    if args.morph_open is not None:
        cfg.morph_open_ksize = args.morph_open
    if args.morph_close is not None:
        cfg.morph_close_ksize = args.morph_close

    if args.adaptive_block_size is not None:
        cfg.adaptive_block_size = args.adaptive_block_size
    if args.adaptive_C is not None:
        cfg.adaptive_C = args.adaptive_C

    if args.dpi is not None:
        cfg.dpi = args.dpi

    pre = DocumentPreprocessor(config=cfg, debug=args.debug)

    exit_code = 0

    try:
        if args.input_file:
            if not args.output_file:
                parser.error(
                    "--output-file обязательно при использовании --input-file "
                    "или задайте DOC_PREPROC_OUTPUT_FILE"
                )
            pre.preprocess_file(args.input_file, args.output_file)
        else:
            if args.include:
                patterns: Tuple[str, ...] = tuple(args.include)
            else:
                patterns = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp")

            _, failed = pre.preprocess_directory(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                suffix=args.suffix,
                patterns=patterns,
                recursive=args.recursive,
                skip_existing=args.skip_existing,
            )
            if failed > 0:
                exit_code = 1
    except DocumentProcessingError as e:
        logger.error(f"{e}")
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    run_from_cli()
