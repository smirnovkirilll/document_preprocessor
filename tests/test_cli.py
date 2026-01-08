import pytest
import sys
from unittest.mock import patch, MagicMock
from pathlib import Path
from document_preprocessor.cli import run_from_cli


@pytest.fixture
def mock_preprocessor():
    """Mock DocumentPreprocessor to avoid needing real images"""
    with patch("document_preprocessor.cli.DocumentPreprocessor") as mock_proc:
        mock_instance = MagicMock()
        mock_proc.return_value = mock_instance
        # Default: preprocess_directory returns (0 processed, 0 failed)
        mock_instance.preprocess_directory.return_value = (0, 0)
        yield mock_instance


# ============================================================================
# 1. BASIC ARGUMENT PARSING TESTS
# ============================================================================

def test_single_file_mode_basic(monkeypatch, tmp_path, mock_preprocessor):
    """Test basic single file processing with CLI args"""
    input_file = tmp_path / "input.jpg"
    output_file = tmp_path / "output.png"
    input_file.touch()

    test_args = [
        "document_preprocessor",
        "--input-file", str(input_file),
        "--output-file", str(output_file),
        "--profile", "default"
    ]

    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as exc_info:
        run_from_cli()

    assert exc_info.value.code == 0
    mock_preprocessor.preprocess_file.assert_called_once_with(str(input_file), str(output_file))


def test_directory_mode_with_output_dir(monkeypatch, tmp_path, mock_preprocessor):
    """Test directory processing with explicit output directory"""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    test_args = [
        "document_preprocessor",
        "--input-dir", str(input_dir),
        "--output-dir", str(output_dir),
    ]

    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as exc_info:
        run_from_cli()

    assert exc_info.value.code == 0
    mock_preprocessor.preprocess_directory.assert_called_once()
    call_kwargs = mock_preprocessor.preprocess_directory.call_args[1]
    assert call_kwargs["input_dir"] == str(input_dir)
    assert call_kwargs["output_dir"] == str(output_dir)


def test_directory_mode_with_suffix(monkeypatch, tmp_path, mock_preprocessor):
    """Test directory processing with suffix instead of output-dir"""
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    test_args = [
        "document_preprocessor",
        "--input-dir", str(input_dir),
        "--suffix", "_processed",
    ]

    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as exc_info:
        run_from_cli()

    assert exc_info.value.code == 0
    call_kwargs = mock_preprocessor.preprocess_directory.call_args[1]
    assert call_kwargs["suffix"] == "_processed"
    assert call_kwargs["output_dir"] is None


def test_profile_selection(monkeypatch, tmp_path, mock_preprocessor):
    """Test that profile is correctly selected"""
    input_file = tmp_path / "input.jpg"
    output_file = tmp_path / "output.png"
    input_file.touch()

    for profile in ["default", "dark", "shadows", "small_text", "small_text_hard", "cardiogram", "ultrasound"]:
        test_args = [
            "document_preprocessor",
            "--input-file", str(input_file),
            "--output-file", str(output_file),
            "--profile", profile,
        ]

        monkeypatch.setattr(sys, "argv", test_args)

        with patch("document_preprocessor.cli.PreprocessorConfig") as mock_config:
            mock_config.from_profile_and_env.return_value = MagicMock()

            with pytest.raises(SystemExit):
                run_from_cli()

            mock_config.from_profile_and_env.assert_called_once_with(profile=profile)
            mock_config.reset_mock()


def test_boolean_flags(monkeypatch, tmp_path, mock_preprocessor):
    """Test boolean flags: debug, verbose, recursive, skip-existing"""
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    test_args = [
        "document_preprocessor",
        "--input-dir", str(input_dir),
        "--debug",
        "--verbose",
        "--recursive",
        "--skip-existing",
    ]

    monkeypatch.setattr(sys, "argv", test_args)

    with patch("document_preprocessor.cli.DocumentPreprocessor") as mock_proc_class:
        mock_instance = MagicMock()
        mock_instance.preprocess_directory.return_value = (0, 0)
        mock_proc_class.return_value = mock_instance

        with pytest.raises(SystemExit):
            run_from_cli()

        # Check DocumentPreprocessor was created with debug=True
        assert mock_proc_class.call_args[1]["debug"] is True

        # Check preprocess_directory was called with correct flags
        call_kwargs = mock_instance.preprocess_directory.call_args[1]
        assert call_kwargs["recursive"] is True
        assert call_kwargs["skip_existing"] is True


def test_include_patterns(monkeypatch, tmp_path, mock_preprocessor):
    """Test --include patterns for file filtering"""
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    test_args = [
        "document_preprocessor",
        "--input-dir", str(input_dir),
        "--include", "*.jpg", "*.png",
    ]

    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit):
        run_from_cli()

    call_kwargs = mock_preprocessor.preprocess_directory.call_args[1]
    assert call_kwargs["patterns"] == ("*.jpg", "*.png")


def test_parameter_overrides(monkeypatch, tmp_path, mock_preprocessor):
    """Test CLI parameter overrides (contrast, binarization, etc.)"""
    input_file = tmp_path / "input.jpg"
    output_file = tmp_path / "output.png"
    input_file.touch()

    test_args = [
        "document_preprocessor",
        "--input-file", str(input_file),
        "--output-file", str(output_file),
        "--contrast", "1.8",
        "--binarization", "adaptive",
        "--long-side", "4000",
        "--median-filter-size", "5",
        "--sharpen-radius", "2.0",
        "--sharpen-percent", "200",
        "--sharpen-threshold", "3",
        "--morph-open", "2",
        "--morph-close", "4",
        "--adaptive-block-size", "45",
        "--adaptive-C", "12",
        "--dpi", "600",
    ]

    monkeypatch.setattr(sys, "argv", test_args)

    with patch("document_preprocessor.cli.DocumentPreprocessor") as mock_proc_class:
        mock_instance = MagicMock()
        mock_proc_class.return_value = mock_instance

        with pytest.raises(SystemExit):
            run_from_cli()

        # Get the config that was passed to DocumentPreprocessor
        config = mock_proc_class.call_args[1]["config"]

        assert config.contrast_factor == 1.8
        assert config.binarization_method == "adaptive"
        assert config.target_long_side_px == 4000
        assert config.median_filter_size == 5
        assert config.sharpen_radius == 2.0
        assert config.sharpen_percent == 200
        assert config.sharpen_threshold == 3
        assert config.morph_open_ksize == 2
        assert config.morph_close_ksize == 4
        assert config.adaptive_block_size == 45
        assert config.adaptive_C == 12
        assert config.dpi == 600


# ============================================================================
# 2. ENVIRONMENT VARIABLE TESTS
# ============================================================================

def test_env_input_output_file(monkeypatch, tmp_path, mock_preprocessor):
    """Test DOC_PREPROC_INPUT_FILE and DOC_PREPROC_OUTPUT_FILE"""
    input_file = tmp_path / "input.jpg"
    output_file = tmp_path / "output.png"
    input_file.touch()

    monkeypatch.setenv("DOC_PREPROC_INPUT_FILE", str(input_file))
    monkeypatch.setenv("DOC_PREPROC_OUTPUT_FILE", str(output_file))

    test_args = ["document_preprocessor"]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as exc_info:
        run_from_cli()

    assert exc_info.value.code == 0
    mock_preprocessor.preprocess_file.assert_called_once_with(str(input_file), str(output_file))


def test_env_input_output_dir(monkeypatch, tmp_path, mock_preprocessor):
    """Test DOC_PREPROC_INPUT_DIR and DOC_PREPROC_OUTPUT_DIR"""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    monkeypatch.setenv("DOC_PREPROC_INPUT_DIR", str(input_dir))
    monkeypatch.setenv("DOC_PREPROC_OUTPUT_DIR", str(output_dir))

    test_args = ["document_preprocessor"]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as exc_info:
        run_from_cli()

    assert exc_info.value.code == 0
    call_kwargs = mock_preprocessor.preprocess_directory.call_args[1]
    assert call_kwargs["input_dir"] == str(input_dir)
    assert call_kwargs["output_dir"] == str(output_dir)


def test_env_boolean_flags(monkeypatch, tmp_path, mock_preprocessor):
    """Test boolean environment variables"""
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    monkeypatch.setenv("DOC_PREPROC_INPUT_DIR", str(input_dir))
    monkeypatch.setenv("DOC_PREPROC_DEBUG", "true")
    monkeypatch.setenv("DOC_PREPROC_VERBOSE", "1")
    monkeypatch.setenv("DOC_PREPROC_RECURSIVE", "yes")
    monkeypatch.setenv("DOC_PREPROC_SKIP_EXISTING", "on")

    test_args = ["document_preprocessor"]
    monkeypatch.setattr(sys, "argv", test_args)

    with patch("document_preprocessor.cli.DocumentPreprocessor") as mock_proc_class:
        mock_instance = MagicMock()
        mock_instance.preprocess_directory.return_value = (0, 0)
        mock_proc_class.return_value = mock_instance

        with pytest.raises(SystemExit):
            run_from_cli()

        assert mock_proc_class.call_args[1]["debug"] is True
        call_kwargs = mock_instance.preprocess_directory.call_args[1]
        assert call_kwargs["recursive"] is True
        assert call_kwargs["skip_existing"] is True


def test_env_profile(monkeypatch, tmp_path, mock_preprocessor):
    """Test DOC_PREPROC_PROFILE environment variable"""
    input_file = tmp_path / "input.jpg"
    output_file = tmp_path / "output.png"
    input_file.touch()

    monkeypatch.setenv("DOC_PREPROC_INPUT_FILE", str(input_file))
    monkeypatch.setenv("DOC_PREPROC_OUTPUT_FILE", str(output_file))
    monkeypatch.setenv("DOC_PREPROC_PROFILE", "shadows")

    test_args = ["document_preprocessor"]
    monkeypatch.setattr(sys, "argv", test_args)

    with patch("document_preprocessor.cli.PreprocessorConfig") as mock_config:
        mock_config.from_profile_and_env.return_value = MagicMock()

        with pytest.raises(SystemExit):
            run_from_cli()

        mock_config.from_profile_and_env.assert_called_once_with(profile="shadows")


def test_env_suffix(monkeypatch, tmp_path, mock_preprocessor):
    """Test DOC_PREPROC_SUFFIX environment variable"""
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    monkeypatch.setenv("DOC_PREPROC_INPUT_DIR", str(input_dir))
    monkeypatch.setenv("DOC_PREPROC_SUFFIX", "_custom")

    test_args = ["document_preprocessor"]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit):
        run_from_cli()

    call_kwargs = mock_preprocessor.preprocess_directory.call_args[1]
    assert call_kwargs["suffix"] == "_custom"


def test_env_include_patterns(monkeypatch, tmp_path, mock_preprocessor):
    """Test DOC_PREPROC_INCLUDE with comma/space separated patterns"""
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    monkeypatch.setenv("DOC_PREPROC_INPUT_DIR", str(input_dir))
    monkeypatch.setenv("DOC_PREPROC_INCLUDE", "*.jpg, *.png *.tiff")

    test_args = ["document_preprocessor"]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit):
        run_from_cli()

    call_kwargs = mock_preprocessor.preprocess_directory.call_args[1]
    assert "*.jpg" in call_kwargs["patterns"]
    assert "*.png" in call_kwargs["patterns"]
    assert "*.tiff" in call_kwargs["patterns"]


# ============================================================================
# 3. CONFIGURATION PRIORITY TESTS
# ============================================================================

def test_cli_overrides_env_vars(monkeypatch, tmp_path, mock_preprocessor):
    """Test that CLI args have higher priority than environment variables"""
    input_file = tmp_path / "input.jpg"
    output_file = tmp_path / "output.png"
    input_file.touch()

    # Set env var for profile
    monkeypatch.setenv("DOC_PREPROC_PROFILE", "dark")

    # But CLI specifies different profile
    test_args = [
        "document_preprocessor",
        "--input-file", str(input_file),
        "--output-file", str(output_file),
        "--profile", "shadows",
    ]

    monkeypatch.setattr(sys, "argv", test_args)

    with patch("document_preprocessor.cli.PreprocessorConfig") as mock_config:
        mock_config.from_profile_and_env.return_value = MagicMock()

        with pytest.raises(SystemExit):
            run_from_cli()

        # CLI arg "shadows" should win over env var "dark"
        mock_config.from_profile_and_env.assert_called_once_with(profile="shadows")


def test_cli_overrides_env_boolean_flags(monkeypatch, tmp_path, mock_preprocessor):
    """Test that CLI boolean flags override environment variables"""
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    # Set env vars
    monkeypatch.setenv("DOC_PREPROC_INPUT_DIR", str(input_dir))
    monkeypatch.setenv("DOC_PREPROC_DEBUG", "false")

    # But CLI explicitly sets --debug
    test_args = [
        "document_preprocessor",
        "--input-dir", str(input_dir),
        "--debug",
    ]

    monkeypatch.setattr(sys, "argv", test_args)

    with patch("document_preprocessor.cli.DocumentPreprocessor") as mock_proc_class:
        mock_instance = MagicMock()
        mock_instance.preprocess_directory.return_value = (0, 0)
        mock_proc_class.return_value = mock_instance

        with pytest.raises(SystemExit):
            run_from_cli()

        # CLI --debug should override env var
        assert mock_proc_class.call_args[1]["debug"] is True


def test_default_profile_when_none_specified(monkeypatch, tmp_path, mock_preprocessor):
    """Test that 'default' profile is used when neither CLI nor env specify one"""
    input_file = tmp_path / "input.jpg"
    output_file = tmp_path / "output.png"
    input_file.touch()

    test_args = [
        "document_preprocessor",
        "--input-file", str(input_file),
        "--output-file", str(output_file),
    ]

    monkeypatch.setattr(sys, "argv", test_args)

    with patch("document_preprocessor.cli.PreprocessorConfig") as mock_config:
        mock_config.from_profile_and_env.return_value = MagicMock()

        with pytest.raises(SystemExit):
            run_from_cli()

        mock_config.from_profile_and_env.assert_called_once_with(profile="default")


def test_default_suffix_when_none_specified(monkeypatch, tmp_path, mock_preprocessor):
    """Test that '_ocr' suffix is used by default"""
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    test_args = [
        "document_preprocessor",
        "--input-dir", str(input_dir),
    ]

    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit):
        run_from_cli()

    call_kwargs = mock_preprocessor.preprocess_directory.call_args[1]
    assert call_kwargs["suffix"] == "_ocr"


def test_default_include_patterns(monkeypatch, tmp_path, mock_preprocessor):
    """Test default include patterns when none specified"""
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    test_args = [
        "document_preprocessor",
        "--input-dir", str(input_dir),
    ]

    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit):
        run_from_cli()

    call_kwargs = mock_preprocessor.preprocess_directory.call_args[1]
    patterns = call_kwargs["patterns"]
    # Default patterns should include common image formats
    assert "*.jpg" in patterns
    assert "*.png" in patterns
    assert "*.tiff" in patterns


# ============================================================================
# 4. ERROR HANDLING TESTS
# ============================================================================

def test_error_no_input_specified(monkeypatch):
    """Test error when no input file or directory is specified"""
    test_args = ["document_preprocessor"]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as exc_info:
        run_from_cli()

    # Should exit with error code (argparse error exits with 2)
    assert exc_info.value.code == 2


def test_error_missing_output_file_in_single_mode(monkeypatch, tmp_path):
    """Test error when --output-file is missing in single file mode"""
    input_file = tmp_path / "input.jpg"
    input_file.touch()

    test_args = [
        "document_preprocessor",
        "--input-file", str(input_file),
        # Missing --output-file
    ]

    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as exc_info:
        run_from_cli()

    assert exc_info.value.code == 2


def test_error_both_input_file_and_dir_env_vars(monkeypatch, tmp_path):
    """Test error when both DOC_PREPROC_INPUT_FILE and DOC_PREPROC_INPUT_DIR are set"""
    input_file = tmp_path / "input.jpg"
    input_dir = tmp_path / "input_dir"
    input_file.touch()
    input_dir.mkdir()

    monkeypatch.setenv("DOC_PREPROC_INPUT_FILE", str(input_file))
    monkeypatch.setenv("DOC_PREPROC_INPUT_DIR", str(input_dir))

    test_args = ["document_preprocessor"]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as exc_info:
        run_from_cli()

    assert exc_info.value.code == 2


def test_mutually_exclusive_input_file_and_dir(monkeypatch, tmp_path):
    """Test that --input-file and --input-dir are mutually exclusive"""
    input_file = tmp_path / "input.jpg"
    input_dir = tmp_path / "input_dir"
    input_file.touch()
    input_dir.mkdir()

    test_args = [
        "document_preprocessor",
        "--input-file", str(input_file),
        "--input-dir", str(input_dir),
    ]

    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as exc_info:
        run_from_cli()

    # argparse should catch this
    assert exc_info.value.code == 2


# ============================================================================
# 5. EXIT CODE TESTS
# ============================================================================

def test_exit_code_success(monkeypatch, tmp_path, mock_preprocessor):
    """Test exit code 0 on successful processing"""
    input_file = tmp_path / "input.jpg"
    output_file = tmp_path / "output.png"
    input_file.touch()

    test_args = [
        "document_preprocessor",
        "--input-file", str(input_file),
        "--output-file", str(output_file),
    ]

    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as exc_info:
        run_from_cli()

    assert exc_info.value.code == 0


def test_exit_code_on_processing_error(monkeypatch, tmp_path):
    """Test exit code 1 when DocumentProcessingError occurs"""
    from document_preprocessor.core import DocumentProcessingError

    input_file = tmp_path / "input.jpg"
    output_file = tmp_path / "output.png"
    input_file.touch()

    test_args = [
        "document_preprocessor",
        "--input-file", str(input_file),
        "--output-file", str(output_file),
    ]

    monkeypatch.setattr(sys, "argv", test_args)

    with patch("document_preprocessor.cli.DocumentPreprocessor") as mock_proc_class:
        mock_instance = MagicMock()
        mock_instance.preprocess_file.side_effect = DocumentProcessingError("Test error")
        mock_proc_class.return_value = mock_instance

        with pytest.raises(SystemExit) as exc_info:
            run_from_cli()

        assert exc_info.value.code == 1


def test_exit_code_on_directory_failures(monkeypatch, tmp_path):
    """Test exit code 1 when directory processing has failures"""
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    test_args = [
        "document_preprocessor",
        "--input-dir", str(input_dir),
    ]

    monkeypatch.setattr(sys, "argv", test_args)

    with patch("document_preprocessor.cli.DocumentPreprocessor") as mock_proc_class:
        mock_instance = MagicMock()
        # Return (5 processed, 3 failed)
        mock_instance.preprocess_directory.return_value = (5, 3)
        mock_proc_class.return_value = mock_instance

        with pytest.raises(SystemExit) as exc_info:
            run_from_cli()

        # Should exit with code 1 because there were failures
        assert exc_info.value.code == 1


def test_exit_code_zero_on_directory_success(monkeypatch, tmp_path, mock_preprocessor):
    """Test exit code 0 when directory processing succeeds with no failures"""
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    # Set return value: (10 processed, 0 failed)
    mock_preprocessor.preprocess_directory.return_value = (10, 0)

    test_args = [
        "document_preprocessor",
        "--input-dir", str(input_dir),
    ]

    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as exc_info:
        run_from_cli()

    assert exc_info.value.code == 0
