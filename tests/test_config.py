from document_preprocessor.core import PreprocessorConfig


def test_config_profile_and_env_override(monkeypatch):
    # Без окружения профиль default
    cfg = PreprocessorConfig.from_profile_and_env("default")
    default_contrast = cfg.contrast_factor

    # Перекрываем через env
    monkeypatch.setenv("DOC_PREPROC_CONTRAST_FACTOR", "2.0")
    cfg2 = PreprocessorConfig.from_profile_and_env("default")

    assert cfg2.contrast_factor == 2.0
    assert cfg2.contrast_factor != default_contrast
