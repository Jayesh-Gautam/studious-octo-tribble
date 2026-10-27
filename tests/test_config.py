from aerofix.config import PROJECT_ROOT, load_config


def test_config_loads_with_required_sections():
    cfg = load_config()
    for key in ("seed", "paths", "split", "filters", "preprocess", "classify", "llm"):
        assert key in cfg


def test_paths_resolve_inside_project():
    cfg = load_config()
    for p in cfg["paths"].values():
        assert p.is_absolute()
        assert PROJECT_ROOT in p.parents


def test_split_years_do_not_overlap():
    s = load_config()["split"]
    assert s["train_max_year"] < s["val_year"] < s["test_year"]
