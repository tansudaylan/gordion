import importlib


def test_import_gordion_main_module():
    main = importlib.import_module('gordion.main')
    assert hasattr(main, 'main')
