import importlib


def test_import_gordion_package():
    package = importlib.import_module('gordion')
    assert hasattr(package, '__file__')


def test_import_gordion_main_module():
    main = importlib.import_module('gordion.main')
    assert hasattr(main, 'main')
