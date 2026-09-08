import gordion


def test_import_gordion_package():
    assert hasattr(gordion, '__file__')