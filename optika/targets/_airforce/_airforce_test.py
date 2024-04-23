import optika


def test_airforce():
    result = optika.targets.airforce("x", "y")
    assert result.ndim == 2
    assert result.sum() > 0
