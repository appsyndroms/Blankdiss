from src.analyze import bucket


def test_bucket_small_increase():
    assert bucket(0.10) == "0–0,25 pp"


def test_bucket_medium_increase():
    assert bucket(0.40) == "0,25–0,50 pp"


def test_bucket_large_increase():
    assert bucket(0.75) == "0,50–1,00 pp"


def test_bucket_very_large_increase():
    assert bucket(1.50) == ">1,00 pp"


def test_bucket_decrease():
    assert bucket(-0.20) == "Minskning"
