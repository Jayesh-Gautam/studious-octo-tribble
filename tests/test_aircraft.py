import pytest

from aerofix.data.aircraft import family_of, parse_variant


@pytest.mark.parametrize("make,model,family", [
    ("AIRBUS", "A321253NX", "A320"), ("AIRBUS", "A319114", "A320"),
    ("BOEING", "7378H4", "B737"), ("BOEING", "737", "B737"),
    ("BOEING", "757222", None), ("AIRBUS", "A330243", None), ("EMB", "ERJ170", None),
])
def test_family_of(make, model, family):
    assert family_of(make, model) == family


@pytest.mark.parametrize("make,model,variant,gen", [
    ("BOEING", "7377H4", "737-700", "NG"),
    ("BOEING", "7378F2", "737-800", "NG"),
    ("BOEING", "737924ER", "737-900ER", "NG"),
    ("BOEING", "7379GPER", "737-900ER", "NG"),
    ("BOEING", "737300", "737-300", "Classic"),
    ("BOEING", "737205", "737-200", "Original"),
    ("BOEING", "7378", "737 MAX 8", "MAX"),
    ("BOEING", "7379", "737 MAX 9", "MAX"),
    ("BOEING", "7378200", "737 MAX 8-200", "MAX"),
    ("BOEING", "737*", "737 (unspecified)", "Unspecified"),
    ("BOEING", "737800*", "737-800", "NG"),
    ("AIRBUS", "A320232", "A320", "ceo"),
    ("AIRBUS", "A321271NX", "A321neo", "neo"),
    ("AIRBUS", "A320251N", "A320neo", "neo"),
    ("AIRBUS", "A320*", "A320 family (unspecified)", "Unspecified"),
])
def test_parse_variant(make, model, variant, gen):
    assert parse_variant(make, model) == (variant, gen)
