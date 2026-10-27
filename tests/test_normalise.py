import math
from collections import Counter

import pytest

from aerofix.config import load_config
from aerofix.preprocess.abbrev import expand, load_abbrev
from aerofix.preprocess.ngram_lm import EOS, NGramLM, UNK
from aerofix.preprocess.normalise import Normaliser
from aerofix.preprocess.spell import SpellCorrector, damerau_levenshtein
from aerofix.preprocess.tokenize import tokenize

DOCS = [s.split() for s in [
    "CORROSION FOUND ON FRAME", "CORROSION FOUND ON SKIN", "CRACK FOUND ON FRAME",
    "REPLACED VALVE OPS CHECK GOOD", "REPLACED PUMP OPS CHECK GOOD", "LEAK FOUND AT VALVE",
] * 5]


def test_tokenize_keeps_domain_tokens():
    assert tokenize("R&R L/H VLV W/ P/N 65-123, [REF]. A/C 737-700 O-RING") == \
        ["R&R", "L/H", "VLV", "W/", "P/N", "65-123", "[REF]", "A/C", "737-700", "O-RING"]


def test_abbrev_file_loads_and_skips_ambiguous():
    table = load_abbrev(load_config()["project_root"] / load_config()["normalise"]["abbrev_path"])
    assert table["RPLCD"] == ["REPLACED"] and table["R&R"] == ["REMOVED", "AND", "REPLACED"]
    assert "LT" not in table and "BS" not in table  # ambiguous / keep-as-is
    assert expand(["RPLCD", "L/H", "VLV"], table) == ["REPLACED", "LEFT", "VALVE"]


@pytest.mark.parametrize("order,smoothing", [(1, "laplace"), (2, "laplace"), (3, "laplace"), (1, "kn"), (2, "kn"), (3, "kn")])
def test_lm_probabilities_sum_to_one(order, smoothing):
    lm = NGramLM(order, smoothing, unk_min_count=1).fit(DOCS)
    for ctx in [("CORROSION", "FOUND"), ("NEVER", "SEEN"), ("REPLACED",)]:
        total = sum(lm.prob(w, ctx) for w in lm.vocab)
        assert math.isclose(total, 1.0, rel_tol=1e-9), (ctx, total)


def test_kn_trigram_beats_unigram_on_seen_text():
    test = [["CORROSION", "FOUND", "ON", "FRAME"]]
    p1 = NGramLM(1, "kn", unk_min_count=1).fit(DOCS).perplexity(test)
    p3 = NGramLM(3, "kn", unk_min_count=1).fit(DOCS).perplexity(test)
    assert p3 < p1


def test_unknown_words_map_to_unk():
    lm = NGramLM(2, "kn", unk_min_count=1).fit(DOCS)
    assert lm.prob("ZZZZ", ("FOUND",)) == lm.prob(UNK, ("FOUND",)) > 0
    assert EOS in lm.vocab


def test_damerau_levenshtein():
    assert damerau_levenshtein("CORRISION", "CORROSION") == 1
    assert damerau_levenshtein("VLAVE", "VALVE") == 1  # transposition = 1 edit
    assert damerau_levenshtein("LIMTS", "LIMITS") == 1


def test_spell_corrects_and_protects():
    counts = Counter(t for d in DOCS for t in d)
    lm = NGramLM(3, "kn", unk_min_count=1).fit(DOCS)
    sc = SpellCorrector(counts, lm, protected={"VLV"}, dict_min_count=1)
    assert sc.correct(["CORRISION", "FOUND"]) == ["CORROSION", "FOUND"]
    assert sc.correct(["VLV", "P/N", "65-123", "ABC"]) == ["VLV", "P/N", "65-123", "ABC"]  # protected / symbols / short
    assert sc.correct(["DETACHABLE", "VALVE"]) == ["DETACHABLE", "VALVE"]  # real English word, rare in data


def test_stem_and_lemma():
    counts = Counter(t for d in DOCS for t in d)
    lm = NGramLM(2, "kn", unk_min_count=1).fit(DOCS)
    norm = Normaliser({}, SpellCorrector(counts, lm, dict_min_count=1), lm)
    assert norm.stem(["CRACKED", "VALVES", "[REF]"]) == ["CRACK", "VALV", "[REF]"]
    assert norm.lemmatise(["REPLACED", "VALVES", "[REF]"]) == ["REPLACE", "VALVE", "[REF]"]
