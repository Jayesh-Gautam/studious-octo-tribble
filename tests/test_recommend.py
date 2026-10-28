import numpy as np

from aerofix.recommend.actions import action_tokens, aggregate, canonicalise, extract_actions
from aerofix.recommend.references import global_rates, recommend

ABBREV = {"RPLCD": ["REPLACED"], "R&R": ["REMOVED", "AND", "REPLACED"]}


def acts(text):
    return extract_actions(action_tokens(text, ABBREV))


def test_extract_basic_and_head_noun():
    assert acts("REMOVED AND REPLACED LH AFT EMERGENCY LIGHT POWER PACK M1673 IAW AMM.") == ["REPLACE POWER PACK"]
    assert acts("SERVICED R1 DOOR ASSIST BOTTLE AS REQUIRED") == ["SERVICE ASSIST BOTTLE"]


def test_extract_abbreviations_and_spaced_r_and_r():
    assert acts("RPLCD VALVE") == ["REPLACE VALVE"]
    assert acts("R&R SEAT TRACK IAW SRM") == ["REPLACE SEAT TRACK"]
    assert acts("R & R EPSU AND BATTERY") == ["REPLACE EPSU"]


def test_object_stops_at_sentence_end_and_checks_ignored():
    assert acts("REPLACED BATTERY PACK ABOVE L-2 DOOR. OPERATIONS CHECK GOOD.") == ["REPLACE BATTERY PACK"]
    assert acts("OPS CHECK GOOD. INSPECTED WORK PERFORMED") == []


def test_verb_only_and_multiple_actions():
    assert acts("REPAIRED IAW SRM. LUBRICATED HINGE.") == ["REPAIR", "LUBRICATE HINGE"]


def test_canonicalise_and_aggregate():
    assert canonicalise(["REPLACE BATTERY PACKS"], {"BATTERY PACKS": "BATTERY PACK"}) == ["REPLACE BATTERY PACK"]
    out = aggregate([["REPLACE VALVE"], ["REPLACE VALVE", "ADJUST CABLE"], ["ADJUST CABLE"]], np.array([0.9, 0.8, 0.1]))
    assert out[0][0] == "REPLACE VALVE" and abs(out[0][1] - 1.7 / 1.8) < 1e-9


def test_specificity_demotes_generic_reference():
    corpus = [["SRM 51-40-02"]] * 50 + [["AMM 52-10-00"]] * 2 + [[]] * 48
    rates = global_rates(corpus)
    cases = [["SRM 51-40-02", "AMM 52-10-00"], ["SRM 51-40-02", "AMM 52-10-00"], ["SRM 51-40-02"]]
    w = np.ones(3)
    assert recommend(cases, w, rates, "count")[0][0] == "SRM 51-40-02"
    assert recommend(cases, w, rates, "specificity")[0][0] == "AMM 52-10-00"
