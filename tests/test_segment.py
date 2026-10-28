from aerofix.segment.common import chunk_bounds, from_bio, join_tokens, seg_tokenize, spans, to_bio
from aerofix.segment.evaluate import span_scores, token_scores
from aerofix.segment.rules import predict, rule_labels


def test_seg_tokenize_keeps_punctuation_and_markers():
    assert seg_tokenize("DOOR LEAKING.<[01] R&R VALVE; P/N 65-12") == \
        ["DOOR", "LEAKING", ".", "<", "[01]", "R&R", "VALVE", ";", "P/N", "65-12"]


def test_bio_roundtrip():
    labels = ["PROBLEM", "PROBLEM", "ACTION", "REF", "REF", "ACTION"]
    assert to_bio(labels) == ["B-PROBLEM", "I-PROBLEM", "B-ACTION", "B-REF", "I-REF", "B-ACTION"]
    assert from_bio(to_bio(labels)) == labels
    assert from_bio(["O", "B-ACTION", "O"]) == ["OTHER", "ACTION", "ACTION"]  # O inherits


def test_spans():
    assert spans(["PROBLEM", "PROBLEM", "ACTION"]) == {(0, 2, "PROBLEM"), (2, 3, "ACTION")}
    assert spans([]) == set()


def test_rules_split_at_first_action_verb():
    toks, labs = rule_labels("L1 DOOR DAMPER LEAKING. REPLACED DAMPER PER AMM 52-10-00. OPS CHECK GOOD")
    assert labs[:5] == ["PROBLEM"] * 5
    assert labs[toks.index("REPLACED")] == "ACTION"
    assert [labs[toks.index("AMM")], labs[toks.index("52-10-00")]] == ["REF", "REF"]
    assert labs[toks.index("PER")] == "ACTION"  # connector stays with the action (guide rule 1)


def test_rules_spaced_r_and_r_and_markers():
    toks, labs = rule_labels("FLOOR SUPPORT CRACKED. R & R SUPPORT IAW SRM <[01] DONE 2022-104311")
    assert labs[toks.index("R")] == "ACTION" and labs[toks.index("SRM")] == "REF"
    assert labs[toks.index("[01]")] == "OTHER" and labs[toks.index("2022-104311")] == "OTHER"


def test_predict_matches_rule_labels_on_tokens():
    text = "CRACK FOUND ON FRAME. REPAIRED IAW SRM 53-00-50."
    assert predict([seg_tokenize(text)])[0] == rule_labels(text)[1]


def test_chunks_start_at_action_verbs():
    toks = seg_tokenize("VALVE LEAKING. REPLACED VALVE")
    assert chunk_bounds(toks) == [0, 3]


def test_metrics_perfect_and_boundary_error():
    gold = [["PROBLEM", "PROBLEM", "ACTION", "ACTION"]]
    assert token_scores(gold, gold)["PROBLEM"] == 1.0 and span_scores(gold, gold)["micro"] == 1.0
    off_by_one = [["PROBLEM", "ACTION", "ACTION", "ACTION"]]
    assert span_scores(gold, off_by_one)["micro"] == 0.0  # strict: both spans wrong
    assert 0 < token_scores(gold, off_by_one)["PROBLEM"] < 1  # token F1 gives partial credit


def test_join_tokens():
    assert join_tokens(["DOOR", "LEAKING", ".", "(", "OPEN", ")"]) == "DOOR LEAKING. (OPEN)"
