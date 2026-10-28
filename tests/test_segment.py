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


def test_chunks_split_station_prefix_and_codes():
    toks = seg_tokenize("PVD - ROWS 10 LIGHT INOP. (W)")
    assert chunk_bounds(toks) == [0, 2, 7]  # "PVD -" | "ROWS 10 LIGHT INOP ." | "( W )"


def test_llm_schema_and_ref_constraint():
    from aerofix.segment.llm_label import apply_ref_constraint, schema
    s = schema(3)
    assert s["required"] == ["1", "2", "3"] and s["properties"]["2"]["enum"] == ["PROBLEM", "ACTION", "REF", "OTHER"]
    toks = seg_tokenize("SPOILER OUT OF ADJUSTMENT. REPAIRED IAW MM.")
    bounds = chunk_bounds(toks)
    ends = bounds[1:] + [len(toks)]
    llm = ["PROBLEM"] * 5 + ["REF", "REF", "OTHER", "OTHER"]  # LLM mislabels "REPAIRED IAW" and "MM ."
    fixed = apply_ref_constraint(toks, llm, bounds, ends)
    assert fixed[toks.index("REPAIRED")] == "ACTION"  # REF chunk starting with an action verb
    assert fixed[toks.index("MM")] == "REF" and fixed[-1] == "REF"  # doc word forced; "." follows it


def test_load_annotations_human_wins(tmp_path):
    import json
    from aerofix.segment.common import load_annotations
    rec = {"row_id": 1, "ann_split": "train", "tokens": ["A"], "skipped": False}
    (tmp_path / "llm.jsonl").write_text(json.dumps({**rec, "labels": ["ACTION"], "annotator": "qwen"}) + "\n")
    (tmp_path / "human.jsonl").write_text(json.dumps({**rec, "labels": ["PROBLEM"]}) + "\n")
    data, sources = load_annotations(tmp_path / "llm.jsonl", tmp_path / "human.jsonl")
    assert data["train"] == [(["A"], ["PROBLEM"])] and sources == {("train", "human"): 1}


def test_join_tokens():
    assert join_tokens(["DOOR", "LEAKING", ".", "(", "OPEN", ")"]) == "DOOR LEAKING. (OPEN)"
