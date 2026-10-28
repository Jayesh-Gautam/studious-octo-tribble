from aerofix.rag.ata import title
from aerofix.rag.generate import Evidence, validate


def _ev():
    return Evidence(family="A320", variant="A321", problem="L1 DOOR DAMPER LEAKING", ata2="52", ata2_conf=0.9,
                    ata4="5210", condition="LEAKING", condition_conf=0.7, actions=[("SERVICE DOOR DAMPER", 0.4)],
                    refs=[("AMM 52-10-00", 0.3)],
                    cases=[{"problem": "DAMPER LEAK", "action": "SERVICED", "refs": ["AMM 52-11-14"], "year": 2020,
                            "variant": "A320"}],
                    passages=[{"label": "AMT Airframe p.763", "heading": "Dampers", "text": "A damper ..."}])


def test_valid_summary_passes():
    s = "Most likely ATA 52 doors [Case 1]. Technicians cited AMM 52-10-00 [Case 1] [AMT Airframe p.763]."
    v = validate(s, _ev())
    assert v["ok"] and v["n_citations"] == 3


def test_invented_citation_and_reference_are_flagged():
    v = validate("Replace the damper [Case 7]; refer to AMM 52-99-99.", _ev())
    assert v["invalid_citations"] == ["Case 7"] and v["hallucinated_refs"] == ["AMM 52-99-99"] and not v["ok"]


def test_no_citation_fails():
    assert not validate("Service the damper.", _ev())["ok"]


def test_trim_to_sentence():
    from aerofix.rag.generate import trim_to_sentence
    assert trim_to_sentence("A [Case 1]. B [Case 2]. Refer to SRM in the") == "A [Case 1]. B [Case 2]."
    assert trim_to_sentence("Complete [Case 1].") == "Complete [Case 1]."


def test_render_labels_and_titles():
    text = _ev().render()
    assert "[Case 1]" in text and "[AMT Airframe p.763]" in text and "ATA 52 DOORS" in text
    assert title("52") == "DOORS" and title("99") == "OTHER SYSTEMS"
