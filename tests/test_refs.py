from aerofix.preprocess.refs import extract_refs, mask_refs, ref_chapters


def test_simple_amm():
    t = "L1 DOOR DAMPER LEAKING. REPLACED DAMPER PER AMM 52-10-00."
    assert extract_refs(t) == ["AMM 52-10-00"]
    assert mask_refs(t) == "L1 DOOR DAMPER LEAKING. REPLACED DAMPER PER [REF]."


def test_fim_with_task_is_one_reference():
    t = "SHIMMY PER FIM 32-51 TASK 804 STEP D"
    assert extract_refs(t) == ["FIM 32-51"]
    assert mask_refs(t) == "SHIMMY PER [REF] STEP D"


def test_alias_and_suffix():
    assert extract_refs("REPLACED PER M/M 33-51-06.") == ["AMM 33-51-06"]
    assert mask_refs("IAW SRM 53-30-15-2R-1, REPAIR 1") == "IAW [REF], REPAIR 1"


def test_airbus_task_number():
    t = "IAW AMM TASK 32-11-00-200-801 OPS CHK GOOD"
    assert extract_refs(t) == ["AMM 32-11-00"]
    assert mask_refs(t) == "IAW [REF] OPS CHK GOOD"


def test_chained_refs_inherit_type():
    t = "IAW B737-700 SRM 51-10-02-0G; 51-40-02-0G AND DWG 115A2714"
    assert extract_refs(t) == ["SRM 51-10-02", "SRM 51-40-02"]
    assert mask_refs(t) == "IAW B737-700 [REF]; [REF] AND DWG 115A2714"


def test_bare_number_after_per():
    t = "REPLACED VALVE PER 21-51-11. OPS CHECK GOOD"
    assert extract_refs(t) == ["REF 21-51-11"]
    assert mask_refs(t) == "REPLACED VALVE PER [REF]. OPS CHECK GOOD"


def test_aircraft_and_part_between_type_and_number():
    assert extract_refs("TREATMENT IAW SRM B737-700 51-20-01;MATE") == ["SRM 51-20-01"]
    assert extract_refs("IAW-B737-700-SRM- 51-10-02-0G-0") == ["SRM 51-10-02"]
    assert extract_refs("REF NDT PT 6-51-00-00-PR23") == ["NDT 51-00-00"]
    assert extract_refs("CHECK.REFERENCE: TASK 25-66-01-200-801") == ["REF 25-66-01"]


def test_leftover_numbers_masked_but_not_extracted():
    t = "AFTER REWORK PER PART 6 51-00-00 PROC 23"
    assert extract_refs(t) == []
    assert mask_refs(t) == "AFTER REWORK PER PART 6 [REF] PROC 23"


def test_no_false_positive_on_part_numbers_and_dates():
    t = "P/N 65-12345 REMOVED. DUE DATE 25-OCT-25"
    assert extract_refs(t) == []
    assert mask_refs(t) == t


def test_ref_chapters():
    assert ref_chapters(["AMM 52-10-00", "FIM 32-51"]) == ["52", "32"]
