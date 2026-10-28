from aerofix.eval.final import section, table

MD = """# Title

## Results

Intro text.

| method | split | score |
|---|---|---|
| rules | test | 0.19 |
| CRF | test | 0.68 |

## Next

other
"""


def test_section_and_table(tmp_path):
    p = tmp_path / "r.md"
    p.write_text(MD, encoding="utf-8")
    s = section(p, "Results")
    assert "Intro text." in s and "other" not in s
    t = table(p, "Results")
    assert t.method.tolist() == ["rules", "CRF"] and t.score.tolist() == [0.19, 0.68]
    assert "not found" in section(p, "Missing")
    assert table(p, "Next").empty
