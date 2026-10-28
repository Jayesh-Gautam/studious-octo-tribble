# Annotation guide: problem / action segmentation

Label every chunk with **one** label. If a chunk contains two labels, use **split** to cut it before the word where the label changes.

| Label       | Question to ask                                                              | Examples                                                                                                                                                    |
| ----------- | ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **PROBLEM** | What was wrong, and where or how it was found?                               | `L1 DOOR DAMPER LEAKING`, `DURING SCHEDULED C-CHECK, FOUND CORROSION ON FRAME`, `CREW REPORTED ...`, `DEFECT SIZE: .187 INCH`, `REPEAT WRITEUP`             |
| **ACTION**  | What did maintenance do about it, including checks and the release decision? | `REMOVED AND REPLACED DAMPER`, `OPS CHECK GOOD`, `AIRCRAFT RELEASED FOR CONTINUED SERVICE`, `DEFERRED PER`, `TASK CARD COMPLETED`, `OK TO CONTINUE`         |
| **REF**     | Which document was cited? Only the document itself, not "PER" or "IAW".      | `AMM 52-10-00`, `SRM 51-10-02 ITEM 9`, `SRM` in `REPAIRED IAW SRM`, `MEL`, `DWG 115A2714`, `EA 97321`                                                       |
| **OTHER**   | Admin text that is neither fault nor fix.                                    | `[01]` step markers, control numbers `2022-104311`, `(OPEN)`, `LAX - FLT 36 -`, `GROUNDED - YES`, tool calibration lines (`TOOL USED P/N ... DUE DATE ...`) |

## Rules for unclear cases

1. **The connector word goes with the action:** `REPAIRED PER` → ACTION, `SRM 53-30-15` → REF.
2. **Finding vs fixing:** `INSPECTED AND FOUND CRACK` → PROBLEM, because the inspection is how the fault was found. `INSPECTED REPAIR, NO DEFECTS` → ACTION.
3. **The problem restated inside the action** stays ACTION: `REPLACED CRACKED ANGLE` → ACTION.
4. **Punctuation** takes the label of the words before it (the `.` ending a problem sentence is PROBLEM).
5. **No action at all** is fine: label everything PROBLEM / REF / OTHER.
6. **Location blocks** (`LOCATION: FRAME 500C AT STR 14R`) describe the fault → PROBLEM.
7. If the report is unreadable or is not a fault report, click **Skip (unusable)**.

Aim for consistency over perfection: when unsure, choose the label a technician would need in order to understand the *fault*.
