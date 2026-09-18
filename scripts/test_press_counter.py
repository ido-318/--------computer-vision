"""בדיקות ממוקדות למכונת המצבים של לחיצת כתפיים (press_logic.PressCounter),
עם רצפי מדד/זמנים מדומים בלבד — בלי להריץ YOLO ובלי מצלמה בכלל.

שימוש:
    venv/bin/python scripts/test_press_counter.py
"""

from press_logic import PressCounter, START_RATIO, TOP_RATIO, CONFIRM_SEC, MAX_MISSING_SEC

START_M = START_RATIO - 0.15   # ברור מתחת לסף ההתחלה (משקולות למטה)
TOP_M = TOP_RATIO + 0.15        # ברור מעל סף ה"למעלה"
MID_M = (START_RATIO + TOP_RATIO) / 2  # אמצע הדרך, לא חוצה אף סף


def hold(metric, seconds, dt=0.02):
    n = max(1, round(seconds / dt))
    return [(dt, metric)] * n


def run_sequence(seq):
    counter = PressCounter()
    history = []
    for dt, metric in seq:
        info = counter.step(dt, metric)
        history.append(info["rep_count"])
    return counter, history


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    return condition


def test_full_sequence_one_rep():
    seq = hold(START_M, 0.2) + hold(TOP_M, 0.2) + hold(START_M, 0.2)
    counter, _ = run_sequence(seq)
    return check("רצף מלא (התחלה->למעלה->התחלה) -> חזרה אחת", counter.rep_count == 1, f"rep_count={counter.rep_count}")


def test_partial_press_no_full_rep():
    # עלייה שלא מגיעה לסף ה"למעלה" וחזרה להתחלה -> לא נספרת חזרה
    seq = hold(START_M, 0.2) + hold(MID_M, 0.2) + hold(START_M, 0.2)
    counter, _ = run_sequence(seq)
    return check("לחיצה חלקית (לא מגיעה ל'למעלה') -> 0", counter.rep_count == 0, f"rep_count={counter.rep_count}")


def test_hidden_hand_mid_press():
    # יד מוסתרת (metric=None) לזמן קצר באמצע העלייה, אמורה רק להשהות את המדידה
    # (לא לאפס), והחזרה עדיין אמורה להיספר כשהיד חוזרת ומתגלה
    seq = (
        hold(START_M, 0.2)
        + [(0.1, 0.5)]           # בדרך למעלה
        + [(0.15, None)]         # יד מוסתרת לרגע קצר (פחות מ-MAX_MISSING_SEC)
        + hold(TOP_M, 0.2)
        + hold(START_M, 0.2)
    )
    counter, _ = run_sequence(seq)
    return check("יד מוסתרת לרגע קצר באמצע העלייה -> החזרה עדיין נספרת", counter.rep_count == 1, f"rep_count={counter.rep_count}")


def test_jitter_near_thresholds_no_double_count():
    jitter_top = [(0.04, TOP_RATIO + 0.05), (0.04, TOP_RATIO - 0.05)] * 3
    jitter_start = [(0.04, START_RATIO - 0.05), (0.04, START_RATIO + 0.05)] * 3
    seq = (
        hold(START_M, 0.2)
        + jitter_top
        + hold(TOP_M, 0.2)     # מתייצב באמת מעל הסף -> אישור אמיתי
        + jitter_start
        + hold(START_M, 0.2)   # מתייצב באמת מתחת לסף -> אישור אמיתי, חזרה אחת בלבד
    )
    counter, history = run_sequence(seq)
    never_counted_early = all(c == 0 for c in history[: len(seq) - len(hold(START_M, 0.2))])
    return check(
        "תנודות קצרות ליד שני הספים -> ללא ספירה כפולה (בדיוק 1)",
        counter.rep_count == 1 and never_counted_early,
        f"rep_count={counter.rep_count}, לא נספר מוקדם מדי: {never_counted_early}",
    )


def test_known_timing_sequence():
    """רצף עם זמנים ידועים מראש (מחושב ידנית), לבדיקת תחילת-עלייה/שיא/סיום ומשכים."""
    seq = (
        [(0.1, START_M), (0.1, START_M)]  # t=0.1, 0.2 - בסיס "התחלה" מאושר
        + [(0.1, 0.5)]     # t=0.3 - יציאה מ"התחלה" -> t_start צפוי = 0.3
        + [(0.1, 0.9)]     # t=0.4
        + [(0.1, 1.4)]     # t=0.5 - כאן "top" מאושר (0.1+0.1 מ-t=0.6 ואילך, לא כאן עדיין)
        + [(0.1, 1.8)]     # t=0.6 - השיא -> t_top צפוי = 0.6 (ערך מקסימלי 1.8)
        + [(0.1, 1.4)]     # t=0.7
        + [(0.1, 0.9)]     # t=0.8
        + [(0.1, 0.5)]     # t=0.9
        + [(0.1, START_M)] # t=1.0 - חזרה ל"התחלה" (גולמי) -> t_end צפוי = 1.0
        + [(0.1, START_M), (0.1, START_M)]  # t=1.1, 1.2 - "start" מאושר -> חזרה נספרת
    )
    counter = PressCounter()
    summaries = []
    for dt, metric in seq:
        info = counter.step(dt, metric)
        if info["rep_summary"] is not None:
            summaries.append(info["rep_summary"])

    ok_count = check("רצף זמנים ידוע: בדיוק חזרה אחת נספרה", counter.rep_count == 1, f"rep_count={counter.rep_count}")
    ok_summary_exists = check("...והתקבל סיכום חזרה אחד", len(summaries) == 1, f"מספר סיכומים={len(summaries)}")
    if not ok_summary_exists:
        return ok_count and ok_summary_exists

    s = summaries[0]
    eps = 1e-9
    checks = [
        check("t_start == 0.3", abs(s["t_start"] - 0.3) < eps, f"t_start={s['t_start']:.4f}"),
        check("t_top == 0.6 (המקסימום, לא חציית סף TOP)", abs(s["t_top"] - 0.6) < eps, f"t_top={s['t_top']:.4f}"),
        check("t_end == 1.0", abs(s["t_end"] - 1.0) < eps, f"t_end={s['t_end']:.4f}"),
        check("משך עלייה == 0.3s (0.6-0.3)", abs(s["ascent_duration"] - 0.3) < eps, f"ascent={s['ascent_duration']:.4f}"),
        check("משך ירידה == 0.4s (1.0-0.6)", abs(s["descent_duration"] - 0.4) < eps, f"descent={s['descent_duration']:.4f}"),
        check("לא מסומן כ'משוער'", s["estimated"] is False),
        check("rep_number == 1", s["rep_number"] == 1),
    ]
    return ok_count and ok_summary_exists and all(checks)


def test_long_missing_after_top_resets_partial_rep():
    seq = (
        hold(START_M, 0.2)
        + hold(TOP_M, 0.2)               # מגיעים ל"למעלה", reached_top=True
        + [(MAX_MISSING_SEC + 0.1, None)]  # אובדן זיהוי ממושך (למשל שני הידיים יצאו מהפריים)
        + hold(START_M, 0.2)              # חוזרים ל"התחלה" -> לא אמור להיספר
    )
    counter, _ = run_sequence(seq)
    ok1 = check(
        "אובדן זיהוי ממושך אחרי 'למעלה', ואז חזרה ל'התחלה' -> 0",
        counter.rep_count == 0,
        f"rep_count={counter.rep_count}",
    )
    seq2 = hold(TOP_M, 0.2) + hold(START_M, 0.2)
    for dt, metric in seq2:
        counter.step(dt, metric)
    ok2 = check("...וחזרה מלאה חדשה אחר כך -> 1 (סה\"כ)", counter.rep_count == 1, f"rep_count סופי={counter.rep_count}")
    return ok1 and ok2


def test_only_counts_sequence_starting_from_confirmed_start():
    # מתחילים ישר ב"למעלה", בלי "התחלה" מאושרת קודם -> לא אמור להיספר
    seq = hold(TOP_M, 0.2) + hold(START_M, 0.2)
    counter, _ = run_sequence(seq)
    ok1 = check(
        "רצף שמתחיל ב'למעלה' (בלי 'התחלה' מאושרת קודם) -> 0",
        counter.rep_count == 0,
        f"rep_count={counter.rep_count}, have_start_baseline={counter.have_start_baseline}",
    )
    seq2 = hold(TOP_M, 0.2) + hold(START_M, 0.2)
    for dt, metric in seq2:
        counter.step(dt, metric)
    ok2 = check("...ואחרי שיש בסיס 'התחלה', חזרה אמיתית כן נספרת -> 1", counter.rep_count == 1, f"rep_count={counter.rep_count}")
    return ok1 and ok2


def main():
    tests = [
        test_full_sequence_one_rep,
        test_partial_press_no_full_rep,
        test_hidden_hand_mid_press,
        test_jitter_near_thresholds_no_double_count,
        test_known_timing_sequence,
        test_long_missing_after_top_resets_partial_rep,
        test_only_counts_sequence_starting_from_confirmed_start,
    ]
    results = [t() for t in tests]
    print(f"\n{sum(results)}/{len(results)} בדיקות עברו.")
    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
