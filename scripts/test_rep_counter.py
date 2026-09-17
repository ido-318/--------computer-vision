"""בדיקות ממוקדות למכונת המצבים של ספירת החזרות (rep_logic.RepCounter),
עם רצפי זוויות וזמנים מדומים בלבד — בלי להריץ YOLO ובלי תמונות בכלל.

שימוש:
    venv/bin/python scripts/test_rep_counter.py
"""

from rep_logic import RepCounter, STAND_ANGLE, DEPTH_ANGLE, CONFIRM_SEC, MAX_MISSING_SEC

STAND_A = STAND_ANGLE + 10   # ~170, ברור מעל סף העמידה
BOTTOM_A = DEPTH_ANGLE - 10  # ~90, ברור מתחת לסף העומק
MID_A = (STAND_ANGLE + DEPTH_ANGLE) / 2  # ~130, אמצע הדרך, לא חוצה אף סף


def hold(angle, seconds, dt=0.02):
    """רשימת (dt, angle) שמכסה יחד לפחות `seconds` שניות."""
    n = max(1, round(seconds / dt))
    return [(dt, angle)] * n


def run_sequence(seq):
    counter = RepCounter()
    history = []
    for dt, angle in seq:
        info = counter.step(dt, angle)
        history.append(info["rep_count"])
    return counter, history


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    return condition


def test_two_full_reps():
    seq = (
        hold(STAND_A, 0.2)
        + hold(BOTTOM_A, 0.2)
        + hold(STAND_A, 0.2)
        + hold(BOTTOM_A, 0.2)
        + hold(STAND_A, 0.2)
    )
    counter, _ = run_sequence(seq)
    return check("שתי חזרות מלאות ברצף -> 2", counter.rep_count == 2, f"rep_count={counter.rep_count}")


def test_shallow_no_depth():
    seq = hold(STAND_A, 0.2) + hold(MID_A, 0.2) + hold(STAND_A, 0.2)
    counter, _ = run_sequence(seq)
    return check(
        "ירידה שלא מגיעה לסף העומק וחזרה לעמידה -> 0",
        counter.rep_count == 0,
        f"rep_count={counter.rep_count}",
    )


def test_jitter_near_thresholds_no_double_count():
    jitter_bottom = [(0.04, 101.0), (0.04, 99.0)] * 3          # מתנודד מסביב ל-100, אף פעם לא מספיק זמן רציף
    jitter_stand = [(0.04, 161.0), (0.04, 159.0)] * 3           # מתנודד מסביב ל-160, אותו דבר
    seq = (
        hold(STAND_A, 0.2)
        + jitter_bottom
        + hold(BOTTOM_A, 0.2)   # עכשיו מתייצב באמת מתחת לסף -> אישור אמיתי
        + jitter_stand
        + hold(STAND_A, 0.2)    # מתייצב באמת מעל הסף -> אישור אמיתי, סופר חזרה אחת בלבד
    )
    counter, history = run_sequence(seq)
    never_counted_during_jitter = all(c == 0 for c in history[: len(seq) - len(hold(STAND_A, 0.2))])
    return check(
        "תנודות קצרות ליד הספים -> ללא ספירה כפולה (בדיוק 1)",
        counter.rep_count == 1 and never_counted_during_jitter,
        f"rep_count={counter.rep_count}, לא נספר מוקדם מדי: {never_counted_during_jitter}",
    )


def test_short_missing_does_not_complete_confirmation():
    counter = RepCounter()
    # בסיס עמידה
    for dt, angle in hold(STAND_A, 0.2):
        counter.step(dt, angle)

    # 0.05s תקף + 0.10s חסר (פער קצר) + 0.05s תקף = 0.10s תקף מצטבר בלבד (<0.12 -> עדיין לא מאושר)
    counter.step(0.05, BOTTOM_A)
    counter.step(0.10, None)
    counter.step(0.05, BOTTOM_A)
    not_yet_confirmed = not counter.reached_depth
    check(
        "מדידה חסרה קצרה לא משלימה את זמן האישור (0.05+0.05=0.10s < 0.12s)",
        not_yet_confirmed,
        f"reached_depth={counter.reached_depth} (צפוי False)",
    )

    # עוד 0.05s תקף -> עכשיו מצטבר 0.15s (>=0.12) -> אמור להתאשר
    counter.step(0.05, BOTTOM_A)
    now_confirmed = counter.reached_depth
    ok2 = check(
        "...ואחרי עוד קצת זמן תקף (סה\"כ 0.15s תקף) כן מאושר",
        now_confirmed,
        f"reached_depth={counter.reached_depth} (צפוי True)",
    )

    for dt, angle in hold(STAND_A, 0.2):
        counter.step(dt, angle)
    completed = counter.rep_count == 1
    ok3 = check("...והחזרה מסתיימת ונספרת כרגיל (1)", completed, f"rep_count={counter.rep_count}")
    return not_yet_confirmed and ok2 and ok3


def test_long_missing_after_bottom_resets_partial_rep():
    seq = (
        hold(STAND_A, 0.2)
        + hold(BOTTOM_A, 0.2)     # מגיעים לעומק, reached_depth=True
        + [(MAX_MISSING_SEC + 0.1, None)]  # אובדן זיהוי ממושך אחרי הגעה לעומק (>0.5s)
        + hold(STAND_A, 0.2)      # חוזרים לעמידה -> לא אמור להיספר (התקדמות אופסה)
    )
    counter, history = run_sequence(seq)
    ok1 = check(
        "אובדן זיהוי >0.5s אחרי הגעה לעומק, ואז חזרה לעמידה -> 0",
        counter.rep_count == 0,
        f"rep_count אחרי החזרה הראשונה לעמידה={counter.rep_count}",
    )

    # אחר כך חזרה מלאה חדשה ואמיתית
    seq2 = hold(BOTTOM_A, 0.2) + hold(STAND_A, 0.2)
    for dt, angle in seq2:
        counter.step(dt, angle)
    ok2 = check(
        "...וחזרה מלאה חדשה אחר כך -> 1 (סה\"כ)",
        counter.rep_count == 1,
        f"rep_count סופי={counter.rep_count}",
    )
    return ok1 and ok2


def test_only_counts_sequence_starting_from_confirmed_stand():
    # מתחילים ישר בעומק, בלי עמידה מאושרת קודם -> לא אמור להיחשב כ"הגעה לעומק" לצורך חזרה
    seq = hold(BOTTOM_A, 0.2) + hold(STAND_A, 0.2)
    counter, _ = run_sequence(seq)
    ok1 = check(
        "רצף שמתחיל בעומק (בלי עמידה מאושרת קודם) -> 0",
        counter.rep_count == 0,
        f"rep_count={counter.rep_count}, have_stand_baseline={counter.have_stand_baseline}",
    )

    # מעכשיו יש כבר בסיס עמידה מאושר (מהעמידה שקרתה בסוף הרצף הקודם) -> חזרה אמיתית כן תיספר
    seq2 = hold(BOTTOM_A, 0.2) + hold(STAND_A, 0.2)
    for dt, angle in seq2:
        counter.step(dt, angle)
    ok2 = check(
        "...ואחרי שיש בסיס עמידה, חזרה מלאה כן נספרת -> 1",
        counter.rep_count == 1,
        f"rep_count={counter.rep_count}",
    )
    return ok1 and ok2


def main():
    tests = [
        test_two_full_reps,
        test_shallow_no_depth,
        test_jitter_near_thresholds_no_double_count,
        test_short_missing_does_not_complete_confirmation,
        test_long_missing_after_bottom_resets_partial_rep,
        test_only_counts_sequence_starting_from_confirmed_stand,
    ]
    results = [t() for t in tests]
    print(f"\n{sum(results)}/{len(results)} בדיקות עברו.")
    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
