"""בדיקות ממוקדות למכונת המצבים של ספירת החזרות (rep_logic.RepCounter),
עם רצפי זוויות וזמנים מדומים בלבד — בלי להריץ YOLO ובלי תמונות בכלל.

שימוש:
    venv/bin/python scripts/test_rep_counter.py
"""

from rep_logic import RepCounter, STAND_ANGLE, DEPTH_ANGLE, CONFIRM_SEC, MAX_MISSING_SEC, pace_feedback, PACE_TARGET_EXAMPLE

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


def test_known_timing_sequence():
    """רצף עם זמנים ידועים מראש (מחושבים ידנית), לבדיקת תחילת-ירידה/תחתית/סיום ומשכים."""
    seq = (
        [(0.1, 170), (0.1, 170)]  # t=0.1, 0.2 - בסיס עמידה מאושר (elapsed 0.2>=0.12)
        + [(0.1, 150)]            # t=0.3 - יציאה מעמידה -> t_start צפוי = 0.3
        + [(0.1, 120)]            # t=0.4
        + [(0.1, 90)]             # t=0.5
        + [(0.1, 70)]             # t=0.6 - כאן bottom מאושר (0.1+0.1=0.2>=0.12)
        + [(0.1, 50)]             # t=0.7 - המינימום -> t_bottom צפוי = 0.7
        + [(0.1, 70)]             # t=0.8
        + [(0.1, 90)]             # t=0.9
        + [(0.1, 120)]            # t=1.0
        + [(0.1, 150)]            # t=1.1
        + [(0.1, 170)]            # t=1.2 - חזרה לעמידה (גולמי) -> t_end צפוי = 1.2
        + [(0.1, 170), (0.1, 170)]  # t=1.3, 1.4 - stand מאושר (0.1+0.1=0.2>=0.12) -> חזרה נספרת כאן, ב-t=1.3
    )
    counter = RepCounter()
    summaries = []
    for dt, angle in seq:
        info = counter.step(dt, angle)
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
        check("t_bottom == 0.7 (המינימום, לא חציית סף העומק)", abs(s["t_bottom"] - 0.7) < eps, f"t_bottom={s['t_bottom']:.4f}"),
        check("t_end == 1.2", abs(s["t_end"] - 1.2) < eps, f"t_end={s['t_end']:.4f}"),
        check("משך ירידה == 0.4s (0.7-0.3)", abs(s["descent_duration"] - 0.4) < eps, f"descent={s['descent_duration']:.4f}"),
        check("משך עלייה == 0.5s (1.2-0.7)", abs(s["ascent_duration"] - 0.5) < eps, f"ascent={s['ascent_duration']:.4f}"),
        check("לא מסומן כ'משוער' (אין מדידות חסרות)", s["estimated"] is False),
        check("rep_number == 1", s["rep_number"] == 1),
    ]
    return ok_count and ok_summary_exists and all(checks)


def test_estimated_flag_on_short_missing_during_rep():
    seq = (
        hold(STAND_A, 0.2)
        + [(0.1, 150), (0.05, None), (0.1, 90), (0.1, 60)]  # פער קצר (0.05s) באמצע הירידה
        + hold(BOTTOM_A, 0.2)
        + hold(STAND_A, 0.2)
    )
    counter = RepCounter()
    summaries = []
    for dt, angle in seq:
        info = counter.step(dt, angle)
        if info["rep_summary"] is not None:
            summaries.append(info["rep_summary"])
    ok1 = check("חזרה נספרה למרות הפער הקצר", counter.rep_count == 1, f"rep_count={counter.rep_count}")
    ok2 = check("...וסומנה 'משוער' (estimated=True) בגלל המדידה החסרה", bool(summaries) and summaries[0]["estimated"] is True)
    return ok1 and ok2


def test_long_loss_suppresses_summary_of_that_rep_only():
    counter = RepCounter()
    seq1 = (
        hold(STAND_A, 0.2)
        + hold(BOTTOM_A, 0.2)
        + [(MAX_MISSING_SEC + 0.1, None)]
        + hold(STAND_A, 0.2)
    )
    summaries1 = []
    for dt, angle in seq1:
        info = counter.step(dt, angle)
        if info["rep_summary"] is not None:
            summaries1.append(info["rep_summary"])
    ok1 = check(
        "אחרי אובדן זיהוי ממושך: לא נספרה חזרה ולא הופק סיכום",
        counter.rep_count == 0 and len(summaries1) == 0,
        f"rep_count={counter.rep_count}, summaries={len(summaries1)}",
    )

    seq2 = hold(BOTTOM_A, 0.2) + hold(STAND_A, 0.2)
    summaries2 = []
    for dt, angle in seq2:
        info = counter.step(dt, angle)
        if info["rep_summary"] is not None:
            summaries2.append(info["rep_summary"])
    ok2 = check(
        "...אבל החזרה המלאה הבאה כן נספרת וכן מקבלת סיכום תקין",
        counter.rep_count == 1 and len(summaries2) == 1,
        f"rep_count={counter.rep_count}, summaries={len(summaries2)}",
    )
    return ok1 and ok2


def _summary(descent, ascent, estimated=False):
    return {"descent_duration": descent, "ascent_duration": ascent, "estimated": estimated, "rep_number": 1}


def test_pace_feedback_no_target():
    msg = pace_feedback(_summary(1.28, 0.88), target=None)
    return check("אין יעד מוגדר -> אין דגש (None), רק מדידות", msg is None, f"msg={msg!r}")


def test_pace_feedback_estimated_timing():
    msg = pace_feedback(_summary(2.5, 1.5, estimated=True), target=PACE_TARGET_EXAMPLE)
    ok = msg is not None and "ודאות" in msg
    return check("תזמון משוער -> משפט 'אין מספיק ודאות', בלי השוואה", ok, f"msg={msg!r}")


def test_pace_feedback_descent_below_range():
    # דוגמת המשתמש: ירידה 1.28s (מתחת לטווח 2-3), עלייה 0.88s (בתוך 1-2, לא רלוונטי - יש עדיפות לירידה)
    msg = pace_feedback(_summary(1.28, 0.88), target=PACE_TARGET_EXAMPLE)
    ok = msg is not None and "ירידה" in msg and "האט" in msg and "1.28" in msg and "2–3" in msg
    return check("ירידה מתחת לטווח -> דגש על האטת הירידה", ok, f"msg={msg!r}")


def test_pace_feedback_descent_above_range():
    msg = pace_feedback(_summary(3.6, 1.5), target=PACE_TARGET_EXAMPLE)
    ok = msg is not None and "ירידה" in msg and "האץ" in msg and "3.60" in msg
    return check("ירידה מעל הטווח -> דגש על האצת הירידה", ok, f"msg={msg!r}")


def test_pace_feedback_within_range():
    msg = pace_feedback(_summary(2.5, 1.5), target=PACE_TARGET_EXAMPLE)
    ok = msg is not None and "ירידה" in msg and "עלייה" in msg and "טווחי היעד" in msg and "האט" not in msg and "האץ" not in msg
    return check("שני השלבים בתוך הטווח -> משפט עובדתי בלבד, בלי דגש תיקון", ok, f"msg={msg!r}")


def test_pace_feedback_only_ascent_out_of_range():
    # ירידה בתוך הטווח (2.5), עלייה מעל הטווח (2.4) -> דגש על העלייה בלבד
    msg = pace_feedback(_summary(2.5, 2.4), target=PACE_TARGET_EXAMPLE)
    ok = msg is not None and "עלייה" in msg and "ירידה" not in msg.split(";")[0] and "האץ" in msg
    return check("רק העלייה מחוץ לטווח -> דגש על העלייה בלבד", ok, f"msg={msg!r}")


def test_pace_feedback_both_out_of_range_prioritizes_descent():
    # שני השלבים מחוץ לטווח: ירידה מהירה מדי (1.0, <2) ועלייה איטית מדי (2.5, >2) -> עדיפות לירידה
    msg = pace_feedback(_summary(1.0, 2.5), target=PACE_TARGET_EXAMPLE)
    ok = msg is not None and "ירידה" in msg and "האט" in msg and "2.50" not in msg
    return check("שני השלבים מחוץ לטווח -> עדיפות לדגש על הירידה בלבד", ok, f"msg={msg!r}")


def main():
    tests = [
        test_two_full_reps,
        test_shallow_no_depth,
        test_jitter_near_thresholds_no_double_count,
        test_short_missing_does_not_complete_confirmation,
        test_long_missing_after_bottom_resets_partial_rep,
        test_only_counts_sequence_starting_from_confirmed_stand,
        test_known_timing_sequence,
        test_estimated_flag_on_short_missing_during_rep,
        test_long_loss_suppresses_summary_of_that_rep_only,
        test_pace_feedback_no_target,
        test_pace_feedback_estimated_timing,
        test_pace_feedback_descent_below_range,
        test_pace_feedback_descent_above_range,
        test_pace_feedback_within_range,
        test_pace_feedback_only_ascent_out_of_range,
        test_pace_feedback_both_out_of_range_prioritizes_descent,
    ]
    results = [t() for t in tests]
    print(f"\n{sum(results)}/{len(results)} בדיקות עברו.")
    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
