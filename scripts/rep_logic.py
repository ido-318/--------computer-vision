"""לוגיקת חישוב הזווית, ספירת החזרות ותזמון ירידה/עלייה — בלי תלות ב-YOLO/ultralytics
כדי שאפשר יהיה לבדוק אותה עם רצפים מדומים של זוויות וזמנים, בלי להריץ מודל בכלל.

מיובא על ידי rep_counter_gif.py (שמוסיף את עיבוד ה-GIF/YOLO סביב זה) וגם על ידי
test_rep_counter.py (בדיקות עם נתונים מדומים).
"""

import numpy as np

L_HIP, L_KNEE, L_ANKLE = 11, 13, 15
KP_CONF_THRESHOLD = 0.7
DEGENERATE_EPS_PX = 2.0

# --- ספי ספירה, ניתנים לשינוי לצורך ניסוי בלבד (לא קביעה על איכות הסקוואט) ---
STAND_ANGLE = 160.0    # מעל זה -> נחשב "עמידה"
DEPTH_ANGLE = 100.0    # מתחת לזה -> נחשב "הגיע לעומק"
CONFIRM_SEC = 0.12      # כמה זמן רציף *של מדידה תקפה* (לא כולל פערי זיהוי) נדרש לפני שמצב "מאושר"
MAX_MISSING_SEC = 0.5   # אובדן זיהוי רציף מעבר לזה -> איפוס התקדמות חלקית של החזרה הנוכחית


def knee_angle_deg(hip_xy, knee_xy, ankle_xy) -> float | None:
    v1 = np.array(hip_xy) - np.array(knee_xy)
    v2 = np.array(ankle_xy) - np.array(knee_xy)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < DEGENERATE_EPS_PX or n2 < DEGENERATE_EPS_PX:
        return None
    cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def measure_frame(keypoints_xy, keypoints_conf) -> float | None:
    confs = [keypoints_conf[i] for i in (L_HIP, L_KNEE, L_ANKLE)]
    if min(confs) < KP_CONF_THRESHOLD:
        return None
    hip, knee, ankle = (keypoints_xy[i] for i in (L_HIP, L_KNEE, L_ANKLE))
    return knee_angle_deg(hip, knee, ankle)


def raw_class(angle: float | None) -> str | None:
    if angle is None:
        return None
    if angle >= STAND_ANGLE:
        return "stand"
    if angle <= DEPTH_ANGLE:
        return "bottom"
    return "mid"


class RepCounter:
    """מכונת מצבים לספירת חזרות + מדידת משך ירידה/עלייה לכל חזרה.

    שימוש: counter.step(dt, angle) לכל פריים לפי הסדר, כאשר dt הוא משך הפריים
    *הזה* בשניות (לא זמן מצטבר) ו-angle הוא הזווית שנמדדה או None אם המדידה חסרה.

    --- ספירת חזרות (כמו קודם, ללא שינוי עקרוני) ---
    מצב (STAND/BOTTOM) "מאושר" רק אחרי שהתנאי הגולמי החזיק ברציפות לפחות CONFIRM_SEC
    שניות של מדידה תקפה בפועל (מדידה חסרה מקפיאה את השעון, לא מקדמת ולא מאפסת אותו
    אם היא קצרה). חזרה נספרת רק במעבר מאושר לעמידה אחרי שהגיעו לעומק מאושר, ורק אם
    כבר הייתה עמידה מאושרת קודם (have_stand_baseline).

    --- תזמון ירידה/עלייה (חדש) ---
    בלי קשר להשהיית האישור שלמעלה (שמיועדת רק למניעת ספירה כפולה): לכל מחזור
    בפועל עוקבים אחרי *זמני האירועים הגולמיים עצמם*:
    - תחילת ירידה = הפריים הראשון שבו הזווית יוצאת ממצב "עמידה" (חוצה מתחת ל-STAND_ANGLE).
    - נקודת ההפרדה (תחתית) = הפריים עם הזווית *המינימלית* לאורך כל המחזור (לא סף העומק!).
    - סיום = הפריים הראשון שבו הזווית חוזרת למצב "עמידה".
    אם הייתה מדידה חסרה כלשהי (קצרה) במהלך המחזור -> מסומן estimated=True (תזמון משוער).
    אם הייתה מדידה חסרה *ממושכת* (אותה MAX_MISSING_SEC שמאפס את reached_depth) -> ה"סיכום"
    של המחזור הזה מבוטל לגמרי (None), גם אם בסופו של דבר עוד מחזור מאוחר יותר כן נספר.
    """

    def __init__(self):
        self.rep_count = 0
        self.reached_depth = False
        self.have_stand_baseline = False
        self.display_phase = "DESCEND"
        self.t = 0.0  # זמן מצטבר; גם משמש כ"זמן אירוע" לתזמון (לא כולל השהיית אישור)

        self._candidate_target: str | None = None
        self._candidate_elapsed = 0.0
        self._candidate_confirmed = False
        self._missing_elapsed = 0.0
        self._missing_reset_done = False

        self._last_raw: str | None = None  # מחלקת הפריים התקף האחרון (לא מתעדכן במדידה חסרה)

        # מעקב אחרי המחזור הפעיל (ירידה בתהליך, עדיין לא חזרו לעמידה)
        self._cycle_t_start: float | None = None
        self._cycle_min_angle: float | None = None
        self._cycle_min_angle_t: float | None = None
        self._cycle_had_missing = False
        self._cycle_long_loss = False

        self._pending_summary: dict | None = None  # מחזור שנסגר (חזר לעמידה) וממתין לאישור הספירה

    def step(self, dt: float, angle: float | None) -> dict:
        self.t += dt
        r = raw_class(angle)
        just_completed_rep = False
        rep_summary = None

        if r is None:
            self._missing_elapsed += dt
            if self._cycle_t_start is not None:
                self._cycle_had_missing = True
            if not self._missing_reset_done and self._missing_elapsed >= MAX_MISSING_SEC:
                self.reached_depth = False
                self._candidate_target = None
                self._candidate_elapsed = 0.0
                self._candidate_confirmed = False
                self._missing_reset_done = True
                self.display_phase = "LOST"
                if self._cycle_t_start is not None:
                    self._cycle_long_loss = True
            # מדידה חסרה: לא בודקים מעברי stand/non-stand ולא מקדמים מועמדות לאישור
        else:
            self._missing_elapsed = 0.0
            self._missing_reset_done = False

            # --- תזמון גולמי: יציאה מעמידה, עדכון מינימום, חזרה לעמידה ---
            if self._last_raw == "stand" and r != "stand" and self._cycle_t_start is None:
                self._cycle_t_start = self.t
                self._cycle_min_angle = angle
                self._cycle_min_angle_t = self.t
                self._cycle_had_missing = False
                self._cycle_long_loss = False
            elif self._cycle_t_start is not None and angle < self._cycle_min_angle:
                self._cycle_min_angle = angle
                self._cycle_min_angle_t = self.t

            if (
                self._last_raw is not None
                and self._last_raw != "stand"
                and r == "stand"
                and self._cycle_t_start is not None
            ):
                if self._cycle_long_loss:
                    self._pending_summary = None
                else:
                    self._pending_summary = {
                        "t_start": self._cycle_t_start,
                        "t_bottom": self._cycle_min_angle_t,
                        "t_end": self.t,
                        "descent_duration": self._cycle_min_angle_t - self._cycle_t_start,
                        "ascent_duration": self.t - self._cycle_min_angle_t,
                        "estimated": self._cycle_had_missing,
                    }
                self._cycle_t_start = None
                self._cycle_min_angle = None
                self._cycle_min_angle_t = None

            self._last_raw = r

            # --- מכונת המצבים לספירה (עם השהיית אישור, כמו קודם) ---
            if r in ("stand", "bottom"):
                if self._candidate_target != r:
                    self._candidate_target = r
                    self._candidate_elapsed = 0.0
                    self._candidate_confirmed = False
                self._candidate_elapsed += dt
                if not self._candidate_confirmed and self._candidate_elapsed >= CONFIRM_SEC:
                    self._candidate_confirmed = True
                    if r == "bottom":
                        if self.have_stand_baseline:
                            self.reached_depth = True
                    elif r == "stand":
                        self.have_stand_baseline = True
                        if self.reached_depth:
                            self.rep_count += 1
                            just_completed_rep = True
                            if self._pending_summary is not None:
                                rep_summary = dict(self._pending_summary, rep_number=self.rep_count)
                            self._pending_summary = None
                        self.reached_depth = False
            else:
                self._candidate_target = None
                self._candidate_elapsed = 0.0
                self._candidate_confirmed = False

            if r == "stand":
                self.display_phase = "STAND"
            elif r == "bottom":
                self.display_phase = "BOTTOM"
            else:
                self.display_phase = "ASCEND" if self.reached_depth else "DESCEND"

        return {
            "phase": self.display_phase,
            "rep_count": self.rep_count,
            "just_completed_rep": just_completed_rep,
            "rep_summary": rep_summary,
        }


# --- יעד קצב אופציונלי (מוגדר ע"י המאמן) ---
# ערכי הדוגמה כאן הם *לצורך בדיקת התוכנה בלבד* — לא המלצת אימון. המאמן יכול לשנות
# את הטווחים האלה, או להשאיר PACE_TARGET=None כדי לא לקבל משוב קצב בכלל (רק מדידות).
PACE_TARGET_EXAMPLE = {
    "descent_range": (2.0, 3.0),  # שניות - טווח יעד לירידה, לצורך בדיקה בלבד
    "ascent_range": (1.0, 2.0),   # שניות - טווח יעד לעלייה, לצורך בדיקה בלבד
}


def _range_status(value: float, lo: float, hi: float) -> str:
    if value < lo:
        return "below"
    if value > hi:
        return "above"
    return "within"


def _fmt_range(lo: float, hi: float) -> str:
    return f"{lo:g}–{hi:g}"


def pace_feedback(summary: dict, target: dict | None) -> str | None:
    """מחזירה דגש אחד בעברית שמשווה בין המדידה ליעד הקצב, או None אם אין יעד מוגדר.

    - אין יעד מוגדר (target=None) -> None (הקוד הקורא מציג רק את המדידות, בלי דגש).
    - התזמון משוער (summary["estimated"]) -> משפט שאין מספיק ודאות למשוב, בלי השוואה בפועל.
    - אחרת: משווים ירידה ועלייה לטווחי היעד. אם שני השלבים מחוץ לטווח -> עדיפות לדגש
      על הירידה. אם שניהם בתוך הטווח -> משפט עובדתי שהקצב בתוך הטווח (לא קביעה על טכניקה).
    """
    if target is None:
        return None

    if summary.get("estimated"):
        return "התזמון משוער בגלל מדידות חסרות במהלך החזרה — אין מספיק ודאות למשוב על הקצב."

    descent, ascent = summary["descent_duration"], summary["ascent_duration"]
    d_lo, d_hi = target["descent_range"]
    a_lo, a_hi = target["ascent_range"]
    d_status = _range_status(descent, d_lo, d_hi)
    a_status = _range_status(ascent, a_lo, a_hi)

    def cue(phase: str, value: float, lo: float, hi: float, status: str) -> str:
        action = f"האט מעט את ה{phase}" if status == "below" else f"האץ מעט את ה{phase}"
        return (
            f"ה{phase} נמשכה {value:.2f} שניות; היעד שהוגדר הוא {_fmt_range(lo, hi)} שניות. "
            f"בחזרה הבאה {action}."
        )

    if d_status != "within":
        return cue("ירידה", descent, d_lo, d_hi, d_status)
    if a_status != "within":
        return cue("עלייה", ascent, a_lo, a_hi, a_status)
    return (
        f"הירידה ({descent:.2f}s) והעלייה ({ascent:.2f}s) בתוך טווחי היעד שהוגדרו "
        f"(ירידה {_fmt_range(d_lo, d_hi)}s, עלייה {_fmt_range(a_lo, a_hi)}s)."
    )
