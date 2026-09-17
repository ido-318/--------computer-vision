"""לוגיקת חישוב הזווית וספירת החזרות, בלי תלות ב-YOLO/ultralytics — כדי שאפשר יהיה
לבדוק אותה עם רצפים מדומים של זוויות וזמנים, בלי להריץ מודל בכלל.

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
    """מכונת מצבים לספירת חזרות: עמידה -> ירידה -> הגעה לעומק -> עלייה -> עמידה.

    שימוש: counter.step(dt, angle) לכל פריים, לפי הסדר, כאשר dt הוא משך הפריים
    *הזה* בשניות (לא זמן מצטבר) ו-angle הוא הזווית שנמדדה או None אם המדידה חסרה.

    עקרונות:
    - מצב (STAND/BOTTOM) "מאושר" רק אחרי שהתנאי הגולמי החזיק ברציפות לפחות
      CONFIRM_SEC שניות *של מדידה תקפה בפועל* — זמן שבו המדידה חסרה לא נספר בתוך
      הרציפות הזו (לא מקדם את השעון), אבל גם לא מאפס אותו אם הוא קצר; רק חריגה
      מהתנאי הגולמי עצמו (למשל המעבר מ-"עומק" ל-"אמצע") מאפסת את השעון.
    - חזרה נספרת רק במעבר מאושר ל-STAND, אם וכאשר reached_depth==True (כלומר עברנו
      דרך BOTTOM מאושר באותו מחזור) *וגם* כבר הייתה בעבר לפחות עמידה אחת מאושרת
      (have_stand_baseline) — כדי לא לספור רצף שמתחיל באמצע התנועה בלי בסיס עמידה.
    - אובדן זיהוי רציף מעבר ל-MAX_MISSING_SEC מאפס reached_depth ואת המועמדות
      הממתינה לאישור (לא את have_stand_baseline — זה נשאר ברגע שהושג פעם אחת).
    """

    def __init__(self):
        self.rep_count = 0
        self.reached_depth = False
        self.have_stand_baseline = False
        self.display_phase = "DESCEND"
        self.t = 0.0  # זמן מצטבר, רק לצורכי לוג/תצוגה - לא משמש בלוגיקת האישור

        self._candidate_target: str | None = None
        self._candidate_elapsed = 0.0
        self._candidate_confirmed = False
        self._missing_elapsed = 0.0
        self._missing_reset_done = False

    def step(self, dt: float, angle: float | None) -> dict:
        self.t += dt
        r = raw_class(angle)
        just_completed_rep = False

        if r is None:
            self._missing_elapsed += dt
            if not self._missing_reset_done and self._missing_elapsed >= MAX_MISSING_SEC:
                self.reached_depth = False
                self._candidate_target = None
                self._candidate_elapsed = 0.0
                self._candidate_confirmed = False
                self._missing_reset_done = True
                self.display_phase = "LOST"
            # מדידה חסרה (קצרה או ארוכה): לא מקדמת שום מועמדות לאישור, ולא מאפסת
            # מועמדות קיימת אם היא קצרה מ-MAX_MISSING_SEC (השעון פשוט מוקפא).
        else:
            self._missing_elapsed = 0.0
            self._missing_reset_done = False

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
                        # אם עוד לא הייתה עמידה מאושרת -> "הגעה לעומק" הזו לא נספרת
                        # לצורך חזרה (אין רצף שמתחיל בעמידה מאושרת).
                    elif r == "stand":
                        self.have_stand_baseline = True
                        if self.reached_depth:
                            self.rep_count += 1
                            just_completed_rep = True
                        self.reached_depth = False
            else:  # "mid" -> משאירים את BOTTOM/STAND האחרון בעינו, אבל מבטלים מועמדות ממתינה
                self._candidate_target = None
                self._candidate_elapsed = 0.0
                self._candidate_confirmed = False

            if r == "stand":
                self.display_phase = "STAND"
            elif r == "bottom":
                self.display_phase = "BOTTOM"
            else:
                self.display_phase = "ASCEND" if self.reached_depth else "DESCEND"

        return {"phase": self.display_phase, "rep_count": self.rep_count, "just_completed_rep": just_completed_rep}
