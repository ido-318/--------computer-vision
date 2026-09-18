"""לוגיקת חישוב מדד הלחיצה, ספירת חזרות ותזמון עלייה/ירידה עבור "לחיצת כתפיים
בעמידה עם משקולות", בצילום חזיתי (שתי הידיים גלויות) — בלי תלות ב-YOLO/ultralytics.

**מודול עצמאי לגמרי מ-rep_logic.py (הסקוואט)** — לא מייבא ממנו ולא משתמש בשום
סף שהוגדר שם. זה תרגיל אחר, עם מדד תנועה אחר (מיקום אנכי של פרק כף היד ביחס
לכתפיים, לא זווית ברך), ולכן גם ספי ספירה משלו.

מיובא על ידי webapp/server.py וגם על ידי scripts/test_press_counter.py
(בדיקות עם רצפים מדומים, בלי מצלמה/מודל).
"""

L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10

KP_CONF_THRESHOLD = 0.7
MIN_SHOULDER_WIDTH_PX = 10.0  # רוחב כתפיים מתחת לזה -> כתפיים חופפות/מדידה לא אמינה

# --- ספי ספירה, ניתנים לשינוי לצורך ניסוי בלבד (ערכי דוגמה - טרם כוילו מול אדם
# אמיתי, ראו "מה עוד דורש בדיקה" בדוח) ---
START_RATIO = 0.3     # יחס מתחת לזה -> "התחלה" (משקולות ליד/מתחת לגובה הכתפיים)
TOP_RATIO = 1.3        # יחס מעל זה -> "למעלה" (לחיצה מעל הראש)
CONFIRM_SEC = 0.12      # זמן רציף של מדידה תקפה שנדרש לפני שמצב "מאושר" (מניעת ספירה כפולה)
MAX_MISSING_SEC = 0.5   # אובדן זיהוי רציף מעבר לזה -> איפוס התקדמות חלקית של החזרה הנוכחית


def measure_frame(keypoints_xy, keypoints_conf) -> float | None:
    """מדד הלחיצה: כמה גבוה ממוצע שתי כפות הידיים מעל קו הכתפיים, מנורמל לפי
    רוחב הכתפיים (כדי להיות בלתי-תלוי במרחק מהמצלמה - בניגוד לפיקסלים גולמיים).

    ערך 0 בערך = כפות הידיים בגובה הכתפיים. ערך חיובי גדל ככל שהידיים גבוהות
    יותר מעל הכתפיים (לקראת/בלחיצה מעל הראש).

    דורש שכל שש הנקודות (כתפיים, מרפקים, פרקי ידיים - שני הצדדים) יזוהו בביטחון
    מספיק. המרפקים לא נכנסים לחישוב המספרי עצמו, אבל נדרשים גלויים כדי לוודא
    שרואים את כל הזרוע לאורך התנועה, כמבוקש.
    """
    required = (L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST)
    if min(keypoints_conf[i] for i in required) < KP_CONF_THRESHOLD:
        return None

    l_sh, r_sh = keypoints_xy[L_SHOULDER], keypoints_xy[R_SHOULDER]
    l_wr, r_wr = keypoints_xy[L_WRIST], keypoints_xy[R_WRIST]

    shoulder_width = abs(l_sh[0] - r_sh[0])
    if shoulder_width < MIN_SHOULDER_WIDTH_PX:
        return None  # כתפיים חופפות/כמעט-חופפות -> לא ניתן לנרמל באופן אמין

    avg_shoulder_y = (l_sh[1] + r_sh[1]) / 2.0
    avg_wrist_y = (l_wr[1] + r_wr[1]) / 2.0
    return float((avg_shoulder_y - avg_wrist_y) / shoulder_width)


def raw_class(metric: float | None) -> str | None:
    if metric is None:
        return None
    if metric <= START_RATIO:
        return "start"
    if metric >= TOP_RATIO:
        return "top"
    return "mid"


class PressCounter:
    """מכונת מצבים לספירת חזרות לחיצת כתפיים: התחלה -> עלייה -> למעלה -> ירידה
    -> התחלה, + מדידת משך עלייה/ירידה לכל חזרה. אותו עיקרון בדיוק כמו
    rep_logic.RepCounter (הוכח ונבדק שם), אבל מודול/מצב עצמאי לגמרי, עם כיוון
    הפוך (השיא הוא *מקסימום* המדד - "למעלה" - במקום מינימום כמו בסקוואט).

    שימוש: counter.step(dt, metric) לכל פריים, כאשר dt הוא משך הפריים הזה
    בשניות (לא זמן מצטבר) ו-metric הוא הערך מ-measure_frame(), או None אם
    המדידה חסרה (למשל בגלל הסתרה או כמה אנשים בפריים).

    - מצב ("start"/"top") "מאושר" רק אחרי שהתנאי הגולמי החזיק ברציפות לפחות
      CONFIRM_SEC שניות של מדידה תקפה בפועל - מונע ספירה כפולה מרעידות ליד הסף.
    - חזרה נספרת רק במעבר מאושר ל-"התחלה" אחרי שהגיעו ל-"למעלה" מאושר באותו
      מחזור, ורק אם כבר הייתה "התחלה" מאושרת בעבר (have_start_baseline) - אין
      ספירה לרצף שמתחיל באמצע התנועה.
    - אובדן זיהוי ממושך (MAX_MISSING_SEC) מאפס את reached_top ואת המועמדות
      הממתינה לאישור - לא סופר חזרה חלקית/מדומה מפער בזיהוי.
    - תזמון עלייה/ירידה נמדד לפי זמני האירועים הגולמיים (לא מושהים): תחילת
      עלייה = יציאה מ"התחלה"; נקודת השיא = המדד *המקסימלי* בכל המחזור (לא
      חציית סף ה-TOP); סיום = חזרה ל"התחלה".
    """

    def __init__(self):
        self.rep_count = 0
        self.reached_top = False
        self.have_start_baseline = False
        self.display_phase = "RAISING"
        self.t = 0.0

        self._candidate_target: str | None = None
        self._candidate_elapsed = 0.0
        self._candidate_confirmed = False
        self._missing_elapsed = 0.0
        self._missing_reset_done = False

        self._last_raw: str | None = None

        self._cycle_t_start: float | None = None
        self._cycle_max_metric: float | None = None
        self._cycle_max_metric_t: float | None = None
        self._cycle_had_missing = False
        self._cycle_long_loss = False

        self._pending_summary: dict | None = None

    def step(self, dt: float, metric: float | None) -> dict:
        self.t += dt
        r = raw_class(metric)
        just_completed_rep = False
        rep_summary = None

        if r is None:
            self._missing_elapsed += dt
            if self._cycle_t_start is not None:
                self._cycle_had_missing = True
            if not self._missing_reset_done and self._missing_elapsed >= MAX_MISSING_SEC:
                self.reached_top = False
                self._candidate_target = None
                self._candidate_elapsed = 0.0
                self._candidate_confirmed = False
                self._missing_reset_done = True
                self.display_phase = "LOST"
                if self._cycle_t_start is not None:
                    self._cycle_long_loss = True
        else:
            self._missing_elapsed = 0.0
            self._missing_reset_done = False

            # --- תזמון גולמי: יציאה מ"התחלה", עדכון שיא הלחיצה, חזרה ל"התחלה" ---
            if self._last_raw == "start" and r != "start" and self._cycle_t_start is None:
                self._cycle_t_start = self.t
                self._cycle_max_metric = metric
                self._cycle_max_metric_t = self.t
                self._cycle_had_missing = False
                self._cycle_long_loss = False
            elif self._cycle_t_start is not None and metric > self._cycle_max_metric:
                self._cycle_max_metric = metric
                self._cycle_max_metric_t = self.t

            if (
                self._last_raw is not None
                and self._last_raw != "start"
                and r == "start"
                and self._cycle_t_start is not None
            ):
                if self._cycle_long_loss:
                    self._pending_summary = None
                else:
                    self._pending_summary = {
                        "t_start": self._cycle_t_start,
                        "t_top": self._cycle_max_metric_t,
                        "t_end": self.t,
                        "ascent_duration": self._cycle_max_metric_t - self._cycle_t_start,
                        "descent_duration": self.t - self._cycle_max_metric_t,
                        "estimated": self._cycle_had_missing,
                    }
                self._cycle_t_start = None
                self._cycle_max_metric = None
                self._cycle_max_metric_t = None

            self._last_raw = r

            # --- מכונת המצבים לספירה (עם השהיית אישור) ---
            if r in ("start", "top"):
                if self._candidate_target != r:
                    self._candidate_target = r
                    self._candidate_elapsed = 0.0
                    self._candidate_confirmed = False
                self._candidate_elapsed += dt
                if not self._candidate_confirmed and self._candidate_elapsed >= CONFIRM_SEC:
                    self._candidate_confirmed = True
                    if r == "top":
                        if self.have_start_baseline:
                            self.reached_top = True
                    elif r == "start":
                        self.have_start_baseline = True
                        if self.reached_top:
                            self.rep_count += 1
                            just_completed_rep = True
                            if self._pending_summary is not None:
                                rep_summary = dict(self._pending_summary, rep_number=self.rep_count)
                            self._pending_summary = None
                        self.reached_top = False
            else:
                self._candidate_target = None
                self._candidate_elapsed = 0.0
                self._candidate_confirmed = False

            if r == "start":
                self.display_phase = "START"
            elif r == "top":
                self.display_phase = "TOP"
            else:
                self.display_phase = "LOWERING" if self.reached_top else "RAISING"

        return {
            "phase": self.display_phase,
            "rep_count": self.rep_count,
            "just_completed_rep": just_completed_rep,
            "rep_summary": rep_summary,
        }
