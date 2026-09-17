"""ניתוח סקוואט חי ממצלמת המחשב (OpenCV) - ספירת חזרות ומדידת קצב.

שימוש:
    venv/bin/python scripts/live_squat.py
    venv/bin/python scripts/live_squat.py --side L
    venv/bin/python scripts/live_squat.py --pace-descent 2.0 3.0 --pace-ascent 1.0 2.0

בקרה:
    L / R - בחירת צד (שמאל/ימין) בתחילת ההפעלה בלבד. קבוע למשך הסט.
    Q     - יציאה בכל שלב, גם במסך בחירת הצד.

לוגיקת הזווית, הספירה, התזמון ומשוב הקצב מיובאת מ-rep_logic.py ללא שינוי עקרוני
(רק נוסף לה פרמטר side ל-measure_frame, כדי לתמוך גם בצד ימין). לא נשמר וידאו
לדיסק - תצוגה חיה בלבד.
"""

import argparse
import sys
import time

import cv2
from ultralytics import YOLO

from rep_logic import RepCounter, measure_frame, pace_feedback

MODEL_NAME = "yolo11n-pose.pt"
WINDOW_NAME = "Live Squat Analysis"
CAMERA_READ_FAILURE_LIMIT = 30  # פריימים רצופים כושלים לפני שמוותרים על החיבור למצלמה

INSTRUCTIONS_HE = (
    "עמוד מהצד למצלמה, כשכל הגוף וכפות הרגליים בפריים.\n"
    "בחר צד למדידה: הקש L לצד שמאל, או R לצד ימין (הבחירה תישאר קבועה למשך הסט).\n"
    "בכל שלב אפשר להקיש Q ליציאה."
)


def parse_args():
    p = argparse.ArgumentParser(description="ניתוח סקוואט חי ממצלמת המחשב")
    p.add_argument("--side", choices=["L", "R"], default=None,
                    help="קביעת צד מראש, בלי בחירה אינטראקטיבית במסך הפתיחה")
    p.add_argument("--camera-index", type=int, default=0, help="אינדקס מצלמה ל-cv2.VideoCapture (ברירת מחדל 0)")
    p.add_argument("--pace-descent", nargs=2, type=float, metavar=("MIN", "MAX"), default=None,
                    help="הפעלת יעד קצב לירידה בשניות. כבוי כברירת מחדל.")
    p.add_argument("--pace-ascent", nargs=2, type=float, metavar=("MIN", "MAX"), default=None,
                    help="הפעלת יעד קצב לעלייה בשניות. כבוי כברירת מחדל. יש להגדיר יחד עם --pace-descent.")
    args = p.parse_args()
    if (args.pace_descent is None) != (args.pace_ascent is None):
        p.error("יש להגדיר גם --pace-descent וגם --pace-ascent יחד (או לא להגדיר בכלל - יעד הקצב כבוי כברירת מחדל).")
    return args


def build_pace_target(args) -> dict | None:
    if args.pace_descent is None:
        return None
    return {"descent_range": tuple(args.pace_descent), "ascent_range": tuple(args.pace_ascent)}


def draw_lines(frame, lines, origin=(10, 24), color=(0, 255, 255), scale=0.55, thickness=1, line_gap=22):
    x, y = origin
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        cv2.rectangle(frame, (x - 4, y - th - 4), (x + tw + 4, y + 6), (0, 0, 0), -1)
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        y += line_gap
    return y


def print_camera_permission_help():
    print("שגיאה: לא ניתן לקבל תמונה מהמצלמה.")
    print("סיבה סבירה ביותר: אין הרשאת מצלמה לאפליקציית הטרמינל שממנה מריצים את הסקריפט.")
    print("ב-macOS: System Settings -> Privacy & Security -> Camera -> ודא שהאפליקציה "
          "(Terminal / iTerm / VS Code וכו') מסומנת שם, ואז הרץ שוב את הסקריפט.")


def select_side(cap, preselected: str | None) -> str | None:
    """מחזירה 'L' או 'R', או None אם המשתמש יצא (Q) או שהמצלמה נכשלה."""
    if preselected is not None:
        print(f"צד נבחר מראש: {'שמאל' if preselected == 'L' else 'ימין'}")
        return preselected

    print("ממתין לבחירת צד (L / R), או Q ליציאה...")
    failures = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            failures += 1
            if failures > CAMERA_READ_FAILURE_LIMIT:
                return None
            continue
        failures = 0
        frame = cv2.flip(frame, 1)
        draw_lines(frame, [
            "Stand sideways to the camera",
            "Full body + feet must be in frame",
            "Press L = left side / R = right side",
            "Press Q to quit",
        ], color=(0, 255, 255))
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("l"), ord("L")):
            print("נבחר צד שמאל. הצד יישאר קבוע למשך הסט.")
            return "L"
        if key in (ord("r"), ord("R")):
            print("נבחר צד ימין. הצד יישאר קבוע למשך הסט.")
            return "R"
        if key in (ord("q"), ord("Q")):
            return None


def guidance_for_frame(num_people: int, angle) -> str | None:
    if num_people == 0:
        return "No person detected - step into frame"
    if num_people > 1:
        return "Multiple people detected - analysis paused"
    if angle is None:
        return "Body not fully visible - stand sideways, full body + feet in frame, check lighting"
    return None


def main() -> None:
    args = parse_args()
    pace_target = build_pace_target(args)

    print(INSTRUCTIONS_HE)
    if pace_target is None:
        print("יעד קצב: כבוי (ברירת מחדל). אפשר להפעיל עם --pace-descent MIN MAX --pace-ascent MIN MAX.")
    else:
        print(f"יעד קצב פעיל (הוגדר במפורש): ירידה {pace_target['descent_range']}s, "
              f"עלייה {pace_target['ascent_range']}s.")

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print_camera_permission_help()
        sys.exit(1)

    cv2.namedWindow(WINDOW_NAME)

    side = select_side(cap, args.side)
    if side is None:
        cap.release()
        cv2.destroyAllWindows()
        print("יציאה.")
        sys.exit(0)

    print("טוען מודל...")
    model = YOLO(MODEL_NAME)
    print(f"מוכן. עמוד בצד ({'שמאל' if side == 'L' else 'ימין'} פונה למצלמה) ועמוד ישר כדי להתחיל "
          "(הספירה תתחיל רק אחרי שהמערכת מזהה עמידה יציבה ברורה).")

    counter = RepCounter()
    last_t = time.monotonic()
    banner_text: str | None = None
    banner_until = 0.0
    read_failures = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                read_failures += 1
                if read_failures > CAMERA_READ_FAILURE_LIMIT:
                    print("שגיאה: אובדן חיבור למצלמה (או שנשללה הרשאה תוך כדי ריצה). עוצר.")
                    break
                continue
            read_failures = 0
            frame = cv2.flip(frame, 1)

            now = time.monotonic()
            dt = now - last_t
            last_t = now

            result = model(frame, verbose=False)[0]
            num_people = 0 if result.keypoints is None else len(result.keypoints)

            angle = None
            if num_people == 1:
                xy = result.keypoints[0].xy[0].tolist()
                conf = result.keypoints[0].conf[0].tolist()
                angle = measure_frame(xy, conf, side=side)
            # num_people == 0 או > 1 -> angle נשאר None (משהים/לא זמין), מנגנון
            # המדידות החסרות הקיים ב-RepCounter מטפל בזה בלי צורך בקוד נוסף כאן.

            info = counter.step(dt, angle)

            if info["rep_summary"] is not None:
                s = info["rep_summary"]
                tag = " (משוער - הייתה מדידה חסרה במהלך החזרה)" if s["estimated"] else ""
                print(f"  >>> חזרה {s['rep_number']}: משך ירידה {s['descent_duration']:.2f} שניות, "
                      f"משך עלייה {s['ascent_duration']:.2f} שניות{tag}")
                feedback = pace_feedback(s, pace_target)
                if feedback is not None:
                    print(f"      דגש קצב: {feedback}")
                banner_text = (f"Rep {s['rep_number']}: descent {s['descent_duration']:.2f}s / "
                                f"ascent {s['ascent_duration']:.2f}s")
                banner_until = now + 3.0

            display = result.plot()

            lines = [
                f"Side: {side}",
                f"Angle: {angle:.1f} deg" if angle is not None else "Angle: N/A",
                f"State: {info['phase']}",
                f"Reps: {info['rep_count']}",
            ]
            if not counter.have_stand_baseline:
                lines.append("Stand straight and still to begin")
            guidance = guidance_for_frame(num_people, angle)
            if guidance is not None:
                lines.append(guidance)
            y_after = draw_lines(display, lines)

            if banner_text is not None and now < banner_until:
                draw_lines(display, [banner_text], origin=(10, y_after + 10), color=(0, 255, 0))
            elif now >= banner_until:
                banner_text = None

            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"המצלמה שוחררה. סה\"כ חזרות בסט: {counter.rep_count}. להתראות.")


if __name__ == "__main__":
    main()
