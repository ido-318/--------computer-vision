"""ספירת חזרות סקוואט על בסיס מכונת מצבים של זווית ברך (אגן-ברך-קרסול, צד שמאל).

שימוש:
    venv/bin/python scripts/rep_counter_gif.py <נתיב ל-GIF>

לוגיקת הזווית והספירה עצמה נמצאת ב-rep_logic.py (בלי תלות ב-YOLO, כדי שאפשר לבדוק
אותה בנפרד עם רצפים מדומים — ראו test_rep_counter.py). הקובץ הזה רק מריץ YOLO על
כל פריים ומעביר את הזווית שנמדדה ל-RepCounter.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from rep_logic import STAND_ANGLE, DEPTH_ANGLE, CONFIRM_SEC, MAX_MISSING_SEC, RepCounter, measure_frame

MODEL_NAME = "yolo11n-pose.pt"


def draw_overlay(frame: Image.Image, angle: float | None, phase: str, rep_count: int) -> Image.Image:
    frame = frame.copy()
    draw = ImageDraw.Draw(frame)
    font = ImageFont.load_default(size=38)
    angle_text = f"L knee angle: {angle:.1f} deg" if angle is not None else "L knee angle: N/A"
    lines = [angle_text, f"State: {phase}", f"Reps: {rep_count}"]
    pad = 8
    y = 0
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        box_w, box_h = bbox[2] - bbox[0] + 2 * pad, bbox[3] - bbox[1] + 2 * pad
        draw.rectangle((0, y, box_w, y + box_h), fill=(0, 0, 0))
        draw.text((pad, y + pad // 2), line, font=font, fill=(0, 255, 255))
        y += box_h
    return frame


def run(gif_path: Path, model: YOLO, verbose: bool = True):
    src = Image.open(gif_path)
    n_frames = getattr(src, "n_frames", 1)
    loop = src.info.get("loop", 0)

    durations: list[int] = []
    annotated_frames: list[Image.Image] = []
    angles: list[float | None] = []
    phases: list[str] = []
    rep_counts: list[int] = []

    counter = RepCounter()

    for i in range(n_frames):
        src.seek(i)
        duration = src.info.get("duration", 100)
        durations.append(duration)

        frame_rgb = np.array(src.convert("RGB"))
        frame_bgr = frame_rgb[:, :, ::-1]
        result = model(frame_bgr, verbose=False)[0]

        if result.keypoints is not None and len(result.keypoints) > 0:
            xy = result.keypoints[0].xy[0].tolist()
            conf = result.keypoints[0].conf[0].tolist()
        else:
            xy = [[0.0, 0.0]] * 17
            conf = [0.0] * 17

        angle = measure_frame(xy, conf)
        info = counter.step(duration / 1000.0, angle)
        angles.append(angle)
        phases.append(info["phase"])
        rep_counts.append(info["rep_count"])

        if verbose and info["just_completed_rep"]:
            print(f"  -> חזרה הושלמה בפריים {i}/{n_frames - 1} (t={counter.t:.2f}s), סה\"כ עד כה: {info['rep_count']}")

        plotted_bgr = result.plot()
        frame_annotated = Image.fromarray(plotted_bgr[:, :, ::-1])
        annotated_frames.append(draw_overlay(frame_annotated, angle, info["phase"], info["rep_count"]))

    return annotated_frames, durations, loop, angles, phases, rep_counts


def main() -> None:
    if len(sys.argv) != 2:
        print(f"שימוש: {sys.argv[0]} <נתיב ל-GIF>")
        sys.exit(1)

    gif_path = Path(sys.argv[1])
    if not gif_path.exists():
        print(f"קובץ לא נמצא: {gif_path}")
        sys.exit(1)

    print(f"טוען מודל: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)

    print(f"ספי ספירה (לניסוי בלבד): STAND_ANGLE>={STAND_ANGLE}, DEPTH_ANGLE<={DEPTH_ANGLE}, "
          f"CONFIRM_SEC={CONFIRM_SEC}, MAX_MISSING_SEC={MAX_MISSING_SEC}")

    annotated_frames, durations, loop, angles, phases, rep_counts = run(gif_path, model)
    n_frames = len(annotated_frames)

    out_gif = gif_path.with_name(f"{gif_path.stem}_reps.gif")
    annotated_frames[0].save(
        out_gif, save_all=True, append_images=annotated_frames[1:], duration=durations, loop=loop
    )
    print(f"\nנשמר: {out_gif} ({n_frames} פריימים, תזמון מקורי נשמר)")
    print(f"ספירה סופית: {rep_counts[-1]} חזרות")

    print("\n--- בדיקה: קטע שנחתך בתחתית לא אמור להיספר כחזרה מלאה ---")
    bottom_idx = phases.index("BOTTOM") if "BOTTOM" in phases else None
    if bottom_idx is not None:
        cutoff = min(bottom_idx + 2, n_frames)
        truncated_count = rep_counts[cutoff - 1]
        print(f"אם הסרטון היה נחתך בפריים {cutoff - 1} (סמוך לתחתית, לפני עלייה חזרה): "
              f"ספירה = {truncated_count} (צפוי: 0)")


if __name__ == "__main__":
    main()
