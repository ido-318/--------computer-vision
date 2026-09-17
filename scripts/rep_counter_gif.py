"""ספירת חזרות סקוואט + מדידת משך ירידה/עלייה לכל חזרה, על בסיס מכונת המצבים
וזווית הברך (אגן-ברך-קרסול, צד שמאל).

שימוש:
    venv/bin/python scripts/rep_counter_gif.py <נתיב ל-GIF>

לוגיקת הזווית, הספירה והתזמון נמצאת ב-rep_logic.py (בלי תלות ב-YOLO, כדי שאפשר
לבדוק אותה בנפרד עם רצפים מדומים — ראו test_rep_counter.py). הקובץ הזה רק מריץ
YOLO על כל פריים, מעביר את הזווית שנמדדה ל-RepCounter, ומפיק את הפלטים.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from rep_logic import (
    STAND_ANGLE,
    DEPTH_ANGLE,
    CONFIRM_SEC,
    MAX_MISSING_SEC,
    PACE_TARGET_EXAMPLE,
    RepCounter,
    measure_frame,
    pace_feedback,
)

MODEL_NAME = "yolo11n-pose.pt"

# יעד קצב אופציונלי שהמאמן יכול להגדיר (טווח שניות לירידה/עלייה). PACE_TARGET_EXAMPLE
# הם ערכי דוגמה *לצורך בדיקת התוכנה בלבד* (ראו rep_logic.py) - לא המלצת אימון.
# אפשר לשנות כאן, או להציב PACE_TARGET = None כדי לקבל מדידות בלבד בלי דגש קצב.
PACE_TARGET = PACE_TARGET_EXAMPLE


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


def print_rep_summary_hebrew(summary: dict) -> None:
    n = summary["rep_number"]
    descent = summary["descent_duration"]
    ascent = summary["ascent_duration"]
    tag = " (משוער - הייתה מדידה חסרה במהלך החזרה)" if summary["estimated"] else ""
    print(f"  >>> חזרה {n}: משך ירידה {descent:.2f} שניות, משך עלייה {ascent:.2f} שניות{tag}")
    feedback = pace_feedback(summary, PACE_TARGET)
    if feedback is not None:
        print(f"      דגש קצב: {feedback}")


def run(gif_path: Path, model: YOLO, verbose: bool = True):
    src = Image.open(gif_path)
    n_frames = getattr(src, "n_frames", 1)
    loop = src.info.get("loop", 0)

    durations: list[int] = []
    annotated_frames: list[Image.Image] = []
    angles: list[float | None] = []
    phases: list[str] = []
    rep_counts: list[int] = []
    rep_summaries: list[dict] = []
    times: list[float] = []

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
        times.append(counter.t)

        if info["rep_summary"] is not None:
            rep_summaries.append(info["rep_summary"])
            if verbose:
                print_rep_summary_hebrew(info["rep_summary"])

        plotted_bgr = result.plot()
        frame_annotated = Image.fromarray(plotted_bgr[:, :, ::-1])
        annotated_frames.append(draw_overlay(frame_annotated, angle, info["phase"], info["rep_count"]))

    return annotated_frames, durations, loop, angles, phases, rep_counts, rep_summaries, times


def plot_angle_with_rep_markers(times, angles, rep_summaries, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    angles_arr = np.array([a if a is not None else np.nan for a in angles])
    ax.plot(times, angles_arr, linewidth=1.3, color="#1f77b4", zorder=1)
    ax.axhline(STAND_ANGLE, color="gray", linestyle="--", linewidth=0.8, label=f"STAND_ANGLE={STAND_ANGLE:.0f}")
    ax.axhline(DEPTH_ANGLE, color="gray", linestyle=":", linewidth=0.8, label=f"DEPTH_ANGLE={DEPTH_ANGLE:.0f}")

    # נקודות start/bottom/end מצוירות לפי הזווית האמיתית שלהן (לא לפי קו הסף)
    for idx, s in enumerate(rep_summaries):
        # ערך הזווית בפועל ברגע t_bottom / t_end = מוצא את הפריים הקרוב ביותר בזמן
        t_arr = np.array(times)
        i_bottom = int(np.argmin(np.abs(t_arr - s["t_bottom"])))
        i_end = int(np.argmin(np.abs(t_arr - s["t_end"])))
        i_start = int(np.argmin(np.abs(t_arr - s["t_start"])))
        marker_edge = "orange" if s["estimated"] else "black"
        ax.scatter([times[i_start]], [angles_arr[i_start]], marker="o", s=80, facecolor="lime",
                   edgecolor=marker_edge, zorder=3, label="תחילת ירידה" if idx == 0 else None)
        ax.scatter([times[i_bottom]], [angles_arr[i_bottom]], marker="v", s=90, facecolor="red",
                   edgecolor=marker_edge, zorder=3, label="תחתית (מינימום)" if idx == 0 else None)
        ax.scatter([times[i_end]], [angles_arr[i_end]], marker="s", s=80, facecolor="cyan",
                   edgecolor=marker_edge, zorder=3, label="סיום (חזרה לעמידה)" if idx == 0 else None)

    ax.set_xlabel("זמן (שניות, לפי תזמון הפריימים המקורי)")
    ax.set_ylabel("זווית ברך שמאל (מעלות)")
    ax.set_title("זווית ברך + נקודות תחילה/תחתית/סיום לכל חזרה")
    ax.set_ylim(0, 190)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)


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

    annotated_frames, durations, loop, angles, phases, rep_counts, rep_summaries, times = run(gif_path, model)
    n_frames = len(annotated_frames)

    out_gif = gif_path.with_name(f"{gif_path.stem}_reps.gif")
    annotated_frames[0].save(
        out_gif, save_all=True, append_images=annotated_frames[1:], duration=durations, loop=loop
    )
    print(f"\nנשמר: {out_gif} ({n_frames} פריימים, תזמון מקורי נשמר)")
    print(f"ספירה סופית: {rep_counts[-1]} חזרות")

    out_png = gif_path.with_name(f"{gif_path.stem}_reps_timing.png")
    plot_angle_with_rep_markers(times, angles, rep_summaries, out_png)
    print(f"נשמר: {out_png}")

    print("\n--- בדיקה: קטע שנחתך בתחתית לא אמור להיספר כחזרה מלאה ---")
    bottom_idx = phases.index("BOTTOM") if "BOTTOM" in phases else None
    if bottom_idx is not None:
        cutoff = min(bottom_idx + 2, n_frames)
        truncated_count = rep_counts[cutoff - 1]
        print(f"אם הסרטון היה נחתך בפריים {cutoff - 1} (סמוך לתחתית, לפני עלייה חזרה): "
              f"ספירה = {truncated_count} (צפוי: 0)")


if __name__ == "__main__":
    main()
