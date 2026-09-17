"""חישוב זווית ברך דו-ממדית (אגן-ברך-קרסול, צד שמאל) על כל פריימי GIF.

שימוש:
    venv/bin/python scripts/knee_angle_gif.py <נתיב ל-GIF>

איך החישוב עובד:
- לכל פריים לוקחים שלוש נקודות בצד שמאל בלבד (אותו צד לאורך כל האנימציה): אגן (hip),
  ברך (knee), קרסול (ankle). הברך היא קודקוד הזווית.
- בונים שני וקטורים מהברך: אחד לכיוון האגן (v1 = hip - knee), אחד לכיוון הקרסול
  (v2 = ankle - knee). הזווית ביניהם מחושבת עם נוסחת הקוסינוס:
      angle = arccos( (v1 . v2) / (|v1| * |v2|) )
  בטווח 0-180 מעלות. רגל ישרה לגמרי -> אגן/ברך/קרסול כמעט על קו ישר -> הזווית קרובה ל-180.
  ככל שהברך מתכופפת יותר, הזווית קטנה יותר.
- מחשבים רק אם שלוש הנקודות (אגן, ברך, קרסול) עברו סף ביטחון של 0.7 ומעלה; אחרת המדידה
  מסומנת כחסרה (None / NaN בגרף, "לא זמין" על האנימציה) במקום להציג מספר לא אמין.
- מטפלים בנקודות חופפות/כמעט-חופפות (אורך אחד מהווקטורים קרוב לאפס, למשל אם שתי נקודות
  זוהו כמעט באותו פיקסל) — גם זה מסומן כמדידה חסרה, כדי למנוע חלוקה באפס/זווית לא יציבה.

שומר:
- <שם>_angle.gif  — האנימציה עם השלד + הזווית כתובה על כל פריים, תזמון מקורי נשמר.
- <שם>_angle.png  — גרף הזווית לאורך זמן (ציר X בשניות אמיתיות, לפי משכי הפריימים המקוריים).

אין כאן ספירת חזרות ואין קביעה אם התרגיל תקין — רק המדידה עצמה.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

MODEL_NAME = "yolo11n-pose.pt"
L_HIP, L_KNEE, L_ANKLE = 11, 13, 15
CONF_THRESHOLD = 0.7
DEGENERATE_EPS_PX = 2.0  # אורך וקטור מתחת לזה (בפיקסלים) נחשב "נקודות חופפות"


def knee_angle_deg(hip_xy, knee_xy, ankle_xy) -> float | None:
    v1 = np.array(hip_xy) - np.array(knee_xy)
    v2 = np.array(ankle_xy) - np.array(knee_xy)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < DEGENERATE_EPS_PX or n2 < DEGENERATE_EPS_PX:
        return None  # נקודות חופפות/כמעט-חופפות -> לא ניתן לחשב כיוון אמין
    cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def measure_frame(keypoints_xy, keypoints_conf) -> float | None:
    confs = [keypoints_conf[i] for i in (L_HIP, L_KNEE, L_ANKLE)]
    if min(confs) < CONF_THRESHOLD:
        return None  # מתחת לסף ביטחון -> מדידה חסרה
    hip, knee, ankle = (keypoints_xy[i] for i in (L_HIP, L_KNEE, L_ANKLE))
    return knee_angle_deg(hip, knee, ankle)


def draw_angle_label(frame: Image.Image, angle: float | None) -> Image.Image:
    frame = frame.copy()
    draw = ImageDraw.Draw(frame)
    font = ImageFont.load_default(size=42)
    # טקסט אנגלי בלבד על גבי הפיקסלים כדי להימנע מבעיות כיווניות RTL/עיצוב גופן בעברית ב-PIL
    text = f"L knee angle: {angle:.1f} deg" if angle is not None else "L knee angle: N/A"
    pad = 10
    bbox = draw.textbbox((0, 0), text, font=font)
    box_w, box_h = bbox[2] - bbox[0] + 2 * pad, bbox[3] - bbox[1] + 2 * pad
    draw.rectangle((0, 0, box_w, box_h), fill=(0, 0, 0))
    draw.text((pad, pad), text, font=font, fill=(0, 255, 255) if angle is not None else (255, 80, 80))
    return frame


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

    src = Image.open(gif_path)
    n_frames = getattr(src, "n_frames", 1)
    loop = src.info.get("loop", 0)
    print(f"קובץ GIF עם {n_frames} פריימים")

    durations: list[int] = []
    annotated_frames: list[Image.Image] = []
    angles: list[float | None] = []
    hip_ys: list[float] = []  # לבחירת פריימי עמידה/אמצע/תחתית, אותו צד (שמאל) לאורך כל הדרך

    for i in range(n_frames):
        src.seek(i)
        durations.append(src.info.get("duration", 100))

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
        angles.append(angle)
        hip_ys.append(xy[L_HIP][1] if conf[L_HIP] > 0.1 else float("nan"))

        plotted_bgr = result.plot()
        frame_annotated = Image.fromarray(plotted_bgr[:, :, ::-1])
        annotated_frames.append(draw_angle_label(frame_annotated, angle))

    out_gif = gif_path.with_name(f"{gif_path.stem}_angle.gif")
    annotated_frames[0].save(
        out_gif, save_all=True, append_images=annotated_frames[1:], duration=durations, loop=loop
    )
    print(f"\nנשמר: {out_gif}")

    # ציר זמן אמיתי לפי משכי הפריימים המקוריים (מילישניות -> שניות, מצטבר)
    t_seconds = np.cumsum([0] + durations[:-1]) / 1000.0

    fig, ax = plt.subplots(figsize=(10, 4.5))
    angles_arr = np.array([a if a is not None else np.nan for a in angles])
    ax.plot(t_seconds, angles_arr, marker="o", markersize=3, linewidth=1.5, color="#1f77b4")
    missing = np.isnan(angles_arr)
    if missing.any():
        ax.scatter(t_seconds[missing], np.full(missing.sum(), 0), marker="x", color="red", label="מדידה חסרה")
        ax.legend()
    ax.set_xlabel("זמן (שניות, לפי תזמון הפריימים המקורי)")
    ax.set_ylabel("זווית ברך שמאל (מעלות)")
    ax.set_title("זווית ברך לאורך הזמן — אגן-ברך-קרסול (צד שמאל)")
    ax.set_ylim(0, 190)
    ax.axhline(180, color="gray", linestyle="--", linewidth=0.8, label="רגל ישרה (180°)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = gif_path.with_name(f"{gif_path.stem}_angle.png")
    fig.savefig(out_png, dpi=150)
    print(f"נשמר: {out_png}")

    # אותו היגיון כמו ב-test_pose_gif.py לבחירת עמידה/אמצע/תחתית, אבל לפי אגן שמאל בלבד
    hip_arr = np.array(hip_ys)
    t_bottom = int(np.nanargmax(hip_arr))
    t_stand = int(np.nanargmin(hip_arr[: t_bottom + 1]))
    mid_target = (hip_arr[t_stand] + hip_arr[t_bottom]) / 2
    segment = hip_arr[t_stand : t_bottom + 1]
    t_mid = t_stand + int(np.nanargmin(np.abs(segment - mid_target)))

    print("\nזוויות בפריימים מייצגים:")
    for label, idx in [("עמידה", t_stand), ("אמצע הירידה", t_mid), ("תחתית הסקוואט", t_bottom)]:
        a = angles[idx]
        a_str = f"{a:.1f}°" if a is not None else "לא זמין (מתחת לסף ביטחון או נקודות חופפות)"
        print(f"  פריים {idx}/{n_frames - 1} ({label}, t={t_seconds[idx]:.2f}s): {a_str}")


if __name__ == "__main__":
    main()
