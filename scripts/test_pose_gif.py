"""בדיקת זיהוי תנוחה על כל הפריימים של GIF, עם שמירת אנימציה מחודשת.

שימוש:
    venv/bin/python scripts/test_pose_gif.py <נתיב ל-GIF>

שומר:
- <שם>_pose.gif   — כל הפריימים עם נקודות גוף+שלד מצוירים, אותו תזמון (duration) כמו המקור.
- <שם>_stand.jpg / _mid.jpg / _bottom.jpg — שלושה פריימים מייצגים (עמידה/אמצע ירידה/תחתית),
  שנבחרים אוטומטית לפי גובה נקודת האגן על פני הפריימים.

בלי חישוב זוויות, ספירת חזרות או דגשי אימון — רק זיהוי, ציור, ובדיקת ביטחון.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

MODEL_NAME = "yolo11n-pose.pt"
KEYPOINT_NAMES = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
    "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle",
]
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 11, 12, 13, 14, 15, 16
L_EAR, R_EAR = 3, 4


def hip_y(keypoints_xy, keypoints_conf) -> float:
    ys, weights = [], []
    for idx in (L_HIP, R_HIP):
        c = keypoints_conf[idx]
        if c > 0.1:
            ys.append(keypoints_xy[idx][1])
            weights.append(c)
    if not ys:
        return float("nan")
    return float(np.average(ys, weights=weights))


def visible_side(keypoints_conf) -> str:
    """הצד שנוטה להיות פנוי (פחות מוסתר) לפי ביטחון האוזניים — פרופיל טיפוסי."""
    return "שמאל" if keypoints_conf[L_EAR] >= keypoints_conf[R_EAR] else "ימין"


def print_lower_body_confidence(label: str, keypoints_conf, side: str) -> None:
    idx = {"שמאל": (L_HIP, L_KNEE, L_ANKLE), "ימין": (R_HIP, R_KNEE, R_ANKLE)}[side]
    hip_c, knee_c, ankle_c = (keypoints_conf[i] for i in idx)
    print(f"\n--- {label} (צד גלוי: {side}) ---")
    print(f"  אגן:    conf={hip_c:.3f}")
    print(f"  ברך:    conf={knee_c:.3f}")
    print(f"  קרסול:  conf={ankle_c:.3f}")


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
    hip_ys: list[float] = []
    xy_per_frame: list[list] = []
    conf_per_frame: list[list] = []

    for i in range(n_frames):
        src.seek(i)
        durations.append(src.info.get("duration", 100))

        frame_rgb = np.array(src.convert("RGB"))
        frame_bgr = frame_rgb[:, :, ::-1]  # ultralytics מצפה לסדר ערוצים BGR (מוסכמת OpenCV) בקלט מסוג numpy array
        result = model(frame_bgr, verbose=False)[0]

        plotted_bgr = result.plot()
        annotated_frames.append(Image.fromarray(plotted_bgr[:, :, ::-1]))

        if result.keypoints is not None and len(result.keypoints) > 0:
            xy = result.keypoints[0].xy[0].tolist()
            conf = result.keypoints[0].conf[0].tolist()
        else:
            xy = [[0.0, 0.0]] * 17
            conf = [0.0] * 17

        xy_per_frame.append(xy)
        conf_per_frame.append(conf)
        hip_ys.append(hip_y(xy, conf))

    out_gif = gif_path.with_name(f"{gif_path.stem}_pose.gif")
    annotated_frames[0].save(
        out_gif,
        save_all=True,
        append_images=annotated_frames[1:],
        duration=durations,
        loop=loop,
    )
    print(f"\nנשמר: {out_gif} ({n_frames} פריימים, תזמון מקורי נשמר)")

    hip_arr = np.array(hip_ys)
    # תחתית = הפריים העמוק ביותר בכל האנימציה. עמידה = השפל הכי גבוה *לפני* התחתית
    # (כדי לא להתבלבל עם עמידה חוזרת בסוף התנועה, אחרי העלייה).
    t_bottom = int(np.nanargmax(hip_arr))
    t_stand = int(np.nanargmin(hip_arr[: t_bottom + 1]))
    mid_target = (hip_arr[t_stand] + hip_arr[t_bottom]) / 2
    segment = hip_arr[t_stand : t_bottom + 1]
    t_mid = t_stand + int(np.nanargmin(np.abs(segment - mid_target)))

    keyframes = [("עמידה", t_stand, "stand"), ("אמצע הירידה", t_mid, "mid"), ("תחתית הסקוואט", t_bottom, "bottom")]

    for label, idx, suffix in keyframes:
        out_path = gif_path.with_name(f"{gif_path.stem}_{suffix}.jpg")
        annotated_frames[idx].save(out_path, quality=95)
        side = visible_side(conf_per_frame[idx])
        print(f"\nפריים {idx}/{n_frames - 1} -> {out_path}")
        print_lower_body_confidence(label, conf_per_frame[idx], side)


if __name__ == "__main__":
    main()
