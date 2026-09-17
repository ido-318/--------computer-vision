"""בדיקת זיהוי תנוחה (pose) על תמונה בודדת של אדם עם כל הגוף בפריים.

שימוש:
    venv/bin/python scripts/test_pose.py <נתיב לתמונה>

שומר עותק של התמונה עם נקודות הגוף (keypoints) והשלד (skeleton) מצוירים עליה,
לצד הקובץ המקורי, בשם <שם_מקורי>_pose<סיומת>.
"""

import sys
from pathlib import Path

from ultralytics import YOLO

# yolo11n-pose: המודל הקטן ביותר במשפחת YOLO11-Pose (~6MB, 17 נקודות גוף בפורמט COCO).
# נבחר משיקולי מהירות ופשטות לבדיקה ראשונית; יורד אוטומטית בפעם הראשונה שהוא נטען.
MODEL_NAME = "yolo11n-pose.pt"


def main() -> None:
    if len(sys.argv) != 2:
        print(f"שימוש: {sys.argv[0]} <נתיב לתמונה>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"קובץ לא נמצא: {image_path}")
        sys.exit(1)

    print(f"טוען מודל: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)

    results = model(str(image_path))
    result = results[0]

    num_people = 0 if result.keypoints is None else len(result.keypoints)
    print(f"זוהו {num_people} אנשים בתמונה")

    output_path = image_path.with_name(f"{image_path.stem}_pose{image_path.suffix}")
    result.save(filename=str(output_path))
    print(f"נשמר: {output_path}")


if __name__ == "__main__":
    main()
