"""הגדרת מפתח Anthropic API עבור המאמן החכם של "מאמן התנועה שלי", ב-macOS Keychain.

הרצה:
    venv/bin/python webapp/setup_api_key.py

מבקש את המפתח בהקלדה מוסתרת (לא מוצג על המסך, לא בהיסטוריית ה-shell) ושומר
אותו ב-Keychain של macOS (לא בקובץ, לא בקוד, לא במשתנה סביבה קבוע). השרת
(webapp/server.py) יקרא אותו משם אוטומטית בכל הפעלה - אלא אם הגדרתם במפורש
ANTHROPIC_API_KEY לאותה הרצה ספציפית, שאז הוא יקבל עדיפות.

פקודות נוספות:
    venv/bin/python webapp/setup_api_key.py --delete   # מוחק את המפתח השמור
    venv/bin/python webapp/setup_api_key.py --status    # מציג רק אם יש מפתח שמור, לא את הערך
"""

import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from keychain_secret import SERVICE_NAME, delete_stored_key, get_stored_key, set_stored_key  # noqa: E402


def cmd_status() -> None:
    if get_stored_key():
        print(f"יש מפתח שמור ב-Keychain (שירות: {SERVICE_NAME}).")
    else:
        print("אין מפתח שמור ב-Keychain כרגע.")


def cmd_delete() -> None:
    if delete_stored_key():
        print("המפתח נמחק מה-Keychain.")
    else:
        print("לא נמצא מפתח שמור למחיקה.")


def cmd_set() -> None:
    print("הגדרת מפתח Anthropic API עבור המאמן החכם.")
    print(f"המפתח יישמר ב-macOS Keychain (שירות: {SERVICE_NAME}) - לא בקובץ ולא בקוד.")
    if get_stored_key():
        print("שים לב: כבר יש מפתח שמור. הזנה חדשה תחליף אותו.")

    key = getpass.getpass("הדבק/הקלד את מפתח ה-API (ההקלדה לא תוצג על המסך): ").strip()
    if not key:
        print("לא הוזן מפתח - לא בוצע שינוי.")
        sys.exit(1)
    if not key.startswith("sk-ant-"):
        print("אזהרה: המפתח לא מתחיל ב-'sk-ant-' - זה לא נראה כמו מפתח Anthropic תקין. נשמר בכל זאת.")

    set_stored_key(key)
    del key  # לא נחוצה יותר בזיכרון התהליך הזה

    print("נשמר בהצלחה ב-Keychain.")
    print("עכשיו אפשר:")
    print("  1. לבדוק חיבור בפועל:  venv/bin/python webapp/test_api_connection.py")
    print("  2. להפעיל את השרת:    venv/bin/python webapp/server.py")


def main() -> None:
    args = sys.argv[1:]
    if "--delete" in args:
        cmd_delete()
    elif "--status" in args:
        cmd_status()
    else:
        cmd_set()


if __name__ == "__main__":
    main()
