"""בדיקת חיבור קצרה מול Anthropic API - מדווחת רק "החיבור הצליח" או הודעת שגיאה
מועילה. לעולם לא מדפיסה את המפתח עצמו או כל תוכן אחר מהבקשה/התשובה.

הרצה (אחרי שהגדרתם מפתח דרך webapp/setup_api_key.py, או ANTHROPIC_API_KEY):
    venv/bin/python webapp/test_api_connection.py

**לא רצה אוטומטית משום מקום** - זו הפעלה ידנית ומודעת בלבד, כמבוקש.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import coach_llm  # noqa: E402


async def main() -> None:
    if not coach_llm.smart_coach_enabled():
        print("אין מפתח API זמין כרגע (לא ב-Keychain ולא במשתנה הסביבה ANTHROPIC_API_KEY).")
        print("הרצה: venv/bin/python webapp/setup_api_key.py")
        sys.exit(1)

    print(f"נמצא מפתח (מקור: {coach_llm.key_source_label()}). שולח בקשת בדיקה קצרה ל-Anthropic...")
    result = await coach_llm.test_connection()
    if result is True:
        print("החיבור הצליח.")
    else:
        print(f"החיבור נכשל: {result}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
