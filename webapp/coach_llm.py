"""אינטגרציה אופציונלית עם Claude (Anthropic API), בצד השרת בלבד, למאמן החכם
של "מאמן התנועה שלי".

**מקור המפתח (לפי סדר עדיפות):**
1. משתנה הסביבה `ANTHROPIC_API_KEY`, אם הוגדר במפורש להרצה הנוכחית - override מכוון וזמני.
2. אחרת, המפתח השמור ב-macOS Keychain (הוגדר פעם אחת דרך `webapp/setup_api_key.py`,
   נשמר בין הפעלות, לא בקובץ ולא בקוד - ראו `webapp/keychain_secret.py`).
3. אם אין אף אחד מהשניים - המאמן החכם כבוי לגמרי, והאפליקציה ממשיכה לעבוד רק עם
   המשוב מבוסס-הכללים הקיים (`rep_logic.pace_feedback`) - בלי שום שינוי בהתנהגות הקיימת.

המפתח **אף פעם** לא מודפס, לא נכתב ליומן (גם לא בהודעות שגיאה - רק שם סוג
השגיאה, לא תוכן החריגה), לא נשלח לדפדפן, ולא מקודד בקוד.

**לא נשלחות תמונות/וידאו למודל בשלב הזה** - רק נתונים מספריים/טקסטואליים
שכבר קיימים במערכת (מספר חזרה, משכי ירידה/עלייה, יעד קצב, המשוב הקיים וכו').

הנחיות המגבלה על המודל (ב-system prompt): להשתמש רק בנתונים שסופקו בפועל,
לא להמציא נתונים/תצפיות, לא לקבוע אם הטכניקה תקינה, לא לאבחן פציעה, לא לטעון
שראה תמונה/וידאו, ולהגיד בפירוש כשהמידע לא מספיק.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from keychain_secret import get_stored_key  # noqa: E402

MODEL_ID = "claude-haiku-4-5"
REQUEST_TIMEOUT_SEC = 8.0

_client = None
_enabled: bool | None = None
_key_source: str | None = None  # לצורכי לוג בלבד ("env" / "keychain" / None) - אף פעם לא הערך עצמו


def _resolve_api_key() -> str | None:
    """מחזירה את המפתח לשימוש, לפי סדר העדיפות (env קודם, אחר כך Keychain). לא מדפיסה כלום."""
    global _key_source
    env_key = os.environ.get("ANTHROPIC_API_KEY")
    if env_key:
        _key_source = "env"
        return env_key
    stored_key = get_stored_key()
    if stored_key:
        _key_source = "keychain"
        return stored_key
    _key_source = None
    return None


def smart_coach_enabled() -> bool:
    """True אם יש מפתח זמין (env או Keychain). נבדק פעם אחת ונשמר בזיכרון."""
    global _enabled
    if _enabled is None:
        _enabled = bool(_resolve_api_key())
    return _enabled


def key_source_label() -> str:
    """תיאור *מקור* המפתח בעברית, לצורך תצוגה/לוג - אף פעם לא הערך של המפתח עצמו."""
    smart_coach_enabled()  # מוודא ש-_key_source חושב
    return {"env": "משתנה סביבה ANTHROPIC_API_KEY", "keychain": "macOS Keychain"}.get(_key_source, "לא הוגדר")


def _get_client():
    global _client
    if _client is None:
        import anthropic
        key = _resolve_api_key()
        _client = anthropic.AsyncAnthropic(api_key=key)
    return _client


SYSTEM_PROMPT = (
    'אתה עוזר קצר ותמציתי בתוך אפליקציית אימון סקוואט בשם "מאמן התנועה שלי". '
    "אתה מקבל אך ורק נתונים מספריים/טקסטואליים שנמדדו על ידי מערכת ראייה ממוחשבת - "
    "אתה עצמך לא רואה שום תמונה, וידאו, או תנוחה.\n"
    "כללים מחייבים:\n"
    "1. ענה תמיד בעברית, במשפט אחד קצר (עד כ-25 מילים).\n"
    "2. התבסס אך ורק על הנתונים שסופקו לך בהודעה הנוכחית. אל תמציא נתונים, תצפיות, "
    "או פרטים שלא נשלחו אליך במפורש.\n"
    "3. לעולם אל תקבע אם הטכניקה 'תקינה' או 'לא תקינה', ואל תאבחן פציעה או בעיה רפואית כלשהי.\n"
    "4. לעולם אל תטען שראית תנוחה, זווית, וידאו או תמונה - קיבלת רק מספרים.\n"
    "5. אם המידע שסופק לא מספיק כדי לענות על השאלה או לתת דגש משמעותי, אמור זאת "
    "בפירוש (למשל: \"אין עדיין מספיק נתונים לכך\") במקום לנחש או להמציא.\n"
    "6. מותר להתייחס לקצב (מהר/לאט ביחס ליעד שהוגדר, אם הוגדר) ולמספרים בפועל, בטון תומך וממוקד."
)


async def generate_rep_cue(
    summary: dict, pace_target: dict | None, rule_feedback: str | None, total_reps: int
) -> str | None:
    """דגש קצר אחד מה-LLM על חזרה שהושלמה זה עתה, או None אם המאמן החכם כבוי/נכשל.

    לא עוצר את זרם המצלמה - הקריאה הזו מיועדת לרוץ ברקע (fire-and-forget) בזמן
    שהלולאה הראשית ממשיכה לעבד פריימים כרגיל.
    """
    if not smart_coach_enabled():
        return None

    data_lines = [
        f'מספר חזרה בסט: {summary["rep_number"]} (סה"כ חזרות שהושלמו עד כה: {total_reps})',
        f"משך ירידה: {summary['descent_duration']:.2f} שניות",
        f"משך עלייה: {summary['ascent_duration']:.2f} שניות",
        f"מדידה משוערת (הייתה מדידה חסרה במהלך החזרה): {'כן' if summary['estimated'] else 'לא'}",
    ]
    if pace_target:
        data_lines.append(
            f"יעד קצב שהוגדר: ירידה {pace_target['descent_range']} שניות, "
            f"עלייה {pace_target['ascent_range']} שניות"
        )
    else:
        data_lines.append("לא הוגדר יעד קצב לאימון הזה.")
    data_lines.append(f"המשוב הקיים המחושב לפי כללים קבועים: {rule_feedback or 'אין'}")

    user_message = (
        "הנה נתוני החזרה שהושלמה זה עתה. תן דגש אחד קצר בעברית שמתאים בדיוק לנתונים האלה "
        "(אפשר להתייחס למשוב הקיים או להוסיף עליו, לא לסתור אותו):\n" + "\n".join(data_lines)
    )
    return await _call_model(user_message)


async def answer_session_question(
    question: str,
    side: str,
    rep_target: int | None,
    pace_target: dict | None,
    rep_summaries: list[dict],
    current_rep_count: int,
) -> str | None:
    """תשובה מבוססת-נתונים לשאלה חופשית על החזרה האחרונה/האימון הנוכחי, או None."""
    if not smart_coach_enabled():
        return None

    if not rep_summaries:
        reps_desc = "אין עדיין חזרות שהושלמו בסט הזה."
    else:
        reps_desc = "\n".join(
            f"- חזרה {s['rep_number']}: ירידה {s['descent_duration']:.2f}s, "
            f"עלייה {s['ascent_duration']:.2f}s" + (" (משוערת)" if s["estimated"] else " (מדודה)")
            for s in rep_summaries
        )

    context = (
        f"צד מדידה שנבחר: {'שמאל' if side == 'L' else 'ימין'}\n"
        f"יעד חזרות שהוגדר: {rep_target if rep_target else 'לא הוגדר'}\n"
        f"יעד קצב שהוגדר: {pace_target if pace_target else 'לא הוגדר'}\n"
        f"מספר חזרות שהושלמו עד כה בסט: {current_rep_count}\n"
        f"פירוט החזרות שהושלמו:\n{reps_desc}"
    )
    user_message = (
        f"נתוני האימון הנוכחי (כל הנתונים כאן הם המידע היחיד שיש לך - אין לך גישה לשום "
        f"דבר מעבר לזה):\n{context}\n\nשאלת המשתמש: {question}"
    )
    return await _call_model(user_message)


async def _call_model(user_message: str) -> str | None:
    import anthropic

    try:
        client = _get_client()
        response = await client.with_options(timeout=REQUEST_TIMEOUT_SEC, max_retries=0).messages.create(
            model=MODEL_ID,
            max_tokens=200,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        for block in response.content:
            if block.type == "text" and block.text.strip():
                return block.text.strip()
        return None
    except anthropic.AuthenticationError:
        print("[coach_llm] מפתח ה-API לא תקין - המאמן החכם ייכבה לבקשה הזו, נופלים למשוב הקיים.")
        return None
    except anthropic.RateLimitError:
        print("[coach_llm] הגבלת קצב מול ה-API - נופלים למשוב הקיים.")
        return None
    except anthropic.APIConnectionError:
        print("[coach_llm] שגיאת רשת מול ה-API - נופלים למשוב הקיים.")
        return None
    except anthropic.APIStatusError as e:
        print(f"[coach_llm] שגיאת שרת מה-API (status={e.status_code}) - נופלים למשוב הקיים.")
        return None
    except Exception as e:  # noqa: BLE001 - כל כשל אחר (כולל timeout) -> נפילה חזרה, בלי לחשוף פרטים
        print(f"[coach_llm] כשל לא צפוי ({type(e).__name__}) בקריאה ל-API - נופלים למשוב הקיים.")
        return None


async def test_connection():
    """בדיקת חיבור קצרה מול Anthropic. מחזירה True בהצלחה, או מחרוזת שגיאה מועילה
    בעברית שלעולם לא כוללת את המפתח עצמו או כל תוכן אחר מהבקשה/התשובה - רק סוג
    השגיאה. לא נקראת אוטומטית משום מקום - יש להריץ אותה במפורש (webapp/test_api_connection.py)."""
    import anthropic

    key = _resolve_api_key()
    if not key:
        return "לא נמצא מפתח API (לא ב-Keychain ולא במשתנה הסביבה ANTHROPIC_API_KEY)."

    try:
        client = anthropic.AsyncAnthropic(api_key=key)
        await client.with_options(timeout=REQUEST_TIMEOUT_SEC, max_retries=0).messages.create(
            model=MODEL_ID,
            max_tokens=1,
            messages=[{"role": "user", "content": "בדיקת חיבור"}],
        )
        return True
    except anthropic.AuthenticationError:
        return "המפתח שנשמר אינו תקין (שגיאת אימות מ-Anthropic)."
    except anthropic.PermissionDeniedError:
        return "המפתח תקין אך אין לו הרשאה מספיקה לפעולה הזו."
    except anthropic.RateLimitError:
        return "הבקשה נחסמה זמנית עקב הגבלת קצב - נסו שוב בעוד רגע."
    except anthropic.APIConnectionError:
        return "שגיאת רשת מול Anthropic - בדקו את החיבור לאינטרנט ונסו שוב."
    except anthropic.APIStatusError as e:
        return f"שגיאת שרת מה-API (קוד {e.status_code})."
    except Exception as e:  # noqa: BLE001 - הודעה כללית בלבד, בלי לחשוף פרטי החריגה
        return f"כשל לא צפוי ({type(e).__name__}) בעת ניסיון החיבור."
