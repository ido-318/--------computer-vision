"""שמירה/קריאה של מפתח Anthropic API ב-macOS Keychain, באמצעות חבילת `keyring`
(שמשתמשת ב-Security framework המקורי של macOS - לא קובץ, לא משתנה סביבה קבוע).

שום פונקציה כאן לא מדפיסה, רושמת ליומן, או חושפת את ערך המפתח בשום צורה.
"""

SERVICE_NAME = "motion-coach-anthropic-api-key"  # מזהה השירות ב-Keychain (ASCII בלבד, לא המפתח עצמו)
ACCOUNT_NAME = "default"


def get_stored_key() -> str | None:
    """מחזירה את המפתח השמור ב-Keychain, או None אם אין כזה/אין גישה ל-Keychain כרגע."""
    try:
        import keyring
        return keyring.get_password(SERVICE_NAME, ACCOUNT_NAME)
    except Exception:
        return None


def set_stored_key(key: str) -> None:
    import keyring
    keyring.set_password(SERVICE_NAME, ACCOUNT_NAME, key)


def delete_stored_key() -> bool:
    """מוחקת את המפתח מה-Keychain. מחזירה True אם היה מפתח שנמחק, False אם לא היה כזה."""
    try:
        import keyring
        keyring.delete_password(SERVICE_NAME, ACCOUNT_NAME)
        return True
    except Exception:
        return False
