"""שרת מקומי ל'מאמן התנועה שלי' - ממשק web שמריץ את ניתוח הסקוואט הקיים (scripts/rep_logic.py)
בזמן אמת, ומזרים תוצאות (וידאו מעובד + נתוני מצב) לדפדפן דרך WebSocket.

המצלמה נפתחת ישירות בשרת (OpenCV, כמו ב-scripts/live_squat.py) - לא דרך מצלמת הדפדפן.
כל העיבוד מקומי; אין שמירת וידאו לדיסק ואין שליחתו לשום מקום חוץ מהדפדפן על אותו מחשב.

הפעלה:
    venv/bin/python webapp/server.py
ואז פותחים http://localhost:8000 בדפדפן.
"""

import asyncio
import base64
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rep_logic import RepCounter, SIDE_KEYPOINTS, measure_frame, pace_feedback  # noqa: E402
from ultralytics import YOLO  # noqa: E402

MODEL_NAME = "yolo11n-pose.pt"
CAMERA_INDEX = 0
JPEG_QUALITY = 72
CAMERA_READ_FAILURE_LIMIT = 60

PHASE_HE = {"STAND": "עמידה", "DESCEND": "ירידה", "BOTTOM": "תחתית", "ASCEND": "עלייה", "LOST": "אין זיהוי"}
SIDE_HE = {"L": "שמאל", "R": "ימין"}

CAMERA_HELP_HE = (
    "לא ניתן לפתוח את המצלמה. סביר שאין הרשאת מצלמה לאפליקציה שמריצה את השרת "
    "(למשל Terminal/iTerm/VS Code). ב-macOS: System Settings -> Privacy & Security -> "
    "Camera -> אפשרו גישה, ואז הפעילו מחדש את השרת."
)

OPENING_MESSAGE_HE = (
    "שלום! אני המאמן האוטומטי של אפליקציית "
    "מאמן התנועה שלי. "
    "אני עוקב אחרי זווית הברך שלך ומודד משכי ירידה ועלייה ומספר חזרות. "
    "חשוב לדעת: אני כלי מדידה אוטומטי בלבד (לא מודל שפה מחובר), ואני לא קובע אם הטכניקה שלך תקינה "
    "ולא מאבחן פציעות. אפשר לשאול אותי בכל שלב, למשל “איך הייתה החזרה האחרונה?” "
    "או “כמה חזרות עשיתי?”."
)

MODEL: YOLO | None = None  # נטען פעם אחת באתחול השרת


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL
    print(f"טוען מודל {MODEL_NAME} ...")
    MODEL = YOLO(MODEL_NAME)
    print("מוכן. פתחו http://localhost:8000")
    yield


app = FastAPI(lifespan=lifespan)


class SessionState:
    def __init__(self):
        self.cap: cv2.VideoCapture | None = None
        self.model: YOLO | None = None
        self.mode = "idle"  # idle | preview | training | paused
        self.side = "L"
        self.rep_target: int | None = None
        self.pace_target: dict | None = None
        self.counter: RepCounter | None = None
        self.rep_summaries: list[dict] = []
        self.target_reached_announced = False
        self.last_t = 0.0
        self.read_failures = 0

    def release_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def guidance_for(num_people: int, angle) -> str | None:
    if num_people == 0:
        return "לא מזוהה אף אחד בפריים - היכנס לתמונה."
    if num_people > 1:
        return "מזוהים כמה אנשים בפריים - הניתוח מושהה כדי לא לערבב ביניהם."
    if angle is None:
        return "לא ניתן לזהות בבירור את הירך/הברך/הקרסול - עמוד מהצד כשכל הגוף כולל כפות הרגליים בפריים, ובתאורה טובה."
    return None


def coach_rep_message(summary: dict, feedback: str | None) -> str:
    tag = " (מדידה משוערת - הייתה מדידה חסרה במהלך החזרה)" if summary["estimated"] else ""
    msg = (f"חזרה {summary['rep_number']} הושלמה: ירידה {summary['descent_duration']:.2f} שניות, "
           f"עלייה {summary['ascent_duration']:.2f} שניות{tag}.")
    if feedback:
        msg += f" {feedback}"
    return msg


def answer_question(text: str, state: SessionState) -> str:
    t = (text or "").strip()

    if any(k in t for k in ["חזרה אחרונה", "החזרה האחרונה", "איך היה", "איך הייתה"]):
        if not state.rep_summaries:
            return "עדיין לא השלמת אף חזרה מלאה בסט הזה."
        s = state.rep_summaries[-1]
        tag = " (מדידה משוערת - הייתה מדידה חסרה במהלכה)" if s["estimated"] else "."
        base = (f"החזרה האחרונה (מספר {s['rep_number']}): ירידה {s['descent_duration']:.2f} שניות, "
                f"עלייה {s['ascent_duration']:.2f} שניות{tag}")
        fb = pace_feedback(s, state.pace_target)
        if fb:
            base += f" {fb}"
        return base

    if any(k in t for k in ["כמה חזרות", "מספר חזרות"]):
        n = state.counter.rep_count if state.counter is not None else 0
        if n == 0:
            return "עדיין לא נספרה אף חזרה בסט הזה."
        return f"עד כה נספרו {n} חזרות מלאות בסט הזה."

    if any(k in t for k in ["קצב", "יעד"]):
        if state.pace_target is None:
            return "לא הוגדר יעד קצב לאימון הזה - אני מציג רק את המדידות, בלי השוואה."
        if not state.rep_summaries:
            return "עדיין אין חזרה מלאה להשוואה מול יעד הקצב שהוגדר."
        fb = pace_feedback(state.rep_summaries[-1], state.pace_target)
        return fb or "אין נתון להצגה כרגע."

    return ("אני מאמן אוטומטי מבוסס מדידות בלבד (בלי חיבור למודל שפה בשלב הזה). אני יכול לענות כרגע "
            "רק על שאלות לגבי מספר החזרות, משכי הירידה/עלייה של החזרה האחרונה, וההשוואה ליעד הקצב "
            "אם הוגדר.")


def encode_jpeg_base64(frame) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


CAMERA_READ_TIMEOUT_SEC = 5.0  # אם קריאה בודדת מהמצלמה נתקעת מעבר לזה (למשל הרשאה תלויה/מצלמה תפוסה)


async def camera_loop(send_queue: asyncio.Queue, state: SessionState):
    loop = asyncio.get_event_loop()
    state.last_t = time.monotonic()
    state.read_failures = 0

    while state.mode in ("preview", "training", "paused"):
        try:
            ret, frame = await asyncio.wait_for(
                loop.run_in_executor(None, state.cap.read), timeout=CAMERA_READ_TIMEOUT_SEC
            )
        except TimeoutError:
            await send_queue.put({
                "type": "camera_error",
                "message": (
                    "המצלמה לא הגיבה בזמן סביר. סיבות אפשריות: יש בקשת הרשאת מצלמה ממתינה מחוץ "
                    "לדפדפן (בדקו חלונות/התראות אחרות במחשב ואשרו), תוכנה אחרת משתמשת כרגע במצלמה, "
                    "או שהמצלמה עדיין 'תפוסה' משימוש קודם - נסו לסגור אפליקציות אחרות שמשתמשות "
                    "במצלמה ולנסות שוב."
                ),
            })
            state.mode = "idle"
            state.release_camera()
            break
        if not ret:
            state.read_failures += 1
            if state.read_failures > CAMERA_READ_FAILURE_LIMIT:
                await send_queue.put({"type": "camera_error", "message": "אובדן חיבור למצלמה תוך כדי ריצה."})
                state.mode = "idle"
                break
            await asyncio.sleep(0.05)
            continue
        state.read_failures = 0
        frame = cv2.flip(frame, 1)

        if state.mode == "paused":
            jpeg = encode_jpeg_base64(frame)
            await send_queue.put({"type": "tick", "mode": "paused", "jpeg": jpeg})
            await asyncio.sleep(0.05)
            continue

        now = time.monotonic()
        dt = now - state.last_t
        state.last_t = now

        result = (await loop.run_in_executor(None, lambda: state.model(frame, verbose=False)))[0]
        num_people = 0 if result.keypoints is None else len(result.keypoints)

        angle = None
        confidences = None
        if num_people == 1:
            xy = result.keypoints[0].xy[0].tolist()
            conf = result.keypoints[0].conf[0].tolist()
            hip_i, knee_i, ankle_i = SIDE_KEYPOINTS[state.side]
            confidences = {"hip": conf[hip_i], "knee": conf[knee_i], "ankle": conf[ankle_i]}
            angle = measure_frame(xy, conf, side=state.side)

        display = result.plot()
        jpeg = encode_jpeg_base64(display)
        guidance = guidance_for(num_people, angle)

        if state.mode == "preview":
            await send_queue.put({
                "type": "tick", "mode": "preview", "jpeg": jpeg,
                "num_people": num_people, "angle": angle, "ready": angle is not None,
                "guidance": guidance, "confidences": confidences,
            })
        elif state.mode == "training":
            info = state.counter.step(dt, angle)
            await send_queue.put({
                "type": "tick", "mode": "training", "jpeg": jpeg,
                "angle": angle, "phase": info["phase"], "phase_he": PHASE_HE.get(info["phase"], info["phase"]),
                "rep_count": info["rep_count"], "guidance": guidance, "confidences": confidences,
            })

            if info["rep_summary"] is not None:
                s = info["rep_summary"]
                state.rep_summaries.append(s)
                feedback = pace_feedback(s, state.pace_target)
                await send_queue.put({"type": "rep_summary", **s})
                await send_queue.put({"type": "coach_message", "text": coach_rep_message(s, feedback)})

                if (state.rep_target and info["rep_count"] >= state.rep_target
                        and not state.target_reached_announced):
                    state.target_reached_announced = True
                    await send_queue.put({
                        "type": "coach_message",
                        "text": f"הגעת ליעד שהגדרת - {state.rep_target} חזרות! אפשר להמשיך עוד קצת או ללחוץ סיום.",
                    })


async def sender_loop(websocket: WebSocket, queue: asyncio.Queue):
    while True:
        msg = await queue.get()
        if msg is None:
            break
        await websocket.send_json(msg)


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    state = SessionState()
    state.model = MODEL
    send_queue: asyncio.Queue = asyncio.Queue()
    sender_task = asyncio.create_task(sender_loop(websocket, send_queue))
    loop_task: asyncio.Task | None = None

    try:
        while True:
            msg = await websocket.receive_json()
            mtype = msg.get("type")

            if mtype == "start_preview":
                state.side = msg.get("side", "L")
                if state.cap is None:
                    cap = cv2.VideoCapture(CAMERA_INDEX)
                    if not cap.isOpened():
                        await send_queue.put({"type": "camera_error", "message": CAMERA_HELP_HE})
                        continue
                    state.cap = cap
                state.mode = "preview"
                if loop_task is None or loop_task.done():
                    loop_task = asyncio.create_task(camera_loop(send_queue, state))

            elif mtype == "start_training":
                if state.cap is None:
                    await send_queue.put({"type": "camera_error", "message": CAMERA_HELP_HE})
                    continue
                state.rep_target = msg.get("rep_target")
                state.pace_target = msg.get("pace_target")
                state.counter = RepCounter()
                state.rep_summaries = []
                state.target_reached_announced = False
                state.mode = "training"
                state.last_t = time.monotonic()
                await send_queue.put({"type": "coach_message", "text": OPENING_MESSAGE_HE})

            elif mtype == "pause":
                if state.mode == "training":
                    state.mode = "paused"

            elif mtype == "resume":
                if state.mode == "paused":
                    state.mode = "training"
                    state.last_t = time.monotonic()  # לא מזינים את זמן ההשהיה כפער אמיתי

            elif mtype == "stop":
                state.mode = "idle"
                if loop_task is not None:
                    loop_task.cancel()
                    loop_task = None
                state.release_camera()
                reps_with_feedback = [
                    {**s, "feedback": pace_feedback(s, state.pace_target)} for s in state.rep_summaries
                ]
                await send_queue.put({
                    "type": "session_summary",
                    "reps": reps_with_feedback,
                    "total": len(state.rep_summaries),
                })

            elif mtype == "chat_question":
                answer = answer_question(msg.get("text", ""), state)
                await send_queue.put({"type": "coach_message", "text": answer})

    except WebSocketDisconnect:
        pass
    finally:
        if loop_task is not None:
            loop_task.cancel()
        state.release_camera()
        await send_queue.put(None)
        await sender_task


@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
