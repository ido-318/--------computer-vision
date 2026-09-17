(() => {
  "use strict";

  const state = {
    ws: null,
    side: null,
    repTarget: null,
    paceTarget: null,
    finalReps: [],
    everDetected: false,
    previewReady: false,
    previewTimeoutId: null,
  };

  const el = (id) => document.getElementById(id);
  const screens = {
    setup: el("screen-setup"),
    training: el("screen-training"),
    finished: el("screen-finished"),
  };

  function showGlobalError(text) {
    const banner = el("global-error");
    banner.textContent = text;
    banner.classList.remove("hidden");
  }

  function hideGlobalError() {
    el("global-error").classList.add("hidden");
  }

  function showScreen(name) {
    Object.values(screens).forEach((s) => s.classList.remove("active"));
    screens[name].classList.add("active");
  }

  // מחברת WebSocket, ומדלגת על יצירת חיבור כפול אם כבר קיים אחד פתוח/בתהליך פתיחה.
  // onReady נקרא ברגע שהחיבור פתוח בפועל (מיד אם כבר פתוח, או כשה-open מגיע).
  function connectWebSocket(onReady) {
    if (state.ws && (state.ws.readyState === WebSocket.OPEN || state.ws.readyState === WebSocket.CONNECTING)) {
      if (onReady) {
        if (state.ws.readyState === WebSocket.OPEN) onReady();
        else state.ws.addEventListener("open", onReady, { once: true });
      }
      return state.ws;
    }
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.onopen = () => {
      hideGlobalError();
      if (onReady) onReady();
    };
    ws.onmessage = (event) => handleMessage(JSON.parse(event.data));
    ws.onclose = () => {
      state.ws = null;
      showGlobalError('החיבור לשרת נסגר (יתכן שהשרת הופעל מחדש). לחצו שוב על "הפעל מצלמה" כדי להתחבר מחדש.');
    };
    ws.onerror = () => {
      showGlobalError("שגיאת תקשורת עם השרת. ודאו שהשרת רץ (venv/bin/python webapp/server.py) ונסו שוב.");
    };
    state.ws = ws;
    return ws;
  }

  function send(msg) {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
      state.ws.send(JSON.stringify(msg));
    } else {
      showGlobalError("אין כרגע חיבור פעיל לשרת. לחצו שוב על \"הפעל מצלמה\" כדי להתחבר מחדש.");
    }
  }

  function clearPreviewTimeout() {
    if (state.previewTimeoutId) {
      clearTimeout(state.previewTimeoutId);
      state.previewTimeoutId = null;
    }
    el("btn-start-camera").disabled = false;
    el("btn-start-camera").textContent = "הפעל מצלמה";
  }

  // ---------- מסך הגדרה ----------

  document.querySelectorAll(".side-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".side-btn").forEach((b) => b.classList.remove("selected"));
      btn.classList.add("selected");
      state.side = btn.dataset.side;
    });
  });

  el("pace-enable").addEventListener("change", (e) => {
    el("pace-fields").classList.toggle("hidden", !e.target.checked);
  });

  el("btn-start-camera").addEventListener("click", () => {
    if (!state.side) {
      const hint = document.getElementById("side-required-hint");
      if (hint) hint.classList.remove("hidden");
      return;
    }
    const hint = document.getElementById("side-required-hint");
    if (hint) hint.classList.add("hidden");
    if (!state.ws) connectWebSocket();
    const trySend = () => {
      if (state.ws.readyState === WebSocket.OPEN) {
        send({ type: "start_preview", side: state.side });
      } else {
        setTimeout(trySend, 100);
      }
    };
    trySend();
    el("preview-card").classList.remove("hidden");
    el("readiness-banner").className = "readiness-banner readiness-pending";
    el("readiness-banner").textContent = "טוען מצלמה...";
    el("btn-start-training").disabled = true;
  });

  el("btn-start-training").addEventListener("click", () => {
    const repTargetVal = el("rep-target").value;
    state.repTarget = repTargetVal ? parseInt(repTargetVal, 10) : null;

    state.paceTarget = null;
    if (el("pace-enable").checked) {
      const dMin = parseFloat(el("pace-descent-min").value);
      const dMax = parseFloat(el("pace-descent-max").value);
      const aMin = parseFloat(el("pace-ascent-min").value);
      const aMax = parseFloat(el("pace-ascent-max").value);
      if (![dMin, dMax, aMin, aMax].some(Number.isNaN)) {
        state.paceTarget = { descent_range: [dMin, dMax], ascent_range: [aMin, aMax] };
      }
    }

    send({ type: "start_training", rep_target: state.repTarget, pace_target: state.paceTarget });
    clearCoachLog();
    state.everDetected = false;
    resetTrainingReadout();
    showScreen("training");
  });

  // ---------- מסך אימון ----------

  let paused = false;
  let detailsOpen = false;

  el("btn-toggle-details").addEventListener("click", () => {
    detailsOpen = !detailsOpen;
    el("details-panel").classList.toggle("hidden", !detailsOpen);
    el("btn-toggle-details").textContent = detailsOpen ? "פרטים ▴" : "פרטים ▾";
  });

  el("btn-pause-resume").addEventListener("click", () => {
    paused = !paused;
    send({ type: paused ? "pause" : "resume" });
    el("btn-pause-resume").textContent = paused ? "המשך" : "השהה";
    el("paused-badge").classList.toggle("hidden", !paused);
  });

  el("btn-finish").addEventListener("click", () => {
    send({ type: "stop" });
  });

  el("btn-new-session").addEventListener("click", () => {
    paused = false;
    el("btn-pause-resume").textContent = "השהה";
    showScreen("setup");
  });

  function resetTrainingReadout() {
    el("rep-count").textContent = "0";
    el("phase-value").textContent = "—";
    el("detail-angle").textContent = "—";
    el("detail-hip").textContent = "—";
    el("detail-knee").textContent = "—";
    el("detail-ankle").textContent = "—";
    el("live-guidance").classList.add("hidden");
  }

  // ---------- צ'אט המאמן ----------

  function clearCoachLog() {
    el("coach-log").innerHTML = "";
  }

  function addCoachBubble(text, role) {
    const div = document.createElement("div");
    div.className = "coach-bubble" + (role === "user" ? " user" : "");
    div.textContent = text;
    const log = el("coach-log");
    log.appendChild(div);
    log.scrollTop = log.scrollHeight;
  }

  document.querySelectorAll(".quick-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const q = btn.dataset.q;
      addCoachBubble(q, "user");
      send({ type: "chat_question", text: q });
    });
  });

  el("coach-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const input = el("coach-input");
    const text = input.value.trim();
    if (!text) return;
    addCoachBubble(text, "user");
    send({ type: "chat_question", text });
    input.value = "";
  });

  // ---------- טיפול בהודעות מהשרת ----------

  function handleMessage(msg) {
    switch (msg.type) {
      case "tick":
        handleTick(msg);
        break;
      case "rep_summary":
        el("rep-count").textContent = msg.rep_number;
        break;
      case "coach_message":
        addCoachBubble(msg.text);
        break;
      case "session_summary":
        renderFinishScreen(msg);
        break;
      case "camera_error":
        showGlobalError(msg.message);
        break;
    }
  }

  function handleTick(msg) {
    if (msg.mode === "preview") {
      el("video-frame").src = "data:image/jpeg;base64," + msg.jpeg;
      const banner = el("readiness-banner");
      if (msg.ready) {
        banner.className = "readiness-banner readiness-ready";
        banner.textContent = "מוכן! אפשר להתחיל אימון.";
        el("btn-start-training").disabled = false;
      } else {
        banner.className = "readiness-banner readiness-pending";
        banner.textContent = msg.guidance || "ממתין לזיהוי...";
        el("btn-start-training").disabled = true;
      }
    } else if (msg.mode === "training" || msg.mode === "paused") {
      el("video-frame-training").src = "data:image/jpeg;base64," + msg.jpeg;
      if (msg.mode === "training") {
        if (msg.angle != null) state.everDetected = true;
        el("rep-count").textContent = msg.rep_count;
        el("phase-value").textContent = state.everDetected ? (msg.phase_he || "—") : "ממתין לזיהוי";
        el("detail-angle").textContent = msg.angle != null ? msg.angle.toFixed(1) + "°" : "—";
        if (msg.confidences) {
          el("detail-hip").textContent = Math.round(msg.confidences.hip * 100) + "%";
          el("detail-knee").textContent = Math.round(msg.confidences.knee * 100) + "%";
          el("detail-ankle").textContent = Math.round(msg.confidences.ankle * 100) + "%";
        }
        const guidanceEl = el("live-guidance");
        if (msg.guidance) {
          guidanceEl.textContent = msg.guidance;
          guidanceEl.classList.remove("hidden");
        } else {
          guidanceEl.classList.add("hidden");
        }
      }
    }
  }

  function renderFinishScreen(msg) {
    state.finalReps = msg.reps || [];
    el("finish-total").textContent = `${msg.total} חזרות`;
    const body = el("reps-table-body");
    body.innerHTML = "";
    state.finalReps.forEach((r) => {
      const tr = document.createElement("tr");
      const tag = r.estimated
        ? '<span class="estimated-tag">משוערת</span>'
        : '<span class="measured-tag">מדודה</span>';
      tr.innerHTML = `
        <td>${r.rep_number}</td>
        <td>${r.descent_duration.toFixed(2)}s</td>
        <td>${r.ascent_duration.toFixed(2)}s</td>
        <td>${tag}</td>
        <td style="font-size:12px; color:var(--text-muted);">${r.feedback || "—"}</td>
      `;
      body.appendChild(tr);
    });
    showScreen("finished");
  }

  connectWebSocket();
})();
