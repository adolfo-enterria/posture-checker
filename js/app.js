import {
  PostureEngine,
  extractFrame,
  buildBaseline,
  moreUpright,
  issueMessages,
  CALIBRATION_MESSAGES,
  dist,
  mid,
} from "./engine.js";

const VISION = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

const params = new URLSearchParams(location.search);
const DEMO = params.has("demo");
const KEY = DEMO ? "posture.demo." : "posture.";

const TICK_MS = 100;
const CALIB_MS = 3000;
const BUCKET_MS = 30_000;      // timeline resolution
const TIMELINE_BUCKETS = 30;   // 15 minutes
const HISTORY_DAYS = 60;
const STEP_TIMEOUT_MS = 90_000; // abandoned recalibration falls back to monitoring
const STALL_MS = 3000;
const BREAK_RESET_MS = 120_000; // being away this long counts as a break
const HINT_EVERY_MS = 600_000;

const COLORS = { good: "#22c55e", warn: "#f59e0b", bad: "#ef4444", idle: "#5b86ff", away: "#8b94a0" };

// ---------------------------------------------------------------------------

const $ = (id) => document.getElementById(id);
const els = Object.fromEntries(
  [
    "video", "overlay", "stage", "graceWrap", "graceLabel", "graceFill", "coach", "coachIssue",
    "coachFix", "stageNote", "stageNoteTitle", "stageNoteText", "panel", "startBtn", "startMsg",
    "frameChecks", "toCalibrateBtn", "calibrateGoBtn", "prepCancelBtn", "captureArc", "captureTitle",
    "resultTitle", "resultList", "retryBtn", "acceptBtn", "resultCancelBtn", "recalibrateBtn",
    "snoozeBtn", "snoozeLabel", "snoozeMenu", "calibInfo", "status", "statusText", "banner",
    "bannerTitle", "bannerFix", "favicon", "ringArc", "scoreNow", "scoreAvg", "goodPct",
    "alertCount", "streak", "bestStreak", "checks", "timeline", "timelineEmpty", "awayInfo", "week",
    "sensitivity", "grace", "graceVal", "breakEvery", "sound", "notify", "notifyHint", "mirror",
    "skeleton", "testAlertBtn", "toast",
  ].map((id) => [id, $(id)]),
);
const ctx = els.overlay.getContext("2d");

const settings = {
  sensitivity: "medium",
  grace: 5,
  breakEvery: 0,
  sound: true,
  notify: true,
  mirror: true,
  skeleton: true,
  ...loadJSON(KEY + "settings"),
};

const engine = new PostureEngine({ sensitivity: settings.sensitivity, graceMs: settings.grace * 1000 });

const app = {
  phase: "welcome",
  stepSince: 0,
  source: null,
  baseline: loadJSON(KEY + "baseline"),
  calib: null,
  pending: null,
  snoozeUntil: 0,
  blocked: false,
  light: 255,
  lastSnap: null,
  lastFrameAt: 0,
  frameOkSince: 0,
  timeline: new Map(),
  flushed: { goodMs: 0, badMs: 0, alerts: 0 },
  audio: null,
  lastRender: 0,
  lastFlush: 0,
  lastTick: 0,
  sitMs: 0,
  awayRun: 0,
  lastHintAt: -Infinity,
};

const worker = new Worker(new URL("./timer-worker.js", import.meta.url));
worker.onmessage = () => tick();

const landmarkerPromise = DEMO ? null : createLandmarker();
landmarkerPromise?.catch(() => {}); // surfaced when the camera starts

// ---------------------------------------------------------------------------
// Sources

async function createLandmarker() {
  const { PoseLandmarker, FilesetResolver } = await import(`${VISION}/vision_bundle.mjs`);
  const vision = await FilesetResolver.forVisionTasks(`${VISION}/wasm`);
  for (const delegate of ["GPU", "CPU"]) {
    try {
      return await PoseLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_URL, delegate },
        runningMode: "VIDEO",
        numPoses: 1,
      });
    } catch (err) {
      console.warn(`Pose model failed on ${delegate}`, err);
    }
  }
  throw new Error("Could not load the pose model.");
}

async function createCameraSource() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
    audio: false,
  });
  const track = stream.getVideoTracks()[0];
  // Auto-framing (Studio Effects, Center Stage…) keeps you centered and sized,
  // which hides leaning and slumping. Turn it off where the browser allows.
  if (track.getCapabilities?.().faceFraming) {
    track.applyConstraints({ advanced: [{ faceFraming: false }] }).catch(() => {});
  }

  const video = els.video;
  video.srcObject = stream;
  await video.play();

  setStartMsg("Loading pose model…");
  const landmarker = await landmarkerPromise;

  // Background tabs may stop painting the <video>; ImageCapture keeps frames coming.
  const capture = "ImageCapture" in window ? new ImageCapture(track) : null;
  const probe = new OffscreenCanvas(32, 24).getContext("2d", { willReadFrequently: true });
  let lastTime = -1;
  let lastAdvance = performance.now();
  let lastLightCheck = 0;
  const s = track.getSettings();

  track.addEventListener("ended", () => {
    toast("Camera disconnected. Reload the page to reconnect.", 8000);
    setStatus("idle", "Camera disconnected");
  });

  return {
    deviceKey: `${s.deviceId || track.label}|${video.videoWidth}x${video.videoHeight}`,
    async detect(now) {
      let input = null;
      let bitmap = null;
      if (video.currentTime !== lastTime && video.readyState >= 2) {
        lastTime = video.currentTime;
        lastAdvance = now;
        input = video;
      } else if (capture && now - lastAdvance > 1000 && track.readyState === "live") {
        try {
          bitmap = await capture.grabFrame();
          input = bitmap;
        } catch { /* camera busy */ }
      }
      if (!input) return null;
      const w = bitmap ? bitmap.width : video.videoWidth;
      const h = bitmap ? bitmap.height : video.videoHeight;

      if (now - lastLightCheck > 1000) {
        lastLightCheck = now;
        probe.drawImage(input, 0, 0, 32, 24);
        const { mean, std } = lumaStats(probe.getImageData(0, 0, 32, 24).data);
        app.light = mean;
        app.blocked = mean < 18 || std < 4;
      }

      const res = landmarker.detectForVideo(input, performance.now());
      bitmap?.close();
      return { landmarks: res.landmarks?.[0], worldLandmarks: res.worldLandmarks?.[0], w, h };
    },
  };
}

async function createDemoSource() {
  const sim = await import("./sim.js");
  const script = [
    [12, "upright"], [12, "slouch"], [8, "upright"], [12, "textNeck"],
    [8, "upright"], [10, "hidden"], [8, "upright"], [12, "leanIn"],
  ];
  const total = script.reduce((a, [s]) => a + s, 0);
  const named = { ...sim.POSES, hidden: { visibility: { 11: 0.1, 12: 0.1 } } };
  const fixed = params.get("demo");
  let override = fixed && named[fixed] ? named[fixed] : null;
  let seed = 1;
  const t0 = performance.now();

  window.__posture.setPose = (p) => { override = p === "away" || typeof p !== "string" ? p : named[p]; };

  return {
    deviceKey: "demo",
    poseAt(now) {
      if (override) return override;
      let t = ((now - t0) / 1000) % total;
      for (const [secs, name] of script) {
        if (t < secs) return named[name];
        t -= secs;
      }
      return sim.UPRIGHT;
    },
    async detect(now) {
      const pose = this.poseAt(now);
      if (pose === "away") return { landmarks: null, worldLandmarks: null, w: 640, h: 480 };
      const { landmarks, worldLandmarks } = sim.synthesize({ noise: 0.0015, seed: seed++, ...pose });
      return { landmarks, worldLandmarks, w: 640, h: 480 };
    },
    calibrationFrames() {
      return Array.from({ length: 30 }, () => {
        const { landmarks, worldLandmarks } = sim.synthesize({ noise: 0.0015, seed: seed++ });
        return extractFrame(landmarks, worldLandmarks, 640, 480);
      });
    },
  };
}

// ---------------------------------------------------------------------------
// Startup

els.startBtn.addEventListener("click", start);

async function start() {
  els.startBtn.disabled = true;
  try {
    initAudio();
    requestNotifications();
    setStartMsg("Starting camera…");
    app.source = await createCameraSource();
    begin();
  } catch (err) {
    console.error(err);
    els.startBtn.disabled = false;
    const msg =
      err.name === "NotAllowedError" ? "Camera access was blocked. Allow it from the camera icon in the address bar, then try again."
      : err.name === "NotFoundError" ? "No camera found. Plug one in and try again."
      : err.name === "NotReadableError" ? "The camera is in use by another app. Close it and try again."
      : `Something went wrong: ${err.message}`;
    setStartMsg(msg, true);
  }
}

async function startDemo() {
  els.video.hidden = true;
  app.source = await createDemoSource();
  if (!loadJSON(KEY + "history")) seedDemoHistory();
  renderWeek();
  if (!params.has("onboard")) {
    const { baseline } = buildBaseline(app.source.calibrationFrames());
    applyBaseline(baseline, { quiet: true });
  }
  begin();
}

/** Returning users who already granted the camera skip the Start click. */
async function maybeAutoStart() {
  try {
    const perm = await navigator.permissions?.query({ name: "camera" });
    if (perm?.state === "granted" && app.baseline) {
      // Audio needs a user gesture; unlock it on the first interaction.
      const unlock = () => initAudio();
      document.addEventListener("pointerdown", unlock, { once: true });
      document.addEventListener("keydown", unlock, { once: true });
      start();
    }
  } catch { /* permissions API unsupported */ }
}

function begin() {
  worker.postMessage({ cmd: "start", interval: TICK_MS });

  const b = app.baseline;
  if (b && b.deviceKey && b.deviceKey !== app.source.deviceKey) {
    toast("Different camera detected. Let's calibrate for this one.", 5000);
    enterStep("frame");
  } else if (b) {
    engine.setBaseline(b);
    enterMonitor();
    if (!DEMO) toast(`Using your calibration from ${fmtDate(b.at)}. Recalibrate if your setup changed.`, 5000);
  } else {
    enterStep("frame");
  }
}

// ---------------------------------------------------------------------------
// Main loop

let busy = false;
async function tick() {
  if (busy || !app.source) return;
  busy = true;
  try {
    await step(performance.now());
  } catch (err) {
    console.error(err);
  } finally {
    busy = false;
  }
}

async function step(now) {
  const dt = app.lastTick ? Math.min(now - app.lastTick, 1000) : 0;
  app.lastTick = now;

  if (app.snoozeUntil) {
    if (now < app.snoozeUntil) {
      accumulateTimeline(null, now, dt);
      return renderSnooze(now);
    }
    endSnooze();
  }

  // An abandoned recalibration shouldn't switch monitoring off for good.
  if (app.baseline && app.phase !== "monitor" && app.phase !== "capture" && now - app.stepSince > STEP_TIMEOUT_MS) {
    engine.setBaseline(app.baseline);
    enterMonitor();
    toast("Kept your previous calibration and resumed monitoring.");
  }

  const raw = await app.source.detect(now);
  if (!raw) {
    if (app.phase === "monitor" && now - app.lastFrameAt > STALL_MS) {
      showStageNote("Camera paused", "The browser stopped sending frames. Keep this tab in its own window (it can be in the background) or click here.");
    }
    return;
  }
  if (app.lastFrameAt && now - app.lastFrameAt > STALL_MS) hideStageNote();
  app.lastFrameAt = now;

  let frame = extractFrame(raw.landmarks, raw.worldLandmarks, raw.w, raw.h);
  if (app.blocked && frame.presence !== "ok") frame = { presence: "partial", reason: "blocked" };

  switch (app.phase) {
    case "frame":
      renderFrameChecks(frame, now);
      break;
    case "capture":
      captureFrame(frame, now);
      break;
    case "monitor": {
      const snap = engine.update(frame, now);
      app.lastSnap = snap;
      handleSnap(snap, now, dt);
      break;
    }
  }
  draw(raw, frame, app.phase === "monitor" ? app.lastSnap : null);
}

function handleSnap(snap, now, dt) {
  const { status, issues, alert } = snap;

  if (alert?.type === "start" || alert?.type === "repeat") fireAlert(issues);
  if (alert?.type === "clear") {
    clearAlertUI();
    if (status !== "away") toast("Nice! Back to good posture.");
  }

  const labels = {
    good: "Good posture",
    warn: "Check your posture",
    bad: "Fix your posture",
    away: "Away",
    uncalibrated: "Not calibrated",
  };
  setStatus(status === "uncalibrated" || status === "away" ? "idle" : status, labels[status]);
  setFavicon(status);

  const msgs = issueMessages(issues, 2);
  if (snap.alerting && msgs.length) {
    // One loud message: the banner. The on-video coach steps aside.
    setText(els.bannerTitle, msgs.map((m) => m.issue).join(" · "));
    setText(els.bannerFix, msgs[0].fix);
    els.banner.hidden = false;
    document.body.classList.add("alerting");
    els.coach.hidden = true;
  } else if (msgs.length && status === "warn") {
    setText(els.coachIssue, msgs[0].issue);
    setText(els.coachFix, msgs[0].fix);
    els.coach.hidden = false;
  } else {
    els.coach.hidden = true;
  }

  // Countdown to alert
  if (!snap.alerting && snap.level > 0 && issues.length) {
    const left = Math.max(0, Math.ceil((snap.graceMs - snap.level) / 1000));
    els.graceWrap.hidden = false;
    setText(els.graceLabel, left > 0 ? `Alert in ${left}s unless you adjust` : "Alert now");
    els.graceFill.style.width = `${Math.min(100, (snap.level / snap.graceMs) * 100)}%`;
  } else {
    els.graceWrap.hidden = true;
  }

  if (snap.hint === "setup" && now - app.lastHintAt > HINT_EVERY_MS) {
    app.lastHintAt = now;
    toast("Looks like the camera or your chair moved. Recalibrate if so.", 6000);
  }

  trackBreaks(status, now, dt);
  accumulateTimeline(snap, now, dt);

  if (now - app.lastRender > 250) {
    app.lastRender = now;
    renderScore(snap.score, status);
    renderChecks(snap);
    renderStats(snap.stats);
    renderTimeline(now);
  }
  maybeFlushHistory(snap.stats, now);
}

// ---------------------------------------------------------------------------
// Onboarding & calibration

function enterStep(step) {
  app.phase = step;
  app.stepSince = performance.now();
  showStep(step);
  setMonitorControls(false);
  resetLiveUI();
  setStatus("idle", step === "frame" ? "Setting up" : "Calibrating");
  setFavicon("idle");
}

function enterMonitor() {
  app.phase = "monitor";
  showStep(null);
  setMonitorControls(true);
}

function setMonitorControls(on) {
  els.recalibrateBtn.disabled = !on;
  els.snoozeBtn.disabled = !on;
  if (!on) closeMenu();
}

function showStep(step) {
  els.panel.hidden = !step;
  els.panel.classList.toggle("center", step === "welcome");
  let shown = null;
  for (const el of els.panel.querySelectorAll("[data-step]")) {
    el.hidden = el.dataset.step !== step;
    if (!el.hidden) shown = el;
  }
  // Keep keyboard focus inside the panel when the focused button disappears.
  if (shown && !shown.contains(document.activeElement) && document.activeElement !== document.body) {
    const target = shown.querySelector("button:not([disabled]):not([hidden])") ?? shown.querySelector("h2");
    target?.setAttribute("tabindex", target.tagName === "H2" ? "-1" : target.getAttribute("tabindex") ?? "0");
    target?.focus();
  } else if (shown && document.activeElement === document.body && step !== "welcome") {
    const h2 = shown.querySelector("h2");
    h2?.setAttribute("tabindex", "-1");
    h2?.focus({ preventScroll: true });
  }
}

/** Clear everything that describes "right now" so nothing stale lingers. */
function resetLiveUI() {
  els.coach.hidden = true;
  els.graceWrap.hidden = true;
  clearAlertUI();
  renderScore(null, "idle");
  renderChecks(null);
}

function renderFrameChecks(frame, now) {
  const ok = frame.presence === "ok";
  const m = frame.metrics;
  const seen = frame.presence !== "away" && frame.reason !== "blocked" && frame.reason !== "frozen";
  const checks = {
    head: seen && frame.reason !== "head",
    shoulders: seen && frame.reason !== "shoulders",
    facing: ok && Math.abs(m.yaw) < 15 && Math.abs(m.headYaw ?? 0) < 15,
    distance: ok && m.lean >= 0.18 && m.lean <= 0.75,
    light: !app.blocked && app.light > 35,
  };
  for (const li of els.frameChecks.children) {
    const pass = checks[li.dataset.check];
    li.classList.toggle("ok", pass);
    li.classList.toggle("fail", !pass);
  }
  const all = Object.values(checks).every(Boolean);
  if (!all) app.frameOkSince = 0;
  else if (!app.frameOkSince) app.frameOkSince = now;
  els.toCalibrateBtn.disabled = !(all && now - app.frameOkSince > 600);
}

els.toCalibrateBtn.addEventListener("click", () => {
  els.prepCancelBtn.hidden = !app.baseline;
  enterStep("prep");
});
els.calibrateGoBtn.addEventListener("click", () => {
  if (DEMO && !params.has("onboard")) return finishCalibration(app.source.calibrationFrames());
  app.calib = { frames: [], start: performance.now() };
  enterStep("capture");
  els.captureTitle.textContent = "Hold still…";
});
for (const btn of [els.prepCancelBtn, els.resultCancelBtn]) {
  btn.addEventListener("click", () => {
    engine.setBaseline(app.baseline);
    enterMonitor();
  });
}
els.recalibrateBtn.addEventListener("click", () => {
  if (app.snoozeUntil) endSnooze(true);
  els.prepCancelBtn.hidden = !app.baseline;
  enterStep("prep");
});
els.retryBtn.addEventListener("click", () => {
  els.prepCancelBtn.hidden = !app.baseline;
  enterStep("prep");
});
els.acceptBtn.addEventListener("click", () => {
  applyBaseline(app.pending.baseline, { warnings: app.pending.warnings });
  app.pending = null;
});

function captureFrame(frame, now) {
  const c = app.calib;
  c.frames.push(frame);
  const p = Math.min(1, (now - c.start) / CALIB_MS);
  els.captureArc.style.strokeDashoffset = String(119.4 * (1 - p));
  const left = Math.ceil((CALIB_MS - (now - c.start)) / 1000);
  setText(els.captureTitle, left > 0 ? `Hold still… ${left}` : "Done");
  if (p >= 1) {
    app.calib = null;
    finishCalibration(c.frames);
  }
}

function bestBaselines() {
  return loadJSON(KEY + "best") || {};
}

function finishCalibration(frames) {
  const reference = bestBaselines()[app.source.deviceKey] ?? null;
  const { baseline, errors, warnings } = buildBaseline(frames, reference);
  if (!errors.length && !warnings.length) {
    applyBaseline(baseline);
    return;
  }
  app.pending = { baseline, warnings };
  els.resultTitle.textContent = errors.length ? "Let's try that again" : "Double-check your posture";
  els.resultList.replaceChildren(
    ...errors.map((e) => li(CALIBRATION_MESSAGES[e], "err")),
    ...warnings.map((w) => li(CALIBRATION_MESSAGES[w], "warn")),
  );
  els.acceptBtn.hidden = errors.length > 0;
  els.acceptBtn.textContent = warnings.includes("worseThanBefore") ? "I moved my camera or chair" : "Use anyway";
  els.resultCancelBtn.hidden = !app.baseline;
  enterStep("result");
}

function applyBaseline(baseline, { warnings = [], quiet = false } = {}) {
  const deviceKey = app.source?.deviceKey ?? "demo";
  app.baseline = { ...baseline, at: Date.now(), overridden: warnings.length > 0, deviceKey };
  saveJSON(KEY + "baseline", app.baseline);

  // Track the most upright calibration per camera. Claiming a setup change
  // resets the reference; otherwise only a better calibration replaces it.
  const best = bestBaselines();
  best[deviceKey] = warnings.includes("worseThanBefore") ? app.baseline : moreUpright(best[deviceKey], app.baseline);
  saveJSON(KEY + "best", best);

  engine.setBaseline(app.baseline);
  document.body.classList.add("has-baseline");
  updateCalibInfo();
  enterMonitor();
  if (!quiet) toast(warnings.length ? "Calibrated (with warnings). You can recalibrate anytime." : "Calibrated. Monitoring your posture.");
}

function updateCalibInfo() {
  const b = app.baseline;
  els.calibInfo.textContent = b ? `Calibrated ${fmtDate(b.at)}${b.overridden ? " · with warnings" : ""}` : "Not calibrated";
}

// ---------------------------------------------------------------------------
// Alerts

function fireAlert(issues, title = "Posture check") {
  const msgs = issueMessages(issues, 2);
  if (settings.sound) beep();
  if (settings.notify && "Notification" in window && Notification.permission === "granted") {
    try {
      const n = new Notification(title, {
        body: msgs.map((m) => `${m.issue}. ${m.fix}`).join("\n"),
        tag: "posture",
        renotify: true,
        icon: faviconHref(COLORS.bad),
      });
      n.onclick = () => { window.focus(); n.close(); };
    } catch (err) {
      console.warn("Notification failed", err);
    }
  }
  document.title = "⚠ Sit up straight! · Posture";
}

function clearAlertUI() {
  els.banner.hidden = true;
  document.body.classList.remove("alerting");
  document.title = "Posture Checker";
}

els.testAlertBtn.addEventListener("click", () => {
  initAudio();
  requestNotifications();
  const sample = [{ key: "headSize" }];
  fireAlert(sample, "Posture check (preview)");
  const [m] = issueMessages(sample);
  setText(els.bannerTitle, `${m.issue} (preview)`);
  setText(els.bannerFix, m.fix);
  els.banner.hidden = false;
  document.body.classList.add("alerting");
  setTimeout(() => { if (!engine.alerting) clearAlertUI(); }, 3500);
});

function trackBreaks(status, now, dt) {
  if (status === "away") {
    app.awayRun += dt;
    if (app.awayRun >= BREAK_RESET_MS) app.sitMs = 0;
    return;
  }
  app.awayRun = 0;
  app.sitMs += dt;
  const every = settings.breakEvery * 60_000;
  if (every && app.sitMs >= every) {
    app.sitMs = 0;
    if (settings.sound) beep();
    if (settings.notify && "Notification" in window && Notification.permission === "granted") {
      try { new Notification("Time for a break", { body: `You've been sitting for ${settings.breakEvery} minutes. Stand up, stretch, look far away.`, tag: "posture-break" }); } catch { /* ignore */ }
    }
    toast(`You've been sitting for ${settings.breakEvery} minutes. Time to stand up and stretch.`, 10_000);
  }
}

function initAudio() {
  try {
    app.audio ??= new AudioContext();
    app.audio.resume();
  } catch { /* no audio */ }
}

function beep() {
  const ac = app.audio;
  if (!ac) return;
  if (ac.state === "suspended") ac.resume();
  const t = ac.currentTime;
  [[784, 0], [587, 0.22]].forEach(([freq, offset]) => {
    const osc = ac.createOscillator();
    const gain = ac.createGain();
    osc.type = "sine";
    osc.frequency.value = freq;
    gain.gain.setValueAtTime(0.0001, t + offset);
    gain.gain.exponentialRampToValueAtTime(0.22, t + offset + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.0001, t + offset + 0.25);
    osc.connect(gain).connect(ac.destination);
    osc.start(t + offset);
    osc.stop(t + offset + 0.27);
  });
}

function requestNotifications() {
  if (settings.notify && "Notification" in window && Notification.permission === "default") {
    Notification.requestPermission().then(updateNotifyHint);
  }
}

function updateNotifyHint() {
  if (!("Notification" in window)) els.notifyHint.textContent = "not supported";
  else if (Notification.permission === "denied") els.notifyHint.textContent = "blocked in browser settings";
  else if (Notification.permission === "default") els.notifyHint.textContent = "permission needed";
  else els.notifyHint.textContent = "";
}

// ---------------------------------------------------------------------------
// Snooze

els.snoozeBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  if (app.snoozeUntil) return endSnooze();
  if (els.snoozeMenu.hidden) openMenu();
  else closeMenu();
});
els.snoozeMenu.addEventListener("click", (e) => {
  const min = Number(e.target.dataset.min);
  if (!min) return;
  closeMenu();
  els.snoozeBtn.focus();
  app.snoozeUntil = performance.now() + min * 60_000;
  resetLiveUI();
  ctx.clearRect(0, 0, els.overlay.width, els.overlay.height);
  els.stage.classList.add("dim");
  setStatus("idle", "Snoozed");
  setFavicon("idle");
  renderSnooze(performance.now());
});
els.snoozeMenu.addEventListener("keydown", (e) => {
  const items = [...els.snoozeMenu.querySelectorAll("button")];
  const i = items.indexOf(document.activeElement);
  if (e.key === "ArrowDown") { e.preventDefault(); items[(i + 1) % items.length].focus(); }
  if (e.key === "ArrowUp") { e.preventDefault(); items[(i - 1 + items.length) % items.length].focus(); }
});
document.addEventListener("click", closeMenu);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !els.snoozeMenu.hidden) {
    closeMenu();
    els.snoozeBtn.focus();
  }
});

function openMenu() {
  els.snoozeMenu.hidden = false;
  els.snoozeBtn.setAttribute("aria-expanded", "true");
  els.snoozeMenu.querySelector("button").focus();
}
function closeMenu() {
  els.snoozeMenu.hidden = true;
  els.snoozeBtn.setAttribute("aria-expanded", "false");
}

function renderSnooze(now) {
  const left = fmtDuration(app.snoozeUntil - now);
  setText(els.snoozeLabel, `Resume (${left})`);
  showStageNote("Snoozed", `Monitoring resumes in ${left}`);
  els.snoozeBtn.classList.add("active");
}

function endSnooze(silent = false) {
  app.snoozeUntil = 0;
  engine.reset();
  els.stage.classList.remove("dim");
  hideStageNote();
  els.snoozeLabel.textContent = "Snooze";
  els.snoozeBtn.classList.remove("active");
  if (!silent) toast("Monitoring resumed.");
}

function showStageNote(title, text) {
  setText(els.stageNoteTitle, title);
  setText(els.stageNoteText, text);
  els.stageNote.hidden = false;
}
function hideStageNote() {
  els.stageNote.hidden = true;
}
els.stageNote.addEventListener("click", () => window.focus());

// ---------------------------------------------------------------------------
// Rendering

let statusKey = "";
function setStatus(kind, text) {
  if (statusKey === kind + text) return;
  statusKey = kind + text;
  els.status.dataset.kind = kind;
  els.statusText.textContent = text;
}

/** Only touch the DOM when text actually changes (keeps live regions quiet). */
function setText(el, text) {
  if (el.textContent !== text) el.textContent = text;
}

let faviconKey = "";
function setFavicon(status) {
  const color = COLORS[status] ?? COLORS.idle;
  if (faviconKey === color) return;
  faviconKey = color;
  els.favicon.href = faviconHref(color);
}
function faviconHref(color) {
  return `data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><circle cx='50' cy='50' r='46' fill='${encodeURIComponent(color)}'/></svg>`;
}

function scoreColor(score) {
  if (score == null) return COLORS.away;
  return score >= 75 ? COLORS.good : score >= 60 ? COLORS.warn : COLORS.bad;
}

function renderScore(score, status) {
  setText(els.scoreNow, score == null ? "–" : String(score));
  els.ringArc.style.strokeDashoffset = String(326.7 * (1 - (score ?? 0) / 100));
  // Colour follows the verdict, so the ring and the status pill always agree.
  els.ringArc.style.stroke = COLORS[status] ?? COLORS.away;
}

function renderStats(s) {
  const total = s.goodMs + s.badMs;
  setText(els.goodPct, total > 2000 ? `${Math.round((s.goodMs / total) * 100)}%` : "–");
  setText(els.scoreAvg, s.scoredMs > 2000 ? String(Math.round(s.scoreMs / s.scoredMs)) : "–");
  setText(els.alertCount, String(s.alerts));
  setText(els.streak, fmtDuration(s.streakMs));
  setText(els.bestStreak, fmtDuration(s.bestStreakMs));
  setText(els.awayInfo, s.awayMs > 60_000 ? `away ${fmtDuration(s.awayMs)}` : "");
}

const CHECKS = [
  { label: "Head height", keys: ["neck"] },
  { label: "Head forward", keys: ["headSize", "depth"] },
  { label: "Sitting height", keys: ["sink"] },
  { label: "Distance to screen", keys: ["lean"] },
  { label: "Head angle", keys: ["pitch"] },
  { label: "Shoulders level", keys: ["shoulderTilt"] },
  { label: "Head tilt", keys: ["headTilt"] },
];
let checkEls = null;
function renderChecks(snap) {
  if (!checkEls) {
    checkEls = CHECKS.map((c) => {
      const row = document.createElement("li");
      row.className = "check na";
      row.innerHTML = `<span class="check-name"></span><span class="check-val">–</span>
        <div class="check-bar"><div class="check-fill"></div><div class="check-limit"></div></div>`;
      row.querySelector(".check-name").textContent = c.label;
      return { row, val: row.querySelector(".check-val"), fill: row.querySelector(".check-fill") };
    });
    els.checks.replaceChildren(...checkEls.map((c) => c.row));
  }
  const active = new Set(snap?.issues?.map((i) => i.key));
  CHECKS.forEach((c, i) => {
    const { row, val, fill } = checkEls[i];
    const sevs = c.keys.map((k) => snap?.severities?.[k]).filter((v) => v != null);
    if (!sevs.length) {
      row.className = "check na";
      setText(val, "–");
      fill.style.width = "0";
      return;
    }
    const sev = Math.max(...sevs);
    const on = c.keys.some((k) => active.has(k));
    row.className = `check ${on ? "off" : sev >= 0.75 ? "warn" : ""}`;
    setText(val, on ? "Fix" : sev >= 0.75 ? "Drifting" : sev >= 0.4 ? "OK" : "Good");
    fill.style.width = `${Math.max(3, Math.min(1, sev / 1.5) * 100)}%`;
  });
}

function accumulateTimeline(snap, now, dt) {
  const idx = Math.floor(now / BUCKET_MS);
  let b = app.timeline.get(idx);
  if (!b) {
    b = { scoreMs: 0, ms: 0, awayMs: 0 };
    app.timeline.set(idx, b);
    for (const k of app.timeline.keys()) if (k < idx - TIMELINE_BUCKETS) app.timeline.delete(k);
  }
  if (!snap || snap.status === "away") b.awayMs += dt;
  else if (snap.score != null) { b.scoreMs += snap.score * dt; b.ms += dt; }
  else if (snap.issues?.length) b.ms += dt; // can't-see-you time scores zero
}

function renderTimeline(now) {
  const c = els.timeline;
  const dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth;
  const h = c.clientHeight;
  if (!w) return;
  if (c.width !== Math.round(w * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
  const g = c.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);

  const css = getComputedStyle(document.documentElement);
  g.fillStyle = css.getPropertyValue("--border");
  g.fillRect(0, h - 1, w, 1);

  const slot = w / TIMELINE_BUCKETS;
  const barW = Math.max(2, slot - 3);
  const last = Math.floor(now / BUCKET_MS);
  let any = false;
  for (let i = 0; i < TIMELINE_BUCKETS; i++) {
    const b = app.timeline.get(last - (TIMELINE_BUCKETS - 1 - i));
    if (!b) continue;
    const x = i * slot + 1;
    if (b.ms < 1000) {
      if (b.awayMs > 0) {
        g.fillStyle = css.getPropertyValue("--text-3");
        g.globalAlpha = 0.4;
        g.fillRect(x, h - 4, barW, 3);
        g.globalAlpha = 1;
      }
      continue;
    }
    any = true;
    const score = b.scoreMs / b.ms;
    const bh = Math.max(4, (score / 100) * (h - 2));
    g.fillStyle = scoreColor(score);
    roundRect(g, x, h - 1 - bh, barW, bh, Math.min(3, barW / 2));
  }
  els.timelineEmpty.hidden = any;
}

function roundRect(g, x, y, w, h, r) {
  g.beginPath();
  g.moveTo(x + r, y);
  g.arcTo(x + w, y, x + w, y + h, r);
  g.lineTo(x + w, y + h);
  g.lineTo(x, y + h);
  g.arcTo(x, y, x + w, y, r);
  g.closePath();
  g.fill();
}

// ---------------------------------------------------------------------------
// History (per-day totals)

function maybeFlushHistory(stats, now) {
  if (now - app.lastFlush < 10_000) return;
  app.lastFlush = now;
  const hist = loadJSON(KEY + "history") || {};
  const day = (hist[todayKey()] ??= { goodMs: 0, badMs: 0, alerts: 0 });
  for (const k of ["goodMs", "badMs", "alerts"]) {
    day[k] += Math.max(0, stats[k] - app.flushed[k]);
    app.flushed[k] = stats[k];
  }
  const keys = Object.keys(hist).sort();
  while (keys.length > HISTORY_DAYS) delete hist[keys.shift()];
  saveJSON(KEY + "history", hist);
  renderWeek();
}

function renderWeek() {
  const hist = loadJSON(KEY + "history") || {};
  const days = [];
  for (let i = 6; i >= 0; i--) {
    const d = new Date();
    d.setDate(d.getDate() - i);
    const rec = hist[todayKey(d)];
    const total = rec ? rec.goodMs + rec.badMs : 0;
    const label = d.toLocaleDateString("en-US", { weekday: "short" });
    days.push({
      label: label.slice(0, 2),
      pct: total > 60_000 ? rec.goodMs / total : null,
      title: total > 0 ? `${label}: ${Math.round((rec.goodMs / total) * 100)}% sitting well, ${rec.alerts} alerts, ${fmtDuration(total)} tracked` : `${label}: no data`,
      today: i === 0,
    });
  }
  els.week.replaceChildren(
    ...days.map((d) => {
      const el = document.createElement("div");
      el.className = `day${d.today ? " today" : ""}`;
      el.title = d.title;
      el.setAttribute("aria-label", d.title);
      const pct = d.pct == null ? 0 : Math.max(4, d.pct * 100);
      el.innerHTML = `<div class="day-bar"><div class="day-fill" style="height:${pct}%;background:${scoreColor(d.pct == null ? null : d.pct * 100)}"></div></div><div class="day-lbl">${d.label}</div>`;
      return el;
    }),
  );
}

function seedDemoHistory() {
  const hist = {};
  const pcts = [0.62, 0.71, 0.58, 0.77, 0.83, 0.8];
  pcts.forEach((p, i) => {
    const d = new Date();
    d.setDate(d.getDate() - (6 - i));
    const total = (3 + (i % 3)) * 3_600_000;
    hist[todayKey(d)] = { goodMs: total * p, badMs: total * (1 - p), alerts: Math.round(20 * (1 - p)) };
  });
  saveJSON(KEY + "history", hist);
}

// ---------------------------------------------------------------------------
// Drawing

function draw(raw, frame, snap) {
  if (document.hidden) return;
  const c = els.overlay;
  if (c.width !== raw.w || c.height !== raw.h) { c.width = raw.w; c.height = raw.h; }
  ctx.clearRect(0, 0, raw.w, raw.h);

  const lm = raw.landmarks;
  if (DEMO) drawDemoScene(lm, raw.w, raw.h);
  if (!lm || !settings.skeleton) return;

  const p = (i) => ({ x: lm[i].x * raw.w, y: lm[i].y * raw.h, v: lm[i].visibility ?? 1 });
  const ls = p(11), rs = p(12), le = p(7), re = p(8), nose = p(0);
  const shoulderMid = mid(ls, rs);
  const earMid = mid(le, re);
  const shoulderW = dist(ls, rs);
  const status = snap ? snap.status : "idle";
  const color = COLORS[status] ?? COLORS.idle;
  const unit = raw.w / 640;

  // Where your ears were when calibrated, relative to your shoulders now.
  if (app.phase === "monitor" && app.baseline && frame.presence === "ok") {
    const cos = Math.cos((frame.metrics.yaw * Math.PI) / 180);
    const sw = shoulderW / cos;
    const ty = shoulderMid.y - app.baseline.neck * sw;
    const half = (app.baseline.headSize * sw) / 2;
    ctx.save();
    ctx.setLineDash([8 * unit, 7 * unit]);
    ctx.lineWidth = 2.5 * unit;
    ctx.strokeStyle = "rgba(255,255,255,0.75)";
    ctx.beginPath();
    ctx.moveTo(shoulderMid.x - half * 1.6, ty);
    ctx.lineTo(shoulderMid.x + half * 1.6, ty);
    ctx.stroke();
    ctx.setLineDash([]);
    for (const dx of [-half, half]) {
      ctx.beginPath();
      ctx.arc(shoulderMid.x + dx, ty, 7 * unit, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.restore();
  }

  ctx.lineCap = "round";
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.shadowColor = "rgba(0,0,0,0.4)";
  ctx.shadowBlur = 6 * unit;
  ctx.lineWidth = 5 * unit;
  const seg = (a, b) => { ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke(); };
  if (ls.v > 0.5 && rs.v > 0.5) seg(ls, rs);
  if (le.v > 0.5 && re.v > 0.5) seg(le, re);
  if (Math.min(ls.v, rs.v, le.v, re.v) > 0.5) seg(shoulderMid, earMid);
  for (const pt of [ls, rs, le, re, nose]) {
    if (pt.v < 0.5) continue;
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, 6.5 * unit, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.shadowBlur = 0;
}

function drawDemoScene(lm, w, h) {
  const g = ctx.createLinearGradient(0, 0, 0, h);
  g.addColorStop(0, "#2a303a");
  g.addColorStop(1, "#15181d");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, w, h);
  if (!lm) return;
  const p = (i) => ({ x: lm[i].x * w, y: lm[i].y * h });
  const ls = p(11), rs = p(12), le = p(7), re = p(8);
  const sw = dist(ls, rs);
  const em = mid(le, re);
  const ew = dist(le, re);
  ctx.fillStyle = "#4a5361";
  ctx.beginPath();
  ctx.moveTo(rs.x - sw * 0.12, h + 10);
  ctx.lineTo(rs.x - sw * 0.1, rs.y + sw * 0.12);
  ctx.quadraticCurveTo(rs.x - sw * 0.06, rs.y - sw * 0.06, rs.x + sw * 0.12, rs.y - sw * 0.08);
  ctx.lineTo(ls.x - sw * 0.12, ls.y - sw * 0.08);
  ctx.quadraticCurveTo(ls.x + sw * 0.06, ls.y - sw * 0.06, ls.x + sw * 0.1, ls.y + sw * 0.12);
  ctx.lineTo(ls.x + sw * 0.12, h + 10);
  ctx.closePath();
  ctx.fill();
  const sm = mid(ls, rs);
  ctx.fillRect(sm.x - ew * 0.33, em.y, ew * 0.66, sm.y - em.y);
  ctx.beginPath();
  ctx.ellipse(em.x, em.y - ew * 0.15, ew * 0.62, ew * 0.8, 0, 0, Math.PI * 2);
  ctx.fill();
}

// ---------------------------------------------------------------------------
// Settings

function initSettings() {
  const segButtons = [...els.sensitivity.querySelectorAll("button")];
  for (const b of segButtons) {
    b.setAttribute("role", "radio");
    b.addEventListener("click", () => {
      settings.sensitivity = b.dataset.v;
      apply();
    });
  }
  els.grace.value = settings.grace;
  els.breakEvery.value = String(settings.breakEvery);
  for (const k of ["sound", "notify", "mirror", "skeleton"]) els[k].checked = settings[k];

  function apply() {
    settings.grace = Number(els.grace.value);
    settings.breakEvery = Number(els.breakEvery.value);
    for (const k of ["sound", "notify", "mirror", "skeleton"]) settings[k] = els[k].checked;
    els.graceVal.textContent = settings.grace;
    els.stage.classList.toggle("mirrored", settings.mirror);
    for (const o of segButtons) {
      const on = o.dataset.v === settings.sensitivity;
      o.classList.toggle("on", on);
      o.setAttribute("aria-checked", String(on));
    }
    engine.sensitivity = settings.sensitivity;
    engine.graceMs = settings.grace * 1000;
    saveJSON(KEY + "settings", settings);
  }
  for (const k of ["grace", "breakEvery", "sound", "notify", "mirror", "skeleton"]) els[k].addEventListener("input", apply);
  els.notify.addEventListener("change", () => { if (els.notify.checked) requestNotifications(); });
  apply();
  updateNotifyHint();
}

// ---------------------------------------------------------------------------
// Helpers

function lumaStats(data) {
  let sum = 0;
  let sq = 0;
  const n = data.length / 4;
  for (let i = 0; i < data.length; i += 4) {
    const y = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    sum += y;
    sq += y * y;
  }
  const mean = sum / n;
  return { mean, std: Math.sqrt(Math.max(0, sq / n - mean * mean)) };
}

function li(text, cls) {
  const el = document.createElement("li");
  el.className = cls;
  el.textContent = text;
  return el;
}

function setStartMsg(text, error = false) {
  els.startMsg.textContent = text;
  els.startMsg.classList.toggle("error", error);
}

let toastTimer = 0;
function toast(text, ms = 3000) {
  els.toast.textContent = text;
  els.toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { els.toast.hidden = true; }, ms);
}

function fmtDuration(ms) {
  const total = Math.max(0, Math.floor(ms / 1000));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = String(total % 60).padStart(2, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${s}` : `${m}:${s}`;
}

function fmtDate(ts) {
  const d = new Date(ts);
  const sameDay = d.toDateString() === new Date().toDateString();
  const time = d.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" });
  return sameDay ? `today ${time}` : `${d.toLocaleDateString("en-US", { month: "short", day: "numeric" })} ${time}`;
}

function todayKey(d = new Date()) {
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

function loadJSON(key) {
  try { return JSON.parse(localStorage.getItem(key)); } catch { return null; }
}
function saveJSON(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage unavailable */ }
}

// ---------------------------------------------------------------------------
// Boot

// Debug / test hook
window.__posture = { app, engine, settings };

initSettings();
renderChecks(null);
renderWeek();
updateCalibInfo();
document.body.classList.toggle("has-baseline", !!app.baseline);
showStep("welcome");
if (DEMO) startDemo();
else maybeAutoStart();
