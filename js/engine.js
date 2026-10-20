// Posture engine: turns pose landmarks into posture metrics, judges them
// against a calibrated baseline, and runs the alert state machine.
// Pure logic (no DOM) so it can be unit-tested in Node.
//
// Design notes (from red-team + real-world review):
// - 2D image geometry is the primary signal. MediaPipe's world landmarks are
//   largely inferred when the hips are out of frame, so 3D is only used for
//   torso yaw and as corroborating evidence, never to raise an alert alone.
// - Camera-position-sensitive metrics (sink) need corroboration from
//   camera-independent ones before they can alert.
// - Short dropouts (a tracking blip, ducking out of view) neither end an alert
//   nor raise one; sustained ones do.

export const LM = { NOSE: 0, L_EAR: 7, R_EAR: 8, L_SHOULDER: 11, R_SHOULDER: 12 };

const MIN_VIS = 0.5;
const EAR_VIS = 0.65;          // both ears needed for ear-width / tilt metrics
const MAX_YAW_DEG = 30;        // torso turned further than this → can't judge
const TILT_MAX_YAW_DEG = 15;   // beyond this, 2D shoulder tilt is perspective, not posture
const HEAD_YAW_LIMIT_DEG = 30; // beyond this, ear-width metrics are unreliable
const NOSE_EAR_RATIO = 1.5;    // ear-to-ear width / nose-ahead-of-ears depth (≈15 cm / 10 cm)
const FOCAL_GUESS = 0.65;      // focal length / image width for a typical webcam
const SHOULDER_M = 0.38;       // typical shoulder-landmark span, metres
const NOSE_DEPTH_M = 0.1;
const SMOOTH_TAU_MS = 350;
const HYSTERESIS = 0.75;       // an issue clears only once severity drops below this
const COMPOSITE_TH = 1.25;     // combined severity across groups that counts as bad
const RECOVERY_RATE = 0.5;     // good posture drains the bad-time bucket at half speed
const CLEAR_AFTER_MS = 2000;   // sustained good posture needed to end an alert
const AWAY_AFTER_MS = 3000;    // "nobody there" must last this long to count
const PARTIAL_AFTER_MS = 1500; // "can't see you properly" must last this long to count
const FROZEN_AFTER_MS = 60_000;
const FROZEN_EPS_PX = 0.05;
const SETUP_HINT_MS = 20_000;
const MIN_SCORE_SEVERITY = 0.15;

export const SENSITIVITY = { low: 1.4, medium: 1, high: 0.7 };

/**
 * kind:
 *  relLower   – bad when the value shrinks (relative change)
 *  relHigher  – bad when the value grows (relative change)
 *  diffHigher – bad when the value grows (absolute change)
 *  diffBoth   – bad in either direction (absolute change)
 */
export const METRICS = {
  neck: {
    label: "Head height", kind: "relLower", th: 0.14, group: "head",
    issue: "Head sinking toward your shoulders",
    fix: "Grow tall: imagine a string pulling the crown of your head up.",
  },
  headSize: {
    label: "Head forward", kind: "relHigher", th: 0.12, group: "head",
    issue: "Head jutting forward",
    fix: "Glide your chin straight back, ears over shoulders.",
  },
  depth: {
    label: "Head forward (3D)", kind: "diffHigher", th: 0.18, group: "head",
    issue: "Head jutting forward",
    fix: "Glide your chin straight back, ears over shoulders.",
  },
  lean: {
    label: "Distance to screen", kind: "relHigher", th: 0.28, group: "lean",
    issue: "Too close to the screen",
    fix: "Sit back so your back meets the chair.",
  },
  sink: {
    label: "Sitting height", kind: "diffHigher", th: 0.22, group: "sink",
    issue: "Slumping down in your chair",
    fix: "Scoot your hips back and sit up tall.",
  },
  pitch: {
    label: "Head angle", kind: "diffBoth", th: 0.3, group: "pitch",
    issue: "Looking down (text neck)", issueNeg: "Chin tilted up",
    fix: "Bring your gaze level with the top of the screen.", fixNeg: "Level your chin.",
  },
  shoulderTilt: {
    label: "Shoulders level", kind: "diffBoth", th: 8, group: "shoulders",
    issue: "Shoulders uneven",
    fix: "Drop both shoulders and relax them evenly.",
  },
  headTilt: {
    label: "Head tilt", kind: "diffBoth", th: 9, group: "tilt",
    issue: "Head tilted to one side",
    fix: "Center your head over your spine.",
  },
};
export const METRIC_KEYS = Object.keys(METRICS);

export const OVERALL = {
  issue: "Slouching overall",
  fix: "Reset: hips back, sit tall, shoulders relaxed, ears over shoulders.",
};

// Problems that stop us from judging posture at all. Once sustained they count
// as bad time and alert, so hiding from the camera isn't a way to silence it.
export const PRESENCE_ISSUES = {
  shoulders: { issue: "Can't see your shoulders", fix: "Move back or tilt the camera down so both shoulders are in view." },
  head: { issue: "Can't see your head", fix: "Adjust the camera so your whole head is in view." },
  turned: { issue: "You're turned away", fix: "Face the screen squarely." },
  far: { issue: "You're too far from the camera", fix: "Move closer so your shoulders fill more of the frame." },
  blocked: { issue: "Camera is covered or too dark", fix: "Uncover the camera or turn on a light." },
  frozen: { issue: "Camera image looks frozen", fix: "Check the camera feed; the picture hasn't changed in a minute." },
};

// ---------------------------------------------------------------------------
// Frame extraction

/**
 * Convert one PoseLandmarker result into posture metrics.
 * @returns {{presence: 'ok'|'partial'|'away', reason?: string, metrics?: object, points?: object}}
 */
export function extractFrame(landmarks, worldLandmarks, w, h) {
  if (!landmarks || landmarks.length < 13) return { presence: "away" };
  const vis = (i) => landmarks[i]?.visibility ?? 1;
  const lv = vis(LM.L_EAR), rv = vis(LM.R_EAR), nv = vis(LM.NOSE);
  const shoulderVis = Math.min(vis(LM.L_SHOULDER), vis(LM.R_SHOULDER));
  const faceVis = Math.max(nv, Math.max(lv, rv));

  if (faceVis < MIN_VIS && shoulderVis < MIN_VIS) return { presence: "away" };
  if (shoulderVis < MIN_VIS) return { presence: "partial", reason: "shoulders" };
  // Need the nose and at least one ear (the far ear hides when looking sideways).
  if (nv < MIN_VIS || Math.max(lv, rv) < MIN_VIS) return { presence: "partial", reason: "head" };

  const px = (i) => ({ x: landmarks[i].x * w, y: landmarks[i].y * h });
  const ls = px(LM.L_SHOULDER), rs = px(LM.R_SHOULDER);
  const le = px(LM.L_EAR), re = px(LM.R_EAR), nose = px(LM.NOSE);
  const points = { ls, rs, le, re, nose };

  // Shoulders at the very edge of the frame are extrapolated, not seen.
  const edge = (p) => p.y > 0.97 * h || p.x < 0.01 * w || p.x > 0.99 * w;
  if (edge(ls) || edge(rs)) return { presence: "partial", reason: "shoulders", points };

  // Torso yaw from 3D: turning shrinks the apparent shoulder width, which would
  // otherwise make every ratio look "better". Depth is corroboration only.
  let yaw = 0;
  let depth = null;
  if (worldLandmarks?.length > 12) {
    const wl = worldLandmarks[LM.L_SHOULDER], wr = worldLandmarks[LM.R_SHOULDER];
    const wle = worldLandmarks[LM.L_EAR], wre = worldLandmarks[LM.R_EAR];
    const wsw = Math.hypot(wl.x - wr.x, wl.y - wr.y, wl.z - wr.z);
    if (wsw > 1e-6) {
      yaw = deg(Math.asin(clamp((wl.z - wr.z) / wsw, -1, 1)));
      depth = ((wl.z + wr.z) / 2 - (wle.z + wre.z) / 2) / wsw;
    }
  }
  if (Math.abs(yaw) > MAX_YAW_DEG) return { presence: "partial", reason: "turned", points };

  const shoulderW = dist(ls, rs) / Math.cos(rad(yaw));
  if (shoulderW < 0.12 * w) return { presence: "partial", reason: "far", points };
  const shoulderMid = mid(ls, rs);

  const bothEars = Math.min(lv, rv) >= EAR_VIS;
  let headSize = null, pitch = null, headTilt = null, neck = null, headYaw = null;
  if (bothEars) {
    const earMid = mid(le, re);
    const earWraw = dist(le, re);
    // Head yaw from how far the nose sits off the ear midline, after removing
    // the parallax you get from simply sitting off-centre (the nose is closer
    // to the camera than the ears, so it shifts toward the image centre less).
    const z = (FOCAL_GUESS * w * SHOULDER_M) / shoulderW;
    const parallax = (earMid.x - w / 2) * (NOSE_DEPTH_M / Math.max(z - NOSE_DEPTH_M, 0.1));
    const noseOff = nose.x - earMid.x - parallax;
    headYaw = deg(Math.atan((NOSE_EAR_RATIO * noseOff) / Math.max(earWraw, 1)));
    neck = (shoulderMid.y - earMid.y) / shoulderW;
    if (Math.abs(headYaw) <= HEAD_YAW_LIMIT_DEG) {
      const earW = earWraw / Math.cos(rad(headYaw));
      headSize = earW / shoulderW;
      pitch = (nose.y - earMid.y) / earW;
      headTilt = angleDeg(re, le);
    }
  } else {
    // One ear hidden (looking sideways): only the visible ear's height is usable.
    const ear = lv >= rv ? le : re;
    neck = (shoulderMid.y - ear.y) / shoulderW;
    depth = null;
  }

  return {
    presence: "ok",
    points,
    metrics: {
      neck,
      headSize,
      depth: headSize == null ? null : depth,
      lean: shoulderW / w,
      sink: (shoulderMid.y - h / 2) / shoulderW,
      pitch,
      shoulderTilt: Math.abs(yaw) <= TILT_MAX_YAW_DEG ? angleDeg(rs, ls) : null,
      headTilt,
      yaw,
      headYaw,
    },
  };
}

// ---------------------------------------------------------------------------
// Calibration

export const CALIBRATION_MESSAGES = {
  notVisible: "Couldn't see your head and shoulders clearly. Check the framing and lighting.",
  moved: "You moved during calibration. Hold still and try again.",
  turned: "Face the camera squarely while calibrating.",
  shoulderTilt: "Your shoulders aren't level. Relax them evenly (or level the camera).",
  headTilt: "Your head is tilted. Center it and try again.",
  lookingDown: "Your head looks tilted down. Look straight at the screen (or the camera is mounted high).",
  chinUp: "Your chin looks raised. Level your head (or the camera is mounted low).",
  neckShort: "Your head looks low relative to your shoulders. Are you sitting fully upright?",
  forward: "Your head looks forward of your shoulders. Glide your chin back.",
  worseThanBefore: "This looks more slouched than your best calibration on this camera.",
};

/**
 * Build a baseline from frames captured while the user sits upright.
 * Errors block the calibration; warnings need the user's confirmation.
 * @param reference the best (most upright) earlier baseline for this camera
 */
export function buildBaseline(frames, reference = null) {
  const errors = [];
  const warnings = [];
  const ok = frames.filter((f) => f.presence === "ok");
  if (ok.length < 12 || ok.length < frames.length * 0.7) {
    const turned = frames.filter((f) => f.reason === "turned").length > frames.length / 3;
    return { baseline: null, errors: [turned ? "turned" : "notVisible"], warnings };
  }

  const series = (k) => ok.map((f) => f.metrics[k]).filter((v) => v != null && Number.isFinite(v));
  const baseline = {};
  for (const k of [...METRIC_KEYS, "yaw", "headYaw"]) {
    const s = series(k);
    baseline[k] = s.length >= ok.length * 0.6 ? median(s) : null;
  }
  if (baseline.headSize == null || baseline.neck == null) {
    return { baseline: null, errors: ["notVisible"], warnings };
  }

  // Stability: the median absolute deviation must be small relative to the value.
  const unstable =
    mad(series("neck")) / Math.abs(baseline.neck) > 0.05 ||
    mad(series("sink")) > 0.05 ||
    mad(series("lean")) / baseline.lean > 0.05;
  if (unstable) errors.push("moved");

  if (Math.abs(baseline.yaw) > 15 || Math.abs(baseline.headYaw ?? 0) > 15) errors.push("turned");
  if (baseline.shoulderTilt != null && Math.abs(baseline.shoulderTilt) > 7) errors.push("shoulderTilt");
  if (baseline.headTilt != null && Math.abs(baseline.headTilt) > 10) errors.push("headTilt");

  // Camera height shifts these, so they only warn.
  if (baseline.pitch != null && baseline.pitch > 0.6) warnings.push("lookingDown");
  if (baseline.pitch != null && baseline.pitch < -0.2) warnings.push("chinUp");
  if (baseline.neck < 0.33) warnings.push("neckShort");
  if (baseline.depth != null && baseline.depth > 0.16) warnings.push("forward");

  // Compare with the best calibration so far: ratcheting the target down a
  // little at a time still gets caught. These ratios don't depend on distance.
  if (reference && isWorse(baseline, reference)) warnings.push("worseThanBefore");

  return { baseline: errors.length ? null : baseline, errors, warnings };
}

function isWorse(b, ref) {
  return (
    (ref.neck && b.neck < ref.neck * 0.92) ||
    (ref.headSize && b.headSize > ref.headSize * 1.08) ||
    (ref.sink != null && b.sink > ref.sink + 0.15) ||
    (ref.depth != null && b.depth != null && b.depth > ref.depth + 0.1)
  );
}

/** Of two baselines, the more upright one (higher head, smaller forward head). */
export function moreUpright(a, b) {
  if (!a) return b;
  if (!b) return a;
  const score = (x) => x.neck / x.headSize;
  return score(b) > score(a) ? b : a;
}

// ---------------------------------------------------------------------------
// Live judging + alert state machine

export class PostureEngine {
  constructor({ sensitivity = "medium", graceMs = 5000, repeatMs = 30000 } = {}) {
    this.sensitivity = sensitivity;
    this.graceMs = graceMs;
    this.repeatMs = repeatMs;
    this.baseline = null;
    this.stats = { goodMs: 0, badMs: 0, awayMs: 0, alerts: 0, scoreMs: 0, scoredMs: 0, streakMs: 0, bestStreakMs: 0 };
    this.reset();
  }

  setBaseline(baseline) {
    this.baseline = baseline;
    this.reset();
  }

  reset() {
    this.smooth = null;
    this.active = new Set();
    this.level = 0;
    this.alerting = false;
    this.goodRun = 0;
    this.lastAlertAt = -Infinity;
    this.lastNow = null;
    this.absent = null;          // { kind, since } for away/partial debouncing
    this.prevPoints = null;
    this.stillSince = null;
    this.setupOddSince = null;
    this.last = null;
  }

  /**
   * Advance the engine with one frame.
   * @param frame result of extractFrame(), or {presence:'partial', reason:'blocked'}
   * @param now   ms timestamp (monotonic)
   */
  update(frame, now) {
    const dt = this.lastNow == null ? 0 : clamp(now - this.lastNow, 0, 1000);
    this.lastNow = now;
    const s = this.stats;

    if (frame.presence === "ok") {
      frame = this.checkContinuity(frame, now);
      if (!frame) return this.hold();
    }

    // Debounce dropouts: a blip neither ends an alert nor counts as bad.
    if (frame.presence !== "ok") {
      const kind = frame.presence === "away" ? "away" : frame.reason;
      if (this.absent?.kind !== kind) this.absent = { kind, since: now };
      const needed = frame.presence === "away" ? AWAY_AFTER_MS : PARTIAL_AFTER_MS;
      if (now - this.absent.since < needed) return this.hold();
    } else {
      this.absent = null;
    }

    if (frame.presence === "away") {
      s.awayMs += dt;
      this.level = Math.max(0, this.level - dt * RECOVERY_RATE);
      const alert = this.alerting ? this.endAlert() : null;
      return this.snapshot({ status: "away", presence: "away", issues: [], alert });
    }

    let issues = [];
    let devs = null;
    let severities = null;
    let score = null;
    let hint = null;

    if (frame.presence === "partial") {
      issues = [{ key: frame.reason, presence: true, severity: 1 }];
    } else {
      this.smooth = smoothMetrics(this.smooth, frame.metrics, dt);
      if (!this.baseline) {
        return this.snapshot({ status: "uncalibrated", presence: "ok", issues: [], alert: null });
      }
      const ev = this.evaluate(this.smooth);
      ({ devs, severities } = ev);
      for (const k of METRIC_KEYS) {
        const sev = severities[k];
        if (sev == null) { this.active.delete(k); continue; }
        if (sev >= 1) this.active.add(k);
        else if (sev < HYSTERESIS) this.active.delete(k);
      }
      issues = [...this.active]
        .filter((k) => k !== "overall")
        .map((k) => ({ key: k, severity: severities[k], negative: devs[k] < 0 }))
        .sort((a, b) => b.severity - a.severity);

      // Several metrics each just under their limit is still a slouch.
      const comp = ev.composite / COMPOSITE_TH;
      if (comp >= 1 || (this.active.has("overall") && comp >= HYSTERESIS)) this.active.add("overall");
      else this.active.delete("overall");
      if (this.active.has("overall") && !issues.length) issues = [{ key: "overall", severity: comp }];

      score = scoreFrom(Math.max(ev.worst, comp));
      hint = this.setupHint(severities, devs, now);
    }

    const bad = issues.length > 0;
    let alert = null;

    if (bad) {
      this.level = Math.min(this.level + dt, this.graceMs * 1.5);
      this.goodRun = 0;
      s.badMs += dt;
      s.streakMs = 0;
      if (!this.alerting && this.level >= this.graceMs) {
        this.alerting = true;
        s.alerts++;
        this.lastAlertAt = now;
        alert = { type: "start", issues };
      } else if (this.alerting && now - this.lastAlertAt >= this.repeatMs) {
        this.lastAlertAt = now;
        alert = { type: "repeat", issues };
      }
    } else {
      this.level = Math.max(0, this.level - dt * RECOVERY_RATE);
      this.goodRun += dt;
      s.goodMs += dt;
      s.streakMs += dt;
      s.bestStreakMs = Math.max(s.bestStreakMs, s.streakMs);
      if (this.alerting && this.goodRun >= CLEAR_AFTER_MS) alert = this.endAlert();
    }

    if (score != null) {
      s.scoreMs += score * dt;
      s.scoredMs += dt;
    } else if (bad) {
      s.scoredMs += dt; // can't-see-you time scores zero
    }

    const status = this.alerting ? "bad" : bad ? "warn" : "good";
    return this.snapshot({ status, presence: frame.presence, issues, devs, severities, score, alert, hint });
  }

  /** Detect frozen feeds and identity jumps (someone else stepping in). */
  checkContinuity(frame, now) {
    const p = frame.points;
    const prev = this.prevPoints;
    this.prevPoints = p;
    if (!prev) { this.stillSince = now; return frame; }

    const sw = dist(p.ls, p.rs), psw = dist(prev.ls, prev.rs);
    const jump = Math.abs(sw / psw - 1) > 0.25 || dist(mid(p.ls, p.rs), mid(prev.ls, prev.rs)) > 0.5 * psw;
    if (jump) {
      this.smooth = null;
      return null;
    }

    const moved = ["ls", "rs", "le", "re", "nose"].some((k) => dist(p[k], prev[k]) > FROZEN_EPS_PX);
    if (moved) this.stillSince = now;
    else if (now - this.stillSince >= FROZEN_AFTER_MS) return { presence: "partial", reason: "frozen", points: p };
    return frame;
  }

  /** Big shifts in camera-dependent metrics with a fine head = probably moved the camera/chair. */
  setupHint(sev, devs, now) {
    const headFine = ["neck", "headSize"].every((k) => sev[k] == null || sev[k] < 0.5);
    const odd = headFine && ((devs.sink != null && Math.abs(devs.sink) > 0.5) || (devs.lean != null && Math.abs(devs.lean) > 0.4));
    if (!odd) { this.setupOddSince = null; return null; }
    this.setupOddSince ??= now;
    return now - this.setupOddSince >= SETUP_HINT_MS ? "setup" : null;
  }

  hold() {
    return this.last
      ? { ...this.last, alert: null, held: true, stats: { ...this.stats } }
      : this.snapshot({ status: this.baseline ? "good" : "uncalibrated", presence: "ok", issues: [], alert: null, held: true });
  }

  endAlert() {
    this.alerting = false;
    this.level = 0;
    return { type: "clear", issues: [] };
  }

  evaluate(m) {
    const mult = SENSITIVITY[this.sensitivity] ?? 1;
    const b = this.baseline;
    const devs = {};
    const severities = {};
    for (const k of METRIC_KEYS) {
      const def = METRICS[k];
      if (m[k] == null || b[k] == null) { devs[k] = null; severities[k] = null; continue; }
      let d;
      switch (def.kind) {
        case "relLower": d = (b[k] - m[k]) / Math.abs(b[k]); break;
        case "relHigher": d = (m[k] - b[k]) / Math.abs(b[k]); break;
        default: d = m[k] - b[k];
      }
      devs[k] = d;
      const magnitude = def.kind === "diffBoth" ? Math.abs(d) : Math.max(0, d);
      severities[k] = magnitude / (def.th * mult);
    }

    // Corroboration: 3D depth and camera-sensitive sink can't alert on their own.
    const headEvidence = Math.max(severities.neck ?? 0, severities.headSize ?? 0);
    if (severities.depth != null) severities.depth = Math.min(severities.depth, headEvidence + 0.5);
    if (severities.sink != null) {
      severities.sink = Math.min(severities.sink, Math.max(headEvidence, severities.pitch ?? 0) + 0.5);
    }

    // Combined severity. Within a group, independent measurements that agree
    // add confidence (strongest + 0.6 × next + 0.3 × third); across groups,
    // root-sum-square.
    const byGroup = {};
    for (const k of METRIC_KEYS) {
      if (severities[k] == null) continue;
      (byGroup[METRICS[k].group] ??= []).push(severities[k]);
    }
    const vals = Object.values(byGroup).map((list) => {
      const [a = 0, b = 0, c = 0] = list.sort((x, y) => y - x);
      return a + 0.6 * b + 0.3 * c;
    });
    const composite = Math.sqrt(vals.reduce((a, v) => a + v * v, 0));
    // Score badness matches the alert rule: 1 = a metric or the composite is at its limit.
    const worst = Math.max(0, ...Object.values(severities).filter((v) => v != null));
    return { devs, severities, composite, worst };
  }

  snapshot(extra) {
    const snap = {
      devs: null,
      severities: null,
      score: null,
      hint: null,
      ...extra,
      level: this.level,
      graceMs: this.graceMs,
      alerting: this.alerting,
      stats: { ...this.stats },
    };
    this.last = snap;
    return snap;
  }
}

/** 100 = perfect, 60 = right at the alert threshold, 0 = far past it. */
export function scoreFrom(badness) {
  const worst = Math.max(0, badness - MIN_SCORE_SEVERITY);
  return Math.round(clamp(100 - 47 * worst, 0, 100));
}

/** Human-readable text for an issue. */
export function describeIssue(issue) {
  if (issue.key === "overall") return OVERALL;
  const p = PRESENCE_ISSUES[issue.key];
  if (p) return p;
  const def = METRICS[issue.key];
  return issue.negative && def.issueNeg
    ? { issue: def.issueNeg, fix: def.fixNeg }
    : { issue: def.issue, fix: def.fix };
}

/** Unique issue descriptions (metrics in the same group share a message). */
export function issueMessages(issues, limit = 2) {
  const seen = new Set();
  const out = [];
  for (const i of issues) {
    const d = describeIssue(i);
    if (seen.has(d.issue)) continue;
    seen.add(d.issue);
    out.push(d);
    if (out.length >= limit) break;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Helpers

function smoothMetrics(prev, raw, dt) {
  if (!prev || dt === 0) return { ...raw };
  const a = 1 - Math.exp(-dt / SMOOTH_TAU_MS);
  const out = {};
  for (const k of Object.keys(raw)) {
    if (raw[k] == null) out[k] = null;
    else if (prev[k] == null) out[k] = raw[k];
    else out[k] = prev[k] + a * (raw[k] - prev[k]);
  }
  return out;
}

export function median(arr) {
  const s = [...arr].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function mad(arr) {
  if (!arr.length) return 0;
  const m = median(arr);
  return median(arr.map((v) => Math.abs(v - m)));
}

const rad = (d) => (d * Math.PI) / 180;
const deg = (r) => (r * 180) / Math.PI;
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
export const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
export const mid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
// Angle of the line from a to b in degrees; 0 when level.
const angleDeg = (a, b) => deg(Math.atan2(b.y - a.y, b.x - a.x));
