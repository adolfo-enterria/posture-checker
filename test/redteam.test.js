// Red-team suite: realistic ways a seated user can sit badly without being
// alerted ("CHEAT:") or sit well and still be alerted ("FALSE ALARM:").
// Each test asserts the CORRECT behaviour, so a failing test is a weakness.
import { test } from "node:test";
import assert from "node:assert/strict";
import { body, FX, FY, POSES } from "../js/sim.js";
import { extractFrame, buildBaseline, PostureEngine, moreUpright } from "../js/engine.js";
import { W, H, calibrate, calibratedEngine, run, starts } from "./helpers.js";

// ---------------------------------------------------------------------------
// Local sim extension (js/sim.js is left untouched):
//   protract    – m, shoulder joints rolled forward (rounded shoulders); the
//                 neck base and head are not moved by it
//   cameraPitch – deg, camera tilted up (+, laptop lid pushed back) or down (-)
// World landmarks are rotated with the camera, like MediaPipe's camera-aligned
// world frame.

const rad = (d) => (d * Math.PI) / 180;
let seed = 10_000;

function rng(s) {
  s = s >>> 0 || 1;
  return () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
}

function synth(pose = {}) {
  const { p, hip, points } = body(pose);
  const protract = pose.protract ?? 0;
  const cp = rad(pose.cameraPitch ?? 0);
  const rand = rng(p.seed);
  const gauss = () => Math.sqrt(-2 * Math.log(Math.max(rand(), 1e-9))) * Math.cos(2 * Math.PI * rand());
  const jitter = () => (p.noise ? gauss() * p.noise : 0);
  const cam = (q) => ({ x: q.x, y: q.y * Math.cos(cp) + q.z * Math.sin(cp), z: -q.y * Math.sin(cp) + q.z * Math.cos(cp) });
  const pts = { ...points };
  for (const i of [11, 12]) pts[i] = { ...pts[i], z: pts[i].z - protract };
  const hc = cam(hip);
  const landmarks = [];
  const worldLandmarks = [];
  for (let i = 0; i < 33; i++) {
    if (!pts[i]) {
      landmarks.push({ x: 0.5, y: 0.9, z: 0, visibility: 0.05 });
      worldLandmarks.push({ x: 0, y: 0, z: 0, visibility: 0.05 });
      continue;
    }
    const c = cam(pts[i]);
    const visibility = p.visibility[i] ?? 0.98;
    landmarks.push({ x: 0.5 + (FX * c.x) / c.z + jitter(), y: 0.5 + (FY * c.y) / c.z + jitter(), z: 0, visibility });
    worldLandmarks.push({ x: c.x - hc.x + jitter() * 0.5, y: c.y - hc.y + jitter() * 0.5, z: c.z - hc.z + jitter() * 0.5, visibility });
  }
  return { landmarks, worldLandmarks };
}

/** Extracted frame from the extended sim, with realistic jitter. */
function xframe(pose = {}) {
  const { landmarks, worldLandmarks } = synth({ noise: 0.0015, seed: seed++, ...pose });
  return extractFrame(landmarks, worldLandmarks, W, H);
}

function xcalibrate(pose = {}, previous = null) {
  return buildBaseline(Array.from({ length: 30 }, () => xframe(pose)), previous);
}

function engineOn(pose, opts = {}) {
  const { baseline, errors } = xcalibrate(pose);
  assert.deepEqual(errors, [], "calibration itself should succeed");
  const e = new PostureEngine({ graceMs: 5000, ...opts });
  e.setBaseline(baseline);
  return e;
}

/** Run with the extended sim. */
const xrun = (engine, script, ms, o) =>
  run(engine, (t) => {
    const item = typeof script === "function" ? script(t) : script;
    return item?.presence ? item : xframe(item);
  }, ms, o);

const scaled = (k) => ({ torsoFlex: 14 * k, neckFlex: 8 + 22 * k, shrug: -0.01 * k }); // k=1 → POSES.slouch
const LAPTOP = { hipY: 0.4, distance: 0.6 }; // camera on a laptop lid: about shoulder height

// ---------------------------------------------------------------------------
// CHEATS (bad posture, no alert)

test("CHEAT: moderate all-round slouch stays just under every single threshold", () => {
  // 75% of the named slouch: torso 10.5° forward, head 16° further forward.
  // Every metric sits at 0.6–0.9 severity, the score is already below 60
  // ("at the alert threshold"), yet no single metric reaches 1.
  const e = calibratedEngine();
  const { alerts, last } = run(e, scaled(0.75), 60_000);
  assert.ok(last.score < 60, `score ${last.score}`);
  assert.equal(starts(alerts).length, 1, "a clearly visible slouch should alert");
});

test("CHEAT: ratcheting the calibration down in small steps never warns", () => {
  // Recalibrate a little more slouched each time. The app compares against the
  // best (most upright) calibration so far, not just the previous one.
  let best = calibrate().baseline;
  const warned = [];
  for (const k of [0.25, 0.5, 0.75, 1]) {
    const r = calibrate(scaled(k), best);
    assert.deepEqual(r.errors, []);
    warned.push(...r.warnings);
    best = moreUpright(best, r.baseline);
  }
  assert.ok(warned.includes("worseThanBefore"), "drifting the baseline to a full slouch should be flagged");
});

test("CHEAT: recalibrating slouched while sitting closer skips the 'worse than before' check", () => {
  // sameSetup() needs shoulder width within 20%, so scooting 15 cm closer for
  // the calibration disables the comparison. Lean is "relHigher", so sitting
  // back at the usual distance afterwards never trips it either.
  const good = calibrate().baseline;
  const r = calibrate({ ...POSES.slouch, distance: 0.5 }, good);
  assert.ok(r.warnings.includes("worseThanBefore"), `warnings: ${r.warnings}`);
});

test("CHEAT: a first-ever calibration done in a full slouch raises no warning", () => {
  // The absolute sanity checks (neck < 0.26, depth > 0.3) only trip for an
  // extreme posture; a typical slouch / forward head passes silently.
  assert.ok(calibrate(POSES.slouch).warnings.length > 0, "slouch calibration accepted silently");
  assert.ok(calibrate(POSES.forwardHead).warnings.length > 0, "forward-head calibration accepted silently");
});

// KNOWN LIMITATION: with the camera at shoulder height, rounded shoulders move
// toward the camera together with the head, so all 2D head-forward signals
// mostly cancel (score drops to ~70, no alert). Catching it would need trusted
// 3D depth, which MediaPipe doesn't give with hips out of frame.
test("CHEAT: laptop camera – forward head with rounded shoulders goes unnoticed", { todo: true }, () => {
  // Camera at shoulder height: the 'neck' metric is blind (ears move toward the
  // camera, not down in the image), and rolling the shoulders forward cancels
  // both 'depth' (shoulders move toward the camera too) and 'headSize'.
  const e = engineOn(LAPTOP);
  const { alerts } = xrun(e, { ...LAPTOP, neckFlex: 34, protract: 0.04 }, 10_000);
  assert.equal(starts(alerts).length, 1);
});

test("CHEAT: laptop camera misses a forward head that an eye-level camera catches", () => {
  // Same 22° forward head that alerts with an eye-level camera.
  const eye = calibratedEngine();
  assert.equal(starts(run(eye, { neckFlex: 30 }, 10_000).alerts).length, 1, "eye-level camera catches it");
  const e = engineOn(LAPTOP);
  const { alerts } = xrun(e, { ...LAPTOP, neckFlex: 30 }, 10_000);
  assert.equal(starts(alerts).length, 1, "laptop camera misses it");
});

// Accepted, not a cheat: hinging forward from the hips with a neutral spine
// (torso tilted, head stacked over shoulders) from a chair pushed back. With
// hips out of frame it is indistinguishable from sitting closer, and it isn't
// the spinal flexion this app is meant to catch.

test("CHEAT: ducking out of frame for an instant dismisses the active alert", () => {
  // Any single 'away' frame calls endAlert(): banner gone and the bucket emptied.
  // Tracking dropouts do the same, so instead of one alert repeating every 30 s
  // the user gets a fresh 'start' beep every ~5 s.
  const e = calibratedEngine();
  const { alerts } = run(e, (t) => (t > 0 && t % 8000 === 0 ? { presence: "away" } : POSES.slouch), 60_000);
  assert.equal(alerts.filter((a) => a.type === "clear").length, 0, "a 100 ms blip should not end the alert");
  assert.equal(starts(alerts).length, 1, `starts at ${starts(alerts).map((a) => a.t)}`);
});

test("CHEAT: a frozen frame (photo / virtual camera loop) is judged good forever", () => {
  // Bit-identical landmarks for 10 minutes: no living person is that still.
  const e = calibratedEngine();
  const { last } = run(e, { noise: 0, seed: 1 }, 600_000);
  assert.notEqual(last.status, "good", "zero landmark motion for 10 min should be flagged");
});

// ---------------------------------------------------------------------------
// FALSE ALARMS (good posture, alert)

test("FALSE ALARM: sitting upright but with the chair 13 cm closer than at calibration", () => {
  // 'lean' is just shoulder width / frame width, so any closer seating
  // (e.g. after a break) reads as 'Leaning in toward the screen'.
  const e = calibratedEngine();
  assert.equal(run(e, { distance: 0.52 }, 20_000).alerts.length, 0);
});

test("FALSE ALARM: laptop lid tilted back 10° (camera pitched up)", () => {
  // 'sink' is measured from the image centre, so any camera pitch shifts it.
  const e = engineOn({});
  assert.equal(xrun(e, { cameraPitch: 10 }, 20_000).alerts.length, 0);
});

test("FALSE ALARM: sitting 30 cm to one side of the camera", () => {
  // Perspective parallax shifts the nose relative to the ears; the head-yaw
  // correction misreads it as a head turn and inflates ear width → headSize.
  const e = calibratedEngine();
  assert.equal(run(e, { x: 0.3 }, 20_000).alerts.length, 0);
});

test("FALSE ALARM: reading a side monitor, far ear occluded", () => {
  // At ~50° head turn the far ear's visibility drops; min(earVis) < 0.3 makes
  // the whole frame 'Can't see your head', which alerts after 5 s.
  const e = calibratedEngine();
  assert.equal(run(e, { headYaw: 50, visibility: { 8: 0.25 } }, 20_000).alerts.length, 0);
});

// ---------------------------------------------------------------------------
// Regression guards (currently pass)

test("guard: slowly drifting into a slouch over 10 minutes is caught", () => {
  // The EMA (tau 350 ms) must not act as a baseline adapter.
  const e = calibratedEngine();
  const { alerts } = run(e, (t) => scaled(Math.min(1, t / 600_000)), 660_000);
  assert.equal(starts(alerts).length, 1);
  assert.ok(starts(alerts)[0].t < 600_000, `alert only at ${starts(alerts)[0].t}`);
});

test("guard: forward head while looking at a second monitor still alerts", () => {
  const e = calibratedEngine();
  assert.equal(starts(run(e, { neckFlex: 34, headYaw: 40 }, 8000).alerts).length, 1);
});

test("guard: reclining 20° against the backrest is fine", () => {
  const e = calibratedEngine();
  assert.equal(run(e, { torsoFlex: -20 }, 20_000).alerts.length, 0);
});
