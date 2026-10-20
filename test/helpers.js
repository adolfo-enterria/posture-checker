import { synthesize, POSES } from "../js/sim.js";
import { extractFrame, buildBaseline, PostureEngine } from "../js/engine.js";

export const W = 640;
export const H = 480;
export { POSES };

let seed = 1;

/** One extracted frame for a pose, with realistic landmark jitter by default. */
export function frame(pose = {}) {
  const { landmarks, worldLandmarks } = synthesize({ noise: 0.0015, seed: seed++, ...pose });
  return extractFrame(landmarks, worldLandmarks, W, H);
}

/** Calibrate on `pose` for 3 s at 10 fps. */
export function calibrate(pose = {}, previous = null) {
  const frames = Array.from({ length: 30 }, () => frame(pose));
  return buildBaseline(frames, previous);
}

/** Engine calibrated on an upright posture. */
export function calibratedEngine(opts = {}) {
  const engine = new PostureEngine({ graceMs: 5000, ...opts });
  const { baseline } = calibrate();
  engine.setBaseline(baseline);
  return engine;
}

/**
 * Drive the engine for `ms` milliseconds at 10 fps.
 * `script(t)` returns a pose, or a ready-made frame object ({presence: ...}).
 */
export function run(engine, script, ms, { start = 0, step = 100 } = {}) {
  const alerts = [];
  const snaps = [];
  for (let t = start; t <= start + ms; t += step) {
    const item = typeof script === "function" ? script(t - start) : script;
    const f = item && item.presence ? item : frame(item);
    const snap = engine.update(f, t);
    snaps.push(snap);
    if (snap.alert) alerts.push({ t: t - start, ...snap.alert });
  }
  return { alerts, snaps, last: snaps[snaps.length - 1], end: start + ms };
}

export const starts = (alerts) => alerts.filter((a) => a.type === "start");
