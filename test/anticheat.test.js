// Ways a user might try to silence the checker without actually sitting well.
import { test } from "node:test";
import assert from "node:assert/strict";
import { buildBaseline } from "../js/engine.js";
import { POSES, calibrate, calibratedEngine, frame, run, starts } from "./helpers.js";

test("flicker: sitting up for 1 s every few seconds doesn't reset the timer", () => {
  const e = calibratedEngine();
  const { alerts } = run(e, (t) => (t % 5000 < 4000 ? POSES.slouch : POSES.upright), 30_000);
  assert.ok(starts(alerts).length >= 1);
  assert.ok(starts(alerts)[0].t < 12_000, `alert only at ${starts(alerts)[0].t}`);
});

test("hiding shoulders out of frame still alerts", () => {
  const e = calibratedEngine();
  const { alerts, last } = run(e, { visibility: { 11: 0.1, 12: 0.1 } }, 8000);
  assert.equal(starts(alerts).length, 1);
  assert.equal(last.issues[0].key, "shoulders");
});

test("hiding the head still alerts", () => {
  const e = calibratedEngine();
  const { alerts } = run(e, { visibility: { 0: 0.1, 7: 0.1, 8: 0.1 } }, 8000);
  assert.equal(starts(alerts).length, 1);
});

test("covering the camera still alerts", () => {
  const e = calibratedEngine();
  const { alerts } = run(e, { presence: "partial", reason: "blocked" }, 8000);
  assert.equal(starts(alerts).length, 1);
});

test("turning the torso away still alerts", () => {
  const e = calibratedEngine();
  const { alerts, last } = run(e, { yaw: 45 }, 8000);
  assert.equal(starts(alerts).length, 1);
  assert.equal(last.issues[0].key, "turned");
});

test("slight torso turn doesn't hide a slouch", () => {
  const e = calibratedEngine();
  const { alerts } = run(e, { ...POSES.slouch, yaw: 22 }, 8000);
  assert.equal(starts(alerts).length, 1);
});

test("slight torso turn while upright is fine", () => {
  const e = calibratedEngine();
  assert.equal(run(e, { yaw: 22 }, 20_000).alerts.length, 0);
});

test("lifting the chin while slouching doesn't hide it", () => {
  const e = calibratedEngine();
  const { alerts } = run(e, { ...POSES.slouch, pitch: -20 }, 8000);
  assert.equal(starts(alerts).length, 1);
});

test("pushing shoulders down while jutting the head forward doesn't hide it", () => {
  const e = calibratedEngine();
  const { alerts } = run(e, { neckFlex: 32, shrug: -0.03 }, 8000);
  assert.equal(starts(alerts).length, 1);
});

test("leaning back far while craning the neck doesn't hide it", () => {
  const e = calibratedEngine();
  const { alerts } = run(e, { torsoFlex: -15, neckFlex: 36, distance: 0.8 }, 8000);
  assert.equal(starts(alerts).length, 1);
});

test("stepping away ends an active alert (legit), returning slouched re-alerts", () => {
  const e = calibratedEngine();
  const a = run(e, POSES.slouch, 7000);
  const b = run(e, { presence: "away" }, 3000, { start: a.end + 100 });
  assert.equal(b.alerts.at(-1)?.type, "clear");
  const c = run(e, POSES.slouch, 8000, { start: b.end + 100 });
  assert.equal(starts(c.alerts).length, 1);
});

test("recalibrating while slouched is flagged against the previous calibration", () => {
  const good = calibrate();
  const { warnings } = calibrate(POSES.forwardHead, good.baseline);
  assert.ok(warnings.includes("worseThanBefore"), `warnings: ${warnings}`);
});

test("calibrating with a tilted head or uneven shoulders is rejected", () => {
  assert.ok(calibrate({ headRoll: 15 }).errors.includes("headTilt"));
  assert.ok(calibrate({ shoulderRoll: 10 }).errors.includes("shoulderTilt"));
});

test("calibrating while looking down is challenged (a warning: a high camera looks the same)", () => {
  assert.ok(calibrate({ pitch: 35 }).warnings.includes("lookingDown"));
});

test("moving during calibration is rejected", () => {
  const frames = Array.from({ length: 30 }, (_, i) => frame(i % 2 ? POSES.upright : POSES.slouch));
  assert.ok(buildBaseline(frames).errors.includes("moved"));
});

test("calibrating out of frame is rejected", () => {
  const frames = Array.from({ length: 30 }, () => frame({ visibility: { 11: 0.1, 12: 0.1 } }));
  assert.ok(buildBaseline(frames).errors.includes("notVisible"));
});
