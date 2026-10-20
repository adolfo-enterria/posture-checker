import { test } from "node:test";
import assert from "node:assert/strict";
import { PostureEngine, issueMessages } from "../js/engine.js";
import { POSES, calibrate, calibratedEngine, run, starts } from "./helpers.js";

test("upright posture never alerts over 3 minutes of noisy frames", () => {
  const e = calibratedEngine();
  const { alerts, last } = run(e, (t) => ({ noise: 0.003, headYaw: 10 * Math.sin(t / 7000) }), 180_000);
  assert.equal(alerts.length, 0);
  assert.equal(last.status, "good");
  assert.ok(last.score >= 90, `score ${last.score}`);
});

for (const name of ["slouch", "forwardHead", "leanIn", "sinking", "textNeck", "tiltedHead", "unevenShoulders"]) {
  test(`${name} alerts after the grace period, not before`, () => {
    const e = calibratedEngine();
    const { alerts } = run(e, POSES[name], 8000);
    const s = starts(alerts);
    assert.equal(s.length, 1, `expected an alert for ${name}`);
    assert.ok(s[0].t >= 5000 && s[0].t <= 6500, `alert at ${s[0].t}ms`);
  });
}

test("alert repeats while still slouching, then clears after sustained good posture", () => {
  const e = calibratedEngine({ repeatMs: 30000 });
  const bad = run(e, POSES.slouch, 40_000);
  assert.deepEqual(bad.alerts.map((a) => a.type), ["start", "repeat"]);
  const good = run(e, POSES.upright, 4000, { start: bad.end + 100 });
  assert.deepEqual(good.alerts.map((a) => a.type), ["clear"]);
  assert.equal(good.last.status, "good");
});

test("a quick reach or stretch does not alert", () => {
  const e = calibratedEngine();
  const { alerts } = run(e, (t) => (t % 60_000 < 3000 ? POSES.leanIn : POSES.upright), 180_000);
  assert.equal(alerts.length, 0);
});

test("distance from the camera doesn't change the verdict", () => {
  const e = calibratedEngine();
  assert.equal(run(e, { distance: 0.8 }, 20_000).alerts.length, 0, "upright but further back");
  e.reset();
  assert.equal(starts(run(e, { ...POSES.slouch, distance: 0.8, hipY: 0.68 }, 8000).alerts).length, 1, "slouching further back");
});

test("looking at a second monitor (head turned) is fine", () => {
  const e = calibratedEngine();
  assert.equal(run(e, { headYaw: 35 }, 20_000).alerts.length, 0);
});

test("sensitivity changes how much deviation is tolerated", () => {
  const mild = { neckFlex: 26 };
  const high = calibratedEngine({ sensitivity: "high" });
  const low = calibratedEngine({ sensitivity: "low" });
  assert.equal(starts(run(high, mild, 8000).alerts).length, 1);
  assert.equal(run(low, mild, 8000).alerts.length, 0);
});

test("stats track good, bad and away time", () => {
  const e = calibratedEngine();
  run(e, POSES.upright, 10_000);
  run(e, POSES.slouch, 10_000, { start: 10_100 });
  const { last } = run(e, { presence: "away" }, 8000, { start: 20_200 }); // first 3 s are debounced
  const s = last.stats;
  assert.ok(Math.abs(s.goodMs - 10_000) < 600, `good ${s.goodMs}`);
  assert.ok(Math.abs(s.badMs - 10_000) < 600, `bad ${s.badMs}`);
  assert.ok(Math.abs(s.awayMs - 5000) < 300, `away ${s.awayMs}`);
  assert.equal(s.alerts, 1);
});

test("issue messages are deduplicated by group", () => {
  const msgs = issueMessages([{ key: "headSize" }, { key: "depth" }, { key: "lean" }]);
  assert.equal(msgs.length, 2);
  assert.match(msgs[0].issue, /forward/);
});

test("uncalibrated engine reports uncalibrated and never alerts", () => {
  const e = new PostureEngine();
  const { alerts, last } = run(e, POSES.slouch, 10_000);
  assert.equal(alerts.length, 0);
  assert.equal(last.status, "uncalibrated");
});

test("calibration on a clean upright posture succeeds", () => {
  const { baseline, errors, warnings } = calibrate();
  assert.deepEqual(errors, []);
  assert.deepEqual(warnings, []);
  assert.ok(baseline.neck > 0.3);
});
