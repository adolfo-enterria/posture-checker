// Verifies posture checks keep running when the tab is in the background.
// Uses plain Chrome over raw CDP (Playwright fakes page focus/visibility).
// Requires the app served at http://127.0.0.1:8000 (npm start).
import { spawn } from "node:child_process";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

const CHROME = process.env.CHROME ?? "C:/Program Files/Google/Chrome/Application/chrome.exe";
const PORT = 9333;
const URL_ = process.argv[2] ?? "http://127.0.0.1:8000/";
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

const dir = await mkdtemp(join(tmpdir(), "posture-bg-"));
const chrome = spawn(CHROME, [
  `--remote-debugging-port=${PORT}`, `--user-data-dir=${dir}`, "--no-first-run", "--no-default-browser-check",
  "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
  "--window-size=800,600", "about:blank",
], { stdio: "ignore" });

let last;
async function json(path, method = "GET") {
  for (let i = 0; i < 150; i++) {
    try { const t = await (await fetch(`http://127.0.0.1:${PORT}${path}`, { method, signal: AbortSignal.timeout(2000) })).text(); try { return JSON.parse(t); } catch { return t; } } catch (e) { last = e; await wait(200); }
  }
  throw new Error(`Chrome did not start: ${last?.message}`);
}

function connect(wsUrl) {
  const ws = new WebSocket(wsUrl);
  let id = 0;
  const pending = new Map();
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.id && pending.has(msg.id)) { pending.get(msg.id)(msg); pending.delete(msg.id); }
  };
  return new Promise((resolve) => ws.onopen = () => resolve({
    send: (method, params = {}) => new Promise((r) => { pending.set(++id, r); ws.send(JSON.stringify({ id, method, params })); }),
    eval: async function (expr) {
      const r = await this.send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true });
      if (r.result?.exceptionDetails) throw new Error(JSON.stringify(r.result.exceptionDetails));
      return r.result?.result?.value;
    },
    close: () => ws.close(),
  }));
}

let ok = false;
try {
  await json("/json/version");
  const tab = await json(`/json/new?${encodeURIComponent(URL_)}`, "PUT");
  const page = await connect(tab.webSocketDebuggerUrl);
  await wait(1500);
  await page.eval(`document.getElementById("startBtn").click()`);
  for (let i = 0; i < 120 && (await page.eval("window.__posture.app.phase")) !== "frame"; i++) await wait(500);

  // Skip onboarding: synthetic baseline, then monitor real camera frames.
  await page.eval(`(async () => {
    const p = window.__posture;
    const sim = await import("./js/sim.js");
    const eng = await import("./js/engine.js");
    const frames = Array.from({ length: 30 }, (_, i) => {
      const r = sim.synthesize({ seed: i + 1, noise: 0.001 });
      return eng.extractFrame(r.landmarks, r.worldLandmarks, 640, 480);
    });
    p.app.baseline = eng.buildBaseline(frames).baseline;
    p.engine.setBaseline(p.app.baseline);
    p.app.phase = "monitor";
  })()`);
  await wait(1500);
  const total = `(() => { const s = window.__posture.engine.stats; return s.awayMs + s.goodMs + s.badMs; })()`;

  const before = await page.eval(total);
  // Background the tab: open another tab over it and minimize the window.
  const other = await json(`/json/new?about:blank`, "PUT");
  await json(`/json/activate/${other.id}`);
  const { result } = await page.send("Browser.getWindowForTarget");
  await page.send("Browser.setWindowBounds", { windowId: result.windowId, bounds: { windowState: "minimized" } });
  await wait(15000);
  const vis = await page.eval("document.visibilityState");
  const processed = (await page.eval(total)) - before;
  ok = vis === "hidden" && processed > 12000;
  console.log(`${ok ? "PASS" : "FAIL"}  background processing: visibility=${vis}, processed ${Math.round(processed)}ms of 15000ms`);
  page.close();
} catch (err) {
  console.log("FAIL ", err.message);
} finally {
  chrome.kill();
  await wait(500);
  await rm(dir, { recursive: true, force: true }).catch(() => {});
}
process.exit(ok ? 0 : 1);
