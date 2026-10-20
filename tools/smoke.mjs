// End-to-end smoke test in real Chrome (headless) with a fake webcam.
// - real mode: camera + MediaPipe model load, background-tab processing
// - demo mode (?demo): synthetic poses drive the full alert flow
// Screenshots land in tools/out/.
//
//   node tools/smoke.mjs            run everything
//   node tools/smoke.mjs --headed   watch it
import { chromium } from "playwright-core";
import { createServer } from "node:http";
import { readFile, mkdir } from "node:fs/promises";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const OUT = join(ROOT, "tools", "out");
const PORT = 8123;
const BASE = `http://localhost:${PORT}`;
const headed = process.argv.includes("--headed");

const TYPES = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css", ".json": "application/json" };
const server = createServer(async (req, res) => {
  const path = normalize(decodeURIComponent(new URL(req.url, BASE).pathname)).replace(/^([/\\])+/, "");
  try {
    const file = path || "index.html";
    const body = await readFile(join(ROOT, file));
    res.writeHead(200, { "content-type": TYPES[extname(file)] ?? "application/octet-stream" });
    res.end(body);
  } catch {
    res.writeHead(404).end();
  }
});
await new Promise((r) => server.listen(PORT, "127.0.0.1", r));
await mkdir(OUT, { recursive: true });

const browser = await chromium.launch({
  channel: "chrome",
  headless: !headed,
  args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", "--autoplay-policy=no-user-gesture-required"],
});

const results = [];
const check = (name, ok, detail = "") => {
  results.push({ name, ok: !!ok, detail });
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? `  (${detail})` : ""}`);
};

async function newPage(url, { viewport = { width: 1400, height: 950 }, colorScheme = "light", init } = {}) {
  const context = await browser.newContext({ viewport, colorScheme, permissions: ["camera", "notifications"] });
  if (init) await context.addInitScript(init);
  const page = await context.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
  await page.goto(BASE + url);
  return { page, context, errors };
}

const state = (page) =>
  page.evaluate(() => {
    const p = window.__posture;
    return {
      phase: p.app.phase,
      status: document.getElementById("statusText").textContent,
      banner: document.getElementById("banner").hidden ? null : document.getElementById("bannerTitle").textContent,
      coach: document.getElementById("coach").hidden ? null : document.getElementById("coachIssue").textContent,
      stats: p.engine.stats,
      visibility: document.visibilityState,
      title: document.title,
    };
  });
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

try {
  // ---------------------------------------------------------------- demo flow
  {
    const { page, context, errors } = await newPage("/?demo=upright");
    await page.waitForFunction(() => window.__posture?.app.phase === "monitor", null, { timeout: 15000 });
    await wait(2500);
    let s = await state(page);
    check("demo: upright is good", s.status === "Good posture" && !s.banner, JSON.stringify(s.status));
    await page.screenshot({ path: join(OUT, "1-good.png") });

    await page.evaluate(() => window.__posture.setPose("slouch"));
    await wait(2500);
    s = await state(page);
    check("demo: slouch shows a warning before the alert", s.status === "Check your posture" && s.coach && !s.banner, `${s.status} / ${s.coach}`);
    await page.screenshot({ path: join(OUT, "2-warning.png") });

    await wait(4500);
    s = await state(page);
    check("demo: slouch alerts after the grace period", s.banner && s.status === "Fix your posture", s.banner);
    check("demo: tab title changes on alert", s.title.includes("Sit up"), s.title);
    await page.screenshot({ path: join(OUT, "3-alert.png") });

    await page.evaluate(() => window.__posture.setPose("upright"));
    await wait(4000);
    s = await state(page);
    check("demo: alert clears after sitting up", !s.banner && s.status === "Good posture", s.status);

    await page.evaluate(() => window.__posture.setPose("hidden"));
    await wait(7500);
    s = await state(page);
    check("demo: hiding shoulders alerts", s.banner?.includes("shoulders"), s.banner);

    await page.evaluate(() => window.__posture.setPose("away"));
    await wait(4500); // away is debounced for 3 s
    s = await state(page);
    check("demo: leaving clears alert and shows Away", !s.banner && s.status === "Away", s.status);

    // Snooze
    await page.evaluate(() => window.__posture.setPose("slouch"));
    await page.click("#snoozeBtn");
    await page.click("#snoozeMenu button[data-min='5']");
    await wait(7000);
    s = await state(page);
    check("demo: snooze suppresses alerts", !s.banner && s.status.startsWith("Snoozed"), s.status);
    await page.click("#snoozeBtn");
    await wait(300);
    s = await state(page);
    check("demo: resume from snooze", !s.status.startsWith("Snoozed"), s.status);

    check("demo: no console errors", errors.length === 0, errors.join(" | "));
    await context.close();
  }

  // ---------------------------------------------------------- demo onboarding
  {
    const { page, context, errors } = await newPage("/?demo=upright&onboard");
    await page.waitForFunction(() => window.__posture?.app.phase === "frame", null, { timeout: 15000 });
    await page.waitForSelector("#toCalibrateBtn:not([disabled])", { timeout: 5000 });
    await page.screenshot({ path: join(OUT, "4-onboard-frame.png") });
    await page.click("#toCalibrateBtn");
    await page.screenshot({ path: join(OUT, "5-onboard-prep.png") });
    await page.click("#calibrateGoBtn");
    await wait(1500);
    await page.screenshot({ path: join(OUT, "6-onboard-capture.png") });
    await page.waitForFunction(() => window.__posture.app.phase === "monitor", null, { timeout: 6000 });
    check("onboarding: calibrates and starts monitoring", true);

    // Cheat: recalibrate while jutting the head forward.
    await page.evaluate(() => window.__posture.setPose("forwardHead"));
    await page.click("#recalibrateBtn");
    await page.click("#calibrateGoBtn");
    await page.waitForFunction(() => window.__posture.app.phase !== "capture", null, { timeout: 6000 });
    const s = await state(page);
    const result = await page.locator("#resultList").innerText().catch(() => "");
    check("onboarding: slouched recalibration is challenged", s.phase === "result" && /slouched|forward|low/i.test(result), result.replace(/\n/g, " | "));
    await page.screenshot({ path: join(OUT, "7-recalibrate-warning.png") });
    check("onboarding: no console errors", errors.length === 0, errors.join(" | "));
    await context.close();
  }

  // ------------------------------------------------------ dark + mobile shots
  {
    const { page, context } = await newPage("/?demo=textNeck", { colorScheme: "dark" });
    await page.waitForFunction(() => window.__posture?.app.phase === "monitor", null, { timeout: 15000 });
    await wait(8000);
    await page.screenshot({ path: join(OUT, "8-dark-alert.png") });
    await context.close();

    const m = await newPage("/?demo", { viewport: { width: 390, height: 844 } });
    await m.page.waitForFunction(() => window.__posture?.app.phase === "monitor", null, { timeout: 15000 });
    await wait(2000);
    await m.page.screenshot({ path: join(OUT, "9-mobile.png"), fullPage: true });
    const overflow = await m.page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth);
    check("mobile: no horizontal scroll", !overflow);
    await m.context.close();
  }

  // ------------------------------------------------- real camera + pose model
  // (Background-tab behaviour is covered by tools/background-check.mjs.)
  {
    const { page, context, errors } = await newPage("/");
    await page.screenshot({ path: join(OUT, "0-welcome.png") });
    await page.click("#startBtn");
    await page.waitForFunction(() => window.__posture.app.phase === "frame", null, { timeout: 60000 });
    check("real: camera starts and pose model loads", true);
    await wait(2500);
    const s = await state(page);
    check("real: empty fake-camera frame is not mistaken for a person", s.phase === "frame" && (await page.locator("#toCalibrateBtn").isDisabled()));
    await page.screenshot({ path: join(OUT, "10-real-frame-step.png") });
    check("real: no console errors", errors.filter((e) => !/WebGL|GL_|gpu/i.test(e)).length === 0, errors.join(" | "));
    await context.close();
  }
} catch (err) {
  check("smoke run completed", false, err.message);
} finally {
  await browser.close();
  server.close();
}

const failed = results.filter((r) => !r.ok);
console.log(`\n${results.length - failed.length}/${results.length} checks passed. Screenshots in tools/out/`);
process.exit(failed.length ? 1 : 0);
