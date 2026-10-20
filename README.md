# Posture Checker

A private, browser-based posture coach. It uses your webcam to learn your best seated posture, then gives you a gentle alert when you drift away from it.

**Live app:** https://adolfo-enterria.github.io/posture-checker/

## What it does

- Runs MediaPipe pose detection locally in the browser; webcam frames are not uploaded or recorded.
- Calibrates to the user and camera instead of assuming one posture fits everyone.
- Detects slouching, forward-head posture, leaning toward the screen, sinking, head tilt, and uneven shoulders.
- Waits through a configurable grace period to avoid nagging on brief movements.
- Supports sound, desktop notifications, break reminders, snoozing, dark mode, and a responsive mobile layout.
- Stores only settings, calibration measurements, and daily summary totals in browser storage.
- Includes adversarial tests for false alarms and common ways posture detection can be fooled.

No account or backend is required.

## Use it

Open the [live app](https://adolfo-enterria.github.io/posture-checker/) in a current desktop version of Chrome or Edge, then:

1. Select **Start camera** and allow camera access.
2. Frame your head and shoulders.
3. Sit in your best posture and complete the three-second calibration.
4. Leave the tab open while you work.

The app is a general wellness aid, not a medical device. It cannot diagnose or treat a condition. Persistent pain or neurological symptoms should be discussed with a qualified clinician.

## Privacy

Pose estimation happens in the browser. Video and images are never sent to this project or stored. The browser downloads the MediaPipe runtime and pose model from their public CDNs; after that, the model evaluates camera frames on-device. There is no analytics, telemetry, account system, or application server.

To remove saved calibration and history, clear site data for the app in your browser.

## Browser support and limitations

- Best supported: current Chrome, Edge, and other Chromium browsers on desktop.
- Camera access requires HTTPS or localhost. GitHub Pages supplies HTTPS.
- Desktop notifications depend on browser and operating-system permissions.
- Laptop cameras placed well below eye level can make some kinds of forward-head posture harder to distinguish.
- Browser power-saving policies may pause camera frames when a tab is heavily throttled. Keeping the app in its own background window is the most reliable setup.

## Local development

Requirements: Node.js 20+ and Python 3 (for the convenience server).

```bash
npm install
npm test
npm start
```

Then open http://127.0.0.1:8000. A synthetic no-camera demo is available at http://127.0.0.1:8000/?demo.

Useful checks:

```bash
npm run smoke:demo  # end-to-end UI flow without a webcam
npm run smoke       # also loads the real MediaPipe model and fake webcam
npm run bgcheck     # background-tab processing check; run npm start first
```

The browser checks use the locally installed Google Chrome.

## Architecture

- `js/engine.js` contains pose feature extraction, calibration, scoring, debouncing, and alert decisions.
- `js/app.js` owns the webcam/MediaPipe pipeline, interface state, settings, notifications, and local history.
- `js/sim.js` synthesizes deterministic postures for tests and demo mode.
- `test/` contains engine, anti-cheat, regression, and false-positive coverage.
- `.github/workflows/pages.yml` runs the test suite and deploys this static app to GitHub Pages.

## License

[MIT](LICENSE)
