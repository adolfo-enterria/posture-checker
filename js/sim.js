// Synthetic pose generator: a simple 3D upper-body model projected through a
// pinhole webcam, emitting MediaPipe-shaped landmarks. Used by the tests and
// by the in-browser demo mode (?demo), so posture logic can be exercised
// without a real person in front of the camera.

export const FX = 0.6; // focal length as a fraction of image width
export const FY = 0.8; // same focal length expressed in image-height units (4:3)

export const UPRIGHT = Object.freeze({
  distance: 0.65,      // m, camera to hips
  x: 0,                // m, lateral offset of the body
  hipY: 0.68,          // m, hips below the camera
  torsoLen: 0.5,       // m, hips to shoulder line
  torsoFlex: 0,        // deg, torso leaning toward the camera (+) or reclining (-)
  neckLen: 0.15,       // m, shoulder line to ear line
  neckFlex: 8,         // deg, head carried forward of vertical (world frame)
  pitch: 0,            // deg, head nod: + chin down, - chin up
  headRoll: 0,         // deg, ear-to-shoulder tilt
  headYaw: 0,          // deg, head turned left/right
  shoulderRoll: 0,     // deg, one shoulder higher than the other
  shrug: 0,            // m, shoulders raised (+) or pushed down (-)
  yaw: 0,              // deg, whole torso rotated away from the camera
  shoulderWidth: 0.38, // m
  headWidth: 0.15,     // m, ear to ear
  visibility: {},      // { index: visibility } overrides
  noise: 0,            // std-dev of image jitter, normalized units
  seed: 1,
});

const IDX = { NOSE: 0, L_EAR: 7, R_EAR: 8, L_SHOULDER: 11, R_SHOULDER: 12 };
const rad = (d) => (d * Math.PI) / 180;

function rng(seed) {
  let s = seed >>> 0 || 1;
  return () => {
    s ^= s << 13; s >>>= 0;
    s ^= s >>> 17;
    s ^= s << 5; s >>>= 0;
    return s / 4294967296;
  };
}

function gauss(rand) {
  const u = Math.max(rand(), 1e-9);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}

const add = (a, b) => ({ x: a.x + b.x, y: a.y + b.y, z: a.z + b.z });
const scale = (a, k) => ({ x: a.x * k, y: a.y * k, z: a.z * k });

/** Build a 3D body (camera coordinates: x right, y down, z away from camera). */
export function body(pose = {}) {
  const p = { ...UPRIGHT, ...pose };
  const hip = { x: p.x, y: p.hipY, z: p.distance };

  const tf = rad(p.torsoFlex);
  const up = { x: 0, y: -Math.cos(tf), z: -Math.sin(tf) };
  const shoulderMid = add(add(hip, scale(up, p.torsoLen)), { x: 0, y: -p.shrug, z: 0 });

  // The person's left side appears on the image's right when facing the camera.
  const yaw = rad(p.yaw);
  const sr = rad(p.shoulderRoll);
  const halfW = p.shoulderWidth / 2;
  const leftAxis = { x: Math.cos(yaw) * Math.cos(sr), y: -Math.sin(sr), z: Math.sin(yaw) };
  const ls = add(shoulderMid, scale(leftAxis, halfW));
  const rs = add(shoulderMid, scale(leftAxis, -halfW));

  const nf = rad(p.neckFlex);
  const earMid = add(shoulderMid, scale({ x: 0, y: -Math.cos(nf), z: -Math.sin(nf) }, p.neckLen));

  const hyaw = rad(p.yaw + p.headYaw);
  const hr = rad(p.headRoll);
  const earAxis = { x: Math.cos(hyaw) * Math.cos(hr), y: -Math.sin(hr), z: Math.sin(hyaw) };
  const le = add(earMid, scale(earAxis, p.headWidth / 2));
  const re = add(earMid, scale(earAxis, -p.headWidth / 2));

  // Nose: ~10 cm in front of the ear line and ~3 cm below it, rotated by pitch/yaw.
  const pt = rad(p.pitch);
  const fwd = { x: Math.sin(hyaw) * Math.cos(pt), y: Math.sin(pt), z: -Math.cos(hyaw) * Math.cos(pt) };
  const down = { x: 0, y: Math.cos(pt), z: Math.sin(pt) };
  const nose = add(add(earMid, scale(fwd, 0.1)), scale(down, 0.03));

  return { p, hip, points: { [IDX.NOSE]: nose, [IDX.L_EAR]: le, [IDX.R_EAR]: re, [IDX.L_SHOULDER]: ls, [IDX.R_SHOULDER]: rs } };
}

/** Project a pose to { landmarks, worldLandmarks } like PoseLandmarker's result[0]. */
export function synthesize(pose = {}) {
  const { p, hip, points } = body(pose);
  const rand = rng(p.seed);
  const landmarks = [];
  const worldLandmarks = [];

  for (let i = 0; i < 33; i++) {
    const pt = points[i];
    if (!pt) {
      landmarks.push({ x: 0.5, y: 0.9, z: 0, visibility: 0.05 });
      worldLandmarks.push({ x: 0, y: 0, z: 0, visibility: 0.05 });
      continue;
    }
    const jitter = () => (p.noise ? gauss(rand) * p.noise : 0);
    const vis = p.visibility[i] ?? 0.98;
    landmarks.push({
      x: 0.5 + (FX * pt.x) / pt.z + jitter(),
      y: 0.5 + (FY * pt.y) / pt.z + jitter(),
      z: ((pt.z - hip.z) / p.distance) * FX,
      visibility: vis,
    });
    worldLandmarks.push({
      x: pt.x - hip.x + jitter() * 0.5,
      y: pt.y - hip.y + jitter() * 0.5,
      z: pt.z - hip.z + jitter() * 0.5,
      visibility: vis,
    });
  }
  return { landmarks, worldLandmarks };
}

/** Named postures used by tests and the demo. */
export const POSES = {
  upright: {},
  slouch: { torsoFlex: 14, neckFlex: 30, shrug: -0.01 },
  forwardHead: { neckFlex: 34 },
  leanIn: { distance: 0.45, torsoFlex: 12, neckFlex: 18 },
  sinking: { hipY: 0.78, torsoFlex: -10, neckFlex: 22 },
  textNeck: { pitch: 28, neckFlex: 22 },
  tiltedHead: { headRoll: 16 },
  unevenShoulders: { shoulderRoll: 9, headRoll: 3 },
};
