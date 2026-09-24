// PRISM home page — interactive 3D glass prism over the title.
// Drag to rotate; it springs back to the dispersion pose where the
// incoming white beam splits into a (subtle) spectrum.

import * as THREE from "three";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

const canvas = document.getElementById("prism-canvas");
const title = document.querySelector(".hero-title");
const hero = document.querySelector(".hero");
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// Statement text lights up when scrolled into view (independent of WebGL).
const statement = document.getElementById("statement");
if (statement && "IntersectionObserver" in window) {
  new IntersectionObserver(([e]) => statement.classList.toggle("lit", e.isIntersecting), { threshold: 0.6 })
    .observe(statement);
}

let renderer;
try {
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
} catch (err) {
  canvas.remove(); // no WebGL: the typographic hero still stands on its own
  throw err;
}
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.1;
renderer.setClearColor(0x000000, 0);

const scene = new THREE.Scene();
const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(renderer), 0.04).texture;

const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 100);
camera.position.set(0, 0, 14);

// ------------------------------------------------------------ prism
const SIDE = 2;                     // triangle side in local units (scaled later)
const H = (Math.sqrt(3) / 2) * SIDE;
const DEPTH = 1.35;

const tri = new THREE.Shape();
tri.moveTo(-SIDE / 2, -H / 3);
tri.lineTo(SIDE / 2, -H / 3);
tri.lineTo(0, (2 * H) / 3);
tri.closePath();

const prismGeo = new THREE.ExtrudeGeometry(tri, {
  depth: DEPTH, bevelEnabled: true, bevelThickness: 0.03, bevelSize: 0.03, bevelSegments: 3,
});
prismGeo.translate(0, 0, -DEPTH / 2);
prismGeo.computeVertexNormals();

const glass = new THREE.MeshPhysicalMaterial({
  color: 0xffffff,
  metalness: 0,
  roughness: 0.06,
  transparent: true,
  opacity: 0.1,
  clearcoat: 1,
  clearcoatRoughness: 0.05,
  iridescence: 0.55,
  iridescenceIOR: 1.35,
  envMapIntensity: 1.6,
  side: THREE.DoubleSide,
  depthWrite: false,
});

const prism = new THREE.Group();
const body = new THREE.Mesh(prismGeo, glass);
prism.add(body);

// crisp edges, like cut glass catching light
const edgeGeo = new THREE.EdgesGeometry(new THREE.ExtrudeGeometry(tri, { depth: DEPTH, bevelEnabled: false }), 20);
edgeGeo.translate(0, 0, -DEPTH / 2);
const edges = new THREE.LineSegments(edgeGeo, new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.75 }));
prism.add(edges);

// faint inner glow core
const core = new THREE.Mesh(
  new THREE.ExtrudeGeometry(tri, { depth: DEPTH * 0.98, bevelEnabled: false }).translate(0, 0, -DEPTH * 0.49).scale(0.55, 0.55, 1),
  new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.02, depthWrite: false, blending: THREE.AdditiveBlending })
);
prism.add(core);

const stage = new THREE.Group();  // positioned/scaled to sit over the title
stage.add(prism);
scene.add(stage);

// ------------------------------------------------------------ light path
// Beam enters the left face, travels through, and leaves the right face as a fan.
const entry = new THREE.Vector3(-SIDE / 4 + 0.02, H / 6 - 0.05, 0);
const exit = new THREE.Vector3(SIDE / 4 - 0.02, H / 6 - 0.12, 0);

function ribbon(points, width, color, opacity) {
  // flat quad strip between two points, always facing the camera plane
  const [a, b] = points;
  const dir = new THREE.Vector3().subVectors(b, a).normalize();
  const n = new THREE.Vector3(-dir.y, dir.x, 0).multiplyScalar(width / 2);
  const g = new THREE.BufferGeometry().setFromPoints([
    a.clone().add(n), a.clone().sub(n), b.clone().add(n), b.clone().sub(n),
  ]);
  g.setIndex([0, 1, 2, 2, 1, 3]);
  return new THREE.Mesh(g, new THREE.MeshBasicMaterial({
    color, transparent: true, opacity, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
  }));
}

const light = new THREE.Group();
const beamIn = ribbon([new THREE.Vector3(-9, -1.05, 0), entry], 0.035, 0xffffff, 0.9);
const beamInside = ribbon([entry, exit], 0.03, 0xffffff, 0.2);
light.add(beamIn, beamInside);

// spectrum fan: a set of thin triangles from the exit point
const spectrumColors = [0xff3b30, 0xff8a2a, 0xffd60a, 0x34c759, 0x32ade6, 0x5856d6, 0xaf52de];
const fan = new THREE.Group();
spectrumColors.forEach((c, i) => {
  const t0 = i / spectrumColors.length, t1 = (i + 1) / spectrumColors.length;
  const reach = 9;
  const y0 = exit.y - 0.35 - t0 * 1.9, y1 = exit.y - 0.35 - t1 * 1.9;
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute([
    exit.x, exit.y, 0, exit.x + reach, y0, 0, exit.x + reach, y1, 0,
  ], 3));
  // alpha fade along the ray via vertex colors (bright near the prism)
  g.setAttribute("color", new THREE.Float32BufferAttribute([1, 1, 1, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15], 3));
  const m = new THREE.MeshBasicMaterial({
    color: c, vertexColors: true, transparent: true, opacity: 0.55,
    depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
  });
  fan.add(new THREE.Mesh(g, m));
});
light.add(fan);
light.renderOrder = -1;
stage.add(light);

// ------------------------------------------------------------ layout
function layout() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();

  const visH = 2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * camera.position.z;
  const unitsPerPx = visH / h;

  // centre the prism on the title's middle letters
  const c = canvas.getBoundingClientRect();
  const t = title.getBoundingClientRect();
  const cy = t.top + t.height * 0.47 - c.top;
  stage.position.y = (h / 2 - cy) * unitsPerPx;
  stage.position.x = 0;

  // prism height ≈ 1.2 × the title's line box
  const targetPx = Math.min(t.height * 1.2, w * 0.38);
  stage.scale.setScalar((targetPx * unitsPerPx) / H);
}
new ResizeObserver(layout).observe(canvas);
if (document.fonts) document.fonts.ready.then(layout);

// ------------------------------------------------------------ interaction
const rest = { x: 0.0, y: 0.0 };
const offset = { x: 0, y: 0 };   // user-driven rotation, springs back
const vel = { x: 0, y: 0 };
const hover = { x: 0, y: 0 };
let dragging = false, last = null;

canvas.addEventListener("pointerdown", (e) => {
  dragging = true; last = { x: e.clientX, y: e.clientY };
  canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener("pointermove", (e) => {
  const r = canvas.getBoundingClientRect();
  hover.x = ((e.clientX - r.left) / r.width) * 2 - 1;
  hover.y = ((e.clientY - r.top) / r.height) * 2 - 1;
  if (!dragging) return;
  const dx = e.clientX - last.x, dy = e.clientY - last.y;
  last = { x: e.clientX, y: e.clientY };
  vel.y = dx * 0.008; vel.x = dy * 0.008;
  offset.y += vel.y; offset.x += vel.x;
});
const release = () => { dragging = false; };
canvas.addEventListener("pointerup", release);
canvas.addEventListener("pointercancel", release);
canvas.addEventListener("pointerleave", () => { hover.x = hover.y = 0; });

// keyboard access: arrow keys nudge the prism
canvas.tabIndex = 0;
canvas.addEventListener("keydown", (e) => {
  const k = { ArrowLeft: [0, -0.3], ArrowRight: [0, 0.3], ArrowUp: [-0.3, 0], ArrowDown: [0.3, 0] }[e.key];
  if (k) { offset.x += k[0]; offset.y += k[1]; e.preventDefault(); }
});

// ------------------------------------------------------------ loop
let visible = true;
new IntersectionObserver(([e]) => { visible = e.isIntersecting; }).observe(hero);

const clock = new THREE.Clock();
function frame() {
  requestAnimationFrame(frame);
  if (!visible) return;
  const t = clock.getElapsedTime();

  if (!dragging) {
    // inertia, then spring back to the dispersion pose
    offset.x += vel.x; offset.y += vel.y;
    vel.x *= 0.92; vel.y *= 0.92;
    offset.x *= 0.965; offset.y *= 0.965;
  }
  const idle = reduceMotion ? 0 : 1;
  const rx = rest.x + offset.x + hover.y * 0.12 + Math.sin(t * 0.5) * 0.05 * idle;
  const ry = rest.y + offset.y + hover.x * 0.2 + Math.sin(t * 0.35) * 0.22 * idle;
  prism.rotation.set(rx, ry, 0);
  prism.position.y = Math.sin(t * 0.8) * 0.03 * idle;

  // light path only makes sense near the dispersion pose
  const align = Math.max(0, Math.cos(rx)) * Math.max(0, Math.cos(ry));
  const strength = Math.pow(align, 6);
  beamIn.material.opacity = 0.85 * strength;
  beamInside.material.opacity = 0.16 * strength;
  fan.children.forEach((m, i) => {
    m.material.opacity = (0.3 + 0.06 * Math.sin(t * 1.3 + i * 0.6) * idle) * strength;
  });

  renderer.render(scene, camera);
}
layout();
frame();
