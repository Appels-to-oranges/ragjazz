/**
 * 3D embedding viewer (Three.js).
 *
 * Idle: global PCA cloud (topic overview).
 * After ask: local cosine-MDS layout so nearest neighbors sit near the query.
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const canvasHost = document.getElementById("viz-canvas");
const metaEl = document.getElementById("viz-meta");
const legendEl = document.getElementById("viz-legend");

let renderer, scene, camera, controls, cloudPoints, queryMesh, neighborGroup, lineGroup, contextPoints;
let cloudData = null;
let animationId = null;
let resizeObserver = null;
let mode = "global"; // global | local

const PALETTE = [
  "#c45c26", "#d4a017", "#2f6f6a", "#7a4e2d", "#3d5a80",
  "#9b2226", "#52796f", "#bc6c25", "#415a77", "#6a994e",
  "#b56576", "#6d597a", "#355070", "#eaac8b", "#e56b6f",
  "#355070", "#b5651d", "#606c38", "#283618", "#a4161a",
];

function colorForIndex(index) {
  return new THREE.Color(PALETTE[index % PALETTE.length]);
}

export async function initViz() {
  if (!canvasHost) return;
  if (cloudData) return;
  try {
    const res = await fetch("/api/embeddings-3d");
    if (!res.ok) throw new Error("3D map not ready");
    cloudData = await res.json();
    setupScene();
    drawGlobalCloud();
    drawLegend(cloudData.titles || []);
    if (metaEl) {
      metaEl.textContent =
        "Overview: PCA of the full corpus. After you ask, this switches to a local cosine layout so true neighbors sit near your query.";
    }
  } catch (err) {
    if (metaEl) metaEl.textContent = err.message || "Visualization unavailable";
  }
}

function setupScene() {
  const width = canvasHost.clientWidth || 640;
  const height = canvasHost.clientHeight || 360;

  scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x120e0b, 0.045);

  camera = new THREE.PerspectiveCamera(55, width / height, 0.01, 100);
  camera.position.set(2.4, 1.6, 2.8);

  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(width, height, false);
  renderer.setClearColor(0x000000, 0);
  canvasHost.innerHTML = "";
  canvasHost.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.minDistance = 0.5;
  controls.maxDistance = 12;

  scene.add(new THREE.AmbientLight(0xfff2dd, 0.85));
  const key = new THREE.DirectionalLight(0xffd7a8, 0.65);
  key.position.set(3, 4, 2);
  scene.add(key);

  const grid = new THREE.GridHelper(6, 12, 0x5a4634, 0x2a2118);
  grid.position.y = -1.2;
  scene.add(grid);

  neighborGroup = new THREE.Group();
  lineGroup = new THREE.Group();
  scene.add(neighborGroup);
  scene.add(lineGroup);

  resizeObserver = new ResizeObserver(() => {
    if (!renderer || !camera) return;
    const w = canvasHost.clientWidth;
    const h = canvasHost.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h, false);
  });
  resizeObserver.observe(canvasHost);

  const animate = () => {
    animationId = requestAnimationFrame(animate);
    controls.update();
    if (queryMesh) queryMesh.rotation.y += 0.012;
    renderer.render(scene, camera);
  };
  animate();
}

function disposePoints(obj) {
  if (!obj) return;
  scene.remove(obj);
  obj.geometry?.dispose();
  obj.material?.dispose();
}

function drawGlobalCloud() {
  if (!cloudData?.points?.length || !scene) return;
  disposePoints(cloudPoints);
  disposePoints(contextPoints);
  contextPoints = null;

  const n = cloudData.points.length;
  const positions = new Float32Array(n * 3);
  const colors = new Float32Array(n * 3);

  cloudData.points.forEach((p, i) => {
    positions[i * 3] = p.x;
    positions[i * 3 + 1] = p.y;
    positions[i * 3 + 2] = p.z;
    const c = colorForIndex(p.color_index || 0);
    colors[i * 3] = c.r;
    colors[i * 3 + 1] = c.g;
    colors[i * 3 + 2] = c.b;
  });

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

  cloudPoints = new THREE.Points(
    geometry,
    new THREE.PointsMaterial({
      size: 0.035,
      vertexColors: true,
      transparent: true,
      opacity: 0.5,
      sizeAttenuation: true,
    })
  );
  scene.add(cloudPoints);
  mode = "global";
}

function drawLocalContext(points) {
  disposePoints(cloudPoints);
  cloudPoints = null;
  disposePoints(contextPoints);

  if (!points?.length) {
    contextPoints = null;
    return;
  }

  const n = points.length;
  const positions = new Float32Array(n * 3);
  const colors = new Float32Array(n * 3);
  points.forEach((p, i) => {
    positions[i * 3] = p.x;
    positions[i * 3 + 1] = p.y;
    positions[i * 3 + 2] = p.z;
    const c = colorForIndex(p.color_index || 0);
    colors[i * 3] = c.r;
    colors[i * 3 + 1] = c.g;
    colors[i * 3 + 2] = c.b;
  });

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

  contextPoints = new THREE.Points(
    geometry,
    new THREE.PointsMaterial({
      size: 0.04,
      vertexColors: true,
      transparent: true,
      opacity: 0.35,
      sizeAttenuation: true,
    })
  );
  scene.add(contextPoints);
}

function drawLegend(titles) {
  if (!legendEl) return;
  legendEl.innerHTML = "";
  titles.slice(0, 12).forEach((title, i) => {
    const item = document.createElement("span");
    item.className = "viz-legend-item";
    item.innerHTML = `<i style="background:${PALETTE[i % PALETTE.length]}"></i>${escapeHtml(title)}`;
    legendEl.appendChild(item);
  });
  if (titles.length > 12) {
    const more = document.createElement("span");
    more.className = "viz-legend-item muted";
    more.textContent = `+${titles.length - 12} more articles`;
    legendEl.appendChild(more);
  }
}

function clearHighlights() {
  if (queryMesh) {
    scene.remove(queryMesh);
    queryMesh.geometry.dispose();
    queryMesh.material.dispose();
    queryMesh = null;
  }
  while (neighborGroup.children.length) {
    const child = neighborGroup.children.pop();
    child.geometry?.dispose();
    child.material?.dispose();
  }
  while (lineGroup.children.length) {
    const child = lineGroup.children.pop();
    child.geometry?.dispose();
    child.material?.dispose();
  }
}

export function updateViz(visualization) {
  if (!visualization?.available || !scene) return;
  const q = visualization.query;
  if (!q) return;

  // Switch to local cosine-MDS coordinates for this query.
  if (visualization.method === "cosine_mds") {
    drawLocalContext(visualization.context || []);
    mode = "local";
  }

  clearHighlights();

  const queryGeo = new THREE.SphereGeometry(0.07, 24, 24);
  const queryMat = new THREE.MeshStandardMaterial({
    color: 0xfff1c9,
    emissive: 0xc45c26,
    emissiveIntensity: 0.7,
    metalness: 0.2,
    roughness: 0.35,
  });
  queryMesh = new THREE.Mesh(queryGeo, queryMat);
  queryMesh.position.set(q.x, q.y, q.z);
  scene.add(queryMesh);

  const neighbors = visualization.neighbors || [];
  neighbors.forEach((n, idx) => {
    if (n.x == null) return;
    const geo = new THREE.SphereGeometry(0.05, 18, 18);
    const mat = new THREE.MeshStandardMaterial({
      color: 0xffb347,
      emissive: 0x9a3f14,
      emissiveIntensity: 0.45 + idx * 0.05,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(n.x, n.y, n.z);
    neighborGroup.add(mesh);

    const lineGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(q.x, q.y, q.z),
      new THREE.Vector3(n.x, n.y, n.z),
    ]);
    const lineMat = new THREE.LineBasicMaterial({
      color: 0xe8c39e,
      transparent: true,
      opacity: 0.7,
    });
    lineGroup.add(new THREE.Line(lineGeo, lineMat));
  });

  if (metaEl) {
    const labels = neighbors
      .map((n) => `${n.title} (${Math.round((n.similarity || 0) * 100)}%)`)
      .join(" · ");
    metaEl.textContent = visualization.note
      ? `${visualization.note}${labels ? ` Selected: ${labels}` : ""}`
      : labels || "";
  }

  controls.target.set(q.x, q.y, q.z);
  camera.position.set(q.x + 1.6, q.y + 1.1, q.z + 1.8);
  controls.update();
}

function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}
