/**
 * 3D embedding viewer (Three.js).
 * Stable PCA corpus cloud + query/neighbor highlights.
 * Click a point for article/section + cosine score vs the current query.
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const canvasHost = document.getElementById("viz-canvas");
const metaEl = document.getElementById("viz-meta");
const legendEl = document.getElementById("viz-legend");
const tooltipEl = document.getElementById("viz-tooltip");

let renderer, scene, camera, controls, cloudPoints, queryMesh, neighborGroup, lineGroup;
let cloudData = null;
let resizeObserver = null;
let raycaster = null;
let pointer = new THREE.Vector2();
let lastQuestion = "";
let neighborById = new Map();
let pointerDown = null;

const PALETTE = [
  "#c45c26", "#d4a017", "#2f6f6a", "#7a4e2d", "#3d5a80",
  "#9b2226", "#52796f", "#bc6c25", "#415a77", "#6a994e",
  "#b56576", "#6d597a", "#355070", "#eaac8b", "#e56b6f",
  "#355070", "#b5651d", "#606c38", "#283618", "#a4161a",
];

// Mild visual scale-up: more breathing room + easier click targets.
const SPREAD = 1.4;
const CLOUD_POINT_SIZE = 0.05;
const NEIGHBOR_RADIUS = 0.065;
const QUERY_RADIUS = 0.09;

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
        "PCA map of the knowledge base. Ask a question, then click any point for its topic and cosine score.";
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
  // Keep tooltip as a sibling overlay; only replace/append the canvas.
  const existingTooltip = tooltipEl;
  canvasHost.querySelectorAll("canvas").forEach((c) => c.remove());
  canvasHost.appendChild(renderer.domElement);
  if (existingTooltip && existingTooltip.parentElement !== canvasHost) {
    canvasHost.appendChild(existingTooltip);
  }

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.minDistance = 0.5;
  controls.maxDistance = 12;

  scene.add(new THREE.AmbientLight(0xfff2dd, 0.85));
  const key = new THREE.DirectionalLight(0xffd7a8, 0.65);
  key.position.set(3, 4, 2);
  scene.add(key);

  const grid = new THREE.GridHelper(8, 16, 0x5a4634, 0x2a2118);
  grid.position.y = -1.6;
  scene.add(grid);

  neighborGroup = new THREE.Group();
  lineGroup = new THREE.Group();
  scene.add(neighborGroup);
  scene.add(lineGroup);

  raycaster = new THREE.Raycaster();
  raycaster.params.Points = { threshold: 0.09 };

  renderer.domElement.style.cursor = "pointer";
  renderer.domElement.addEventListener("pointerdown", onPointerDown);
  renderer.domElement.addEventListener("pointerup", onPointerUp);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") hideTooltip();
  });

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
    requestAnimationFrame(animate);
    controls.update();
    if (queryMesh) queryMesh.rotation.y += 0.012;
    renderer.render(scene, camera);
  };
  animate();
}

function onPointerDown(event) {
  pointerDown = { x: event.clientX, y: event.clientY };
}

async function onPointerUp(event) {
  if (!pointerDown) return;
  const dx = event.clientX - pointerDown.x;
  const dy = event.clientY - pointerDown.y;
  pointerDown = null;
  // Ignore drags so orbiting doesn't open tooltips.
  if (Math.hypot(dx, dy) > 4) return;
  await handleClick(event);
}

function setPointerFromEvent(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
}

async function handleClick(event) {
  if (!raycaster || !camera) return;
  setPointerFromEvent(event);
  raycaster.setFromCamera(pointer, camera);

  // Prefer neighbor spheres, then corpus points.
  const neighborHits = raycaster.intersectObjects(neighborGroup.children, false);
  if (neighborHits.length) {
    const mesh = neighborHits[0].object;
    const chunkId = mesh.userData?.id;
    if (chunkId) {
      await showChunkTooltip(chunkId, event, mesh.userData);
      return;
    }
  }

  if (cloudPoints) {
    const cloudHits = raycaster.intersectObject(cloudPoints, false);
    if (cloudHits.length) {
      const idx = cloudHits[0].index;
      const point = cloudData?.points?.[idx];
      if (point?.id) {
        await showChunkTooltip(point.id, event, point);
        return;
      }
    }
  }

  hideTooltip();
}

async function showChunkTooltip(chunkId, event, fallback = {}) {
  if (!tooltipEl) return;
  tooltipEl.hidden = false;
  tooltipEl.classList.add("is-loading");
  tooltipEl.innerHTML = `<strong>${escapeHtml(fallback.title || "Loading…")}</strong>`;
  placeTooltip(event);

  const cached = neighborById.get(chunkId);
  try {
    const res = await fetch("/api/chunk-info", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        id: chunkId,
        question: lastQuestion || null,
      }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || "Could not load chunk");
    if (data.similarity == null && cached?.similarity != null) {
      data.similarity = cached.similarity;
    }
    renderTooltip(data);
  } catch (err) {
    renderTooltip({
      title: fallback.title || cached?.title || "Chunk",
      section: fallback.section || cached?.section || "",
      preview: fallback.preview || err.message,
      similarity: cached?.similarity ?? null,
    });
  }
}

function renderTooltip(data) {
  if (!tooltipEl) return;
  tooltipEl.classList.remove("is-loading");
  const sim =
    data.similarity == null
      ? lastQuestion
        ? "Cosine: —"
        : "Ask a question to see cosine score"
      : `Cosine: ${Math.round(data.similarity * 100)}%`;
  tooltipEl.innerHTML = `
    <strong>${escapeHtml(data.title || "Untitled")}</strong>
    <span class="viz-tip-section">${escapeHtml(data.section || "")}</span>
    <span class="viz-tip-score">${escapeHtml(sim)}</span>
    <p>${escapeHtml(data.preview || "")}</p>
  `;
}

function placeTooltip(event) {
  if (!tooltipEl) return;
  const rect = canvasHost.getBoundingClientRect();
  const x = event.clientX - rect.left + 12;
  const y = event.clientY - rect.top + 12;
  tooltipEl.style.left = `${Math.min(x, rect.width - 220)}px`;
  tooltipEl.style.top = `${Math.min(y, rect.height - 120)}px`;
}

function hideTooltip() {
  if (!tooltipEl) return;
  tooltipEl.hidden = true;
  tooltipEl.classList.remove("is-loading");
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

  const n = cloudData.points.length;
  const positions = new Float32Array(n * 3);
  const colors = new Float32Array(n * 3);

  cloudData.points.forEach((p, i) => {
    positions[i * 3] = p.x * SPREAD;
    positions[i * 3 + 1] = p.y * SPREAD;
    positions[i * 3 + 2] = p.z * SPREAD;
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
      size: CLOUD_POINT_SIZE,
      vertexColors: true,
      transparent: true,
      opacity: 0.55,
      sizeAttenuation: true,
    })
  );
  scene.add(cloudPoints);
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

  lastQuestion = visualization.question || "";
  neighborById = new Map(
    (visualization.neighbors || [])
      .filter((n) => n.id)
      .map((n) => [n.id, n])
  );
  hideTooltip();

  drawGlobalCloud();
  clearHighlights();

  const qx = q.x * SPREAD;
  const qy = q.y * SPREAD;
  const qz = q.z * SPREAD;

  const queryGeo = new THREE.SphereGeometry(QUERY_RADIUS, 24, 24);
  const queryMat = new THREE.MeshStandardMaterial({
    color: 0xfff1c9,
    emissive: 0xc45c26,
    emissiveIntensity: 0.7,
    metalness: 0.2,
    roughness: 0.35,
  });
  queryMesh = new THREE.Mesh(queryGeo, queryMat);
  queryMesh.position.set(qx, qy, qz);
  scene.add(queryMesh);

  const neighbors = visualization.neighbors || [];
  neighbors.forEach((n, idx) => {
    if (n.x == null) return;
    const nx = n.x * SPREAD;
    const ny = n.y * SPREAD;
    const nz = n.z * SPREAD;
    const geo = new THREE.SphereGeometry(NEIGHBOR_RADIUS, 18, 18);
    const mat = new THREE.MeshStandardMaterial({
      color: 0xffb347,
      emissive: 0x9a3f14,
      emissiveIntensity: 0.45 + idx * 0.05,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(nx, ny, nz);
    mesh.userData = {
      id: n.id,
      title: n.title,
      section: n.section,
      similarity: n.similarity,
    };
    neighborGroup.add(mesh);

    const lineGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(qx, qy, qz),
      new THREE.Vector3(nx, ny, nz),
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

  controls.target.set(qx, qy, qz);
  camera.position.set(qx + 2.0, qy + 1.35, qz + 2.2);
  controls.update();
}

function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}
