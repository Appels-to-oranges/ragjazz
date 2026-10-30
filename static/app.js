import { initViz, updateViz } from "./viz.js";

const form = document.getElementById("ask-form");
const questionEl = document.getElementById("question");
const askBtn = document.getElementById("ask-btn");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");
const noticeEl = document.getElementById("notice");
const answerCard = document.getElementById("answer-card");
const answerEl = document.getElementById("answer");
const sourcesEl = document.getElementById("sources");
const chunksEl = document.getElementById("chunks");

initViz().catch(() => {});

document.querySelectorAll(".examples button").forEach((btn) => {
  btn.addEventListener("click", () => {
    questionEl.value = btn.dataset.q;
    questionEl.focus();
  });
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionEl.value.trim();
  if (!question) return;

  askBtn.disabled = true;
  statusEl.textContent = "Searching the knowledge base…";
  resultEl.hidden = true;

  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data.detail || "Request failed");
    }

    renderResult(data);
    await initViz();
    if (data.visualization) updateViz(data.visualization);
    statusEl.textContent = data.mode === "llm" ? "Answer ready" : "Passages ready";
  } catch (err) {
    statusEl.textContent = err.message || "Something went wrong";
  } finally {
    askBtn.disabled = false;
  }
});

function renderResult(data) {
  resultEl.hidden = false;

  if (data.notice) {
    noticeEl.hidden = false;
    noticeEl.textContent = data.notice;
  } else {
    noticeEl.hidden = true;
    noticeEl.textContent = "";
  }

  if (data.answer) {
    answerCard.hidden = false;
    answerEl.textContent = data.answer;
  } else {
    answerCard.hidden = true;
    answerEl.textContent = "";
  }

  sourcesEl.innerHTML = "";
  (data.sources || []).forEach((source) => {
    const li = document.createElement("li");
    if (source.url) {
      const a = document.createElement("a");
      a.href = source.url;
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      a.textContent = `${source.title} › ${source.section}`;
      li.appendChild(a);
    } else {
      li.textContent = `${source.title} › ${source.section}`;
    }
    sourcesEl.appendChild(li);
  });

  chunksEl.innerHTML = "";
  (data.chunks || []).forEach((chunk) => {
    const div = document.createElement("div");
    div.className = "chunk";
    const meta = document.createElement("div");
    meta.className = "chunk-meta";
    meta.innerHTML = `<strong>${escapeHtml(chunk.title)}</strong><span>${escapeHtml(chunk.section)}</span><span>${Math.round((chunk.similarity || 0) * 100)}% match</span>`;
    const text = document.createElement("p");
    text.textContent = chunk.text;
    div.appendChild(meta);
    div.appendChild(text);
    chunksEl.appendChild(div);
  });
}

function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}
