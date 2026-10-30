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
const examplesEl = document.getElementById("examples");

const EXAMPLE_POOL = [
  { q: "What is a tritone substitution?", label: "Tritone substitution" },
  { q: "How does stride piano differ from boogie-woogie?", label: "Stride vs boogie" },
  { q: "Explain the ii-V-I progression", label: "ii–V–I" },
  { q: "What is modal jazz?", label: "Modal jazz" },
  { q: "What are Coltrane changes?", label: "Coltrane changes" },
  { q: "How do altered dominant chords work?", label: "Altered dominants" },
  { q: "What is a backdoor progression?", label: "Backdoor progression" },
  { q: "Explain upper structure triads on piano", label: "Upper structures" },
  { q: "What is quartal harmony?", label: "Quartal harmony" },
  { q: "How do you voice a seventh chord on piano?", label: "Seventh chords" },
  { q: "What is the chord-scale system?", label: "Chord-scale system" },
  { q: "Explain the bebop scale", label: "Bebop scale" },
  { q: "What makes a good jazz piano voicing?", label: "Jazz voicings" },
  { q: "How does voice leading work in jazz harmony?", label: "Voice leading" },
  { q: "What is a turnaround in jazz?", label: "Turnarounds" },
  { q: "Explain rhythm changes", label: "Rhythm changes" },
  { q: "What is a slash chord?", label: "Slash chords" },
  { q: "How does the circle of fifths help with jazz keys?", label: "Circle of fifths" },
  { q: "What is comping in jazz piano?", label: "Comping" },
  { q: "Explain secondary dominants", label: "Secondary dominants" },
  { q: "What is an extended chord?", label: "Extended chords" },
  { q: "How is the blues scale used in jazz piano?", label: "Blues scale" },
  { q: "What is a block chord?", label: "Block chords" },
  { q: "Explain walking bass under a jazz piano left hand", label: "Walking bass" },
];

const EXAMPLES_SHOWN = 4;

let initViz = async () => {};
let updateViz = () => {};

function shuffleInPlace(items) {
  for (let i = items.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [items[i], items[j]] = [items[j], items[i]];
  }
  return items;
}

function renderExamples() {
  if (!examplesEl) return;
  const picks = shuffleInPlace([...EXAMPLE_POOL]).slice(0, EXAMPLES_SHOWN);
  examplesEl.innerHTML = "";
  picks.forEach((item) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.dataset.q = item.q;
    btn.textContent = item.label;
    btn.addEventListener("click", () => {
      questionEl.value = item.q;
      questionEl.focus();
    });
    examplesEl.appendChild(btn);
  });
}

// Examples must not depend on Three.js loading.
renderExamples();

import("./viz.js")
  .then((mod) => {
    initViz = mod.initViz;
    updateViz = mod.updateViz;
    return initViz();
  })
  .catch(() => {
    if (statusEl && !statusEl.textContent) {
      // Keep the page usable even if the viz bundle fails.
    }
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
    try {
      await initViz();
      if (data.visualization) updateViz(data.visualization);
    } catch (_) {
      // Visualization is optional.
    }
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
