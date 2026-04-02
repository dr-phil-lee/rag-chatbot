const statusBadge = document.getElementById("statusBadge");
const pdfInput = document.getElementById("pdfInput");
const uploadBtn = document.getElementById("uploadBtn");
const apiBaseInput = document.getElementById("apiBaseInput");
const saveApiBtn = document.getElementById("saveApiBtn");
const topKInput = document.getElementById("topK");
const topKValue = document.getElementById("topKValue");
const showSnippetsInput = document.getElementById("showSnippets");
const clearBtn = document.getElementById("clearBtn");
const chatArea = document.getElementById("chatArea");
const chatForm = document.getElementById("chatForm");
const questionInput = document.getElementById("questionInput");
const askBtn = document.getElementById("askBtn");

let ready = false;
const STORAGE_KEY = "ragApiBaseUrl";

function resolveApiBase() {
  const fromQuery = new URLSearchParams(window.location.search).get("api");
  if (fromQuery) return fromQuery.replace(/\/$/, "");

  const fromStorage = localStorage.getItem(STORAGE_KEY);
  if (fromStorage) return fromStorage.replace(/\/$/, "");

  if (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1") {
    return "http://127.0.0.1:8000";
  }
  return window.location.origin;
}

let API_BASE = resolveApiBase();

function saveApiBase() {
  const value = apiBaseInput.value.trim();
  if (!value) {
    localStorage.removeItem(STORAGE_KEY);
    API_BASE = resolveApiBase();
    setStatus(`API URL reset. Current: ${API_BASE}`, "loading");
    return;
  }

  localStorage.setItem(STORAGE_KEY, value.replace(/\/$/, ""));
  API_BASE = resolveApiBase();
  setStatus(`API URL saved: ${API_BASE}`, "ready");
}

function describeNetworkError(error) {
  if (error instanceof TypeError && error.message.includes("Failed to fetch")) {
    return `Cannot reach API (${API_BASE}). Check backend URL, CORS, and HTTPS.`;
  }
  return error.message || "Unknown error.";
}

function setStatus(message, type = "loading") {
  statusBadge.textContent = message;
  statusBadge.className = `status-badge ${type}`;
}

function appendUserMessage(text) {
  const wrapper = document.createElement("div");
  wrapper.className = "message user";
  wrapper.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>`;
  chatArea.appendChild(wrapper);
  chatArea.scrollTop = chatArea.scrollHeight;
}

function appendAssistantMessage(answer, refs = []) {
  const wrapper = document.createElement("div");
  wrapper.className = "message assistant";

  const refsHtml = refs
    .map((ref) => `<div class="ref-pill">[Ref ${ref.idx}] Page ${ref.page}</div>`)
    .join("");

  const snippetsHtml = showSnippetsInput.checked
    ? `<div class="snippets">${refs
        .map(
          (ref) =>
            `<div class="snippet"><strong>[Ref ${ref.idx}]</strong> ${escapeHtml(
              ref.snippet
            )}...</div>`
        )
        .join("")}</div>`
    : "";

  wrapper.innerHTML = `
    <div class="bubble">
      ${escapeHtml(answer)}
      ${refs.length ? `<div class="refs">${refsHtml}</div>` : ""}
      ${refs.length ? snippetsHtml : ""}
    </div>
  `;
  chatArea.appendChild(wrapper);
  chatArea.scrollTop = chatArea.scrollHeight;
}

function escapeHtml(input) {
  return input
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function uploadPdf() {
  const file = pdfInput.files[0];
  if (!file) {
    setStatus("Please select a PDF file first.", "error");
    return;
  }

  setStatus("Loading models and indexing... please wait.", "loading");
  uploadBtn.disabled = true;

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch(`${API_BASE}/upload`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || "Upload failed.");
    }

    const data = await response.json();
    setStatus(`✓ Ready — ${data.n_pages} pages · ${data.n_chunks} chunks`, "ready");
    ready = true;
    questionInput.disabled = false;
    askBtn.disabled = false;
  } catch (error) {
    setStatus(`Error: ${describeNetworkError(error)}`, "error");
  } finally {
    uploadBtn.disabled = false;
  }
}

async function askQuestion(event) {
  event.preventDefault();

  if (!ready) {
    setStatus("Please upload a PDF first.", "error");
    return;
  }

  const question = questionInput.value.trim();
  if (!question) return;

  appendUserMessage(question);
  questionInput.value = "";
  askBtn.disabled = true;

  try {
    const response = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        top_k: Number(topKInput.value),
      }),
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || "Question processing failed.");
    }

    const data = await response.json();
    appendAssistantMessage(data.answer, data.refs);
  } catch (error) {
    appendAssistantMessage(`An error occurred: ${describeNetworkError(error)}`, []);
  } finally {
    askBtn.disabled = false;
  }
}

async function clearChat() {
  chatArea.innerHTML = "";
  await fetch(`${API_BASE}/clear`, { method: "POST" });
}

topKInput.addEventListener("input", () => {
  topKValue.textContent = topKInput.value;
});
saveApiBtn.addEventListener("click", saveApiBase);
uploadBtn.addEventListener("click", uploadPdf);
chatForm.addEventListener("submit", askQuestion);
clearBtn.addEventListener("click", clearChat);

if (apiBaseInput) {
  apiBaseInput.value = API_BASE;
}
