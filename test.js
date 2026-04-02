const API_BASE = "http://127.0.0.1:8000";

const statusBadge = document.getElementById("statusBadge");
const pdfInput = document.getElementById("pdfInput");
const uploadBtn = document.getElementById("uploadBtn");
const topKInput = document.getElementById("topK");
const topKValue = document.getElementById("topKValue");
const showSnippetsInput = document.getElementById("showSnippets");
const clearBtn = document.getElementById("clearBtn");
const chatArea = document.getElementById("chatArea");
const chatForm = document.getElementById("chatForm");
const questionInput = document.getElementById("questionInput");
const askBtn = document.getElementById("askBtn");

let ready = false;

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
    setStatus("PDF 파일을 먼저 선택하세요", "error");
    return;
  }

  setStatus("모델 로딩/인덱싱 중... 잠시 기다려 주세요", "loading");
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
      throw new Error(err.detail || "업로드 실패");
    }

    const data = await response.json();
    setStatus(`✓ Ready — ${data.n_pages} pages · ${data.n_chunks} chunks`, "ready");
    ready = true;
    questionInput.disabled = false;
    askBtn.disabled = false;
  } catch (error) {
    setStatus(`오류: ${error.message}`, "error");
  } finally {
    uploadBtn.disabled = false;
  }
}

async function askQuestion(event) {
  event.preventDefault();

  if (!ready) {
    setStatus("먼저 PDF 업로드가 필요합니다", "error");
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
      throw new Error(err.detail || "질문 처리 실패");
    }

    const data = await response.json();
    appendAssistantMessage(data.answer, data.refs);
  } catch (error) {
    appendAssistantMessage(`오류가 발생했습니다: ${error.message}`, []);
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
uploadBtn.addEventListener("click", uploadPdf);
chatForm.addEventListener("submit", askQuestion);
clearBtn.addEventListener("click", clearChat);
