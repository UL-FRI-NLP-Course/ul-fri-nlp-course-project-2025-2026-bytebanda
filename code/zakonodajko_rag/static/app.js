const state = {
  history: [],
  pending: false,
  payloads: {},
  selectedSourceId: null,
  responseCounter: 0,
};

const messagesEl = document.getElementById("messages");
const formEl = document.getElementById("chat-form");
const inputEl = document.getElementById("chat-input");
const sendButtonEl = document.getElementById("send-button");
const healthDotEl = document.getElementById("health-dot");
const healthTextEl = document.getElementById("health-text");
const chunkCountEl = document.getElementById("chunk-count");
const drawerTitleEl = document.getElementById("drawer-title");
const drawerCopyEl = document.getElementById("drawer-copy");
const drawerContentEl = document.getElementById("drawer-content");

function boot() {
  const welcomeTemplate = document.getElementById("welcome-template");
  messagesEl.appendChild(welcomeTemplate.content.cloneNode(true));
  bindPromptChips();
  bindComposer();
  checkHealth();
}

function bindPromptChips() {
  document.querySelectorAll(".prompt-chip").forEach((button) => {
    button.addEventListener("click", () => {
      inputEl.value = button.dataset.prompt || "";
      inputEl.focus();
    });
  });
}

function bindComposer() {
  formEl.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (state.pending) {
      return;
    }
    const message = inputEl.value.trim();
    if (!message) {
      return;
    }
    inputEl.value = "";
    await sendMessage(message);
  });

  inputEl.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      formEl.requestSubmit();
    }
  });
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) {
      throw new Error("health failed");
    }
    const payload = await response.json();
    healthDotEl.classList.add("online");
    healthTextEl.textContent = `${payload.document_count} dokumentov, ${payload.chunk_count} chunkov`;
    chunkCountEl.textContent = payload.generator_model ? "lokalni LLM" : `${payload.chunk_count} chunkov`;
  } catch (_error) {
    healthDotEl.classList.add("offline");
    healthTextEl.textContent = "Strežnik ni dosegljiv";
    chunkCountEl.textContent = "offline";
  }
}

async function sendMessage(message) {
  state.pending = true;
  inputEl.disabled = true;
  sendButtonEl.disabled = true;

  renderUserMessage(message);
  const typingEl = renderTypingMessage(buildPendingUiState(message, state.history));

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message,
        history: state.history,
      }),
    });

    if (!response.ok) {
      throw new Error("API request failed");
    }

    const payload = await response.json();
    const responseId = `assistant-${++state.responseCounter}`;
    disposeTypingMessage(typingEl);
    renderAssistantMessage(responseId, payload);

    state.history.push({ role: "user", content: message });
    state.history.push(buildAssistantHistoryEntry(payload));
    state.payloads[responseId] = payload;

    openSourcesDrawer(responseId);
  } catch (_error) {
    disposeTypingMessage(typingEl);
    renderErrorMessage();
  } finally {
    state.pending = false;
    inputEl.disabled = false;
    sendButtonEl.disabled = false;
    inputEl.focus();
  }
}

function buildAssistantHistoryEntry(payload) {
  return {
    role: "assistant",
    content: payload.message,
    citations: payload.citations || [],
    retrieval_query: payload.retrieval_query || "",
    memory_topic: payload.memory_topic || "",
    calculator_context: payload.calculator_context || null,
  };
}

function renderUserMessage(message) {
  const article = document.createElement("article");
  article.className = "message user";
  article.innerHTML = `
    <div class="message-head">
      <span class="message-role">Ti</span>
      <span class="message-meta">Vprašanje</span>
    </div>
    <div class="message-body">
      <p></p>
    </div>
  `;
  article.querySelector("p").textContent = message;
  messagesEl.appendChild(article);
  scrollMessagesToBottom();
}

function renderTypingMessage(pendingUi) {
  const article = document.createElement("article");
  article.className = "message assistant";
  const phasesHtml = (pendingUi.phases || [])
    .map((phase) => `<span class="thinking-phase${phase.active ? " active" : ""}">${escapeHtml(phase.label)}</span>`)
    .join("");
  article.innerHTML = `
    <div class="message-head">
      <span class="message-role">Zakonodajko</span>
      <span class="message-meta">${escapeHtml(pendingUi.meta || "Obdelujem vprašanje")}</span>
    </div>
    <div class="message-body">
      <div class="typing" aria-label="Nalagam odgovor">
        <span></span>
        <span></span>
        <span></span>
      </div>
      <div class="thinking-phases" aria-label="Faze obdelave">${phasesHtml}</div>
    </div>
  `;
  messagesEl.appendChild(article);
  scrollMessagesToBottom();
  return article;
}

function disposeTypingMessage(article) {
  article?.remove();
}

function renderAssistantMessage(responseId, payload) {
  const article = document.createElement("article");
  article.className = "message assistant";
  article.dataset.responseId = responseId;

  const citationPills = renderCitationPills(payload.citations || []);
  const extraSourceCount = Math.max(0, (payload.used_chunks || []).length - Math.min(2, (payload.citations || []).length));
  const sourceLabel = extraSourceCount > 0 ? `Viri in odlomki (${(payload.used_chunks || []).length})` : "Odpri vire";
  const calculatorPreview = renderCalculatorPreview(payload.calculator_result);
  const workflowBadge = renderWorkflowBadge(payload);

  article.innerHTML = `
    <div class="message-head">
      <span class="message-role">Zakonodajko</span>
      <span class="message-meta">${escapeHtml(messageMeta(payload))}</span>
    </div>
    <div class="message-body">
      <p>${escapeHtml(payload.message)}</p>
      ${workflowBadge}
      ${calculatorPreview}
      <div class="message-foot">
        ${citationPills ? `<div class="citation-pills">${citationPills}</div>` : ""}
        <button type="button" class="source-trigger" data-source-id="${escapeHtml(responseId)}">${escapeHtml(sourceLabel)}</button>
      </div>
    </div>
  `;

  const button = article.querySelector(".source-trigger");
  button.addEventListener("click", () => {
    openSourcesDrawer(responseId);
  });

  messagesEl.appendChild(article);
  scrollMessagesToBottom();
}

function renderCitationPills(citations) {
  return citations
    .slice(0, 2)
    .map((citation) => {
      const label = shortCitationLabel(citation);
      return `<a class="citation-pill" href="${escapeHtml(citation.source_url || "#")}" target="_blank" rel="noreferrer">${escapeHtml(label)}</a>`;
    })
    .join("");
}

function openSourcesDrawer(responseId) {
  const payload = state.payloads[responseId];
  if (!payload) {
    return;
  }
  state.selectedSourceId = responseId;
  setActiveSourceButton(responseId);
  renderSourcesDrawer(payload);
}

function setActiveSourceButton(responseId) {
  document.querySelectorAll(".source-trigger").forEach((button) => {
    button.classList.toggle("active", button.dataset.sourceId === responseId);
  });
}

function renderSourcesDrawer(payload) {
  const primaryCitation = (payload.citations || [])[0];
  drawerTitleEl.textContent = primaryCitation ? shortCitationLabel(primaryCitation) : "Dokazna podlaga odgovora";
  drawerCopyEl.textContent = payload.contextualized
    ? "Odgovor uporablja follow-up kontekst. Spodaj so prikazani uporabljeni viri in interpretirani kontekst."
    : "Odgovor je zgrajen iz spodnjih pravnih virov in podpornih odlomkov.";

  const contextualizationHtml = payload.contextualized
    ? `
      <section class="drawer-section">
        <p class="drawer-section-title">Interpretacija follow-up</p>
        <div class="drawer-note">
          <p>${escapeHtml(payload.retrieval_query || "")}</p>
          ${
            payload.memory_topic
              ? `<p class="drawer-note-meta">Tema spomina: ${escapeHtml(payload.memory_topic)}</p>`
              : ""
          }
        </div>
      </section>
    `
    : "";

  const processingTraceHtml = (payload.processing_trace || [])
    .map((step) => {
      return `
        <article class="trace-card trace-${escapeHtml(step.status || "completed")}">
          <div class="trace-head">
            <p class="trace-title">${escapeHtml(step.label || "")}</p>
            <span class="trace-status">${escapeHtml(traceStatusLabel(step.status))}</span>
          </div>
          <p class="trace-detail">${escapeHtml(step.detail || "")}</p>
        </article>
      `;
    })
    .join("");

  const calculatorHtml = payload.calculator_result
    ? renderCalculatorDrawer(payload.calculator_result)
    : "";

  const citationsHtml = (payload.citations || [])
    .map((citation) => {
      const label = shortCitationLabel(citation);
      return `
        <article class="citation-card">
          <p class="citation-title"><a href="${escapeHtml(citation.source_url || "#")}" target="_blank" rel="noreferrer">${escapeHtml(label)}</a></p>
          <p class="citation-meta">${escapeHtml(citation.title || "")}</p>
        </article>
      `;
    })
    .join("");

  const supportingHtml = (payload.supporting_sentences || [])
    .map((sentence) => {
      return `
        <blockquote class="drawer-quote">
          <p>${escapeHtml(sentence.text || "")}</p>
          <p class="drawer-quote-meta">${escapeHtml(sentence.citation || "")}</p>
        </blockquote>
      `;
    })
    .join("");

  const sourcesHtml = (payload.used_chunks || [])
    .map((chunk) => {
      const label = [chunk.law_ref, chunk.article_number, chunk.article_title].filter(Boolean).join(" ");
      return `
        <article class="source-card">
          <p class="citation-title"><a href="${escapeHtml(chunk.source_url || "#")}" target="_blank" rel="noreferrer">${escapeHtml(label)}</a></p>
          <p class="source-preview">${escapeHtml(chunk.text_preview || "")}</p>
        </article>
      `;
    })
    .join("");

  drawerContentEl.innerHTML = `
    ${contextualizationHtml}
    ${calculatorHtml}
    ${
      processingTraceHtml
        ? `<section class="drawer-section"><p class="drawer-section-title">Faze obdelave</p><div class="trace-list">${processingTraceHtml}</div></section>`
        : ""
    }
    ${
      citationsHtml
        ? `<section class="drawer-section"><p class="drawer-section-title">Citirani členi</p><div class="citation-list">${citationsHtml}</div></section>`
        : ""
    }
    ${
      supportingHtml
        ? `<section class="drawer-section"><p class="drawer-section-title">Podporni stavki</p>${supportingHtml}</section>`
        : ""
    }
    ${
      sourcesHtml
        ? `<section class="drawer-section"><p class="drawer-section-title">Uporabljeni odlomki</p><div class="source-list">${sourcesHtml}</div></section>`
        : ""
    }
    ${
      payload.insufficient_evidence
        ? `<section class="drawer-section"><p class="drawer-section-title">Opozorilo</p><div class="drawer-note"><p>Retriever ni našel dovolj neposredne pravne podlage za zanesljiv odgovor.</p></div></section>`
        : ""
    }
  `;
}

function renderErrorMessage() {
  const article = document.createElement("article");
  article.className = "message assistant";
  article.innerHTML = `
    <div class="message-head">
      <span class="message-role">Zakonodajko</span>
      <span class="message-meta">Napaka</span>
    </div>
    <div class="message-body">
      <p>Pri poizvedbi je prišlo do napake. Preveri, ali strežnik teče in ali so retrieval artefakti zgrajeni.</p>
    </div>
  `;
  messagesEl.appendChild(article);
  scrollMessagesToBottom();
}

function shortCitationLabel(citation) {
  const legalLabel = [citation.law_ref, citation.article_number, citation.article_title].filter(Boolean).join(" ");
  if (legalLabel) {
    return legalLabel;
  }
  if (citation.title) {
    return citation.law_ref ? `${citation.law_ref}, ${citation.title}` : citation.title;
  }
  return citation.law_ref || "Vir";
}

function scrollMessagesToBottom() {
  messagesEl.scrollTo({ top: messagesEl.scrollHeight, behavior: "smooth" });
}

function escapeHtml(text) {
  return String(text || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function messageMeta(payload) {
  if (payload.backend === "calculator") {
    return "Kalkulator";
  }
  if (payload.backend === "calculator_clarification") {
    return "Manjka podatek";
  }
  if (payload.insufficient_evidence) {
    return "Omejena podlaga";
  }
  if (payload.backend === "local_transformer") {
    return "Lokalni model";
  }
  return "Pravni odgovor";
}

function renderCalculatorPreview(calculatorResult) {
  if (!calculatorResult || calculatorResult.status !== "completed") {
    return "";
  }
  const breakdown = (calculatorResult.breakdown || [])
    .slice(0, 2)
    .map((item) => `<div class="calc-preview-row"><span>${escapeHtml(item.label || "")}</span><strong>${escapeHtml(item.value || "")}</strong></div>`)
    .join("");
  return `
    <section class="calc-preview">
      <p class="calc-preview-title">${escapeHtml(calculatorResult.title || "Kalkulator")}</p>
      ${calculatorResult.result_summary ? `<p class="calc-preview-summary">${escapeHtml(calculatorResult.result_summary)}</p>` : ""}
      ${breakdown ? `<div class="calc-preview-grid">${breakdown}</div>` : ""}
    </section>
  `;
}

function renderWorkflowBadge(payload) {
  if (payload.backend === "calculator" || payload.backend === "calculator_clarification") {
    return `<div class="workflow-badge workflow-badge-calculator">Uporabljen kalkulator</div>`;
  }
  if (payload.backend === "local_transformer") {
    return `<div class="workflow-badge">Odgovor je ubesedil lokalni model</div>`;
  }
  return "";
}

function buildPendingUiState(message, history) {
  const normalized = String(message || "").toLowerCase();
  const lastAssistant = [...(history || [])].reverse().find((item) => item.role === "assistant");
  const pendingCalculator = lastAssistant && lastAssistant.calculator_context && lastAssistant.calculator_context.status === "pending";
  if (pendingCalculator) {
    return {
      meta: "Dopolnjujem izračun",
      phases: [
        { label: "Dopolnjujem vhodne podatke", active: true },
        { label: "Pripravljam kalkulator", active: false },
        { label: "Preverjam pravno podlago", active: false },
      ],
    };
  }
  if (looksLikeCalculatorMessage(normalized)) {
    return {
      meta: "Pripravljam izračun",
      phases: [
        { label: "Razumem vprašanje", active: false },
        { label: "Pripravljam kalkulator", active: true },
        { label: "Računam rezultat", active: false },
        { label: "Preverjam pravno podlago", active: false },
      ],
    };
  }
  return {
    meta: "Obdelujem vprašanje",
    phases: [
      { label: "Razumem vprašanje", active: true },
      { label: "Planiram odgovor", active: false },
      { label: "Iščem vire", active: false },
      { label: "Preverjam citate", active: false },
    ],
  };
}

function looksLikeCalculatorMessage(message) {
  return (
    (message.includes("ddv") || message.includes("dohodnin")) &&
    (
      message.includes("izračun") ||
      message.includes("izracun") ||
      message.includes("izračunaj") ||
      message.includes("izracunaj") ||
      message.includes("koliko znaša") ||
      message.includes("koliko znasa") ||
      message.includes("razred") ||
      message.includes("padem") ||
      message.includes("spadam") ||
      message.includes("%") ||
      message.includes("eur")
    )
  );
}

function renderCalculatorDrawer(calculatorResult) {
  const inputs = (calculatorResult.inputs || [])
    .map((item) => `<div class="calc-row"><span>${escapeHtml(item.label || "")}</span><strong>${escapeHtml(item.value || "")}</strong></div>`)
    .join("");
  const breakdown = (calculatorResult.breakdown || [])
    .map((item) => `<div class="calc-row"><span>${escapeHtml(item.label || "")}</span><strong>${escapeHtml(item.value || "")}</strong></div>`)
    .join("");
  const assumptions = (calculatorResult.assumptions || [])
    .map((item) => `<li>${escapeHtml(item)}</li>`)
    .join("");
  const missing = (calculatorResult.missing_params || [])
    .map((item) => `<span class="missing-pill">${escapeHtml(item)}</span>`)
    .join("");
  return `
    <section class="drawer-section">
      <p class="drawer-section-title">Kalkulator</p>
      <div class="calc-card">
        <p class="calc-title">${escapeHtml(calculatorResult.title || "Kalkulator")}</p>
        ${calculatorResult.result_summary ? `<p class="calc-summary">${escapeHtml(calculatorResult.result_summary)}</p>` : ""}
        ${inputs ? `<div class="calc-block"><p class="calc-block-title">Vhodni podatki</p><div class="calc-grid">${inputs}</div></div>` : ""}
        ${breakdown ? `<div class="calc-block"><p class="calc-block-title">Razčlenitev</p><div class="calc-grid">${breakdown}</div></div>` : ""}
        ${assumptions ? `<div class="calc-block"><p class="calc-block-title">Predpostavke</p><ul class="calc-list">${assumptions}</ul></div>` : ""}
        ${missing ? `<div class="calc-block"><p class="calc-block-title">Manjkajoči podatki</p><div class="missing-list">${missing}</div></div>` : ""}
      </div>
    </section>
  `;
}

function traceStatusLabel(status) {
  if (status === "verified") {
    return "preverjeno";
  }
  if (status === "weak") {
    return "pozor";
  }
  if (status === "missing") {
    return "manjka";
  }
  return "opravljeno";
}

boot();
