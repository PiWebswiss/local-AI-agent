// Local AI Agent — web UI controller.
// Talks to web_server.py via JSON + Server-Sent Events for streamed phases.

(function () {
  "use strict";

  // --- helpers --------------------------------------------------------------
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  function escapeHtml(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  function renderMarkdown(text) {
    if (!text) return "";
    if (typeof window.marked !== "undefined") {
      try {
        window.marked.setOptions({ breaks: true, gfm: true });
        return window.marked.parse(String(text));
      } catch (e) { /* fall through */ }
    }
    return "<p>" + escapeHtml(text).replace(/\n/g, "<br/>") + "</p>";
  }

  let toastTimer = null;
  function toast(message, kind) {
    const el = $("#toast");
    el.textContent = message;
    el.className = "toast" + (kind ? " " + kind : "");
    el.hidden = false;
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.hidden = true; }, 3500);
  }

  function formatBytes(n) {
    if (!n && n !== 0) return "";
    if (n < 1024) return n + " B";
    if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " KB";
    if (n < 1024 * 1024 * 1024) return (n / (1024 * 1024)).toFixed(1) + " MB";
    return (n / (1024 * 1024 * 1024)).toFixed(2) + " GB";
  }

  /** Wire a file input + label to show the selected filename and enable drag-and-drop. */
  function bindFileDrop(inputEl, metaEl) {
    if (!inputEl) return;
    const drop = inputEl.closest(".file-drop");
    function update() {
      const f = inputEl.files && inputEl.files[0];
      if (!f) {
        if (metaEl) metaEl.textContent = "";
        if (drop) drop.classList.remove("has-file");
        return;
      }
      if (metaEl) metaEl.textContent = `${f.name} · ${formatBytes(f.size)}`;
      if (drop) drop.classList.add("has-file");
    }
    inputEl.addEventListener("change", update);
    if (drop) {
      ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => {
        e.preventDefault(); e.stopPropagation(); drop.classList.add("dragover");
      }));
      ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => {
        e.preventDefault(); e.stopPropagation(); drop.classList.remove("dragover");
      }));
      drop.addEventListener("drop", (e) => {
        const files = e.dataTransfer && e.dataTransfer.files;
        if (files && files.length) { inputEl.files = files; update(); }
      });
    }
    update();
  }

  async function api(path, opts = {}) {
    const init = Object.assign({ headers: { "Content-Type": "application/json" } }, opts);
    const resp = await fetch(path, init);
    if (!resp.ok) {
      let msg = resp.status + " " + resp.statusText;
      try { const data = await resp.json(); msg = data.detail || data.message || msg; } catch (_) {}
      throw new Error(msg);
    }
    return resp.json();
  }

  /** Stream Server-Sent Events from a POST endpoint with optional JSON or FormData body.
   *  `handlers.signal` (AbortSignal, optional) lets the caller cancel the stream.
   *  Resolves when the stream finishes (event:result / event:done) or rejects on abort/error. */
  function streamSSE(path, body, handlers) {
    return new Promise(async (resolve, reject) => {
      const init = { method: "POST" };
      if (body instanceof FormData) {
        init.body = body;
      } else if (body !== undefined) {
        init.headers = { "Content-Type": "application/json" };
        init.body = JSON.stringify(body);
      }
      if (handlers && handlers.signal) init.signal = handlers.signal;
      let resp;
      try { resp = await fetch(path, init); }
      catch (err) {
        if (err && err.name === "AbortError") reject(new Error("aborted"));
        else reject(err);
        return;
      }
      if (!resp.ok || !resp.body) {
        let msg = resp.status + " " + resp.statusText;
        try { const data = await resp.json(); msg = data.detail || data.message || msg; } catch (_) {}
        reject(new Error(msg));
        return;
      }
      const reader = resp.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";
      let finalResult = null;
      let errored = null;
      try {
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          let idx;
          while ((idx = buffer.indexOf("\n\n")) !== -1) {
            const chunk = buffer.slice(0, idx);
            buffer = buffer.slice(idx + 2);
            let evt = "message";
            const dataLines = [];
            for (const raw of chunk.split("\n")) {
              if (raw.startsWith("event:")) evt = raw.slice(6).trim();
              else if (raw.startsWith("data:")) dataLines.push(raw.slice(5).trim());
            }
            if (!dataLines.length) continue;
            let payload;
            try { payload = JSON.parse(dataLines.join("\n")); }
            catch (_) { payload = dataLines.join("\n"); }
            if (evt === "result" || evt === "done") finalResult = payload;
            if (evt === "error") errored = payload;
            try {
              if (handlers && typeof handlers[evt] === "function") handlers[evt](payload);
              else if (handlers && typeof handlers.message === "function") handlers.message(evt, payload);
            } catch (_) {}
          }
        }
      } catch (err) {
        if (err && err.name === "AbortError") { reject(new Error("aborted")); return; }
        reject(err); return;
      }
      if (errored) reject(new Error(errored.message || "Stream error"));
      else resolve(finalResult);
    });
  }

  // --- tabs -----------------------------------------------------------------
  $$(".nav-item").forEach((btn) => {
    btn.addEventListener("click", () => {
      const tab = btn.dataset.tab;
      $$(".nav-item").forEach((b) => b.classList.toggle("active", b === btn));
      $$(".tab").forEach((t) => t.classList.toggle("active", t.id === "tab-" + tab));
      if (tab === "books") refreshBooks();
      if (tab === "models") refreshModels();
      if (tab === "settings") refreshSettings();
    });
  });

  // --- theme ----------------------------------------------------------------
  const themeToggle = $("#themeToggle");
  const savedTheme = localStorage.getItem("agent.theme");
  if (savedTheme === "light") { document.body.classList.add("light"); themeToggle.checked = true; }
  themeToggle.addEventListener("change", () => {
    document.body.classList.toggle("light", themeToggle.checked);
    localStorage.setItem("agent.theme", themeToggle.checked ? "light" : "dark");
  });

  // --- chat -----------------------------------------------------------------
  const messages = $("#messages");
  const composer = $("#composer");
  const composerInput = $("#composerInput");
  const sendBtn = $("#sendBtn");
  const stopBtn = $("#stopBtn");
  const phaseHint = $("#phaseHint");
  let activeAbort = null;

  function addMessage(role, content, opts = {}) {
    const el = document.createElement("div");
    el.className = "msg " + role;
    const initial = role === "user" ? "You" : "AI";
    el.innerHTML = `
      <div class="msg-avatar">${initial}</div>
      <div class="msg-body">
        <div class="msg-meta"></div>
        <div class="msg-content"></div>
        <div class="msg-sources"></div>
      </div>`;
    messages.appendChild(el);
    if (opts.mode) {
      const tag = document.createElement("span");
      tag.className = "mode-tag";
      tag.textContent = opts.mode;
      $(".msg-meta", el).appendChild(tag);
    }
    setMessageContent(el, content || "");
    if (opts.sources && opts.sources.length) setMessageSources(el, opts.sources);
    messages.scrollTop = messages.scrollHeight;
    return el;
  }

  function setMessageContent(msgEl, text) {
    $(".msg-content", msgEl).innerHTML = renderMarkdown(text);
  }

  function setMessageSources(msgEl, sources) {
    const wrap = $(".msg-sources", msgEl);
    wrap.innerHTML = "";
    if (!sources || !sources.length) return;
    const det = document.createElement("details");
    const sum = document.createElement("summary");
    sum.textContent = `Sources (${sources.length})`;
    det.appendChild(sum);
    const ul = document.createElement("ul");
    for (const s of sources) {
      const li = document.createElement("li");
      const txt = String(s);
      if (/^https?:\/\//i.test(txt)) {
        const a = document.createElement("a");
        a.href = txt; a.target = "_blank"; a.rel = "noopener"; a.textContent = txt;
        li.appendChild(a);
      } else {
        li.textContent = txt;
      }
      ul.appendChild(li);
    }
    det.appendChild(ul);
    wrap.appendChild(det);
  }

  function setMode(msgEl, mode) {
    if (!mode) return;
    let tag = $(".msg-meta .mode-tag", msgEl);
    if (!tag) {
      tag = document.createElement("span");
      tag.className = "mode-tag";
      $(".msg-meta", msgEl).appendChild(tag);
    }
    tag.textContent = mode;
  }

  function setPhase(text) {
    if (!text) { phaseHint.classList.add("hidden"); phaseHint.textContent = ""; return; }
    phaseHint.classList.remove("hidden");
    phaseHint.textContent = text;
  }

  composer.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = composerInput.value.trim();
    if (!text) return;
    composerInput.value = "";
    composerInput.style.height = "auto";
    addMessage("user", text);
    sendBtn.hidden = true;
    stopBtn.hidden = false;
    const placeholder = addMessage("assistant", "_…thinking…_");
    setPhase("Understanding request…");

    // Live-stream buffer: token events append to this; once we finish (or
    // hit the final result), we re-render the message as markdown.
    let streamed = "";
    activeAbort = new AbortController();

    try {
      const result = await streamSSE("/api/chat", { message: text }, {
        signal: activeAbort.signal,
        status: (d) => setPhase(d.text || ""),
        token: (d) => {
          if (!d || typeof d.text !== "string") return;
          if (!streamed) setPhase("");
          streamed += d.text;
          // Plain-text append while streaming keeps it cheap; we re-render
          // markdown once the final result arrives.
          $(".msg-content", placeholder).textContent = streamed;
          messages.scrollTop = messages.scrollHeight;
        },
        result: (r) => {
          setPhase("");
          setMode(placeholder, r.mode || "chat");
          setMessageContent(placeholder, r.answer || streamed || "_(empty response)_");
          if (r.sources && r.sources.length) setMessageSources(placeholder, r.sources);
        },
        error: (e) => {
          setPhase("");
          setMessageContent(placeholder, "**Error:** " + escapeHtml(e.message || "unknown"));
        },
      });
      messages.scrollTop = messages.scrollHeight;
      refreshMemoryBadge();
    } catch (err) {
      setPhase("");
      if (err.message === "aborted") {
        if (streamed) {
          setMessageContent(placeholder, streamed + "\n\n_(stopped)_");
        } else {
          setMessageContent(placeholder, "_(stopped)_");
        }
      } else {
        setMessageContent(placeholder, "**Error:** " + escapeHtml(err.message || String(err)));
      }
    } finally {
      activeAbort = null;
      sendBtn.hidden = false;
      stopBtn.hidden = true;
      composerInput.focus();
    }
  });

  stopBtn.addEventListener("click", () => {
    if (activeAbort) {
      activeAbort.abort();
      toast("Stopped", "success");
    }
  });

  composerInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      composer.dispatchEvent(new Event("submit"));
    }
  });
  composerInput.addEventListener("input", () => {
    composerInput.style.height = "auto";
    composerInput.style.height = Math.min(200, composerInput.scrollHeight) + "px";
  });

  $$(".chip").forEach((c) => c.addEventListener("click", () => {
    const prefix = c.dataset.prefix || "";
    composerInput.value = prefix + composerInput.value;
    composerInput.focus();
    composerInput.dispatchEvent(new Event("input"));
  }));

  // --- voice input (WhisperX) ----------------------------------------------
  const micBtn = $("#micBtn");

  // User preference, separate from server-side capability. Tri-state:
  //   null      → user hasn't been asked yet → show the one-time modal
  //   "true"    → opted in
  //   "false"   → opted out
  // Mic button only appears when (server has whisperx) AND (user opted in).
  const VOICE_PREF_KEY = "agent.voiceOptIn";
  function getVoicePref() {
    const v = localStorage.getItem(VOICE_PREF_KEY);
    if (v === "true") return true;
    if (v === "false") return false;
    return null;
  }
  function setVoicePref(value) {
    localStorage.setItem(VOICE_PREF_KEY, value ? "true" : "false");
  }

  // Capability flag from /api/health, refreshed on each health probe.
  let voiceCapability = false;

  function applyMicVisibility() {
    if (!micBtn) return;
    const pref = getVoicePref();
    const enabled = voiceCapability && pref === true;
    micBtn.hidden = !enabled;
    if (enabled) {
      micBtn.title = "Hold to talk — transcribed by WhisperX";
    }
  }

  // One-time opt-in modal on first visit. The install-time question (in
  // start.cmd / start.sh) handles the docker-side opt-in; this modal is the
  // browser-side counterpart — only meaningful when WhisperX is actually
  // running, so we wait for the capability probe before deciding to show it.
  const voiceModal = $("#voiceOptInModal");
  function maybeShowVoiceModal() {
    if (!voiceModal) return;
    if (getVoicePref() !== null) return;
    if (!voiceCapability) return; // Nothing to enable — don't ask.
    voiceModal.hidden = false;
  }
  if (voiceModal) {
    $("#voiceOptInBtn").addEventListener("click", () => {
      setVoicePref(true);
      voiceModal.hidden = true;
      applyMicVisibility();
      const t = $("#voiceEnabledToggle");
      if (t) t.checked = true;
      toast("Voice input enabled", "success");
    });
    $("#voiceOptOutBtn").addEventListener("click", () => {
      setVoicePref(false);
      voiceModal.hidden = true;
      applyMicVisibility();
      const t = $("#voiceEnabledToggle");
      if (t) t.checked = false;
    });
  }
  let mediaRecorder = null;
  let mediaStream = null;
  let recordedChunks = [];

  function setMicState(state) {
    // state: "idle" | "recording" | "transcribing"
    micBtn.classList.toggle("recording", state === "recording");
    micBtn.classList.toggle("busy", state === "transcribing");
    const label = micBtn.querySelector(".mic-label");
    if (label) {
      label.textContent = state === "recording" ? "Stop" : state === "transcribing" ? "…" : "Talk";
    }
    micBtn.disabled = state === "transcribing";
  }

  function pickMimeType() {
    const candidates = [
      "audio/webm;codecs=opus",
      "audio/webm",
      "audio/ogg;codecs=opus",
      "audio/mp4",
    ];
    if (typeof MediaRecorder === "undefined") return "";
    for (const c of candidates) {
      if (MediaRecorder.isTypeSupported(c)) return c;
    }
    return "";
  }

  async function startRecording() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      toast("Microphone API not available in this browser", "error");
      return;
    }
    if (typeof MediaRecorder === "undefined") {
      toast("Recording not supported in this browser", "error");
      return;
    }
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      toast("Microphone access denied: " + (err.message || err), "error");
      return;
    }
    const mimeType = pickMimeType();
    try {
      mediaRecorder = mimeType
        ? new MediaRecorder(mediaStream, { mimeType })
        : new MediaRecorder(mediaStream);
    } catch (err) {
      toast("Could not start recorder: " + (err.message || err), "error");
      mediaStream.getTracks().forEach((t) => t.stop());
      mediaStream = null;
      return;
    }
    recordedChunks = [];
    mediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) recordedChunks.push(e.data);
    };
    mediaRecorder.onstop = async () => {
      const blobType = mediaRecorder.mimeType || mimeType || "audio/webm";
      if (mediaStream) {
        mediaStream.getTracks().forEach((t) => t.stop());
        mediaStream = null;
      }
      mediaRecorder = null;
      if (!recordedChunks.length) {
        setMicState("idle");
        return;
      }
      const blob = new Blob(recordedChunks, { type: blobType });
      recordedChunks = [];
      await uploadAndTranscribe(blob, blobType);
    };
    mediaRecorder.start();
    setMicState("recording");
    setPhase("Recording — click again to stop");
  }

  function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      mediaRecorder.stop();
    }
  }

  async function uploadAndTranscribe(blob, blobType) {
    setMicState("transcribing");
    setPhase("Transcribing voice…");
    const ext = (blobType.includes("ogg") ? "ogg" : blobType.includes("mp4") ? "m4a" : "webm");
    const fd = new FormData();
    fd.append("file", blob, "voice." + ext);
    try {
      const resp = await fetch("/api/transcribe", { method: "POST", body: fd });
      if (!resp.ok) {
        let msg = resp.status + " " + resp.statusText;
        try { const j = await resp.json(); msg = j.detail || j.message || msg; } catch (_) {}
        throw new Error(msg);
      }
      const data = await resp.json();
      const text = (data && typeof data.text === "string") ? data.text.trim() : "";
      if (!text) {
        toast("No speech detected", "error");
      } else {
        const current = composerInput.value;
        composerInput.value = current ? (current.replace(/\s+$/, "") + " " + text) : text;
        composerInput.dispatchEvent(new Event("input"));
        composerInput.focus();
      }
    } catch (err) {
      toast("Transcription failed: " + (err.message || err), "error");
    } finally {
      setPhase("");
      setMicState("idle");
    }
  }

  if (micBtn) {
    micBtn.addEventListener("click", () => {
      if (mediaRecorder && mediaRecorder.state === "recording") {
        stopRecording();
      } else {
        startRecording();
      }
    });
  }

  // --- memory + book sidebar -----------------------------------------------
  const memoryStatus = $("#memoryStatus");
  const clearMemoryBtn = $("#clearMemoryBtn");
  const activeBookLabel = $("#activeBookLabel");
  const bookOffBtn = $("#bookOffBtn");

  async function refreshMemoryBadge() {
    try {
      const data = await api("/api/memory");
      memoryStatus.textContent = `${data.turns} / ${data.max_turns} turns`;
    } catch (_) { memoryStatus.textContent = "—"; }
  }

  clearMemoryBtn.addEventListener("click", async () => {
    try {
      await api("/api/memory/clear", { method: "POST", body: "{}" });
      refreshMemoryBadge();
      toast("Memory cleared", "success");
    } catch (err) { toast("Failed: " + err.message, "error"); }
  });

  async function refreshActiveBook() {
    try {
      const data = await api("/api/books");
      const name = data.active;
      activeBookLabel.textContent = name || "None";
      bookOffBtn.hidden = !name;
    } catch (_) { activeBookLabel.textContent = "—"; }
  }

  bookOffBtn.addEventListener("click", async () => {
    try {
      await api("/api/books/off", { method: "POST", body: "{}" });
      refreshActiveBook();
      refreshBooks();
      toast("Book mode disabled", "success");
    } catch (err) { toast("Failed: " + err.message, "error"); }
  });

  // --- books tab ------------------------------------------------------------
  const booksList = $("#booksList");
  const indexForm = $("#indexForm");
  const indexFile = $("#indexFile");
  const indexName = $("#indexName");
  const indexPhases = $("#indexPhases");
  $("#refreshBooksBtn").addEventListener("click", refreshBooks);

  async function refreshBooks() {
    try {
      const data = await api("/api/books");
      const active = data.active;
      booksList.innerHTML = "";
      if (!data.books.length) {
        booksList.innerHTML = `<p class="muted">No books indexed yet. Upload a file below to get started.</p>`;
        return;
      }
      for (const name of data.books) {
        const row = document.createElement("div");
        row.className = "book-row" + (name === active ? " active" : "");
        row.innerHTML = `
          <div class="book-name">${escapeHtml(name)}</div>
          <div class="book-actions">
            <button class="ghost-btn small" data-act="load">${name === active ? "Active" : "Use"}</button>
          </div>`;
        $("button[data-act=load]", row).addEventListener("click", async () => {
          try {
            await api("/api/books/load", { method: "POST", body: JSON.stringify({ name }) });
            toast(`Loaded book: ${name}`, "success");
            refreshActiveBook();
            refreshBooks();
          } catch (err) { toast("Failed: " + err.message, "error"); }
        });
        booksList.appendChild(row);
      }
    } catch (err) {
      booksList.innerHTML = `<p class="muted">Error: ${escapeHtml(err.message)}</p>`;
    }
  }

  function pushPhase(target, text, kind) {
    target.classList.add("active");
    const line = document.createElement("div");
    line.className = "line" + (kind ? " " + kind : "");
    line.textContent = text;
    target.appendChild(line);
    target.scrollTop = target.scrollHeight;
  }

  indexForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!indexFile.files.length) return;
    indexPhases.innerHTML = "";
    pushPhase(indexPhases, "Uploading…");
    const fd = new FormData();
    fd.append("file", indexFile.files[0]);
    fd.append("name", indexName.value.trim());
    try {
      const result = await streamSSE("/api/books/index", fd, {
        status: (d) => pushPhase(indexPhases, d.text),
        result: (r) => pushPhase(indexPhases, `Indexed ${r.chunks} chunks → ${r.name}`, "ok"),
        error: (e) => pushPhase(indexPhases, "Error: " + e.message, "error"),
      });
      if (result) {
        toast(`Book indexed: ${result.name} (${result.chunks} chunks)`, "success");
        indexFile.value = "";
        indexName.value = "";
        refreshBooks();
        refreshActiveBook();
      }
    } catch (err) {
      pushPhase(indexPhases, "Error: " + err.message, "error");
      toast("Indexing failed: " + err.message, "error");
    }
  });

  // --- files tab ------------------------------------------------------------
  function bindFileTransform(opts) {
    opts.form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const file = opts.fileEl.files[0];
      const text = (opts.textEl ? opts.textEl.value : "").trim();
      if (!file && !text) { toast("Provide a file or text", "error"); return; }
      opts.phases.innerHTML = "";
      opts.result.classList.remove("active");
      opts.result.textContent = "";
      pushPhase(opts.phases, "Uploading…");
      const fd = new FormData();
      if (file) fd.append("file", file);
      if (text) fd.append("text", text);
      for (const [k, el] of Object.entries(opts.fields || {})) fd.append(k, el.value);
      try {
        const result = await streamSSE(opts.endpoint, fd, {
          status: (d) => pushPhase(opts.phases, d.text),
          result: (r) => {
            if (r.text) {
              opts.result.classList.add("active");
              if (opts.markdown) opts.result.innerHTML = renderMarkdown(r.text);
              else opts.result.textContent = r.text;
            }
            if (r.saved_path) {
              const name = r.download_name || r.saved_path.split(/[\\/]/).pop();
              pushPhase(opts.phases, `Saved: ${r.saved_path}`, "ok");
              const link = document.createElement("a");
              link.href = "/api/files/" + encodeURIComponent(name);
              link.textContent = "Download " + name;
              link.target = "_blank";
              const li = document.createElement("div");
              li.className = "line ok";
              li.appendChild(link);
              opts.phases.appendChild(li);
            }
          },
          error: (e) => pushPhase(opts.phases, "Error: " + e.message, "error"),
        });
        if (result) toast(opts.successLabel + " complete", "success");
      } catch (err) {
        pushPhase(opts.phases, "Error: " + err.message, "error");
        toast(opts.successLabel + " failed: " + err.message, "error");
      }
    });
  }

  bindFileTransform({
    form: $("#correctForm"),
    fileEl: $("#correctFile"),
    textEl: $("#correctText"),
    fields: { language: $("#correctLanguage") },
    endpoint: "/api/correct",
    phases: $("#correctPhases"),
    result: $("#correctResult"),
    markdown: false,
    successLabel: "Correction",
  });

  bindFileTransform({
    form: $("#summarizeForm"),
    fileEl: $("#summarizeFile"),
    textEl: $("#summarizeText"),
    fields: { language: $("#summarizeLanguage"), length: $("#summarizeLength") },
    endpoint: "/api/summarize",
    phases: $("#summarizePhases"),
    result: $("#summarizeResult"),
    markdown: true,
    successLabel: "Summary",
  });

  // --- models tab -----------------------------------------------------------
  const primaryModelsEl = $("#primaryModels");
  const committeeModelsEl = $("#committeeModels");
  const currentPrimary = $("#currentPrimary");
  const multiAgentToggle = $("#multiAgentToggle");
  const pullForm = $("#pullForm");
  const pullName = $("#pullName");
  const pullPhases = $("#pullPhases");
  $("#refreshModelsBtn").addEventListener("click", refreshModels);

  async function refreshModels() {
    try {
      const [models, settings] = await Promise.all([api("/api/models"), api("/api/settings")]);
      currentPrimary.textContent = models.primary || "—";
      multiAgentToggle.checked = !!settings.multi_agent;

      const installed = models.installed || [];
      primaryModelsEl.innerHTML = "";
      committeeModelsEl.innerHTML = "";

      if (!installed.length) {
        primaryModelsEl.innerHTML = `<p class="muted">No models installed. Pull one below.</p>`;
        committeeModelsEl.innerHTML = `<p class="muted">No models available.</p>`;
        return;
      }

      for (const m of installed) {
        const card = document.createElement("div");
        card.className = "model-card" + (m.name === models.primary ? " active" : "");
        const sizeMb = m.size_b ? (m.size_b / (1024 * 1024)).toFixed(0) + " MB" : "";
        card.innerHTML = `
          <div class="model-name">${escapeHtml(m.name)}</div>
          <div class="model-meta">${escapeHtml(sizeMb)}</div>`;
        card.addEventListener("click", async () => {
          try {
            await api("/api/settings", { method: "POST", body: JSON.stringify({ primary_model: m.name }) });
            toast("Primary model: " + m.name, "success");
            refreshModels();
            refreshHealth();
          } catch (err) { toast("Failed: " + err.message, "error"); }
        });
        primaryModelsEl.appendChild(card);
      }

      const committee = new Set(models.committee || []);
      for (const m of installed) {
        const card = document.createElement("label");
        card.className = "model-card committee-card" + (committee.has(m.name) ? " active" : "");
        card.innerHTML = `
          <input type="checkbox" ${committee.has(m.name) ? "checked" : ""} />
          <div>
            <div class="model-name">${escapeHtml(m.name)}</div>
            <div class="model-meta">${committee.has(m.name) ? "in committee" : "not in committee"}</div>
          </div>`;
        const cb = $("input", card);
        cb.addEventListener("change", async () => {
          if (cb.checked) committee.add(m.name); else committee.delete(m.name);
          try {
            await api("/api/settings", {
              method: "POST",
              body: JSON.stringify({ multi_models: Array.from(committee) }),
            });
            refreshModels();
          } catch (err) {
            toast("Failed: " + err.message, "error");
            cb.checked = !cb.checked;
          }
        });
        committeeModelsEl.appendChild(card);
      }
    } catch (err) {
      primaryModelsEl.innerHTML = `<p class="muted">Error: ${escapeHtml(err.message)}</p>`;
    }
  }

  multiAgentToggle.addEventListener("change", async () => {
    try {
      await api("/api/settings", {
        method: "POST",
        body: JSON.stringify({ multi_agent: multiAgentToggle.checked }),
      });
      toast("Multi-agent: " + (multiAgentToggle.checked ? "on" : "off"), "success");
    } catch (err) {
      toast("Failed: " + err.message, "error");
      multiAgentToggle.checked = !multiAgentToggle.checked;
    }
  });

  pullForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const name = pullName.value.trim();
    if (!name) return;
    pullPhases.innerHTML = "";
    pushPhase(pullPhases, `Pulling ${name}…`);
    try {
      await streamSSE("/api/models/pull", { name }, {
        status: (d) => {
          const status = d.status || "";
          const total = d.total || 0;
          const completed = d.completed || 0;
          let line = status;
          if (total > 0) {
            const pct = Math.floor((completed / total) * 100);
            line += ` (${pct}%)`;
          }
          pushPhase(pullPhases, line || JSON.stringify(d));
        },
        done: () => pushPhase(pullPhases, `Done: ${name}`, "ok"),
        error: (e) => pushPhase(pullPhases, "Error: " + e.message, "error"),
      });
      toast(`Model pulled: ${name}`, "success");
      pullName.value = "";
      refreshModels();
    } catch (err) {
      pushPhase(pullPhases, "Error: " + err.message, "error");
      toast("Pull failed: " + err.message, "error");
    }
  });

  // --- settings tab ---------------------------------------------------------
  const qualityToggle = $("#qualityToggle");
  const smartToggle = $("#smartToggle");
  const roleToggle = $("#roleToggle");
  const peerToggle = $("#peerToggle");
  const simpleN = $("#simpleN");
  const mediumN = $("#mediumN");
  const hardN = $("#hardN");
  const memCompressToggle = $("#memCompressToggle");
  const memCompressThreshold = $("#memCompressThreshold");
  const memKeepRecent = $("#memKeepRecent");
  const subagentsToggle = $("#subagentsToggle");
  const subagentsMax = $("#subagentsMax");
  const hostField = $("#hostField");
  const primaryField = $("#primaryField");
  const saveSettingsBtn = $("#saveSettingsBtn");
  const qualityBadge = $("#qualityBadge");
  const modelBadge = $("#modelBadge");

  async function refreshSettings() {
    try {
      const s = await api("/api/settings");
      qualityToggle.checked = !!s.quality_mode;
      smartToggle.checked = !!s.multi_smart;
      roleToggle.checked = !!s.multi_role_mode;
      peerToggle.checked = !!s.multi_peer_review;
      simpleN.value = s.multi_simple_models;
      mediumN.value = s.multi_medium_models;
      hardN.value = s.multi_hard_models;
      if (memCompressToggle) memCompressToggle.checked = !!s.memory_compress;
      if (memCompressThreshold) memCompressThreshold.value = s.memory_compress_threshold;
      if (memKeepRecent) memKeepRecent.value = s.memory_keep_recent_turns;
      if (subagentsToggle) subagentsToggle.checked = !!s.subagents;
      if (subagentsMax) subagentsMax.value = s.subagents_max;
      hostField.value = s.host;
      primaryField.value = s.primary_model;
      qualityBadge.textContent = "quality: " + (s.quality_mode ? "on" : "off");
      modelBadge.textContent = "model: " + s.primary_model;
    } catch (err) { toast("Settings load failed: " + err.message, "error"); }
  }

  saveSettingsBtn.addEventListener("click", async () => {
    try {
      await api("/api/settings", {
        method: "POST",
        body: JSON.stringify({
          quality_mode: qualityToggle.checked,
          multi_smart: smartToggle.checked,
          multi_role_mode: roleToggle.checked,
          multi_peer_review: peerToggle.checked,
          multi_simple_models: parseInt(simpleN.value, 10) || 1,
          multi_medium_models: parseInt(mediumN.value, 10) || 2,
          multi_hard_models: parseInt(hardN.value, 10) || 0,
          memory_compress: memCompressToggle ? memCompressToggle.checked : undefined,
          memory_compress_threshold: memCompressThreshold ? (parseInt(memCompressThreshold.value, 10) || 8000) : undefined,
          memory_keep_recent_turns: memKeepRecent ? (parseInt(memKeepRecent.value, 10) || 4) : undefined,
          subagents: subagentsToggle ? subagentsToggle.checked : undefined,
          subagents_max: subagentsMax ? (parseInt(subagentsMax.value, 10) || 3) : undefined,
        }),
      });
      toast("Settings saved", "success");
      refreshSettings();
    } catch (err) { toast("Save failed: " + err.message, "error"); }
  });

  // --- file inputs (drag-and-drop + filename display) ----------------------
  bindFileDrop($("#indexFile"), $("#indexFileMeta"));
  bindFileDrop($("#correctFile"), $("#correctFileMeta"));
  bindFileDrop($("#summarizeFile"), $("#summarizeFileMeta"));

  // --- health / startup -----------------------------------------------------
  async function refreshHealth() {
    try {
      const h = await api("/api/health");
      const s = $("#brandStatus");
      s.textContent = h.ok ? `Connected · ${h.primary_model}` : "Ollama unreachable";
      s.style.color = h.ok ? "" : "var(--bad)";
      modelBadge.textContent = "model: " + h.primary_model;
      voiceCapability = !!(h.capabilities && h.capabilities.transcription);
      applyMicVisibility();
      updateVoiceCapabilityNote();
      maybeShowVoiceModal();
    } catch (_) {
      $("#brandStatus").textContent = "Server offline";
      $("#brandStatus").style.color = "var(--bad)";
      voiceCapability = false;
      applyMicVisibility();
      updateVoiceCapabilityNote();
    }
  }

  function updateVoiceCapabilityNote() {
    const note = $("#voiceCapabilityNote");
    if (!note) return;
    note.textContent = voiceCapability
      ? "WhisperX is reachable and ready."
      : "WhisperX service is not reachable yet — it may still be downloading model weights on first boot.";
  }

  // Settings: voice-input toggle. This is a browser-side preference (per-user,
  // per-browser); the server-side capability flag still gates the mic button.
  const voiceEnabledToggle = $("#voiceEnabledToggle");
  if (voiceEnabledToggle) {
    const pref = getVoicePref();
    voiceEnabledToggle.checked = pref === true;
    voiceEnabledToggle.addEventListener("change", () => {
      setVoicePref(voiceEnabledToggle.checked);
      applyMicVisibility();
      toast(
        "Voice input " + (voiceEnabledToggle.checked ? "enabled" : "disabled"),
        "success"
      );
    });
  }

  refreshHealth();
  refreshActiveBook();
  refreshMemoryBadge();
  refreshSettings();
  // Note: voice opt-in modal is triggered from inside refreshHealth(), once
  // we know whether WhisperX is actually reachable.
})();
