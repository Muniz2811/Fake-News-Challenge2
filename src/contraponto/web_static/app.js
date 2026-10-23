"use strict";

const $ = (selector) => document.querySelector(selector);
const state = { user: null, messages: [], version: -1, timer: null, sending: false, chats: [], chatId: "group", profiles: [], drafts: {} };
const input = $("#message-input");
const mobileLayout = window.matchMedia("(max-width: 700px)");

function updateNavigation() {
  const listVisible = mobileLayout.matches && !$(".app-shell").classList.contains("chat-open");
  $("#search").placeholder = listVisible ? "Buscar conversas" : "Buscar nesta conversa";
  $("#search").setAttribute("aria-label", listVisible ? "Buscar conversas" : "Buscar mensagens");
  filterMessages();
}

function showChats() {
  state.drafts[state.chatId] = input.value;
  input.blur();
  $(".app-shell").classList.remove("chat-open");
  $("#search").value = "";
  updateNavigation();
  renderChatList();
}

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function brandSymbol(className = "") {
  const image = node("img", `brand-symbol ${className}`);
  image.src = "/brand/Logotipo.png";
  image.alt = "";
  image.width = 73;
  image.height = 73;
  return image;
}

function icon(name) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", `#i-${name}`);
  svg.append(use);
  svg.setAttribute("aria-hidden", "true");
  return svg;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    credentials: "same-origin",
    ...options,
    headers: { "Content-Type": "application/json", ...options.headers },
  });
  if (response.status === 204) return null;
  const data = await response.json();
  if (!response.ok) {
    const error = new Error(typeof data.detail === "string" ? data.detail : "Confira os campos e tente novamente.");
    error.status = response.status;
    throw error;
  }
  return data;
}

function notice(text = "") {
  $("#notice").textContent = text;
  $("#notice").hidden = !text;
}

function showJoin() {
  state.user = null;
  state.version = -1;
  state.messages = [];
  state.drafts = {};
  input.value = "";
  clearTimeout(state.timer);
  $(".app-shell").classList.remove("chat-open");
  $("#messages").replaceChildren();
  $("#empty-state").hidden = false;
  $("#info-dialog").close();
  $("#join-profile").value = "";
  $("#group-consent").checked = false;
  $("#join-submit").disabled = true;
  if (!$("#join-dialog").open) $("#join-dialog").showModal();
  refreshProfiles().catch(() => {});
}

async function enter(user) {
  state.user = user;
  state.version = -1;
  $("#profile-name").textContent = user.name;
  for (const avatar of [$("#profile-avatar"), $("#rail-avatar")]) {
    avatar.textContent = user.id.slice(1);
    avatar.className = `avatar small personal profile-${user.id}`;
  }
  state.chats = (await api("/api/chats")).map((chat) => chat.id === "bot" ? { ...chat, name: "Contra" } : chat);
  $("#join-dialog").close();
  $("#join-code").value = "";
  switchChat("group");
  if (mobileLayout.matches) showChats();
}

async function refreshProfiles() {
  const config = await api("/api/config");
  $("#code-field").hidden = !config.code_required;
  $("#join-code").required = config.code_required;
  state.profiles = config.profiles;
  const selected = $("#join-profile").value;
  if (!config.profiles.some((profile) => profile.id === selected && !profile.online)) $("#join-profile").value = "";
  $("#profile-options").replaceChildren(...config.profiles.map((profile) => {
    const button = node("button", `profile-option${profile.id === $("#join-profile").value ? " active" : ""}`);
    button.type = "button";
    button.disabled = profile.online;
    button.setAttribute("aria-pressed", String(profile.id === $("#join-profile").value));
    button.append(node("span", `avatar small profile-${profile.id}`, profile.id.slice(1)));
    const label = node("div");
    label.append(node("strong", "", profile.name), node("small", "", profile.online ? "Em uso" : "Disponível"));
    button.append(label);
    button.addEventListener("click", () => {
      $("#join-profile").value = profile.id;
      for (const item of document.querySelectorAll(".profile-option")) {
        item.classList.toggle("active", item === button);
        item.setAttribute("aria-pressed", String(item === button));
      }
      $("#join-submit").disabled = false;
    });
    return button;
  }));
  $("#join-submit").disabled = !$("#join-profile").value;
}

function renderChatList() {
  $("#chat-list").replaceChildren(...state.chats.map((chat) => {
    const button = node("button", `room-card${chat.id === state.chatId ? " active" : ""}`);
    button.type = "button";
    button.dataset.search = chat.name.toLocaleLowerCase("pt-BR");
    button.setAttribute("aria-pressed", String(chat.id === state.chatId));
    const avatar = node("span", `avatar ${chat.type === "private" ? `profile-${chat.id}` : "room-avatar"}`);
    if (chat.type === "group") avatar.append(icon("users"));
    else if (chat.type === "bot") {
      avatar.classList.add("bot-avatar");
      avatar.append(brandSymbol());
    }
    else avatar.textContent = chat.id.slice(1);
    const label = node("span", "room-card-text");
    const online = state.profiles.find((profile) => profile.id === chat.id)?.online;
    label.append(node("strong", "", chat.name), node("span", "", chat.type === "group"
      ? "4 perfis · análise por solicitação" : chat.type === "bot" ? "Investigue no privado" : `${online ? "Online" : "Offline"} · conversa privada`));
    button.append(avatar, label);
    if (chat.id === state.chatId) button.append(node("span", "room-dot"));
    button.addEventListener("click", () => switchChat(chat.id));
    return button;
  }));
  filterMessages();
}

function switchChat(id) {
  const chat = state.chats.find((item) => item.id === id);
  if (!chat) return;
  state.drafts[state.chatId] = input.value;
  state.chatId = id;
  $(".app-shell").classList.add("chat-open");
  input.value = state.drafts[id] || "";
  input.style.height = "auto";
  state.messages = [];
  state.version = -1;
  clearTimeout(state.timer);
  $("#chat-title").textContent = chat.name;
  $("#header-auto").hidden = chat.type === "private";
  $("#header-auto").textContent = chat.type === "group" ? "Análise por solicitação" : "Verificação sob solicitação";
  $("#header-avatar").className = `avatar ${chat.type === "private" ? `profile-${id}` : chat.type === "bot" ? "bot-avatar" : "room-avatar"}`;
  $("#header-avatar").replaceChildren(chat.type === "group" ? icon("users")
    : chat.type === "bot" ? brandSymbol() : document.createTextNode(id.slice(1)));
  $("#context-text").textContent = chat.type === "private"
    ? `Conversa privada entre ${state.user.name} e ${chat.name}. Só o texto que você encaminhar ao Contra será analisado.`
    : chat.type === "group" ? "Nenhuma análise automática. Use “Analisar mensagem” no texto que quiser investigar."
      : "Envie ou encaminhe uma mensagem para investigar alegações e consultar fontes.";
  $("#empty-title").textContent = chat.type === "group" ? "A pergunta é o começo."
    : chat.type === "bot" ? "O que vamos investigar?" : `Sua conversa com ${chat.name}.`;
  $("#empty-description").textContent = chat.type === "private" ? "Troquem mensagens por aqui. Apenas os dois perfis têm acesso a esta conversa."
    : chat.type === "bot" ? "Envie uma notícia ou alegação. A verificação fica apenas neste chat com o seu perfil."
      : "Converse livremente. Toque em “Analisar mensagem” para investigar um texto e consultar evidências.";
  $("#empty-tags").hidden = chat.type === "private";
  $("#visibility-hint").textContent = chat.type === "group" ? "Análise só ao clicar · resultado visível ao grupo."
    : "Visível apenas para os perfis desta conversa.";
  input.placeholder = chat.type === "bot" ? "Envie uma notícia para verificar…" : `Mensagem para ${chat.name}…`;
  $("#search").value = "";
  updateNavigation();
  notice();
  renderChatList();
  renderMessages();
  poll();
  if (!mobileLayout.matches) input.focus();
}

function sourceLink(raw) {
  try {
    const url = new URL(raw);
    if (!["https:", "http:"].includes(url.protocol)) return null;
    const link = node("a", "", `${url.hostname.replace(/^www\./, "")} ↗`);
    link.href = url.href;
    link.title = url.href;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    return link;
  } catch { return null; }
}

function analysisCard(message, opened) {
  const card = node("section", "verification");
  card.setAttribute("aria-label", "Verificação do Contra");
  const heading = node("div", "bot-heading");
  heading.append(brandSymbol("mini-mark"), node("span", "", "Contra"), node("span", "bot-label", "VERIFICAÇÃO"));
  card.append(heading);
  if (message.status === "queued" || message.status === "verifying") {
    const progress = node("p", "progress-text");
    progress.append(node("span", "spinner"), node("span", "", message.status === "queued"
      ? "Na fila de verificações…" : "Investigando alegações e procurando evidências…"));
    card.append(progress);
    return card;
  }
  if (message.status === "error") {
    card.append(node("p", "warning", message.error));
    return card;
  }
  const result = message.analysis;
  if (!result) return card;
  card.append(node("p", "analysis-summary", result.claims.length
    ? `${result.claims.length} ${result.claims.length === 1 ? "alegação investigada" : "alegações investigadas"}. Confira as evidências abaixo.`
    : "Nenhuma alegação verificável identificada nesta mensagem."));
  for (const claim of result.claims) {
    const section = node("div", "claim");
    section.append(node("p", "claim-text", `“${claim.text}”`));
    const category = ["apoiada", "refutada", "conflitante"].includes(claim.category) ? claim.category : "";
    section.append(node("span", `verdict ${category}`, claim.label));
    // Todo conteúdo de usuário, motor ou LLM entra como texto, nunca como HTML.
    const rationale = claim.rationale.replace(/^\[ev_[0-9a-f]{64}\] /gm, "");
    section.append(node("p", "rationale", rationale));
    const sources = node("div", "sources");
    for (const source of claim.sources) {
      const link = sourceLink(source);
      if (link) sources.append(link);
    }
    if (sources.childElementCount) section.append(sources);
    card.append(section);
  }
  for (const warning of result.warnings) card.append(node("p", "warning", warning));
  const language = node("details", "language-details");
  language.dataset.messageId = message.id;
  language.open = opened.has(message.id);
  language.append(node("summary", "", "Análise da linguagem · separada da verificação factual"));
  language.append(node("p", "", result.language.text || "Score linguístico indisponível."));
  card.append(language);
  return card;
}

function messageAction(label, action) {
  const button = node("button", "message-action", label);
  button.type = "button";
  button.addEventListener("click", async () => {
    button.disabled = true;
    notice();
    try { await action(); }
    catch (error) {
      if (error.status === 401) showJoin();
      else notice(error.message);
    } finally { button.disabled = false; }
  });
  return button;
}

function renderMessages() {
  const timeline = $("#timeline");
  const nearBottom = timeline.scrollHeight - timeline.scrollTop - timeline.clientHeight < 100;
  const opened = new Set([...document.querySelectorAll("details[open][data-message-id]")].map((el) => el.dataset.messageId));
  const fragment = document.createDocumentFragment();
  for (const message of state.messages) {
    const own = message.author_id === state.user?.id;
    const group = node("article", `message-group${own ? " own" : ""}`);
    group.dataset.search = `${message.author} ${message.text}`.toLocaleLowerCase("pt-BR");
    const bubble = node("div", "message-bubble");
    bubble.append(node("span", "message-author", own ? "Você" : message.author));
    const textRow = node("div", "message-text-row");
    textRow.append(node("p", "message-text", message.text));
    bubble.append(textRow);
    if (state.chatId === "group" && ["sent", "error"].includes(message.status)) {
      bubble.append(messageAction(message.status === "error" ? "Tentar análise novamente" : "Analisar mensagem", async () => {
        await api(`/api/messages/${encodeURIComponent(message.id)}/verify`, { method: "POST" });
        await poll();
      }));
    }
    if (!["group", "bot"].includes(state.chatId)) {
      const chatId = state.chatId;
      bubble.append(messageAction("Encaminhar ao Contra", async () => {
        await api(`/api/messages/${encodeURIComponent(message.id)}/forward`, {
          method: "POST", body: JSON.stringify({ chat_id: chatId }),
        });
        switchChat("bot");
      }));
    }
    const meta = node("div", "message-meta", new Date(message.created_at).toLocaleTimeString("pt-BR", { hour: "2-digit", minute: "2-digit" }));
    if (own) meta.append(icon("check"));
    bubble.append(meta);
    group.append(bubble);
    if (["queued", "verifying", "done", "error"].includes(message.status)) group.append(analysisCard(message, opened));
    fragment.append(group);
  }
  $("#messages").replaceChildren(fragment);
  $("#empty-state").hidden = state.messages.length > 0;
  filterMessages();
  const busy = state.chatId === "bot" && state.messages.some((msg) => msg.author_id === state.user?.id && ["queued", "verifying"].includes(msg.status));
  $("#send").disabled = state.sending || busy;
  $("#send").title = busy ? "Aguarde a verificação da sua última mensagem" : "Enviar mensagem";
  if (nearBottom) timeline.scrollTop = timeline.scrollHeight;
}

function filterMessages() {
  const search = $("#search").value.trim().toLocaleLowerCase("pt-BR");
  const listVisible = mobileLayout.matches && !$(".app-shell").classList.contains("chat-open");
  document.querySelectorAll(".room-card").forEach((card) => {
    card.hidden = listVisible && Boolean(search) && !card.dataset.search.includes(search);
  });
  let visible = 0;
  document.querySelectorAll(".message-group").forEach((group) => {
    group.hidden = !listVisible && search && !group.dataset.search.includes(search);
    if (!group.hidden) visible++;
  });
  $("#no-results").hidden = listVisible || !search || visible > 0;
}

async function poll() {
  if (!state.user) return;
  const chatId = state.chatId;
  const userId = state.user.id;
  try {
    const data = await api(`/api/messages?chat_id=${encodeURIComponent(chatId)}&version=${state.version}`);
    if (state.user?.id !== userId || state.chatId !== chatId) return;
    if (data.version >= state.version && data.messages !== null) {
      state.version = data.version;
      state.messages = data.messages;
      renderMessages();
    }
    const profilesChanged = JSON.stringify(state.profiles) !== JSON.stringify(data.profiles);
    state.profiles = data.profiles;
    const chat = state.chats.find((item) => item.id === chatId);
    const count = data.profiles.filter((profile) => profile.online).length;
    const status = chat.type === "group" ? `${count} de 4 perfis online · grupo compartilhado`
      : chat.type === "bot" ? "Seu assistente de verificação · chat privado"
        : `${data.profiles.find((profile) => profile.id === chatId)?.online ? "Online" : "Offline"} · conversa privada`;
    $("#connection").replaceChildren(node("span", "connection-dot"), document.createTextNode(` ${status}`));
    $("#members").replaceChildren(...data.profiles.map((profile) => node("span", "", `${profile.name} · ${profile.online ? "online" : "offline"}`)));
    if (profilesChanged) renderChatList();
  } catch (error) {
    if (state.user?.id !== userId || state.chatId !== chatId) return;
    if (error.status === 401) { showJoin(); return; }
    $("#connection").textContent = "Reconectando à sala…";
  }
  clearTimeout(state.timer);
  if (state.user?.id === userId && state.chatId === chatId) state.timer = setTimeout(poll, 1800);
}

$("#join-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  $("#join-submit").disabled = true;
  $("#join-error").hidden = true;
  try {
    if (!$("#join-profile").value) throw new Error("Escolha um perfil disponível.");
    await enter(await api("/api/session", { method: "POST", body: JSON.stringify({ profile_id: $("#join-profile").value, access_code: $("#join-code").value, group_consent: $("#group-consent").checked }) }));
  } catch (error) {
    $("#join-error").textContent = error.message;
    $("#join-error").hidden = false;
    if (error.status === 409) await refreshProfiles().catch(() => {});
  } finally { $("#join-submit").disabled = !$("#join-profile").value; }
});

$("#join-dialog").addEventListener("cancel", (event) => event.preventDefault());
$("#composer").addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = input.value.trim();
  const chatId = state.chatId;
  if (!text || state.sending || $("#send").disabled) return;
  state.sending = true;
  $("#send").disabled = true;
  notice();
  try {
    await api("/api/messages", { method: "POST", body: JSON.stringify({ text, chat_id: chatId }) });
    state.drafts[chatId] = "";
    if (state.chatId === chatId && input.value.trim() === text) {
      input.value = "";
      input.style.height = "auto";
    }
    await poll();
    $("#timeline").scrollTop = $("#timeline").scrollHeight;
  } catch (error) {
    if (error.status === 401) showJoin();
    else notice(error.message);
  } finally {
    state.sending = false;
    renderMessages();
    if (!mobileLayout.matches) input.focus();
  }
});
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    $("#composer").requestSubmit();
  }
});
input.addEventListener("input", () => { input.style.height = "auto"; input.style.height = `${Math.min(input.scrollHeight, 160)}px`; });
$("#search").addEventListener("input", filterMessages);
for (const button of document.querySelectorAll(".info-trigger")) button.addEventListener("click", () => $("#info-dialog").showModal());
$("#close-info").addEventListener("click", () => $("#info-dialog").close());
for (const button of document.querySelectorAll("#logout, #logout-mobile")) button.addEventListener("click", async () => {
  try { await api("/api/session", { method: "DELETE" }); showJoin(); }
  catch { notice("Não foi possível sair agora. Tente novamente."); }
});
$(".rail-button.selected").addEventListener("click", () => switchChat("group"));
$("#back-to-chats").addEventListener("click", showChats);
mobileLayout.addEventListener("change", () => {
  $("#search").value = "";
  updateNavigation();
});
$("#refresh-profiles").addEventListener("click", () => refreshProfiles().catch(() => {
  $("#join-error").textContent = "Não consegui atualizar a disponibilidade.";
  $("#join-error").hidden = false;
}));

(async () => {
  try {
    await refreshProfiles();
    try { await enter(await api("/api/session")); }
    catch { showJoin(); }
  } catch {
    showJoin();
    $("#join-error").textContent = "Não consegui conectar ao servidor. Tente entrar novamente em instantes.";
    $("#join-error").hidden = false;
  }
})();
