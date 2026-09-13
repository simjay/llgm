"use strict";
const $ = (id) => document.getElementById(id);
const svgNS = "http://www.w3.org/2000/svg";
let snapshot = null,
  selected = null,
  conversation = null,
  tab = "turns",
  requestId = 0,
  searchId = 0;
let matches = new Set(),
  positions = new Map(),
  camera = { x: 0, y: 0, k: 1 },
  refreshBusy = false;
function el(tag, className, text) {
  const item = document.createElement(tag);
  if (className) item.className = className;
  if (text !== undefined) item.textContent = text;
  return item;
}
function svg(tag, attrs) {
  const item = document.createElementNS(svgNS, tag);
  for (const [key, value] of Object.entries(attrs || {}))
    item.setAttribute(key, value);
  return item;
}
function notify(message) {
  $("notice").textContent = message;
  $("notice").hidden = !message;
}
function run(action) {
  return (...args) =>
    Promise.resolve()
      .then(() => action(...args))
      .catch((error) => notify(error.message));
}
async function api(route, params = {}) {
  const response = await fetch(
    "api/" + route + "?" + new URLSearchParams(params),
    { cache: "no-store" },
  );
  let result;
  try {
    result = await response.json();
  } catch {
    throw new Error(
      "The viewer is unavailable. Keep the LLGM viewer process running.",
    );
  }
  if (!response.ok) throw new Error(result.error || "Unable to read workspace");
  return result;
}
function fileLink(kind, nodeId, recordId, label) {
  const link = el("a", "file-link", label);
  link.href =
    "file?" +
    new URLSearchParams({ kind, node_id: nodeId, record_id: recordId || "" });
  link.rel = "noopener noreferrer";
  return link;
}
function short(id) {
  return id.length > 24 ? id.slice(0, 19) + "…" : id;
}
function applyCamera() {
  $("world").setAttribute(
    "transform",
    `translate(${camera.x} ${camera.y}) scale(${camera.k})`,
  );
}
function fit() {
  const rect = $("graph").getBoundingClientRect(),
    points = [...positions.values()];
  if (!points.length) return;
  const left = Math.min(...points.map((p) => p.x)) - 95,
    right = Math.max(...points.map((p) => p.x)) + 95,
    top = Math.min(...points.map((p) => p.y)) - 45,
    bottom = Math.max(...points.map((p) => p.y)) + 70;
  camera.k = Math.min(
    1.5,
    (rect.width - 40) / (right - left),
    (rect.height - 210) / (bottom - top),
  );
  camera.x = rect.width / 2 - ((left + right) / 2) * camera.k;
  camera.y = (rect.height + 50) / 2 - ((top + bottom) / 2) * camera.k;
  applyCamera();
}
function zoom(factor) {
  const rect = $("graph").getBoundingClientRect();
  const old = camera.k;
  camera.k = Math.max(0.08, Math.min(4, old * factor));
  camera.x = rect.width / 2 - ((rect.width / 2 - camera.x) * camera.k) / old;
  camera.y = rect.height / 2 - ((rect.height / 2 - camera.y) * camera.k) / old;
  applyCamera();
}
function arrange() {
  const ordered = [...snapshot.nodes].sort(
    (a, b) =>
      (b.node_id === snapshot.current_node_id) -
      (a.node_id === snapshot.current_node_id),
  );
  const count = ordered.length;
  ordered.forEach((node, i) => {
    if (positions.has(node.node_id)) return;
    if (i === 0) positions.set(node.node_id, { x: 0, y: 0 });
    else {
      const angle =
        -Math.PI / 2 + ((i - 1) / Math.max(1, count - 1)) * 2 * Math.PI;
      const radius = Math.max(175, (count - 1) * 32);
      positions.set(node.node_id, {
        x: Math.cos(angle) * radius,
        y: Math.sin(angle) * radius,
      });
    }
  });
  for (const id of positions.keys())
    if (!snapshot.nodes.some((n) => n.node_id === id)) positions.delete(id);
}
function nodeClasses(id) {
  return [
    "graph-node",
    id === snapshot.current_node_id ? "current" : "",
    matches.has(id) ? "match" : "",
    id === selected ? "selected" : "",
  ].join(" ");
}
function highlight() {
  $("nodes")
    .querySelectorAll("g.graph-node")
    .forEach((g) => g.setAttribute("class", nodeClasses(g.dataset.id)));
  $("node-list")
    .querySelectorAll("button")
    .forEach((b) => b.classList.toggle("active", b.dataset.id === selected));
}
function drawEdges() {
  $("edges").replaceChildren();
  const pairs = new Map();
  for (const edge of snapshot.edges) {
    const from = positions.get(edge.source_node_id),
      to = positions.get(edge.target_node_id);
    if (!from || !to) continue;
    const key = [edge.source_node_id, edge.target_node_id].sort().join("\0");
    const index = pairs.get(key) || 0;
    pairs.set(key, index + 1);
    let path;
    if (from === to)
      path = `M ${from.x - 17} ${from.y - 12} C ${from.x - 70} ${from.y - 95},${from.x + 70} ${from.y - 95},${from.x + 19} ${from.y - 13}`;
    else {
      const dx = to.x - from.x,
        dy = to.y - from.y,
        length = Math.hypot(dx, dy) || 1;
      const bend = index % 2 ? 34 * Math.ceil(index / 2) : -12 * index;
      path = `M ${from.x + (dx / length) * 25} ${from.y + (dy / length) * 25} Q ${(from.x + to.x) / 2 - (dy / length) * bend} ${(from.y + to.y) / 2 + (dx / length) * bend},${to.x - (dx / length) * 29} ${to.y - (dy / length) * 29}`;
    }
    const line = svg("path", { d: path, class: "edge" });
    const title = svg("title");
    title.textContent = `${edge.source_node_id} → ${edge.target_node_id}\n${edge.edge_id}`;
    line.append(title);
    $("edges").append(line);
  }
}
function drawGraph() {
  arrange();
  drawEdges();
  $("nodes").replaceChildren();
  for (const node of snapshot.nodes) {
    const p = positions.get(node.node_id),
      g = svg("g", {
        class: nodeClasses(node.node_id),
        transform: `translate(${p.x} ${p.y})`,
        tabindex: "0",
        role: "button",
        "aria-label": `Open node ${node.node_id}${node.node_id === snapshot.current_node_id ? ", current topic" : ""}`,
      });
    g.dataset.id = node.node_id;
    const title = svg("title");
    title.textContent = node.node_id;
    g.append(
      title,
      svg("circle", { r: 30, class: "halo" }),
      svg("circle", { r: 25, class: "body" }),
    );
    const icon = svg("text", { y: 6, class: "center-icon" });
    icon.textContent = node.node_id === snapshot.current_node_id ? "↳" : "·";
    const label = svg("text", { y: 48, class: "node-label" });
    label.textContent = short(node.node_id);
    const count = svg("text", { y: 64, class: "node-count" });
    count.textContent = node.journal_count
      ? `${node.journal_count} journal ${node.journal_count === 1 ? "entry" : "entries"}`
      : "";
    g.append(icon, label, count);
    g.addEventListener(
      "keydown",
      run((event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          return openNode(node.node_id);
        }
      }),
    );
    $("nodes").append(g);
  }
  $("empty-graph").hidden = snapshot.nodes.length > 0;
}
function drawExplorer() {
  $("node-list").replaceChildren();
  for (const node of snapshot.nodes) {
    const button = el("button", "node-row");
    button.dataset.id = node.node_id;
    button.title = node.node_id;
    button.append(
      el(
        "i",
        "dot " +
          (node.node_id === snapshot.current_node_id
            ? "current"
            : matches.has(node.node_id)
              ? "match"
              : "ordinary"),
      ),
      el("span", "name", node.node_id),
    );
    if (node.node_id === snapshot.current_node_id)
      button.append(el("span", "current-tag", "CURRENT"));
    button.addEventListener(
      "click",
      run(() => openNode(node.node_id)),
    );
    $("node-list").append(button);
  }
  highlight();
}
async function refresh(manual = false) {
  if (refreshBusy) return;
  refreshBusy = true;
  const requestedConversation = conversation;
  let stale = false;
  try {
    const next = await api(
      "graph",
      conversation === null ? {} : { conversation_id: conversation },
    );
    if (conversation !== requestedConversation) {
      stale = true;
      return;
    }
    const initial = snapshot === null,
      changed = JSON.stringify(next) !== JSON.stringify(snapshot);
    snapshot = next;
    conversation = next.conversation_id;
    if (changed) {
      const options = new Set([
        conversation,
        ...snapshot.conversations.map((c) => c.conversation_id),
      ]);
      $("conversation").replaceChildren(
        ...[...options].map((id) => {
          const option = el("option", "", id);
          option.value = id;
          return option;
        }),
      );
      $("conversation").value = conversation;
      $("node-count").textContent = snapshot.nodes.length;
      $("graph-summary").textContent =
        `${snapshot.nodes.length} nodes · ${snapshot.edges.length} active connections`;
      $("database").textContent = snapshot.database;
      $("database").title = snapshot.database;
      $("current-label").textContent = snapshot.current_node_id
        ? "Current: " + snapshot.current_node_id
        : "No current topic";
      $("current-label").title =
        snapshot.current_node_id || "No saved topic for this conversation";
      $("current-button").disabled = !snapshot.current_node_id;
      drawGraph();
      drawExplorer();
      const badge = $("inspector").querySelector(".detail-status");
      badge?.remove();
      if (selected === snapshot.current_node_id) {
        $("inspector")
          .querySelector(".detail-id")
          ?.after(
            el("span", "detail-status", "● Current topic in " + conversation),
          );
      }
      if (initial) fit();
    }
    if (manual) {
      notify("");
      if (selected) await openNode(selected);
    }
  } finally {
    refreshBusy = false;
    if (stale) run(() => refresh())();
  }
}
function makeRecord(summary, id, load) {
  const details = el("details", "record"),
    heading = el("summary", "", summary);
  heading.append(el("span", "record-id", id));
  const body = el("div", "record-body");
  details.append(heading, body);
  details.addEventListener(
    "toggle",
    run(async () => {
      if (!details.open || details.dataset.loaded) return;
      details.dataset.loaded = "yes";
      body.textContent = "Loading…";
      try {
        await load(body);
      } catch (error) {
        delete details.dataset.loaded;
        body.textContent = error.message;
      }
    }),
  );
  return details;
}
async function turnBody(body, nodeId, turn) {
  let offset = 0;
  const pre = el("pre"),
    more = el("button", "more", "Read next section");
  const position = el("p", "small-note");
  const next = async () => {
    more.disabled = true;
    try {
      const end = Math.min(turn.length, offset + 16384);
      const response = await api("turn", {
        node_id: nodeId,
        turn_id: turn.turn_id,
        offset,
        end,
      });
      pre.textContent = response.text;
      position.textContent = `Characters ${offset.toLocaleString()}–${end.toLocaleString()} of ${turn.length.toLocaleString()}`;
      offset = end;
      more.hidden = offset >= turn.length;
    } finally {
      more.disabled = false;
    }
  };
  more.addEventListener("click", run(next));
  body.replaceChildren(
    pre,
    position,
    more,
    el("p", "storage-location", turn.location),
    fileLink(
      turn.storage,
      nodeId,
      turn.turn_id,
      turn.storage === "turn"
        ? "Open source file ↗"
        : "Open source manifest ↗",
    ),
  );
  await next();
}
async function openNode(nodeId, nextTab = tab) {
  selected = nodeId;
  tab = nextTab;
  const ticket = ++requestId;
  highlight();
  const target = $("inspector");
  target.replaceChildren(el("p", "small-note", "Loading node…"));
  const page = await api("node", { node_id: nodeId });
  if (ticket !== requestId) return;
  const heading = el("div", "inspector-heading");
  heading.append(el("span", "eyebrow", "NODE INSPECTOR"));
  const reload = el("button", "", "Reload");
  reload.addEventListener(
    "click",
    run(() => openNode(nodeId)),
  );
  heading.append(reload);
  const title = el("h2", "detail-id", nodeId);
  target.replaceChildren(heading, title);
  if (nodeId === snapshot.current_node_id)
    target.append(
      el("span", "detail-status", "● Current topic in " + conversation),
    );
  target.append(
    el(
      "p",
      "small-note",
      `${page.total_turns.toLocaleString()} ${page.total_turns === 1 ? "turn" : "turns"} · original evidence`,
    ),
  );
  const tabs = el("div", "tabs");
  for (const name of ["turns", "journals", "connections", "files"]) {
    const button = el(
      "button",
      name === tab ? "active" : "",
      name[0].toUpperCase() + name.slice(1),
    );
    button.addEventListener(
      "click",
      run(() => openNode(nodeId, name)),
    );
    tabs.append(button);
  }
  const content = el("div");
  target.append(tabs, content);
  if (tab === "turns") {
    const append = async (data) => {
      for (const turn of data.turns)
        content.append(
          makeRecord(
            `${turn.role} · ${turn.length.toLocaleString()} characters`,
            turn.turn_id,
            (body) => turnBody(body, nodeId, turn),
          ),
        );
      if (data.next_offset !== null) {
        const more = el("button", "more", "Load more turns");
        more.addEventListener(
          "click",
          run(async () => {
            more.disabled = true;
            try {
              const next = await api("node", {
                node_id: nodeId,
                offset: data.next_offset,
              });
              more.remove();
              await append(next);
            } catch (error) {
              more.disabled = false;
              throw error;
            }
          }),
        );
        content.append(more);
      }
    };
    await append(page);
    if (!page.total_turns)
      content.append(el("p", "small-note", "No turns in this node."));
  }
  if (tab === "journals") {
    content.append(
      el(
        "p",
        "small-note",
        "Retained amendment history. These records live in SQLite, and open as JSON.",
      ),
    );
    const append = async (offset) => {
      const data = await api("journals", { node_id: nodeId, offset });
      for (const entry of data.entries)
        content.append(
          makeRecord(
            `#${entry.journal_sequence} · ${entry.record_kind}`,
            entry.entry_id,
            async (body) => {
              const value = await api("journal", {
                node_id: nodeId,
                entry_id: entry.entry_id,
              });
              let text = value.text;
              if (!value.truncated)
                text = JSON.stringify(JSON.parse(text), null, 2);
              body.replaceChildren(
                el("pre", "", text),
                fileLink(
                  "journal",
                  nodeId,
                  entry.entry_id,
                  "Open journal JSON ↗",
                ),
              );
              if (value.truncated)
                body.append(
                  el(
                    "p",
                    "small-note",
                    "Preview truncated. Open the JSON to see the complete record.",
                  ),
                );
            },
          ),
        );
      if (data.next_offset !== null) {
        const more = el("button", "more", "Load more journal entries");
        more.addEventListener(
          "click",
          run(async () => {
            more.disabled = true;
            try {
              await append(data.next_offset);
              more.remove();
            } catch (error) {
              more.disabled = false;
              throw error;
            }
          }),
        );
        content.append(more);
      }
      if (!data.entries.length && !offset)
        content.append(el("p", "small-note", "No journal entries."));
    };
    await append(0);
  }
  if (tab === "connections") {
    const edges = snapshot.edges.filter(
      (edge) =>
        edge.source_node_id === nodeId || edge.target_node_id === nodeId,
    );
    for (const edge of edges) {
      const outgoing = edge.source_node_id === nodeId,
        other = outgoing ? edge.target_node_id : edge.source_node_id;
      const button = el(
        "button",
        "connection",
        (outgoing ? "→ " : "← ") + other,
      );
      button.title = edge.edge_id;
      button.addEventListener(
        "click",
        run(() => openNode(other)),
      );
      content.append(button);
    }
    if (!edges.length)
      content.append(el("p", "small-note", "No active connections."));
  }
  if (tab === "files") {
    content.append(
      el("p", "small-note", "Source manifest"),
      el("p", "storage-location", page.manifest_location),
      fileLink("manifest", nodeId, "", "Open source manifest ↗"),
      el(
        "p",
        "small-note",
        "Appended turn files are available under Turns. The manifest contains imported base turns and source metadata.",
      ),
      el("p", "small-note", "Journal storage"),
      el(
        "p",
        "storage-location",
        snapshot.database + "\nTable: journal_entries\nNode: " + nodeId,
      ),
      el(
        "p",
        "small-note",
        "Open individual journal records from the Journals tab.",
      ),
    );
  }
}
$("search-form").addEventListener(
  "submit",
  run(async (event) => {
    event.preventDefault();
    const query = $("search").value.trim();
    if (!query) return;
    const ticket = ++searchId;
    $("results").hidden = false;
    $("search-summary").textContent = "Searching the evidence backend…";
    $("result-list").replaceChildren();
    const data = await api("search", { q: query });
    if (ticket !== searchId) return;
    matches = new Set(data.nodes.map((node) => node.node_id));
    highlight();
    drawExplorer();
    const backend = data.backend.source_retriever || data.backend;
    const backendName =
      backend.backend === "H"
        ? "Hybrid · BM25 + ColBERTv2"
        : !data.backend.source_retriever
          ? "BM25"
          : backend.implementation || backend.backend || "configured retriever";
    $("search-summary").textContent =
      `${data.nodes.length} matching nodes · ${backendName} · ranked passages grouped by node`;
    for (const node of data.nodes) {
      const hit = node.matches[0];
      const result = el("button", "result");
      result.append(
        el("strong", "", node.node_id),
        el("p", "", hit.text),
        el(
          "small",
          "",
          `Passage rank ${hit.rank} · score ${Number(hit.score).toPrecision(4)}`,
        ),
      );
      result.addEventListener(
        "click",
        run(() => openNode(node.node_id)),
      );
      $("result-list").append(result);
    }
    if (!data.nodes.length)
      $("result-list").append(
        el("p", "small-note", "No matching evidence. Try another term."),
      );
    notify("");
  }),
);
$("clear-search").addEventListener("click", () => {
  ++searchId;
  matches.clear();
  $("results").hidden = true;
  $("search").value = "";
  highlight();
  drawExplorer();
});
$("conversation").addEventListener(
  "change",
  run(async () => {
    conversation = $("conversation").value;
    await refresh();
    if (selected) await openNode(selected);
  }),
);
$("refresh").addEventListener(
  "click",
  run(() => refresh(true)),
);
$("current-button").addEventListener(
  "click",
  run(() => snapshot.current_node_id && openNode(snapshot.current_node_id)),
);
$("fit").addEventListener("click", fit);
$("zoom-in").addEventListener("click", () => zoom(1.2));
$("zoom-out").addEventListener("click", () => zoom(1 / 1.2));
$("graph").addEventListener(
  "wheel",
  (event) => {
    event.preventDefault();
    zoom(event.deltaY < 0 ? 1.1 : 1 / 1.1);
  },
  { passive: false },
);
let drag = null;
$("graph").addEventListener("pointerdown", (event) => {
  const g = event.target.closest(".graph-node");
  drag = {
    id: g?.dataset.id,
    x: event.clientX,
    y: event.clientY,
    moved: false,
  };
  $("graph").setPointerCapture(event.pointerId);
});
$("graph").addEventListener("pointermove", (event) => {
  if (!drag) return;
  const dx = event.clientX - drag.x,
    dy = event.clientY - drag.y;
  if (Math.abs(dx) + Math.abs(dy) > 2) drag.moved = true;
  if (drag.id) {
    const p = positions.get(drag.id);
    p.x += dx / camera.k;
    p.y += dy / camera.k;
    const g = [...$("nodes").children].find(
      (node) => node.dataset.id === drag.id,
    );
    g.setAttribute("transform", `translate(${p.x} ${p.y})`);
    drawEdges();
  } else {
    camera.x += dx;
    camera.y += dy;
    applyCamera();
  }
  drag.x = event.clientX;
  drag.y = event.clientY;
});
$("graph").addEventListener(
  "pointerup",
  run(async (event) => {
    const previous = drag;
    drag = null;
    $("graph").releasePointerCapture(event.pointerId);
    if (previous?.id && !previous.moved) await openNode(previous.id);
  }),
);
$("graph").addEventListener("pointercancel", () => {
  drag = null;
});
window.addEventListener("resize", fit);
run(() => refresh())();
setInterval(() => {
  if (!document.hidden) run(() => refresh())();
}, 5000);
