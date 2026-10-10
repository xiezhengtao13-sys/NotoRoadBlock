"""Build the blind manual-review page for dataset_noto2024_v0.

Review set: every candidate-blocked sample plus 300 near-clear and 200 random-clear controls, shuffled with a
fixed seed so every reviewer sees the same order. The page shows only the three photos and an outline around
the target road piece: no candidate label, no GSI polygons, no sample id (ids are numbered by type).

Writes dataset_noto2024_v0/review/: review.html, review_list.json, road_outline/<id>.png.
Serve it with road_v2/review_server.py, which stores the answers in review/labels_<reviewer>.csv.

Usage:
  python -I road_v2/make_noto_review.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
DS = ROOT / "dataset_noto2024_v0"
OUT = DS / "review"
N_NEAR, N_RANDOM, SEED = 300, 200, 7
MAGENTA = (255, 43, 214)


def outline(sid, line_px):
    """Wide corridor (~22 m) around the target piece, drawn as a thin line so the road surface stays visible.

    The January orthophotos are often shifted by several metres against the OSM road (and the pre-event photos), so
    the corridor is wide enough to contain the road; the pre-event version also shows the OSM centre line dashed,
    because OSM fits the pre-event photos best.
    """
    from PIL import ImageDraw
    m = Image.open(DS / "masks/road" / f"{sid}.png")
    corridor = np.array(m.filter(ImageFilter.MaxFilter(35))) > 0
    inner = np.array(Image.fromarray(corridor.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(5))) > 0
    edge = corridor & ~inner
    rgba = np.zeros((256, 256, 4), np.uint8)
    rgba[edge, :3] = MAGENTA; rgba[edge, 3] = 255
    Image.fromarray(rgba).save(OUT / "road_outline" / f"{sid}.png", optimize=True)
    pre = Image.fromarray(rgba); d = ImageDraw.Draw(pre)
    p = np.asarray(line_px, float)
    dense = np.concatenate([np.linspace(p[i], p[i + 1], 40, endpoint=False) for i in range(len(p) - 1)])
    for i in range(0, len(dense) - 4, 8):  # dashes of ~4 px
        d.line([tuple(dense[i]), tuple(dense[i + 4])], fill=(255, 230, 0, 255), width=2)
    pre.save(OUT / "road_outline_pre" / f"{sid}.png", optimize=True)


RECHECK_SEED = 11
RECHECK_EXAMPLES = (("debris", "noto_00160", "选 1：看得清土石盖在路面上"), ("snow", "noto_02600", "选 3：积雪盖住，看不清路面"),
                    ("occluded", "noto_02280", "选 3：被雾、树冠或阴影遮住"))


def make_recheck(reviewer="xie"):
    """Recheck only the disputed candidates: first-pass answers that disagree with the independent LiDAR inventory.

    Disputed = candidate judged blocked without a DEM-based deposit within 5 m, or judged clear with one.
    Same three answers as the first pass, with a stricter rule: snow cover and occlusion are 'undeterminable'.
    Answers go to review/labels_<reviewer>_recheck.csv; the first pass stays untouched.
    """
    first = {}
    with open(OUT / f"labels_{reviewer}.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            first[r["sample_id"]] = r["label"]
    rows = {r["sample_id"]: r for r in csv.DictReader(open(DS / "samples.csv", encoding="utf-8-sig"))
            if r["post_date"] and r["tunnel"] == "0" and r["post_event_road"] == "0" and r["sample_type"] == "candidate"}
    ids = [s for s, r in rows.items() if (first.get(s) == "blocked" and r["label_lidar"] == "0")
           or (first.get(s) == "clear" and r["label_lidar"] == "1")]
    rng = np.random.default_rng(RECHECK_SEED)
    ids = [ids[i] for i in rng.permutation(len(ids))]
    items = [dict(id=s, d0=rows[s]["pre_date"], d1=rows[s]["post_date"], d2=rows[s]["post_apr_date"]) for s in ids]
    (OUT / "review_list_recheck.json").write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    (OUT / "examples").mkdir(exist_ok=True)
    for name, sid, _ in RECHECK_EXAMPLES:
        im = np.asarray(Image.open(DS / "chips/post" / f"{sid}.jpg").convert("RGBA")).copy()
        ol = np.asarray(Image.open(OUT / "road_outline" / f"{sid}.png").convert("RGBA"))
        m = ol[..., 3] > 0; im[m] = ol[m]
        Image.fromarray(im).convert("RGB").save(OUT / "examples" / f"{name}.jpg", quality=90)
    page = PAGE
    a = page.index("    <li><kbd>1</kbd> <b>堵</b>"); b = page.index("    <li><kbd>←</kbd>")
    examples = "".join(f'<figure style="margin:0;text-align:center"><img src="examples/{n}.jpg" width="170" alt="">'
                       f'<figcaption class="small">{cap}</figcaption></figure>' for n, _, cap in RECHECK_EXAMPLES)
    rules = ('    <li><b>这是对有争议样本的复核，标准更严：</b>只有<b>看得清</b>土石、倒木或垮塌盖在框内路面上，才选“堵”。'
             '路面被雪盖住或被遮挡、看不清时，一律选 3。</li>\n'
             '    <li><kbd>1</kbd> <b>堵</b>：看得清土石、倒木或垮塌盖住（或冲断）框内路面，哪怕只盖住一部分。</li>\n'
             '    <li><kbd>2</kbd> <b>没堵</b>：框内路面看得清，没有被盖住（滑坡停在路边也算没堵）。</li>\n'
             '    <li><kbd>3</kbd> <b>判断不了</b>：积雪盖住路面；树冠、阴影、雾挡住；或照片上找不到这条路。</li>\n'
             f'    <li style="list-style:none;margin-top:6px"><div style="display:flex;gap:12px;flex-wrap:wrap">{examples}</div></li>\n')
    page = page[:a] + rules + page[b:]
    rep = [("<h1>道路堵塞人工审核</h1>", "<h1>道路堵塞复核：有争议的样本</h1>"),
           ("<title>Road Review</title>", "<title>Road Review Recheck</title>"),
           ("<details>", "<details open>"),
           ('<span>路面被盖住或冲断（哪怕一部分）</span>', '<span>看得清土石/倒木/垮塌盖住路面</span>'),
           ('<span>雪/阴影/云/看不到路</span>', '<span>积雪/遮挡/找不到路</span>'),
           ("const pendingKey = () => `review_pending_${who}`;", "const pendingKey = () => `review_pending_recheck_${who}`;"),
           ("reviewer=${encodeURIComponent(who)}", 'reviewer=${encodeURIComponent(who + "_recheck")}'),
           ('items = await (await fetch("review_list.json")).json();', 'items = await (await fetch("review_list_recheck.json")).json();')]
    for x, y in rep:
        assert page.count(x) == 1, x
        page = page.replace(x, y)
    assert page.count("reviewer: who,") == 2
    page = page.replace("reviewer: who,", 'reviewer: who + "_recheck",')
    (OUT / "review_recheck.html").write_text(page.replace("__N__", str(len(items))), encoding="utf-8")
    n1 = sum(first.get(s) == "blocked" for s in ids)
    print(f"[done] recheck: {len(items)} disputed candidates ({n1} first-pass blocked without a LiDAR deposit, "
          f"{len(items) - n1} first-pass clear with one) -> {OUT / 'review_recheck.html'}")
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "recheck":
        return make_recheck(*(sys.argv[2:3] or []))
    rows = list(csv.DictReader(open(DS / "samples.csv", encoding="utf-8-sig")))
    rng = np.random.default_rng(SEED)
    by = {t: [r for r in rows if r["sample_type"] == t] for t in ("candidate", "near", "random")}
    pick = by["candidate"] + list(rng.choice(by["near"], N_NEAR, replace=False)) + list(rng.choice(by["random"], N_RANDOM, replace=False))
    order = rng.permutation(len(pick))
    (OUT / "road_outline").mkdir(parents=True, exist_ok=True); (OUT / "road_outline_pre").mkdir(exist_ok=True)
    geo = {f["properties"]["sample_id"]: f["geometry"]["coordinates"]
           for f in json.loads((DS / "samples.geojson").read_text(encoding="utf-8"))["features"]}
    npx = 256 * 2 ** 18
    items = []
    for k, j in enumerate(order):
        r = pick[j]
        c = np.asarray(geo[r["sample_id"]])
        line = np.column_stack([(c[:, 0] + 180) / 360 * npx - int(r["x0"]),
                                (1 - np.arcsinh(np.tan(np.radians(c[:, 1]))) / np.pi) / 2 * npx - int(r["y0"])])
        outline(r["sample_id"], line)
        items.append(dict(id=r["sample_id"], d0=r["pre_date"], d1=r["post_date"], d2=r["post_apr_date"]))
    (OUT / "review_list.json").write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    (OUT / "review.html").write_text(PAGE.replace("__N__", str(len(items))), encoding="utf-8")
    print(f"[done] {len(items)} samples ({len(by['candidate'])} candidates, {N_NEAR} near, {N_RANDOM} random) -> {OUT}")
    return 0


PAGE = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Road Review</title>
<style>
:root { --bg:#f7f6f3; --card:#ffffff; --ink:#24231f; --muted:#6b6862; --line:#e2dfd8; --accent:#2f5d8a; --ok:#1b8a5a; --bad:#c23b3a;
        --b1:#c23b3a; --b2:#d9822b; --b3:#1b8a5a; --b4:#7d7a73; --b5:#5b4a8a; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#1b1a18; --card:#25241f; --ink:#ece9e2; --muted:#a29e95; --line:#3a3832; } }
:root[data-theme="dark"] { --bg:#1b1a18; --card:#25241f; --ink:#ece9e2; --muted:#a29e95; --line:#3a3832; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.5 system-ui, "Microsoft YaHei", sans-serif; }
.wrap { max-width:1280px; margin:0 auto; padding:12px 16px 40px; }
header { display:flex; flex-wrap:wrap; gap:8px 16px; align-items:center; margin-bottom:10px; }
h1 { font-size:18px; margin:0; font-weight:600; }
input[type=text] { font:inherit; padding:4px 8px; border:1px solid var(--line); border-radius:6px; background:var(--card); color:var(--ink); }
button { font:inherit; cursor:pointer; border:1px solid var(--line); background:var(--card); color:var(--ink); border-radius:8px; padding:6px 12px; }
button:disabled { opacity:.45; cursor:default; }
.progress { flex:1; min-width:220px; }
.bar { height:8px; background:var(--line); border-radius:4px; overflow:hidden; }
.bar i { display:block; height:100%; background:var(--accent); width:0; }
.small { font-size:12.5px; color:var(--muted); }
.imgs { display:grid; grid-template-columns:repeat(3, 1fr); gap:10px; }
.big .imgs { grid-template-columns:repeat(2, 1fr); }
.fig { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:8px; }
.fig .cap { font-size:13px; margin-bottom:6px; }
.im { position:relative; aspect-ratio:1; background:#ddd; border-radius:4px; overflow:hidden; }
.im img { position:absolute; inset:0; width:100%; height:100%; }
.hide-outline .ol { display:none; }
.labels { display:grid; grid-template-columns:repeat(3, 1fr); gap:8px; margin:12px 0 8px; }
.lb { text-align:left; padding:8px 10px; border-width:2px; }
.lb b { display:block; font-size:15px; }
.lb span { font-size:12px; color:var(--muted); }
.lb.on { color:#fff; }
.lb.on span { color:#fff; }
.lb[data-l=blocked].on { background:var(--b1); border-color:var(--b1); }
.lb[data-l=partial].on { background:var(--b2); border-color:var(--b2); }
.lb[data-l=clear].on { background:var(--b3); border-color:var(--b3); }
.lb[data-l=unclear].on { background:var(--b4); border-color:var(--b4); }
.lb[data-l=invalid].on { background:var(--b5); border-color:var(--b5); }
.row { display:flex; flex-wrap:wrap; gap:8px; align-items:center; }
.note { flex:1; min-width:200px; }
.status { font-size:12.5px; }
.status.ok { color:var(--ok); } .status.bad { color:var(--bad); font-weight:600; }
details { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:8px 12px; margin:10px 0; }
details li { margin:3px 0; }
kbd { font:12px ui-monospace, monospace; border:1px solid var(--line); border-bottom-width:2px; border-radius:4px; padding:0 4px; background:var(--card); }
@media (max-width:760px) { .imgs, .big .imgs { grid-template-columns:1fr; } .labels { grid-template-columns:1fr; } }
</style>
</head>
<body>
<div class="wrap" id="app">
<header>
  <h1>道路堵塞人工审核</h1>
  <label class="small">审核人 <input type="text" id="who" placeholder="填名字或缩写" size="10"></label>
  <div class="progress"><div class="bar"><i id="pbar"></i></div><div class="small" id="ptext"></div></div>
</header>
<details>
  <summary>判断标准与快捷键（第一次审核前请看）</summary>
  <ul>
    <li>只判断<b>紫色框里的那一段路</b>（约 50 m）。主要看<b>震后 1 月</b>的照片；震前照片用来确认路原来在哪；4 月照片作参考（4 月还在抢修，说明当时很可能被堵或被冲断）。</li>
    <li><b>框的位置：</b>震后 1 月的航拍是震后紧急制作的，整体常偏几米、个别偏 10 m 左右，所以框画得比较宽。请先在<b>震前图</b>里认准黄色虚线所在的那条路（那里位置最准），再到震后图里看<b>同一段路</b>；框里如果有两条路，以震前图认出的那条为准。</li>
    <li><kbd>1</kbd> <b>堵</b>：框内路面被土石、倒木盖住或被冲断，哪怕只盖住一部分。</li>
    <li><kbd>2</kbd> <b>没堵</b>：框内路面看得清、没被盖住。滑坡就在路边但没上路面，也算没堵。</li>
    <li><kbd>3</kbd> <b>判断不了</b>：雪、阴影、云挡住了，或者照片上看不到路（隧道、框和路对不上）。</li>
    <li><kbd>←</kbd> <kbd>→</kbd> 上一个/下一个，<kbd>H</kbd> 隐藏/显示紫框，<kbd>Z</kbd> 放大，<kbd>N</kbd> 写备注（回车保存），<kbd>Backspace</kbd> 回到刚审的上一个重新选。每按一次就自动保存到硬盘；审核服务断了也会先暂存在浏览器里，恢复后自动补存。关掉页面再打开，会从第一个未审的样本接着来。</li>
  </ul>
</details>
<div class="row small" style="margin-bottom:6px"><span id="pos"></span><span id="mine"></span><span id="last" style="margin-left:auto"></span></div>
<div class="imgs" id="imgs"></div>
<div class="labels" id="labels">
  <button class="lb" data-l="blocked"><b>1 堵</b><span>路面被盖住或冲断（哪怕一部分）</span></button>
  <button class="lb" data-l="clear"><b>2 没堵</b><span>路面清楚</span></button>
  <button class="lb" data-l="unclear"><b>3 判断不了</b><span>雪/阴影/云/看不到路</span></button>
</div>
<div class="row">
  <button id="prev">← 上一个</button>
  <button id="next">下一个 →</button>
  <button id="jump">跳到第一个未审</button>
  <input type="text" class="note" id="note" placeholder="备注（可选，按 N 进入，回车保存）">
  <span class="status" id="status"></span>
</div>
</div>
<script>
const N = __N__;
const NAMES = {blocked:"堵", clear:"没堵", unclear:"判断不了"};
const KEYS = {"1":"blocked","2":"clear","3":"unclear"};
let items = [], labels = {}, idx = 0, who = "", history = [], pending = [], lastSaved = "";
const $ = id => document.getElementById(id);
function safeGet(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }
function safeSet(k, v) { try { localStorage.setItem(k, v); } catch (e) {} }
// Every answer is first queued in the browser, then written to disk by the review server.
// If the server is down, the queue survives a page reload and is retried every 5 seconds.
const pendingKey = () => `review_pending_${who}`;
function loadPending() { try { pending = JSON.parse(safeGet(pendingKey()) || "[]"); } catch (e) { pending = []; } }
function storePending() { safeSet(pendingKey(), JSON.stringify(pending)); }
function setStatus(text, cls) { $("status").textContent = text; $("status").className = "status " + (cls || ""); }
function showSaveState() {
  if (pending.length) setStatus(`有 ${pending.length} 条还没写进硬盘（已暂存在浏览器里），审核服务恢复后会自动补存`, "bad");
  else if (lastSaved) setStatus(`已自动保存 ✓ ${lastSaved}`, "ok");
}
async function post(p) {
  const r = await fetch("/api/label", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(p)});
  if (!r.ok) throw new Error(await r.text());
}
async function flush() {
  let wrote = false;
  while (pending.length) {
    try { await post(pending[0]); pending.shift(); storePending(); wrote = true; }
    catch (e) { break; }
  }
  if (wrote && !pending.length) lastSaved = new Date().toLocaleTimeString();
  showSaveState();
}
setInterval(flush, 5000);
window.addEventListener("beforeunload", e => { if (pending.length) { e.preventDefault(); e.returnValue = ""; } });
async function loadLabels() {
  labels = {};
  if (!who) return;
  try { labels = await (await fetch(`/api/labels?reviewer=${encodeURIComponent(who)}`)).json(); }
  catch (e) { setStatus("连不上审核服务：答案会先暂存在浏览器里", "bad"); }
  loadPending();
  pending.forEach(p => labels[p.sample_id] = {label: p.label, note: p.note});
}
function firstTodo(from = 0) {
  for (let k = 0; k < items.length; k++) { const j = (from + k) % items.length; if (!labels[items[j].id]) return j; }
  return -1;
}
function render() {
  const it = items[idx];
  const im = (dir, date, cap) => `<div class="fig"><div class="cap">${cap}（${date || "无"}）</div><div class="im">
      ${date ? `<img src="../chips/${dir}/${it.id}.jpg" alt="${cap}">` : ""}
      <img class="ol" src="${dir === "pre" ? "road_outline_pre" : "road_outline"}/${it.id}.png" alt=""></div></div>`;
  $("imgs").innerHTML = im("pre", it.d0, "震前") + im("post", it.d1, "震后 1 月") + im("post_apr", it.d2, "2024 年 4 月");
  const cur = labels[it.id];
  document.querySelectorAll(".lb").forEach(b => b.classList.toggle("on", !!cur && b.dataset.l === cur.label));
  $("note").value = cur ? (cur.note || "") : "";
  $("pos").textContent = `第 ${idx + 1} / ${items.length} 个`;
  $("mine").textContent = cur ? ` · 已选：${NAMES[cur.label]}（按 1/2/3 可改）` : " · 未审";
  const done = Object.keys(labels).length;
  $("pbar").style.width = `${100 * done / items.length}%`;
  const c = {}; Object.values(labels).forEach(v => c[v.label] = (c[v.label] || 0) + 1);
  $("ptext").textContent = `已审 ${done} / ${items.length}` + (done ? "：" + Object.keys(NAMES).filter(k => c[k]).map(k => `${NAMES[k]} ${c[k]}`).join("，") : "");
  $("prev").disabled = idx === 0; $("next").disabled = idx === items.length - 1;
  const nx = items[idx + 1]; if (nx) ["pre", "post", "post_apr"].forEach(d => { const i = new Image(); i.src = `../chips/${d}/${nx.id}.jpg`; });
}
async function choose(label) {
  if (!who) { setStatus("请先在左上角填审核人", "bad"); $("who").focus(); return; }
  const it = items[idx], note = $("note").value.trim();
  labels[it.id] = {label, note};
  pending.push({reviewer: who, sample_id: it.id, label, note}); storePending();
  history.push(idx);
  $("last").textContent = `上一个（第 ${idx + 1} 个）选了：${NAMES[label]}　按 Backspace 回去改`;
  const j = firstTodo(idx + 1);
  idx = j < 0 ? idx : j; render();
  await flush();
}
document.querySelectorAll(".lb").forEach(b => b.onclick = () => choose(b.dataset.l));
$("prev").onclick = () => { if (idx > 0) { idx--; render(); } };
$("next").onclick = () => { if (idx < items.length - 1) { idx++; render(); } };
$("jump").onclick = () => { const j = firstTodo(0); if (j >= 0) { idx = j; render(); } };
$("note").addEventListener("keydown", async e => {
  if (e.key === "Enter") {
    const cur = labels[items[idx].id];
    if (cur) { const it = items[idx]; cur.note = $("note").value.trim(); pending.push({reviewer: who, sample_id: it.id, label: cur.label, note: cur.note}); storePending(); await flush(); }
    else setStatus("先选一个结果，备注会一起保存", "bad");
    $("note").blur();
  }
  if (e.key === "Escape") $("note").blur();
});
$("who").addEventListener("change", async () => {
  who = $("who").value.trim(); safeSet("review_who", who); history = [];
  await loadLabels(); await flush(); const j = firstTodo(0); idx = j < 0 ? 0 : j; render();
});
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT") return;
  if (KEYS[e.key]) { e.preventDefault(); choose(KEYS[e.key]); }
  else if (e.key === "Backspace") { e.preventDefault(); if (history.length) { idx = history.pop(); render(); $("last").textContent = "已回到上一个，重新按 1/2/3 即可覆盖"; } }
  else if (e.key === "ArrowLeft") $("prev").click();
  else if (e.key === "ArrowRight") $("next").click();
  else if (e.key === "h" || e.key === "H") document.body.classList.toggle("hide-outline");
  else if (e.key === "z" || e.key === "Z") $("app").classList.toggle("big");
  else if (e.key === "n" || e.key === "N") { e.preventDefault(); $("note").focus(); }
});
(async () => {
  items = await (await fetch("review_list.json")).json();
  who = safeGet("review_who") || ""; $("who").value = who;
  await loadLabels(); await flush();
  const j = firstTodo(0); idx = j < 0 ? 0 : j;
  render();
  if (!who) setStatus("请先在左上角填审核人", "bad");
  else if (Object.keys(labels).length) setStatus(`欢迎回来：已审 ${Object.keys(labels).length} 个，从第 ${idx + 1} 个接着审`, "ok");
})();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    raise SystemExit(main())
