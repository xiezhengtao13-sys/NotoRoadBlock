"""Static HTML gallery for road_v2/dataset_noto2024_v0: browse every sample (before | January | April) with filters.

Draws the GSI polygon outlines into small transparent PNGs (viz/outline/<id>.png) and writes gallery.html
next to samples.csv; open it in any browser, no server needed.

Usage:
  python -I road_v2/make_noto_gallery.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
DS = Path(os.environ.get("NOTO_DS", ROOT / "dataset_noto2024_v0"))
ORANGE, BLUE = (237, 161, 0), (61, 143, 214)


def edge(mask_path):
    m = np.array(Image.open(mask_path)) > 0
    er = np.array(Image.fromarray(m.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(5))) > 0
    return m & ~er


def outline_png(sid, out):
    rgba = np.zeros((256, 256, 4), np.uint8)
    for sub, color in (("slide_lidar", BLUE), ("slide_aerial", ORANGE)):
        e = edge(DS / "masks" / sub / f"{sid}.png")
        rgba[e, :3] = color; rgba[e, 3] = 255
    Image.fromarray(rgba).save(out, optimize=True)


def main() -> int:
    with open(DS / "samples.csv", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    geo = {f["properties"]["sample_id"]: f["geometry"]["coordinates"]
           for f in json.loads((DS / "samples.geojson").read_text(encoding="utf-8"))["features"]}
    (DS / "viz/outline").mkdir(parents=True, exist_ok=True)
    n = 256 * 2 ** 18
    data = []
    for k, r in enumerate(rows):
        sid = r["sample_id"]
        outline_png(sid, DS / "viz/outline" / f"{sid}.png")
        lon, lat = np.asarray(geo[sid]).T
        x = (lon + 180) / 360 * n - int(r["x0"]); y = (1 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2 * n - int(r["y0"])
        data.append(dict(id=sid, t=r["sample_type"], hw=r["highway"], rg=r["region"], la=int(r["label_aerial"]),
                         ll=int(r["label_lidar"]), ov=float(r["slide_overlap_m"]), md=float(r["mlit_dist_m"]),
                         tn=int(r["tunnel"]), br=int(r["bridge"]), pe=int(r["post_event_road"]), f=int(r["fold"]), nm=r["road_name"],
                         d0=r["pre_date"], d1=r["post_date"], d2=r["post_apr_date"],
                         ln=" ".join(f"{a:.1f},{b:.1f}" for a, b in zip(x, y))))
        if k % 500 == 0:
            print(f"  outlines {k}/{len(rows)}", flush=True)
    html = TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    (DS / "gallery.html").write_text(html, encoding="utf-8")
    print("[done]", DS / "gallery.html", len(data), "samples")
    return 0


TEMPLATE = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Noto Road Samples</title>
<style>
:root { --bg:#fbfaf8; --card:#ffffff; --ink:#24231f; --muted:#6b6862; --line:#e4e1db; --red:#e34948; --green:#1baf7a;
        --orange:#eda100; --blue:#3d8fd6; --chip:#f1efea; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#1b1a18; --card:#25241f; --ink:#ece9e2;
        --muted:#a29e95; --line:#3a3832; --chip:#302e29; } }
:root[data-theme="dark"] { --bg:#1b1a18; --card:#25241f; --ink:#ece9e2; --muted:#a29e95; --line:#3a3832; --chip:#302e29; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.45 system-ui, "Microsoft YaHei", sans-serif; }
header { padding:18px 16px 8px; max-width:1500px; margin:0 auto; }
h1 { font-size:20px; margin:0 0 4px; font-weight:600; }
.sub { color:var(--muted); font-size:13px; }
.legend span { display:inline-flex; align-items:center; gap:5px; margin-right:14px; }
.sw { width:14px; height:4px; border-radius:2px; display:inline-block; }
.bar { position:sticky; top:0; z-index:5; background:var(--bg); border-bottom:1px solid var(--line); }
.controls { max-width:1500px; margin:0 auto; padding:10px 16px; display:flex; flex-wrap:wrap; gap:8px 14px; align-items:center; }
label { color:var(--muted); font-size:13px; }
select, button { font:inherit; font-size:13px; color:var(--ink); background:var(--card); border:1px solid var(--line);
                 border-radius:6px; padding:4px 8px; }
button { cursor:pointer; }
button:disabled { opacity:.4; cursor:default; }
.count { margin-left:auto; color:var(--muted); font-size:13px; }
main { max-width:1500px; margin:0 auto; padding:12px 16px 40px; display:grid; gap:14px;
       grid-template-columns:repeat(auto-fill, minmax(min(100%, 420px), 1fr)); }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:10px; }
.head { display:flex; justify-content:space-between; gap:8px; font-size:13px; }
.head b { font-weight:600; }
.tag { font-size:12px; padding:1px 7px; border-radius:10px; background:var(--chip); color:var(--muted); white-space:nowrap; }
.tag.cand { color:#fff; background:var(--red); }
.imgs { display:grid; grid-template-columns:repeat(3, 1fr); gap:5px; margin:7px 0 6px; }
.im { position:relative; aspect-ratio:1; background:var(--chip); border-radius:4px; overflow:hidden; }
.im img, .im svg { position:absolute; inset:0; width:100%; height:100%; }
.im .cap { position:absolute; left:0; right:0; bottom:0; font-size:11px; color:#fff; background:rgba(0,0,0,.45); padding:1px 5px; }
.hide-ov .ov { display:none; }
.meta { font-size:12px; color:var(--muted); }
.meta .warn { color:var(--red); font-weight:600; }
.pager { max-width:1500px; margin:0 auto; padding:0 16px 30px; display:flex; gap:10px; align-items:center; justify-content:center; }
</style>
</head>
<body>
<header>
  <h1>能登 2024 道路堵塞数据集 v0：样本浏览</h1>
  <div class="sub">每个样本一段 50 m 的 OSM 道路。三张图依次是：震前（国土地理院年度航拍）、震后 1 月（国土地理院灾害正射）、2024 年 4 月。影像出典：国土地理院。</div>
  <div class="sub legend" style="margin-top:6px">
    <span><i class="sw" style="background:var(--red)"></i>候选被堵路段</span>
    <span><i class="sw" style="background:var(--green)"></i>未堵路段</span>
    <span><i class="sw" style="background:var(--orange)"></i>国土地理院航拍判读的滑坡/堆积范围</span>
    <span><i class="sw" style="background:var(--blue)"></i>国土地理院激光雷达判读的堆积范围</span>
  </div>
</header>
<div class="bar"><div class="controls">
  <label>类型 <select id="t"><option value="">全部</option><option value="candidate">候选被堵</option><option value="near">未堵·邻近（≤300 m）</option><option value="random">未堵·随机</option></select></label>
  <label>激光雷达 <select id="ll"><option value="">不限</option><option value="1">也判为堆积</option><option value="0">未判为堆积</option></select></label>
  <label>深入多边形 <select id="ov"><option value="0">不限</option><option value="0.01">&gt;0 m</option><option value="10">≥10 m</option><option value="25">≥25 m</option><option value="45">≥45 m</option></select></label>
  <label>道路等级 <select id="hw"><option value="">全部</option></select></label>
  <label>地区 <select id="rg"><option value="">全部</option></select></label>
  <label><input type="checkbox" id="tn"> 排除隧道和震后新建路</label>
  <label><input type="checkbox" id="ovl" checked> 显示叠加</label>
  <label>排序 <select id="so"><option value="id">编号</option><option value="ov">深入多边形（多→少）</option><option value="md">离国交省受灾点（近→远）</option><option value="rnd">随机</option></select></label>
  <span class="count" id="count"></span>
</div></div>
<main id="grid"></main>
<div class="pager"><button id="prev">上一页</button><span id="page"></span><button id="next">下一页</button></div>
<script>
const D = __DATA__;
const PER = 48; let page = 0, view = [];
const $ = id => document.getElementById(id);
const rnd = new Map(D.map(d => [d.id, Math.random()]));
for (const [key, el] of [["hw", $("hw")], ["rg", $("rg")]]) {
  const counts = {}; D.forEach(d => counts[d[key]] = (counts[d[key]] || 0) + 1);
  Object.entries(counts).sort((a, b) => b[1] - a[1]).forEach(([v, n]) => el.add(new Option(`${v || "(无)"} (${n})`, v)));
}
const typeName = {candidate: "候选被堵", near: "未堵·邻近", random: "未堵·随机"};
function card(d) {
  const color = d.la ? "var(--red)" : "var(--green)";
  const im = (dir, date, cap) => `<div class="im">${date ? `<a href="chips/${dir}/${d.id}.jpg" target="_blank"><img loading="lazy" src="chips/${dir}/${d.id}.jpg" alt="${cap} ${d.id}"></a>` : ""}
     <img class="ov" loading="lazy" src="viz/outline/${d.id}.png" alt="" style="pointer-events:none">
     <svg class="ov" viewBox="0 0 256 256" style="pointer-events:none"><polyline points="${d.ln}" fill="none" stroke="${color}" stroke-width="6" stroke-opacity=".85"/></svg>
     <div class="cap">${cap} ${date || "无影像"}</div></div>`;
  const meta = [d.la ? `${d.ov.toFixed(0)} m 在多边形内` : "", `激光雷达${d.ll ? "：堆积" : "：无"}`,
                `国交省受灾点 ${d.md < 1000 ? d.md.toFixed(0) + " m" : (d.md / 1000).toFixed(1) + " km"}`,
                d.tn ? `<span class="warn">隧道</span>` : "", d.pe ? `<span class="warn">震后新建路</span>` : "", d.br ? "桥梁" : ""].filter(Boolean).join(" · ");
  return `<div class="card"><div class="head"><span><b>${d.id}</b> · ${d.hw}${d.nm ? " · " + d.nm : ""}</span>
    <span class="tag ${d.t === "candidate" ? "cand" : ""}">${typeName[d.t]}</span></div>
    <div class="imgs">${im("pre", d.d0, "震前")}${im("post", d.d1, "震后")}${im("post_apr", d.d2, "4月")}</div>
    <div class="meta">${d.rg || "(无地区)"} · fold ${d.f} · ${meta}</div></div>`;
}
function apply() {
  const t = $("t").value, ll = $("ll").value, ov = +$("ov").value, hw = $("hw").value, rg = $("rg").value, tn = $("tn").checked;
  view = D.filter(d => (!t || d.t === t) && (ll === "" || d.ll === +ll) && (!ov || d.ov >= ov) && (!hw || d.hw === hw) &&
                       (!rg || d.rg === rg) && (!tn || (!d.tn && !d.pe)));
  const so = $("so").value;
  if (so === "ov") view.sort((a, b) => b.ov - a.ov);
  else if (so === "md") view.sort((a, b) => a.md - b.md);
  else if (so === "rnd") view.sort((a, b) => rnd.get(a.id) - rnd.get(b.id));
  page = 0; render();
}
function render() {
  const pages = Math.max(1, Math.ceil(view.length / PER));
  $("grid").innerHTML = view.slice(page * PER, (page + 1) * PER).map(card).join("");
  $("count").textContent = `${view.length} / ${D.length} 个样本`;
  $("page").textContent = `第 ${page + 1} / ${pages} 页`;
  $("prev").disabled = page === 0; $("next").disabled = page >= pages - 1;
  document.body.classList.toggle("hide-ov", !$("ovl").checked);
}
["t", "ll", "ov", "hw", "rg", "tn", "so"].forEach(id => $(id).addEventListener("change", apply));
$("ovl").addEventListener("change", render);
$("prev").onclick = () => { page--; render(); scrollTo(0, 0); };
$("next").onclick = () => { page++; render(); scrollTo(0, 0); };
apply();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    raise SystemExit(main())
