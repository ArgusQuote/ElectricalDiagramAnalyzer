#!/usr/bin/env python3
# MLCellReader/label_server.py
"""
Local web page for labeling harvested breaker-cell crops.

    python MLCellReader/label_server.py          # then open http://127.0.0.1:8765

Each page shows a grid of crops with the current OCR reading pre-filled.
Fix anything that is wrong, then "Save page" (Ctrl+Enter). Rules:
  - a value in the box          -> that is the label
  - empty box                   -> the cell is blank (SPACE / SPARE / '--' / empty)
  - "?" button                  -> you cannot read it either (excluded from training)

Crops are sorted by suggested value, so a page of "20" crops makes the odd one
out easy to spot. Labels are appended to labels.jsonl; nothing is ever deleted,
and re-labeling a cell simply records a newer label.

Binds to 127.0.0.1 only: the crops come from customer drawings.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import parse_qs, unquote, urlparse

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from MLCellReader.cell_data import (  # noqa: E402
    CELL_ROLES,
    CROPS_DIRNAME,
    ROLE_POLES,
    ROLE_TRIP,
    append_jsonl,
    data_root,
    labels_path,
    load_labels,
    load_manifest,
    make_label_record,
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_PAGE_SIZE = 60
MAX_PAGE_SIZE = 300

# An OCR suggestion at or above this confidence lands in the "easy" view.
CONFIDENT_OCR_CONF = 0.85

VIEW_REVIEW = "review"      # unlabeled, OCR missing or unsure
VIEW_EASY = "easy"          # unlabeled, OCR confident
VIEW_LABELED = "labeled"
VIEWS = (VIEW_REVIEW, VIEW_EASY, VIEW_LABELED)


class LabelStore:
    """In-memory view of manifest + labels, persisted by appending to labels.jsonl."""

    def __init__(self, root: Path):
        self.root = root
        self._lock = threading.Lock()
        self.manifest: Dict[str, dict] = {}
        self.labels: Dict[str, dict] = {}
        self._manifest_mtime: Optional[float] = None
        with self._lock:
            self.labels = load_labels(self.root)
        self.refresh_manifest()

    def refresh_manifest(self) -> None:
        """Pick up crops added by a harvest that is still running."""
        path = self.root / "manifest.jsonl"
        mtime = path.stat().st_mtime if path.exists() else None
        if mtime == self._manifest_mtime:
            return
        with self._lock:
            self.manifest = load_manifest(self.root)
            self._manifest_mtime = mtime

    @staticmethod
    def _suggestion(record: dict) -> tuple:
        role = record.get("role")
        if role == ROLE_TRIP:
            return (record.get("weak_amps"),)
        if role == ROLE_POLES:
            return (record.get("weak_poles"),)
        return (record.get("weak_amps"), record.get("weak_poles"))

    def _is_confident(self, record: dict) -> bool:
        suggestion = self._suggestion(record)
        if any(value is None for value in suggestion):
            return False
        conf = record.get("ocr_conf")
        return conf is not None and float(conf) >= CONFIDENT_OCR_CONF

    def summary(self) -> dict:
        with self._lock:
            out = {}
            for role in CELL_ROLES:
                records = [r for r in self.manifest.values() if r.get("role") == role]
                labeled = sum(1 for r in records if r["id"] in self.labels)
                easy = sum(
                    1 for r in records
                    if r["id"] not in self.labels and self._is_confident(r)
                )
                out[role] = {
                    "total": len(records),
                    "labeled": labeled,
                    "easy": easy,
                    "review": len(records) - labeled - easy,
                }
            return out

    def items(self, role: str, view: str, offset: int, limit: int) -> dict:
        with self._lock:
            records = [r for r in self.manifest.values() if r.get("role") == role]
            if view == VIEW_LABELED:
                selected = [r for r in records if r["id"] in self.labels]
            else:
                want_confident = view == VIEW_EASY
                selected = [
                    r for r in records
                    if r["id"] not in self.labels and self._is_confident(r) == want_confident
                ]

            def sort_key(record: dict):
                values = self._suggestion(record)
                return (
                    tuple((value is None, value or 0) for value in values),
                    record.get("source_pdf") or "",
                    record["id"],
                )

            selected.sort(key=sort_key)
            page = selected[offset: offset + limit]
            return {
                "total": len(selected),
                "offset": offset,
                "items": [self._item_payload(r) for r in page],
            }

    def _item_payload(self, record: dict) -> dict:
        return {
            "id": record["id"],
            "role": record["role"],
            "image_url": "/" + record["image"].replace(os.sep, "/"),
            "ocr_text": record.get("ocr_text") or "",
            "ocr_conf": record.get("ocr_conf"),
            "suggest_amps": record.get("weak_amps"),
            "suggest_poles": record.get("weak_poles"),
            "source_pdf": record.get("source_pdf") or "",
            "label": self.labels.get(record["id"]),
        }

    def save(self, raw_labels: List[dict]) -> dict:
        records, errors = [], []
        with self._lock:
            for raw in raw_labels:
                cell_id = raw.get("id")
                if cell_id not in self.manifest:
                    errors.append({"id": cell_id, "error": "unknown cell id"})
                    continue
                try:
                    records.append(
                        make_label_record(
                            cell_id,
                            raw.get("state"),
                            amps=_optional_int(raw.get("amps")),
                            poles=_optional_int(raw.get("poles")),
                        )
                    )
                except (TypeError, ValueError) as exc:
                    errors.append({"id": cell_id, "error": str(exc)})
            if not errors:
                append_jsonl(labels_path(self.root), records)
                for record in records:
                    self.labels[record["id"]] = record
        return {"saved": 0 if errors else len(records), "errors": errors}


def _optional_int(value) -> Optional[int]:
    if value is None or value == "":
        return None
    return int(value)


class LabelRequestHandler(BaseHTTPRequestHandler):
    store: LabelStore = None  # set in main()

    def log_message(self, fmt, *args):  # keep the console quiet; errors still surface
        pass

    def _send_json(self, payload, status=HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, body: bytes, content_type: str, cache_control: str = "max-age=3600") -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache_control)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        query = parse_qs(url.query)

        if url.path == "/":
            self._send_bytes(INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8", cache_control="no-store")
        elif url.path == "/api/summary":
            self.store.refresh_manifest()
            self._send_json(self.store.summary())
        elif url.path == "/api/items":
            role = query.get("role", [ROLE_TRIP])[0]
            view = query.get("view", [VIEW_REVIEW])[0]
            if role not in CELL_ROLES or view not in VIEWS:
                self._send_json({"error": "bad role or view"}, HTTPStatus.BAD_REQUEST)
                return
            try:
                offset = max(0, int(query.get("offset", ["0"])[0]))
                limit = min(MAX_PAGE_SIZE, max(1, int(query.get("limit", [str(DEFAULT_PAGE_SIZE)])[0])))
            except ValueError:
                self._send_json({"error": "bad offset or limit"}, HTTPStatus.BAD_REQUEST)
                return
            self.store.refresh_manifest()
            self._send_json(self.store.items(role, view, offset, limit))
        elif url.path.startswith(f"/{CROPS_DIRNAME}/"):
            self._serve_crop(unquote(url.path.lstrip("/")))
        else:
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self):
        if urlparse(self.path).path != "/api/labels":
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            raw_labels = payload["labels"]
            if not isinstance(raw_labels, list):
                raise ValueError("'labels' must be a list")
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            self._send_json({"error": f"bad request: {exc}"}, HTTPStatus.BAD_REQUEST)
            return
        result = self.store.save(raw_labels)
        status = HTTPStatus.OK if not result["errors"] else HTTPStatus.BAD_REQUEST
        self._send_json(result, status)

    def _serve_crop(self, rel_path: str) -> None:
        crops_root = (self.store.root / CROPS_DIRNAME).resolve()
        target = (self.store.root / rel_path).resolve()
        if crops_root not in target.parents or target.suffix.lower() != ".png" or not target.is_file():
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        self._send_bytes(target.read_bytes(), "image/png")


INDEX_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Breaker cell labeling</title>
<style>
 body{font-family:system-ui,sans-serif;margin:0;background:#f3f4f6}
 header{position:sticky;top:0;background:#111827;color:#fff;padding:10px 16px;display:flex;gap:16px;align-items:center;flex-wrap:wrap;z-index:2}
 header button,header select{font-size:14px;padding:4px 10px}
 .tab.active{background:#2563eb;color:#fff}
 #stats{font-size:13px;opacity:.85}
 #help{font-size:13px;padding:8px 16px;background:#e0e7ff}
 #grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:8px;padding:12px 16px}
 .card{background:#fff;border:3px solid #9ca3af;border-radius:6px;padding:6px;display:flex;flex-direction:column;gap:4px}
 .card.has{border-color:#16a34a}.card.blank{border-color:#d1d5db;background:#f9fafb}.card.unread{border-color:#dc2626;background:#fef2f2}.card.bad{border-color:#f59e0b}
 .card img{width:100%;height:64px;object-fit:contain;background:#fff;image-rendering:auto}
 .row{display:flex;gap:4px;align-items:center}
 .row input{width:64px;font-size:20px;padding:2px 4px}
 .meta{font-size:11px;color:#6b7280;overflow:hidden;white-space:nowrap;text-overflow:ellipsis}
 footer{position:sticky;bottom:0;background:#111827;color:#fff;padding:10px 16px;display:flex;gap:12px;align-items:center}
 footer button{font-size:16px;padding:6px 16px}
 #save{background:#16a34a;color:#fff;border:0;border-radius:4px}
</style></head><body>
<header>
 <strong>Breaker cells</strong>
 <span><button class="tab" data-role="trip">Trip amps</button><button class="tab" data-role="poles">Poles</button><button class="tab" data-role="combo">Combo</button></span>
 <select id="view">
  <option value="review">Needs review (OCR missing/unsure)</option>
  <option value="easy">Easy (OCR confident - just scan)</option>
  <option value="labeled">Already labeled (fix mistakes)</option>
 </select>
 <span id="stats"></span>
</header>
<div id="help">Fix anything wrong. <b>Empty box = blank cell</b> (SPACE/SPARE/--/empty). <b>?</b> = you can't read it either. Poles must be 1, 2 or 3. Cards you change are saved automatically. <b>Ctrl+Enter</b> confirms the rest of the page as shown and loads the next.</div>
<div id="grid"></div>
<footer><button id="save">Save page &amp; next (Ctrl+Enter)</button><span id="pos"></span><span id="msg"></span></footer>
<script>
let role=localStorage.getItem("label_role")||"trip", view=localStorage.getItem("label_view")||"review", offset=0, items=[];
const PAGE=60, AUTOSAVE_DELAY_MS=800, ROLE_NAMES={trip:"Trip amps",poles:"Poles",combo:"Combo"};
const dirty=new Set(); let autosaveTimer=null;
const $=s=>document.querySelector(s);
const esc=s=>String(s).replace(/[&<>"']/g,ch=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[ch]));
function fieldsFor(r){return r==="trip"?["amps"]:r==="poles"?["poles"]:["amps","poles"];}
function initialValue(it,f){
  if(it.label){ if(it.label.state!=="labeled") return ""; const v=it.label[f]; return v==null?"":String(v);}
  const v=f==="amps"?it.suggest_amps:it.suggest_poles; return v==null?"":String(v);
}
function validate(f,v){ if(v==="") return true; if(!/^\d+$/.test(v)) return false; const n=+v;
  return f==="poles"? (n>=1&&n<=3) : (n>=0&&n<=4000); }
function refreshCard(card){
  const unread=card.dataset.unread==="1"; const vals=[...card.querySelectorAll("input")];
  const bad=vals.some(i=>!validate(i.dataset.f,i.value.trim())); const has=vals.some(i=>i.value.trim()!=="");
  card.className="card "+(unread?"unread":bad?"bad":has?"has":"blank");
}
function render(){
  const g=$("#grid"); g.innerHTML="";
  items.forEach((it,idx)=>{
    const c=document.createElement("div"); c.dataset.id=it.id;
    c.dataset.unread=(it.label&&it.label.state==="unreadable")?"1":"0";
    const conf=it.ocr_conf==null?"-":it.ocr_conf.toFixed(2);
    let html=`<img loading="lazy" src="${esc(it.image_url)}"><div class="row">`;
    fieldsFor(role).forEach(f=>{html+=`<label>${f==="amps"?"A":"P"} <input data-f="${f}" value="${initialValue(it,f)}" inputmode="numeric"></label>`;});
    html+=`<button class="unr" title="Can't read it">?</button><button class="clr" title="Blank cell">blank</button></div>`;
    html+=`<div class="meta" title="${esc(it.source_pdf)}">OCR: "${esc(it.ocr_text)}" (${conf}) · ${esc(it.source_pdf)}</div>`;
    c.innerHTML=html; g.appendChild(c);
    c.querySelectorAll("input").forEach(i=>i.addEventListener("input",()=>{c.dataset.unread="0";refreshCard(c);markDirty(c);}));
    c.querySelector(".unr").onclick=()=>{c.dataset.unread=c.dataset.unread==="1"?"0":"1";refreshCard(c);markDirty(c);};
    c.querySelector(".clr").onclick=()=>{c.querySelectorAll("input").forEach(i=>i.value="");c.dataset.unread="0";refreshCard(c);markDirty(c);};
    refreshCard(c);
  });
  window.scrollTo(0,0);
}
async function loadStats(){
  const s=await (await fetch("/api/summary")).json(); const x=s[role];
  document.querySelectorAll(".tab").forEach(b=>{const t=s[b.dataset.role]; b.textContent=`${ROLE_NAMES[b.dataset.role]} (${t.labeled})`;});
  $("#stats").textContent=`${role}: ${x.labeled}/${x.total} labeled · ${x.review} need review · ${x.easy} easy`;
}
function cardLabel(c){
  if(c.dataset.unread==="1") return {id:c.dataset.id,state:"unreadable"};
  const o={id:c.dataset.id}; let any=false;
  c.querySelectorAll("input").forEach(i=>{const v=i.value.trim(); o[i.dataset.f]=v===""?null:+v; if(v!=="") any=true;});
  o.state=any?"labeled":"blank"; return o;
}
function markDirty(c){
  dirty.add(c.dataset.id); $("#msg").textContent="unsaved changes...";
  clearTimeout(autosaveTimer); autosaveTimer=setTimeout(flushDirty,AUTOSAVE_DELAY_MS);
}
function dirtyCards(){return [...document.querySelectorAll(".card")].filter(c=>dirty.has(c.dataset.id));}
async function postLabels(labels){
  const resp=await fetch("/api/labels",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({labels})});
  return {ok:resp.ok, body:await resp.json()};
}
// Saves every changed card that is valid; orange (invalid) cards stay pending until fixed.
async function flushDirty(){
  clearTimeout(autosaveTimer);
  const cards=dirtyCards().filter(c=>!c.classList.contains("bad"));
  if(!cards.length) return dirty.size===0;
  try{
    const r=await postLabels(cards.map(cardLabel));
    if(!r.ok){$("#msg").textContent="AUTOSAVE FAILED: "+JSON.stringify(r.body.errors||r.body.error);return false;}
    cards.forEach(c=>dirty.delete(c.dataset.id));
    $("#msg").textContent=dirty.size?`autosaved ${r.body.saved} · fix orange cards`:`autosaved ${r.body.saved}`;
    loadStats();
  }catch(err){$("#msg").textContent="AUTOSAVE FAILED (is the server running?): "+err;return false;}
  return dirty.size===0;
}
async function leavePage(){
  if(await flushDirty()) return true;
  return confirm("Some changed cards could not be saved (orange or server error). Leave this page anyway?");
}
async function load(){
  dirty.clear(); localStorage.setItem("label_role",role); localStorage.setItem("label_view",view); $("#view").value=view;
  const r=await (await fetch(`/api/items?role=${role}&view=${view}&offset=${offset}&limit=${PAGE}`)).json();
  items=r.items; render();
  $("#pos").textContent=r.total?`showing ${offset+1}-${offset+items.length} of ${r.total}`:"nothing left in this view";
  document.querySelectorAll(".tab").forEach(b=>b.classList.toggle("active",b.dataset.role===role));
  loadStats();
}
async function save(){
  const cards=[...document.querySelectorAll(".card")]; if(!cards.length) return;
  if(cards.some(c=>c.classList.contains("bad"))){$("#msg").textContent="Fix the orange cards first.";return;}
  clearTimeout(autosaveTimer);
  let r;
  try{ r=await postLabels(cards.map(cardLabel)); }
  catch(err){$("#msg").textContent="SAVE FAILED (is the server running?): "+err;return;}
  if(!r.ok){$("#msg").textContent="Save failed: "+JSON.stringify(r.body.errors||r.body.error);return;}
  $("#msg").textContent=`saved ${r.body.saved}`;
  if(view==="labeled") offset+=PAGE;  // labeled view does not shrink, so page forward
  load();
}
document.querySelectorAll(".tab").forEach(b=>b.onclick=async()=>{if(!await leavePage()) return; role=b.dataset.role;offset=0;load();});
$("#view").onchange=async e=>{const next=e.target.value; if(!await leavePage()){e.target.value=view;return;} view=next;offset=0;load();};
$("#save").onclick=save;
document.addEventListener("keydown",e=>{if(e.key==="Enter"&&(e.ctrlKey||e.metaKey)){e.preventDefault();save();}});
window.addEventListener("beforeunload",e=>{
  const cards=dirtyCards().filter(c=>!c.classList.contains("bad"));
  if(cards.length && navigator.sendBeacon("/api/labels",new Blob([JSON.stringify({labels:cards.map(cardLabel)})],{type:"application/json"})))
    cards.forEach(c=>dirty.delete(c.dataset.id));
  if(dirty.size){e.preventDefault(); e.returnValue="";}
});
load();
</script></body></html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local labeling UI for breaker-cell crops.")
    parser.add_argument("--data-root", type=Path, default=None, help="Annotation data root")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = data_root(args.data_root)
    if not (root / "manifest.jsonl").exists():
        print(f"[label] No manifest at {root}. Run MLCellReader/harvest_cell_crops.py first.")
        return 1

    LabelRequestHandler.store = LabelStore(root)
    summary = LabelRequestHandler.store.summary()
    server = ThreadingHTTPServer((DEFAULT_HOST, args.port), LabelRequestHandler)
    print(f"[label] data_root={root}")
    for role, counts in summary.items():
        print(f"[label]   {role:6} {counts}")
    print(f"[label] Open http://{DEFAULT_HOST}:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[label] stopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
