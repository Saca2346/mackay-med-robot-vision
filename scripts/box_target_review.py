"""Local evidence index and owner review form; never updates inventory."""
from collections import Counter
import csv
import json
import os
from pathlib import Path
from urllib.parse import quote


HTML = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Box-by-box evidence review</title>
<style>
body{font:16px system-ui;margin:0;background:#f2f5f8;color:#152538}header,main{max-width:1180px;margin:auto;padding:24px}
header{background:#172f49;color:white;border-radius:0 0 15px 15px}h1{font-size:28px;margin:0 0 12px}h2{font-size:21px}
p{line-height:1.5}button,select,input,textarea{font:inherit}button,select{padding:9px;border:1px solid #99a9b8;border-radius:6px;background:white}
button{cursor:pointer}button.primary{background:#1563a1;color:white;border:0}nav{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:18px 0}
.case{background:white;padding:20px;border-radius:12px;margin:20px 0;border:1px solid #d9e1e8}.badge{display:inline-block;padding:5px 10px;background:#fff0c2;color:#604800;border-radius:5px;font-size:14px}
.pass{background:#d9f3e5;color:#164f34}.fail{background:#fbe2df;color:#7e2924}.pending{background:#e6ebf0;color:#435568}
.media{display:grid;grid-template-columns:1fr 1fr;gap:18px}.media img,.media video{width:100%;max-height:470px;object-fit:contain;background:#edf1f4}
.meta{font-size:14px;color:#4a6073}.links{display:flex;gap:13px;flex-wrap:wrap;margin:15px 0}.review{border-top:1px solid #dae3ec;padding-top:15px;margin-top:15px}
.checks{display:flex;flex-wrap:wrap;gap:15px;margin:12px 0}textarea{width:98%;min-height:65px;padding:7px}details{margin:12px 0}pre{overflow:auto;font-size:12px;background:#f1f4f7;padding:12px}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;border-bottom:1px solid #d4dde7;padding:9px}a{color:#075791}
@media(max-width:760px){.media{grid-template-columns:1fr}header,main{padding:14px}table{font-size:12px}th,td{padding:5px}}
</style>
<header><h1>Box-by-box evidence review</h1><div id="counts"></div>
<p>__PROVENANCE__
Green requires independent evidence and any requested printed date. Machine verification and your approval are separate.</p></header>
<main><p id="batch" class="meta"></p>
<nav><button class="primary" id="csv">Export owner review CSV</button><button id="json">Export owner review JSON</button>
<select id="filter"><option value="all">All boxes</option><option value="PENDING">Owner review pending</option><option value="machine_pending">Machine not verified</option><option value="ACCEPT">Owner accepted</option></select></nav>
<p class="meta" id="storage">Choices stay in this browser when local storage is available. Export your review before closing.
Refreshing loads newly completed cases. No review choice changes inventory or authorizes retrieval.</p>
<div style="overflow:auto"><table><thead><tr><th>Box / requested identity</th><th>Machine result</th><th>Green frame ready</th><th>Sources</th><th>Owner</th></tr></thead><tbody id="matrix"></tbody></table></div>
<div id="cards"></div></main>
<script id="data" type="application/json">__DATA__</script>
<script>
const data=JSON.parse(document.getElementById('data').textContent),key='box-review:'+data.batch_id;
let saved={};try{saved=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){document.getElementById('storage').textContent='Local storage unavailable. Export the review before closing; inventory is never changed.'}
const reviews={};for(const c of data.cases)reviews[c.box_id]=saved[c.box_id]||{decision:'PENDING',correct_box:false,outline_ok:false,sources_ok:false,date_ok:false,notes:''};
const el=(tag,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n};
const persist=()=>{try{localStorage.setItem(key,JSON.stringify(reviews))}catch(e){};matrix()};
const fmt=x=>x==null?'Not verified':Number(x).toFixed(3)+' s';
function matrix(){const body=document.getElementById('matrix');body.replaceChildren();for(const c of data.cases){const r=el('tr');const a=el('a',c.box_id+' '+c.request);a.href='#'+c.box_id;const td=el('td');td.append(a);r.append(td);for(const x of [c.status,fmt(c.green_frame_seconds)+(c.timing_eligible?'':' (diagnostic)'),c.verified_sources||c.last_observed_sources||'No confirmed source',reviews[c.box_id].decision])r.append(el('td',x));body.append(r)}}
document.getElementById('counts').textContent=data.completed+'/'+data.cases.length+' trials complete | '+data.verified_once+' machine VERIFIED at least once | '+data.within_2s+' eligible first green <= 2 s';
document.getElementById('batch').textContent='Batch '+data.batch_id+' | Every owner decision starts PENDING. Missing fields remain unknown. Inventory reference photos are not current OCR proof.';
for(const c of data.cases){
 const card=el('article',undefined,'case');card.id=c.box_id;card.dataset.box=c.box_id;card.append(el('h2',c.box_id+' — '+c.request));
 const badge=el('span',c.status,c.status==='VERIFIED_WITHIN_2S'?'badge pass':c.status==='PROCESS_FAILED'?'badge fail':'badge');card.append(badge);
 card.append(el('p','First verification: '+fmt(c.verification_seconds)+' | Green frame ready: '+fmt(c.green_frame_seconds)+' | Capture frame: '+fmt(c.capture_seconds_at_green)+' | First text ready (unverified): '+(c.text_frame_seconds==null?'Not recorded':Number(c.text_frame_seconds).toFixed(3)+' s'),'meta'));
 card.append(el('p','Sources at first green: '+(c.verified_sources||'not verified')+' | Latest observed sources: '+(c.last_observed_sources||'not confirmed')+' | Later verification losses: '+(c.later_verification_losses||0)+' | All observed fields read: '+Boolean(c.all_fields_read),'meta'));
 if(c.position_warning)card.append(el('p',c.position_warning,'badge'));
 const media=el('div',undefined,'media');
 if(c.links.video){const v=el('video');v.controls=true;v.preload='none';v.src=c.links.video;media.append(v)}
 if(c.links.final_photo){const wrap=el('div');wrap.append(el('p','Final camera/view evidence','meta'));const image=el('img');image.loading='lazy';image.src=c.links.final_photo;wrap.append(image);media.append(wrap)}
 card.append(media);const links=el('div',undefined,'links');for(const [label,url] of Object.entries(c.links)){if(!url)continue;const a=el('a',label.replaceAll('_',' '));a.href=url;a.target='_blank';links.append(a)}card.append(links);
 const details=el('details');details.append(el('summary','Actual reader crops, observed fields and provenance'));
 for(const crop of c.reader_images||[]){const figure=el('figure');figure.append(el('figcaption',crop.label));const a=el('a');a.href=crop.url;a.target='_blank';const img=el('img');img.loading='lazy';img.src=crop.url;img.style.maxWidth='100%';img.style.maxHeight='400px';a.append(img);figure.append(a);details.append(figure)}
 details.append(el('pre',JSON.stringify(c.observed_fields||{},null,2)));card.append(details);
 const review=el('div',undefined,'review');review.append(el('strong','Your validation'));
 const checks=el('div',undefined,'checks');for(const [field,label] of [['correct_box','Correct physical box'],['outline_ok','Full outline is correct'],['sources_ok','Independent sources agree'],['date_ok','Requested date is visible and correct']]){
  const l=el('label');const input=el('input');input.type='checkbox';input.checked=Boolean(reviews[c.box_id][field]);input.disabled=field==='date_ok'&&!c.required_printed_date;input.onchange=()=>{reviews[c.box_id][field]=input.checked;persist()};l.append(input,document.createTextNode(' '+label+(input.disabled?' (not requested)':'')));checks.append(l)
 }review.append(checks);
 const select=el('select');for(const [value,label] of [['PENDING','Pending review'],['ACCEPT','Accept evidence'],['REJECT','Reject evidence'],['NEEDS_CLOSER_VIEW','Need closer image / barcode']]){const o=el('option',label);o.value=value;select.append(o)}select.value=reviews[c.box_id].decision;select.onchange=()=>{reviews[c.box_id].decision=select.value;persist();filter()};review.append(select);
 const note=el('textarea');note.placeholder='Notes: wrong box, clipped end, unreadable size/date/barcode, timing…';note.value=reviews[c.box_id].notes;note.oninput=()=>{reviews[c.box_id].notes=note.value;persist()};review.append(el('p','Notes'),note);card.append(review);document.getElementById('cards').append(card)
}
function filter(){const f=document.getElementById('filter').value;for(const c of data.cases){const visible=f==='all'||(f==='machine_pending'&&!c.machine_verified_once)||reviews[c.box_id].decision===f;document.getElementById(c.box_id).hidden=!visible}}
document.getElementById('filter').onchange=filter;
function records(){return data.cases.map(c=>({box_id:c.box_id,request:c.request,machine_status:c.status,green_frame_seconds:c.green_frame_seconds,timing_eligible:c.timing_eligible,...reviews[c.box_id]}))}
function download(name,text,type){const u=URL.createObjectURL(new Blob([text],{type})),a=el('a');a.href=u;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)}
document.getElementById('json').onclick=()=>download('owner_review_'+data.batch_id+'.json',JSON.stringify({batch_id:data.batch_id,reviewed_at:new Date().toISOString(),inventory_modified:false,rows:records()},null,2),'application/json');
document.getElementById('csv').onclick=()=>{const rows=records(),keys=Object.keys(rows[0]);const q=v=>{let s=v==null?'':String(v);if(/^[=+@-]/.test(s))s="'"+s;return '"'+s.replaceAll('"','""')+'"'};download('owner_review_'+data.batch_id+'.csv','\ufeff'+[keys,...rows.map(r=>keys.map(k=>r[k]))].map(r=>r.map(q).join(',')).join('\r\n'),'text/csv;charset=utf-8')};
matrix();
</script></html>'''


def write_review(output, plan, rows):
    output = Path(output).resolve()
    metadata_file = output / "inputs.json"
    metadata = json.loads(metadata_file.read_text(encoding="utf-8")) if metadata_file.exists() else {}
    live = metadata.get("live", False)
    provenance = ("Live camera trials with fresh observations for every target. " if live else
                  "Recorded replay with fresh observations for every target; this is not a live display acceptance test. ")
    provenance += "The model has seen earlier shelf imagery; overlap with this capture is not established."
    by_id = {r["box_id"]: r for r in rows}
    diagnostics_path = output / "position_diagnostics.json"
    diagnostics = ({r["box_id"]: r for r in json.loads(diagnostics_path.read_text(encoding="utf-8"))}
                   if diagnostics_path.exists() else {})
    cases = []
    def href(path):
        return quote(Path(os.path.relpath(Path(path).resolve(), output)).as_posix(), safe="/") if path else None
    for expected in plan:
        row = {**expected, **by_id.get(expected["box_id"], {}), "owner_validation": "PENDING"}
        row.setdefault("status", "NOT_RUN")
        row["position_warning"] = (
            "The displayed candidate may differ from the assistant-proposed shelf slot. "
            "Check its actual label: the slot proposal is unreviewed and is not identity evidence."
            if diagnostics.get(row["box_id"], {}).get("possible_different_box_requires_owner_check") else None)
        take = Path(row["take"]) if row.get("take") else None
        if take and (take / "target_view.jsonl").is_file():
            events = [json.loads(s) for s in (take / "target_view.jsonl").read_text(encoding="utf-8").splitlines()]
            first = next((r for r in events if r["state"] == "DISPLAY_READY"), None)
            row["capture_seconds_at_green"] = first.get("source_seconds") if first else None
        else:
            row["capture_seconds_at_green"] = None
        row["within_2s_recorded_bounds"] = (row.get("timing_eligible", False)
                    and row.get("machine_verified_at_end", False) and not row.get("later_verification_losses", 0)
                    and not row.get("request_changed", False) and not row.get("exit_code", 0)
                    and row.get("green_frame_seconds") is not None and row["green_frame_seconds"] <= 2
                    and row["capture_seconds_at_green"] is not None and row["capture_seconds_at_green"] <= 2)
        retimed = take / "view_real_time.mp4" if take else None
        browser_video = take / "view_browser.mp4" if take else None
        row["links"] = {"CMD": href(take / "cmd.txt") if take else None,
                        "video": href(browser_video if browser_video and browser_video.exists() else
                                      retimed if retimed and retimed.exists() else take / "view.mp4") if take else None,
                        "original_encoded_video": href(take / "view.mp4") if retimed and retimed.exists() else None,
                        "frame_timing": href(take / "frame_latency.jsonl") if take else None,
                        "final_photo": href(row.get("final_photo")),
                        "first_verified_photo": href(row.get("first_verified_photo")),
                        "first_verified_fields_crops": href(row.get("first_verified_result")),
                        "latest_fields_crops": href(row.get("latest_result")),
                        "inventory_reference_only": href(row.get("reference_photo"))}
        proposed = output / "proposed_spines" / (row["box_id"] + ".png")
        if proposed.exists():
            row["links"]["assistant_proposed_source_spine"] = href(proposed)
        row["reader_images"] = []
        snapshots = [("First VERIFIED", row.get("first_verified_result")),
                     ("Latest observation", row.get("latest_result"))]
        for label, result in snapshots:
            if not result:
                continue
            folder = Path(result).parent
            # Show real saved inputs from the last reading of each snapshot.
            readings = sorted(folder.glob("reading_*"), key=lambda p: int(p.name.split("_")[-1]))
            images = [folder / "panel.png"]
            if readings:
                images += [readings[-1] / name for name in ("spine.png", "spine_ocr_input.png",
                            "band_size_joint.png", "size_label_1.png", "barcode_candidate.png")]
                images += sorted(readings[-1].glob("band_size_retry_*.png"))
            row["reader_images"] += [{"label": label + " / " + p.name, "url": href(p)}
                                     for p in images if p.is_file()]
        cases.append(row)
    data = {"batch_id": output.name, "completed": len(rows), "cases": cases,
            "verified_once": sum(bool(r.get("machine_verified_once")) for r in rows),
            "within_2s": sum(c["within_2s_recorded_bounds"] for c in cases),
            "machine_status_counts": dict(Counter(c["status"] for c in cases)),
            "owner_validation": "PENDING", "inventory_modified": False}
    encoded = json.dumps(data, ensure_ascii=True).replace("<", "\\u003c")
    from html import escape
    (output / "REVIEW.html").write_text(HTML.replace("__DATA__", encoded).replace("__PROVENANCE__", escape(provenance)), encoding="utf-8")
    (output / "review_index.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    with (output / "review_template.csv").open("w", encoding="utf-8-sig", newline="") as f:
        keys = ["box_id", "request", "machine_status", "green_frame_seconds", "timing_eligible",
                "owner_validation", "owner_notes"]
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for c in cases:
            writer.writerow({"box_id": c["box_id"], "request": c["request"], "machine_status": c["status"],
                             "green_frame_seconds": c.get("green_frame_seconds"),
                             "timing_eligible": c.get("timing_eligible", False),
                             "owner_validation": "PENDING", "owner_notes": ""})
    lines = ["# One-by-one shelf target evidence", "",
             f"{len(rows)}/{len(plan)} trials complete; {data['verified_once']} machine VERIFIED at least once; "
             f"{data['within_2s']} eligible first green frames within 2 s.", "",
             "Open [REVIEW.html](REVIEW.html) to inspect video, CMD, photos and actual reader crops and export your "
             "validation. Owner decisions start PENDING. Inventory is never modified. Input dates remain required "
             "printed-date observations. A row's inventory status is reference data, not this test's result.", "",
             provenance + " Every trial starts with a fresh tracker after model loading. "
             "Initialization and full field "
             "completion are separate. Unverified candidates and later losses remain visible.", "",
             "| Box | Request | Machine result | Green ready | Sources | Evidence |",
             "| --- | --- | --- | --- | --- | --- |"]
    for c in cases:
        seconds = c.get("green_frame_seconds")
        timing = f"{seconds:.3f} s" if seconds is not None else "not verified"
        sources = c.get("verified_sources") or c.get("last_observed_sources") or "not confirmed"
        links = " / ".join(f"[{label.replace('_', ' ')}]({url})" for label, url in c["links"].items() if url)
        lines.append(f"| {c['box_id']} | {c['request']} | {c['status']} | {timing} | {sources} | {links} |")
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
