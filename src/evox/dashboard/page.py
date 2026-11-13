"""The single-page dashboard UI (no external scripts, fonts or network calls).

Data is only ever inserted with textContent / setAttribute, never innerHTML, so values read from
run files cannot inject markup.  ``/*__SNAPSHOT__*/null`` is replaced by exported data for offline use.
"""

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EVO-X dashboard</title>
<style>
:root{--bg:#f7f8fa;--card:#fff;--fg:#1c2330;--mut:#667085;--line:#d9dee7;--acc:#2f5bea;--ok:#177245;--bad:#b42318;--warn:#b54708;--chip:#eef1f6}
@media (prefers-color-scheme:dark){:root{--bg:#10141c;--card:#171c26;--fg:#e6e9f0;--mut:#9aa3b5;--line:#2a3140;--acc:#7b9bff;--ok:#4cc38a;--bad:#ff7a6e;--warn:#f3a64f;--chip:#222a38}}
*{box-sizing:border-box}body{margin:0;font:14px/1.45 system-ui,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--fg)}
header{padding:12px 20px;border-bottom:1px solid var(--line);display:flex;gap:16px;align-items:center;flex-wrap:wrap;background:var(--card)}
h1{font-size:17px;margin:0}h2{font-size:15px;margin:0 0 8px}h3{font-size:13px;margin:12px 0 6px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}
select{font:inherit;padding:4px 6px;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px;max-width:360px}
nav{display:flex;gap:4px;padding:8px 20px 0;flex-wrap:wrap}nav button{font:inherit;border:1px solid var(--line);border-bottom:none;background:var(--chip);color:var(--fg);padding:6px 12px;border-radius:8px 8px 0 0;cursor:pointer}
nav button.on{background:var(--card);font-weight:600;color:var(--acc)}
main{padding:16px 20px;max-width:1200px;margin:0 auto}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin-bottom:14px}
.banner{padding:8px 12px;border-radius:8px;margin-bottom:12px;background:#fff4e5;color:#7a4100;border:1px solid #f3c27b}
@media (prefers-color-scheme:dark){.banner{background:#33260f;color:#f3c27b;border-color:#6b4a14}}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:5px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{color:var(--mut);font-weight:600}
.num{text-align:right;font-variant-numeric:tabular-nums}.chip{display:inline-block;padding:1px 8px;border-radius:99px;background:var(--chip);margin:1px 3px 1px 0;font-size:12px}
.pill{display:inline-block;padding:1px 9px;border-radius:99px;font-size:12px;font-weight:600;color:#fff}
.DEPLOYED,.pass,.supported,.promote{background:var(--ok)}.REJECTED,.fail,.not_supported,.ROLLED_BACK{background:var(--bad)}.SUPERSEDED,.inconclusive,.CANDIDATE,.NO_OP{background:#667085}
.row{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:14px}.pick{cursor:pointer}.pick:hover td{background:var(--chip)}.sel td{background:var(--chip)}
.mut{color:var(--mut)}pre{white-space:pre-wrap;background:var(--chip);padding:10px;border-radius:8px;font-size:12px;overflow:auto;max-height:320px}
svg text{fill:var(--fg);font:11px system-ui,sans-serif}svg .m{fill:var(--mut)}.scroll{overflow-x:auto}
</style></head><body>
<header><h1>EVO-X dashboard</h1><span class="mut">read-only replay of stored runs</span>
<label>Run <select id="run"></select></label><label>Registry <select id="reg"></select></label><span id="integrity"></span></header>
<nav id="tabs"></nav><main id="main"></main>
<script>
const SNAP=/*__SNAPSHOT__*/null;
const $=(id)=>document.getElementById(id);
function el(tag,attrs,...kids){const e=document.createElement(tag);for(const[k,v]of Object.entries(attrs||{})){if(k==='class')e.className=v;else if(k==='onclick')e.onclick=v;else e.setAttribute(k,v)}
 for(const c of kids.flat()){if(c==null)continue;e.append(c.nodeType?c:document.createTextNode(String(c)))}return e}
const NS='http://www.w3.org/2000/svg';
function sv(tag,attrs,...kids){const e=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(attrs||{}))e.setAttribute(k,v);for(const c of kids.flat()){if(c==null)continue;e.append(c.nodeType?c:document.createTextNode(String(c)))}return e}
const fmt=(x,d=3)=>x==null?'–':(typeof x==='number'?(Number.isInteger(x)?String(x):x.toFixed(d)):String(x));
async function api(ep,...args){const key=[ep,...args].join('|');
 if(SNAP){if(!(key in SNAP))throw new Error('not in exported snapshot: '+key);return SNAP[key]}
 const names={runs:[],overview:['run'],registries:['run'],timeline:['run','registry'],decisions:['run'],results:['run'],batches:['run','scenario','system']}[ep];
 const q=names.map((n,i)=>n+'='+encodeURIComponent(args[i])).join('&');
 const r=await fetch('/api/'+ep+(q?'?'+q:''));if(!r.ok)throw new Error((await r.text())||r.status);return r.json()}
const pill=(text,cls)=>el('span',{class:'pill '+(cls||text)},text);
function table(cols,rows,onpick,sel){const t=el('table',{},el('tr',{},cols.map(c=>el('th',{class:c.num?'num':''},c.h))));
 rows.forEach((r,i)=>{const tr=el('tr',{class:(onpick?'pick ':'')+(sel===i?'sel':'')},cols.map(c=>{const v=c.f?c.f(r):r[c.k];return el('td',{class:c.num?'num':''},v)}));if(onpick)tr.onclick=()=>onpick(i);t.append(tr)});
 return el('div',{class:'scroll'},t)}
function barChart(items,{w=460,h=170,title='',fmtv=(v)=>fmt(v,3),hi=null}={}){
 const vals=items.map(i=>i.v).filter(v=>v!=null);const mx=Math.max(1e-9,...vals.map(Math.abs));const m={l:8,r:8,t:22,b:54};
 const bw=(w-m.l-m.r)/Math.max(1,items.length);const zero=m.t+(h-m.t-m.b)*(Math.min(...vals,0)<0?0.5:1);const sc=(h-m.t-m.b)/(Math.min(...vals,0)<0?2:1)/mx;
 const s=sv('svg',{viewBox:`0 0 ${w} ${h}`,width:'100%',role:'img','aria-label':title});s.append(sv('text',{x:m.l,y:14},title));
 items.forEach((it,i)=>{const v=it.v||0,x=m.l+i*bw+bw*.15,bh=Math.abs(v)*sc,y=v>=0?zero-bh:zero;
  s.append(sv('rect',{x,y,width:bw*.7,height:Math.max(0.5,bh),fill:it.hi?'var(--acc)':'var(--mut)',opacity:it.hi?1:.7}),
   sv('text',{x:x+bw*.35,y:v>=0?y-3:y+bh+10,'text-anchor':'middle'},fmtv(it.v)));
  const t=sv('text',{x:x+bw*.35,y:h-m.b+14,'text-anchor':'end',transform:`rotate(-35 ${x+bw*.35} ${h-m.b+14})`,class:'m'},it.l);s.append(t)});
 s.append(sv('line',{x1:m.l,x2:w-m.r,y1:zero,y2:zero,stroke:'var(--line)'}));return s}
function lineChart(points,marks,{w=700,h=190,title=''}={}){const m={l:34,r:10,t:22,b:22};const s=sv('svg',{viewBox:`0 0 ${w} ${h}`,width:'100%',role:'img','aria-label':title});s.append(sv('text',{x:m.l,y:14},title));
 const xs=points.map(p=>p.x);const x0=Math.min(...xs),x1=Math.max(...xs,x0+1);const X=x=>m.l+(x-x0)/(x1-x0)*(w-m.l-m.r),Y=y=>m.t+(1-y)*(h-m.t-m.b);
 [0,.5,1].forEach(g=>{s.append(sv('line',{x1:m.l,x2:w-m.r,y1:Y(g),y2:Y(g),stroke:'var(--line)'}),sv('text',{x:2,y:Y(g)+4,class:'m'},g))});
 (marks||[]).forEach(k=>{s.append(sv('line',{x1:X(k.x),x2:X(k.x),y1:m.t,y2:h-m.b,stroke:'var(--bad)','stroke-dasharray':'3 3'}),sv('text',{x:X(k.x)+3,y:m.t+10,class:'m'},k.l))});
 const pts=points.filter(p=>p.y!=null).map(p=>X(p.x)+','+Y(p.y)).join(' ');s.append(sv('polyline',{points:pts,fill:'none',stroke:'var(--acc)','stroke-width':2}));return s}
function meter(label,value,lo,hi,max){const w=300,pct=v=>Math.min(100,Math.max(0,100*v/max));
 const s=sv('svg',{viewBox:`0 0 ${w+120} 26`,width:'100%'});s.append(sv('text',{x:0,y:12},label),sv('rect',{x:110,y:3,width:w,height:10,rx:5,fill:'var(--chip)'}),
  sv('rect',{x:110,y:3,width:w*pct(value||0)/100,height:10,rx:5,fill:'var(--acc)'}));
 [lo,hi].forEach(t=>{if(t!=null)s.append(sv('line',{x1:110+w*pct(t)/100,x2:110+w*pct(t)/100,y1:0,y2:16,stroke:'var(--warn)','stroke-width':2}))});
 s.append(sv('text',{x:110,y:25,class:'m'},fmt(value)));return s}

const state={tab:'Overview',runs:[],run:null,regs:[],reg:null,tl:null,dec:null,res:null,ov:null,pick:0};
const TABS=['Overview','Drift & decisions','Candidates & gate','Versions & rollback','Results'];
function tabs(){const n=$('tabs');n.replaceChildren(...TABS.map(t=>el('button',{class:t===state.tab?'on':'',onclick:()=>{state.tab=t;state.pick=0;render();tabs()}},t)))}
function banner(){const r=state.runs.find(x=>x.id===state.run);if(!r)return null;
 if(r.data_tag==='SYNTHETIC'||r.quick)return el('div',{class:'banner'},'PIPELINE CHECK ONLY: ',r.data_tag==='SYNTHETIC'?'synthetic data. ':'',r.quick?'quick run. ':'','These numbers validate execution, not research performance.');return null}

function viewOverview(){const o=state.ov,c=el('div',{});
 c.append(el('div',{class:'card'},el('h2',{},'Run'),table([{h:'Field',k:'k'},{h:'Value',k:'v'}],[['kind',o.kind],...Object.entries(o.summary||{}).filter(([k,v])=>typeof v!=='object'),...Object.entries(o.protocol||{}).filter(([k,v])=>typeof v!=='object')].map(([k,v])=>({k,v:fmt(v)})))));
 if(o.demo_steps)c.append(el('div',{class:'card'},el('h2',{},'Scripted demonstration (controller inputs are scripted; registry events are real)'),el('ol',{},o.demo_steps.map(x=>el('li',{},el('b',{},x.what+': '),x.detail)))));
 const h=Object.entries(o.hypotheses||{});if(h.length)c.append(el('div',{class:'card'},el('h2',{},'Hypotheses'),table([{h:'',f:r=>r[0]},{h:'Status',f:r=>pill(r[1].status)},{h:'Claim',f:r=>r[1].claim},{h:'Evidence',f:r=>r[1].evidence}],h)));
 if(o.hypotheses&&o.hypotheses.H5)c.append(el('div',{class:'card'},'H5: ',pill(o.hypotheses.H5.status)));
 if(o.report)c.append(el('div',{class:'card'},el('h2',{},'Run report'),el('pre',{},o.report)));return c}

function decisionRows(){ // controller decisions from Phase B events, or from the selected registry's candidate contexts
 if(state.dec&&state.dec.controller_decisions.length)return state.dec.controller_decisions.map(d=>({id:d.event_id,sig:d.signature,rule:d.rule,chosen:d.chosen,eligible:d.eligible,excluded:d.excluded,util:d.utilities,est:d.estimates,weights:d.weights,th:d.thresholds}));
 return ((state.tl&&state.tl.versions)||[]).filter(v=>v.controller&&v.controller.chosen_action).map(v=>({id:v.version+' ('+v.action+')',sig:v.signature,rule:v.controller.rule,chosen:v.controller.chosen_action,eligible:v.controller.eligible,excluded:v.controller.excluded,util:v.controller.utilities,est:v.controller.estimates,weights:v.controller.weights,th:null}))}
function viewDecisions(){const rows=decisionRows(),c=el('div',{});
 if(!rows.length){c.append(el('div',{class:'card mut'},'No controller decisions are stored for this run/registry (fixed policies and the static IDS make no decisions).'));return c}
 const i=Math.min(state.pick,rows.length-1),d=rows[i];
 c.append(el('div',{class:'card'},el('h2',{},'Detected drift events ('+rows.length+')'),table([{h:'Event',k:'id'},{h:'Type',f:r=>(r.sig||{}).type_guess},{h:'Magnitude',num:1,f:r=>fmt((r.sig||{}).magnitude,2)},{h:'Persist.',num:1,f:r=>(r.sig||{}).persistence},{h:'Novelty',num:1,f:r=>fmt((r.sig||{}).novelty,2)},{h:'Rule',k:'rule'},{h:'Chosen',f:r=>el('b',{},r.chosen)}],rows.slice(0,200),(k)=>{state.pick=k;render()},i)));
 const s=d.sig||{},th=d.th||{};
 c.append(el('div',{class:'row'},el('div',{class:'card'},el('h2',{},'Drift state'),
   meter('magnitude',s.magnitude,th.low_magnitude,th.high_magnitude,Math.max(12,(s.magnitude||0)*1.1)),meter('persistence',s.persistence,th.short_persistence,th.long_persistence,Math.max(20,(s.persistence||0)*1.1)),
   meter('novelty',s.novelty,th.novelty,null,1),meter('recurrence',s.recurrence,th.recurrence,null,1),
   el('div',{class:'mut'},'Orange ticks mark the controller thresholds. Type guess: ',s.type_guess||'–','; alarm batch ',fmt(s.alarm_batch),'; analysed at batch ',fmt(s.now_batch))),
  el('div',{class:'card'},el('h2',{},'Eligibility rules'),el('div',{},'Rule fired: ',el('b',{},d.rule||'–')),el('h3',{},'Eligible'),(d.eligible||[]).map(a=>el('span',{class:'chip'},a)),
   el('h3',{},'Excluded'),Object.entries(d.excluded||{}).map(([a,r])=>el('span',{class:'chip'},a+': '+r)),el('h3',{},'Weights'),Object.entries(d.weights||{}).map(([k,v])=>el('span',{class:'chip'},k+' '+fmt(v,2))))));
 const items=Object.entries(d.util||{}).map(([a,u])=>({l:a,v:u.total,hi:a===d.chosen}));
 c.append(el('div',{class:'card'},el('h2',{},'Utility of each eligible action (chosen = highlighted)'),barChart(items,{title:'utility total'}),
   table([{h:'Action',k:'a'},{h:'gain',num:1,f:r=>fmt(r.t.gain)},{h:'forgetting',num:1,f:r=>fmt(r.t.forgetting)},{h:'cost',num:1,f:r=>fmt(r.t.cost)},{h:'ΔFPR',num:1,f:r=>fmt(r.t.fpr)},{h:'latency',num:1,f:r=>fmt(r.t.latency)},{h:'total',num:1,f:r=>fmt(r.u)}],
   Object.entries(d.util||{}).map(([a,u])=>({a,t:u.terms||{},u:u.total})))));return c}

function viewCandidates(){const vs=((state.tl&&state.tl.versions)||[]),c=el('div',{});
 if(!vs.length){c.append(el('div',{class:'card mut'},'No registry selected.'));return c}
 const i=Math.min(state.pick,vs.length-1),v=vs[i];
 c.append(el('div',{class:'card'},el('h2',{},'Model versions and Safety Gate outcomes'),table([{h:'Version',k:'version'},{h:'Parent',k:'parent_version'},{h:'Strategy',k:'action'},{h:'Status',f:r=>pill(r.status)},{h:'Gate',f:r=>r.gate&&r.gate.reason?r.gate.reason:'–'},{h:'Train s',num:1,f:r=>fmt(r.training_seconds,2)}],vs,(k)=>{state.pick=k;render()},i)));
 const checks=Object.entries((v.gate||{}).checks||{});
 c.append(el('div',{class:'card'},el('h2',{},'Gate checks for '+v.version),checks.length?table([{h:'Check',f:r=>r[0]},{h:'Value',num:1,f:r=>fmt(r[1].value,4)},{h:'Bound',num:1,f:r=>fmt(r[1].bound,4)},{h:'Result',f:r=>pill(r[1].passed?'pass':'fail')}],checks):el('div',{class:'mut'},'No gate decision stored (initial model, or a fixed policy that bypasses the gate).')));return c}

function viewVersions(){const tl=state.tl,c=el('div',{});if(!tl){c.append(el('div',{class:'card mut'},'No registry selected.'));return c}
 const ev=tl.events,W=Math.max(700,ev.length*74),col={INITIALIZED:'var(--mut)',PROMOTED:'var(--ok)',REJECTED:'var(--bad)',ROLLED_BACK:'var(--warn)',CANDIDATE_REGISTERED:'var(--acc)'};
 const s=sv('svg',{viewBox:`0 0 ${W} 150`,width:W,height:150});s.append(sv('line',{x1:20,x2:W-20,y1:60,y2:60,stroke:'var(--line)','stroke-width':2}));
 ev.forEach((e,i)=>{const x=40+i*74;s.append(sv('circle',{cx:x,cy:60,r:9,fill:col[e.kind]||'var(--mut)'}),sv('text',{x,y:92,'text-anchor':'middle'},e.kind.replace('_',' ').toLowerCase().slice(0,16)),
  sv('text',{x,y:44,'text-anchor':'middle',class:'m'},'#'+e.sequence),sv('text',{x,y:112,'text-anchor':'middle',class:'m'},'active '+(e.active_after||'–')))});
 c.append(el('div',{class:'card'},el('h2',{},'Version and rollback timeline'),el('div',{class:'scroll'},s),el('div',{class:'mut'},'Final active version: ',el('b',{},tl.active||'–'),'. Events: ',tl.n_events,'. Showing the latest ',ev.length,'.')));
 c.append(el('div',{class:'card'},el('h2',{},'Event journal'),table([{h:'#',num:1,k:'sequence'},{h:'Event',f:r=>pill(r.kind,r.kind)},{h:'Active before',k:'active_before'},{h:'Active after',k:'active_after'},{h:'Detail',f:r=>JSON.stringify(r.payload).slice(0,140)}],ev)));return c}

function viewResults(){const r=state.res,c=el('div',{});
 if(!r||!r.systems.length){c.append(el('div',{class:'card mut'},'This run has no comparative results table.'));return c}
 const by=(k)=>r.systems.map(s=>({l:s.system,v:s[k],hi:/^evox(_native|_transfer)?$/.test(s.system)}));
 c.append(el('div',{class:'row'},[['macro_f1','Macro-F1 (reporting rows)'],['updates','Promoted updates'],['forgetting','Historical forgetting'],['training_seconds','Training seconds']].filter(([k])=>r.systems[0][k]!==undefined).map(([k,t])=>el('div',{class:'card'},barChart(by(k),{title:t})))));
 c.append(el('div',{class:'card'},el('h2',{},'Per-system means'),table(Object.keys(r.systems[0]).map(k=>({h:k,num:k!=='system',f:x=>k==='system'?x[k]:fmt(x[k],3)})),r.systems)));
 if(r.portability)c.append(el('div',{class:'card'},el('h2',{},'Portability'),table([{h:'Dataset',k:'dataset'},{h:'Alg.',k:'algorithm'},{h:'Kind',k:'kind'},{h:'Baseline',k:'baseline'},{h:'Advantage',num:1,f:x=>fmt(x.advantage,4)},{h:'Verdict',k:'portability'}],r.portability.filter(x=>x.kind==='ALL'))));
 if(state.batches&&state.batches.length){const pts=state.batches.map(b=>({x:b.batch,y:b.binary_f1}));let marks=[],prev=null;state.batches.forEach(b=>{if(prev!==null&&b.version!==prev)marks.push({x:b.batch,l:b.version});prev=b.version});
  c.append(el('div',{class:'card'},el('h2',{},'Per-batch binary F1 for '+state.reg),lineChart(pts,marks,{title:'binary F1 by batch (dashed = version change)'})))}
 return c}

async function render(){const m=$('main');m.replaceChildren();const b=banner();if(b)m.append(b);
 try{const v={Overview:viewOverview,'Drift & decisions':viewDecisions,'Candidates & gate':viewCandidates,'Versions & rollback':viewVersions,Results:viewResults}[state.tab];m.append(v())}catch(e){m.append(el('div',{class:'card'},'Cannot render this view: '+e.message))}
 const i=state.tl&&state.tl.integrity;$('integrity').replaceChildren(i?el('span',{class:'chip',title:i.detail},'journal '+(i.ok?'verified ('+i.events+' events)':'NOT verified: '+i.detail)):'')}
async function loadRun(id){state.run=id;state.pick=0;state.tl=state.batches=null;
 [state.ov,state.regs,state.dec,state.res]=await Promise.all([api('overview',id),api('registries',id),api('decisions',id),api('results',id)]);
 const sel=$('reg');sel.replaceChildren(...state.regs.map(r=>el('option',{value:r.id},r.id==='registry'||r.id==='.'?'journal':r.id)));sel.disabled=!state.regs.length;
 await loadReg(state.regs.length?(state.regs.find(r=>/(^|\/)evox$/.test(r.id))||state.regs[0]).id:null);if(state.regs.length)sel.value=state.reg}
async function loadReg(id){state.reg=id;state.pick=0;state.batches=null;try{state.tl=id?await api('timeline',state.run,id):null}catch(e){state.tl=null}
 const r=state.regs.find(x=>x.id===id);if(r&&r.scenario){try{state.batches=await api('batches',state.run,r.scenario,r.system)}catch(e){state.batches=null}}render()}
(async function init(){try{state.runs=await api('runs');const s=$('run');
 s.replaceChildren(...state.runs.map(r=>el('option',{value:r.id},r.id+' ['+r.kind+(r.data_tag?', '+r.data_tag:'')+']')));
 s.onchange=async()=>{await loadRun(s.value);render()};$('reg').onchange=(e)=>loadReg(e.target.value);tabs();
 if(!state.runs.length){$('main').append(el('div',{class:'card'},'No runs found. Run `evox demo` or `evox lifecycle-demo`, or point --runs-dir at a results/runs folder.'));return}
 await loadRun(state.runs[0].id);render()}catch(e){$('main').append(el('div',{class:'card'},'Dashboard error: '+e.message))}})();
</script></body></html>
"""
