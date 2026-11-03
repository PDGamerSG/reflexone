"""Build a self-contained interactive report from the committed experiment JSON."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    names = (
        "support",
        "support-seeds",
        "support-training",
        "support-benchmark",
        "commands",
        "spam",
    )
    data = {name: json.loads((ROOT / "reports" / f"{name}.json").read_text()) for name in names}
    payload = json.dumps(data, allow_nan=False).replace("</", "<\\/")
    html = TEMPLATE.replace("__DATA__", payload)
    (ROOT / "reports/dashboard.html").write_text(html)
    print(ROOT / "reports/dashboard.html")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ReflexOne measured experiments</title><style>
:root{color-scheme:light dark}*{box-sizing:border-box}body{margin:0;font-family:var(--font-sans,system-ui);color:var(--foreground,CanvasText);font-size:14px;line-height:1.5}
button,select{font:inherit;color:inherit;background:var(--secondary,ButtonFace);border:1px solid var(--border,GrayText);border-radius:var(--radius,8px);padding:7px 11px}button{cursor:pointer}button[aria-pressed=true]{outline:2px solid var(--primary,Highlight)}
nav{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.wide{grid-column:1/-1}.box{background:var(--card,Canvas);border:1px solid var(--border,GrayText);padding:16px;border-radius:var(--radius,10px)}
h2{font-size:15px;margin:0 0 10px}p{margin:6px 0}.muted,.note{color:var(--muted-foreground,GrayText);font-size:12px}.note{margin:12px 0}.kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.big{display:block;font-size:26px;font-weight:650;letter-spacing:-.03em}.label{font-size:12px;color:var(--muted-foreground,GrayText)}svg{width:100%;display:block;overflow:visible}svg text{fill:var(--muted-foreground,CanvasText);font:11px var(--font-sans,system-ui)}.axis{stroke:var(--border,GrayText);stroke-width:1}.a{fill:var(--chart-1,#3984d7)}.b{fill:var(--chart-2,#14a18c)}.line-a{fill:none;stroke:var(--chart-1,#3984d7);stroke-width:2.5}.line-b{fill:none;stroke:var(--chart-2,#14a18c);stroke-width:2.5}
table{width:100%;border-collapse:collapse;font-size:12px}td,th{text-align:right;padding:6px 3px;border-bottom:1px solid var(--border,GrayText)}td:first-child,th:first-child{text-align:left}th{font-weight:600}.legend{display:flex;gap:12px;font-size:12px}.legend span:before{content:'';display:inline-block;width:9px;height:9px;margin-right:5px;background:var(--chart-1,#3984d7);border-radius:3px}.legend span:nth-child(2):before{background:var(--chart-2,#14a18c)}
[hidden]{display:none!important}a{color:var(--primary,LinkText)}@media(max-width:520px){.grid{grid-template-columns:1fr}.kpis{grid-template-columns:1fr 1fr}.big{font-size:23px}}
</style></head><body>
<nav aria-label="Experiment profile"><button data-profile="support" aria-pressed="true">Support</button><button data-profile="commands" aria-pressed="false">Commands</button><button data-profile="spam" aria-pressed="false">Spam</button></nav>
<div id="support" class="panel grid">
<section class="box wide"><div class="kpis"><div><span class="big" id="intent-kpi"></span><span class="label">Intent micro-F1 · primary seed 42</span></div><div><span class="big" id="slot-kpi"></span><span class="label">Exact-span F1</span></div><div><span class="big" id="latency-kpi"></span><span class="label">Warm CPU p95 · batch one</span></div></div></section>
<section class="box"><h2>Intent baseline comparison</h2><label class="label" for="metric">Metric </label><select id="metric"><option value="micro_f1">Micro-F1</option><option value="macro_f1">Macro-F1</option><option value="exact_match">Exact intent-set match</option></select><div id="bars"></div><p class="muted">Same 302-request grouped test split; baseline does not extract spans.</p></section>
<section class="box"><h2>Validation learning curves</h2><div id="learning"></div><div class="legend"><span>Intent micro-F1</span><span>Exact-span F1</span></div><p class="muted">Epoch selection uses validation; final scores use held-out test data.</p></section>
<section class="box wide"><h2>Initialization variability</h2><table><thead><tr><th>Seed</th><th>Epoch</th><th>Intent micro-F1</th><th>Span F1</th><th>Full frame</th></tr></thead><tbody id="seed-rows"></tbody></table><p class="muted">Primary 42 was fixed; all runs share one test set. Variation is not a population confidence interval.</p></section>
<section class="box"><h2>Calibration ablation</h2><table><thead><tr><th>Metric</th><th>Raw</th><th>Calibrated</th></tr></thead><tbody id="calibration-rows"></tbody></table><p class="muted">Lower is better. Marginal intent probabilities; rare labels limit reliability.</p></section>
<section class="box"><h2>Deployed review policy</h2><div id="review"></div><p class="muted">Margin + nonempty intent set + no truncation. The validation 10% error target is not met on test. Slot correctness is not guaranteed.</p></section>
</div>
<div id="commands" class="panel grid" hidden>
<section class="box wide"><h2>150 supported intents + labeled unsupported requests</h2><div class="kpis" id="command-kpis"></div><p class="muted">TF-IDF + multiclass logistic regression; temperature and rejection fitted on separate held-out partitions.</p></section>
<section class="box"><h2>In-scope risk versus coverage</h2><div id="risk"></div><p class="muted">Diagnostic thresholds on supported test requests; these points do not tune the deployed rejection rule.</p></section>
<section class="box"><h2>Rejection trade-off</h2><table><tbody id="command-rows"></tbody></table><p class="muted">Class probabilities condition on supported intents. Maximum class probability is not P(in-scope).</p></section>
</div>
<div id="spam" class="panel grid" hidden>
<section class="box wide"><h2>SMS spam screening</h2><div class="kpis" id="spam-kpis"></div><p class="muted">TF-IDF + Complement Naive Bayes, selected on validation spam F1; separate temperature calibration.</p></section>
<section class="box"><h2>Confusion matrix</h2><table><thead><tr><th>Actual / predicted</th><th>Ham</th><th>Spam</th></tr></thead><tbody id="confusion"></tbody></table><p class="muted">Held-out 516 messages; spam is the positive class. The collection is old and does not represent modern scams.</p></section>
<section class="box"><h2>Reliability by confidence bin</h2><div id="reliability"></div><div class="legend"><span>Mean confidence</span><span>Observed accuracy</span></div><p class="muted">Bins use the calibrated test predictions; small bins have high sampling uncertainty.</p></section>
</div>
<p class="note">Measured on local arm64 macOS. Support timing includes tokenization, model, decoding and schema validation; excludes startup/loading/HTTP. Fixed schemas prevent malformed output, not incorrect predictions. Data sources: NLU++, CLINC150, UCI SMS; full provenance and limitations accompany the repository reports.</p>
<script>
const data=__DATA__;
const pct=x=>(x*100).toFixed(2)+'%';const el=id=>document.getElementById(id);
const support=data.support,commands=data.commands.test,spam=data.spam.test;
el('intent-kpi').textContent=pct(support.calibrated.micro_f1);el('slot-kpi').textContent=pct(support.slots.f1);el('latency-kpi').textContent=data['support-benchmark'].p95_ms.toFixed(2)+'ms';
function bars(){const key=el('metric').value;const values=[support.baseline.calibrated[key],support.calibrated[key]];const names=['TF-IDF + logistic','Fine-tuned MiniLM'];el('bars').innerHTML='<svg viewBox="0 0 300 150" role="img" aria-label="Intent baseline comparison">'+values.map((v,i)=>'<text x="0" y="'+(28+i*65)+'">'+names[i]+'</text><rect class="'+(i?'b':'a')+'" x="0" y="'+(38+i*65)+'" width="'+(v*248)+'" height="19" rx="3"/><text x="'+(v*248+6)+'" y="'+(52+i*65)+'">'+(v*100).toFixed(1)+'%</text>').join('')+'</svg>'}
bars();el('metric').addEventListener('change',bars);
function lineChart(target,series){const w=300,h=185,l=28,t=12,r=8,b=27;const n=series[0].values.length;const x=i=>l+i*(w-l-r)/Math.max(1,n-1),y=v=>h-b-v*(h-t-b);let svg='<svg viewBox="0 0 '+w+' '+h+'" role="img" aria-label="'+target+' chart">';[0,.25,.5,.75,1].forEach(v=>{svg+='<line class="axis" x1="'+l+'" y1="'+y(v)+'" x2="'+(w-r)+'" y2="'+y(v)+'"/><text x="0" y="'+(y(v)+4)+'">'+Math.round(v*100)+'</text>'});series.forEach((s,j)=>{svg+='<polyline class="line-'+(j?'b':'a')+'" points="'+s.values.map((v,i)=>x(i)+','+y(v)).join(' ')+'"/>'});svg+='<text x="'+l+'" y="'+(h-4)+'">1</text><text x="'+(w-25)+'" y="'+(h-4)+'">'+n+'</text></svg>';el(target).innerHTML=svg}
lineChart('learning',[{values:data['support-training'].map(x=>x.validation_intent_micro_f1)},{values:data['support-training'].map(x=>x.validation_slot_f1)}]);
el('seed-rows').innerHTML=data['support-seeds'].runs.map(r=>'<tr><td>'+r.seed+(Number(r.seed)===42?' · primary':'')+'</td><td>'+r.best_epoch+'</td><td>'+pct(r.micro_f1)+'</td><td>'+pct(r.slot_f1)+'</td><td>'+pct(r.frame_accuracy)+'</td></tr>').join('');
el('calibration-rows').innerHTML=[['Binary NLL','nll'],['Brier','brier'],['Pooled ECE','ece'],['Supported-label ECE','supported_label_ece']].map(([name,key])=>'<tr><td>'+name+'</td><td>'+support.raw[key].toFixed(5)+'</td><td>'+support.calibrated[key].toFixed(5)+'</td></tr>').join('');
el('review').innerHTML='<p><strong>'+support.review.accepted_count+'/302 accepted</strong> · '+pct(support.review.coverage)+' coverage</p><p>'+pct(support.review.intent_set_risk)+' accepted intent-set error</p><p>'+pct(support.review.complete_frame_risk)+' accepted full-frame error</p>';
const kpi=(value,label)=>'<div><span class="big">'+value+'</span><span class="label">'+label+'</span></div>';
el('command-kpis').innerHTML=kpi(pct(commands.accepted_accuracy),'Accepted accuracy')+kpi(pct(commands.accepted_coverage),'Overall coverage')+kpi(pct(commands.oos_recall),'OOS recall');
el('command-rows').innerHTML=[['Overall accuracy',pct(commands.accuracy)],['Supported requests accepted',pct(commands.in_scope_acceptance)],['OOS AUROC',commands.oos_detection_auroc.toFixed(4)],['Final examples',commands.n]].map(([a,b])=>'<tr><td>'+a+'</td><td>'+b+'</td></tr>').join('');
const curve=data.commands.selective_curve_in_scope;let risk='<svg viewBox="0 0 300 190" role="img" aria-label="Supported intent risk versus coverage">';for(const v of [0,.25,.5,.75,1])risk+='<line class="axis" x1="30" y1="'+(155-v*130)+'" x2="290" y2="'+(155-v*130)+'"/><text x="0" y="'+(159-v*130)+'">'+Math.round(v*100)+'</text>';risk+='<polyline class="line-a" points="'+curve.map(p=>(30+p.coverage*260)+','+(155-p.risk*130)).join(' ')+'"/><text x="110" y="180">Accepted coverage →</text></svg>';el('risk').innerHTML=risk;
el('spam-kpis').innerHTML=kpi(pct(spam.accuracy),'Accuracy')+kpi(pct(spam.spam_f1),'Spam F1')+kpi(pct(spam.spam_recall),'Spam recall');
el('confusion').innerHTML=spam.confusion_matrix.map((r,i)=>'<tr><td>'+(i?'Spam':'Ham')+'</td><td>'+r[0]+'</td><td>'+r[1]+'</td></tr>').join('');
lineChart('reliability',[{values:spam.reliability_bins.map(x=>x.confidence)},{values:spam.reliability_bins.map(x=>x.accuracy)}]);
document.querySelectorAll('nav button').forEach(button=>button.addEventListener('click',()=>{document.querySelectorAll('.panel').forEach(p=>p.hidden=p.id!==button.dataset.profile);document.querySelectorAll('nav button').forEach(b=>b.setAttribute('aria-pressed',String(b===button)))}));
</script></body></html>"""


if __name__ == "__main__":
    main()
