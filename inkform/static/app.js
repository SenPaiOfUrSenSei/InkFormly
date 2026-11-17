const $ = (id) => document.getElementById(id);
let variationPresets = {};
let tRefinementSupported = false;
let tRefinementKnown = false;
let cursiveSelectionSupported = false;
let gapMinimumSupported = false;
let slantAdjustmentSupported = false;
let expandedLayoutSupported = false;
const state = { id: null, busy: false, reference: null, exportId: null, dirty: false, sourceDirty: false, exportDirty: false };
const example = $("text").value;
const pageVariationFields = ["width-variation", "height-variation", "slant-variation", "rhythm-variation", "word-spacing-variation", "line-start-variation", "line-end-variation", "line-spacing-variation", "line-tilt-variation", "baseline-variation"];
const controls = ["text","seed","bias","form-diversity","consistent","quality","t-refinement","prefer-cursive-r","color","width","opacity","roughness","pressure","flow","margin","height","spacing","paper","naturalness","letter-variation","density-variation","variation-seed","min-word-gap","slant-adjustment","fit-lines-individually",...pageVariationFields];
const number = (id) => Number($(id).value);
const files = (id, name) => `/api/jobs/${id}/files/${name}`;
const pageSettings = () => ({margin_mm:number("margin"),writing_height_mm:number("height"),line_spacing_mm:number("spacing"),paper:$("paper").value,naturalness:number("naturalness"),letter_variation:number("letter-variation"),layout_seed:number("variation-seed"),...(gapMinimumSupported?{word_gap_min_mm:number("min-word-gap")}:{}),...(slantAdjustmentSupported?{slant_adjustment_deg:number("slant-adjustment")}:{}),...(expandedLayoutSupported?{fit_lines_individually:$("fit-lines-individually").checked}:{}),...Object.fromEntries(pageVariationFields.filter(id=>id!=="line-end-variation"||expandedLayoutSupported).map(id=>[id.replaceAll("-","_"),number(id)]))});
const inkSettings = () => ({color:$("color").value,width_mm:number("width"),opacity:number("opacity"),roughness:number("roughness"),pressure_variation:number("pressure"),flow:number("flow"),density_variation:number("density-variation"),seed:71});
function updateCursiveSelection(){
  $("prefer-cursive-r").disabled=state.busy||!cursiveSelectionSupported||!$("quality").checked||!$("consistent").checked;
}
function updateLabels(){
  updateCursiveSelection();
  $("t-refinement").disabled=state.busy||!tRefinementSupported||!$("quality").checked;
  $("char-count").textContent = `${$("text").value.length} / 4000`;
  for (const id of ["form-diversity","opacity","roughness","pressure","flow","naturalness","letter-variation","density-variation",...pageVariationFields]) $(`${id}-value`).textContent = `${Math.round(number(id)*100)}%`;
  for (const id of ["width","margin","height","spacing"]) $(`${id}-value`).textContent = `${number(id)} mm`;
  $("bias-value").textContent = number("bias").toFixed(2);
  $("slant-adjustment-value").textContent = `${number("slant-adjustment")}°`;
  $("min-word-gap-value").textContent = `${number("min-word-gap").toFixed(1)} mm`;
}
function dirtyMessage(){
  $("dirty").hidden=!state.dirty;
  $("dirty").textContent=state.sourceDirty?"Words or handwriting settings changed. Generate a new take before exporting.":"Ink or paper settings changed. Apply them to this take before exporting.";
}
function markDirty(source=false){
  if(state.id){state.dirty=true;state.sourceDirty=state.sourceDirty||source;state.exportDirty=true;dirtyMessage();$("downloads").hidden=true;}
}
function busy(value){
  state.busy=value;
  $("variation-preset").disabled=value||!Object.keys(variationPresets).length;$("shuffle-variations").disabled=value;
  $("generate").disabled=value;
  $("rerender").disabled=value||!state.id;
  $("export").disabled=value||!state.id;
  for(const id of controls) $(id).disabled=value;
  updateCursiveSelection();
  $("t-refinement").disabled=value||!tRefinementSupported||!$("quality").checked;
  $("min-word-gap").disabled=value||!gapMinimumSupported;
  $("slant-adjustment").disabled=value||!slantAdjustmentSupported;
  for(const id of ["line-end-variation","fit-lines-individually"])$(id).disabled=value||!expandedLayoutSupported;
  $("reference").disabled=value;$("clear-reference").disabled=value;$("new-take").disabled=value;$("load-example").disabled=value;
  $("progress").hidden=!value;
  $("preview-dot").style.background=value?"#c29652":"#7c927a";
  document.querySelectorAll(".rewrite-line").forEach(button=>button.disabled=value);
}
function error(message){$("error").textContent=message;$("error").hidden=!message;}
async function api(path,body){
  const response=await fetch(path,{method:body===undefined?"GET":"POST",headers:body===undefined?{}:{"Content-Type":"application/json"},body:body===undefined?undefined:JSON.stringify(body)});
  const result=await response.json();
  if(!response.ok){const detail=result.detail;throw new Error(typeof detail==="string"?detail:Array.isArray(detail)?detail.map(x=>`${x.loc.slice(1).join(" / ")}: ${x.msg}`).join("; "):"The request failed.");}
  return result;
}
async function poll(id,kind){
  for(;;){
    const job=await api(`/api/jobs/${id}`);
    if(job.state==="error") throw new Error(job.error);
    const total=job.total_lines;
    $("progress").value=job.state==="ready"?100:total?Math.max(5,job.completed_lines/total*90):8;
    $("status").textContent=job.state==="queued"?"Waiting for the local pen…":job.state==="generating"?`Writing line ${job.completed_lines+1} of ${total}…`:job.state==="rendering"?(kind==="export"?"Preparing 300 dpi print files…":"Laying down the ink…"):"Ready.";
    if(job.state==="ready") return job;
    await new Promise(resolve=>setTimeout(resolve,650));
  }
}
async function showDocument(job){
  state.id=job.id;if(job.operation==="generate")state.sourceDirty=false;state.dirty=state.sourceDirty;state.exportDirty=true;state.exportId=null;
  dirtyMessage();$("empty").hidden=true;$("pages").replaceChildren();$("downloads").hidden=true;
  for(let i=1;i<=job.pages;i++){
    const figure=document.createElement("figure"),img=document.createElement("img"),caption=document.createElement("figcaption");
    img.src=files(job.id,`preview-${String(i).padStart(3,"0")}.png`);img.alt=`Generated handwritten page ${i}`;img.loading=i>1?"lazy":"eager";
    caption.textContent=`PAGE ${String(i).padStart(2,"0")} / ${String(job.pages).padStart(2,"0")}`;
    figure.append(img,caption);$("pages").append(figure);
  }
  $("preview-title").textContent=`Your handwriting / ${job.pages} ${job.pages===1?"page":"pages"}`;
  $("status").textContent=job.review_lines.length?`${job.review_lines.length} lines to compare with the input.`:job.operation==="render"?"Fresh ink. Same handwriting.":"Your take is ready. Take a closer look.";
  $("metrics").textContent=`${job.seconds.toFixed(1)}s GENERATION · 120 DPI PREVIEW`;
  $("zoom").disabled=false;$("diagnostics").hidden=false;$("warnings").replaceChildren();
  for(const warning of job.warnings){const li=document.createElement("li");li.textContent=warning;$("warnings").append(li);}
  $("trajectory-download").href=files(job.id,"document.json");$("style-download").href=files(job.id,"reference.json");
  const review=await api(`/api/jobs/${job.id}/review`);
  $("readback-panel").hidden=false;$("readback-panel").open=job.review_lines.length>0;$("readback-lines").replaceChildren();
  $("readback-summary").textContent=review.quality_checks?`Wording readback · ${job.readback_matches} matches · ${job.review_lines.length} to review`:"Compare the wording · automatic readback is off";
  for(const line of review.lines){
    const row=document.createElement("section"),heading=document.createElement("div"),label=document.createElement("strong"),badge=document.createElement("span");
    const wordReview=line.word_clarity?.status==="review";row.className=`readback-row ${wordReview?"review":line.readback.status}`;heading.className="readback-heading";
    label.textContent=`Line ${line.index+1}`;badge.textContent=wordReview?"Check t words":line.readback.numbers_match===false?"Check the numbers":line.readback.status==="match"?"Readback match":line.readback.status==="review"?"Compare wording":"Unchecked";heading.append(label,badge);row.append(heading);
    const link=document.createElement("a"),image=document.createElement("img");link.href=files(job.id,line.preview);link.target="_blank";link.rel="noopener";image.src=link.href;image.alt=`Handwriting for line ${line.index+1}`;image.loading="lazy";link.append(image);row.append(link);
    for(const [title,value] of [["Input",line.text],["Readback",line.readback.status==="unchecked"?"Not checked":line.readback.recognized_text||"No text recognized"]]){
      const text=document.createElement("p"),name=document.createElement("span"),content=document.createElement("span");name.textContent=title;content.textContent=value;text.append(name,content);row.append(text);
    }
    if(line.word_clarity){const note=document.createElement("p");note.textContent=line.word_clarity.words.length?"Separate t-word readback: "+line.word_clarity.words.map(w=>`${w.word} → ${w.recognized_text}${w.exact?" ✓":" (review)"}`).join("; "):line.word_clarity.reason;row.append(note);}
    const button=document.createElement("button");button.className="rewrite-line secondary";button.textContent="Try another version of this line ↻";button.disabled=state.busy;
    button.addEventListener("click",()=>perform(async()=>{const result=await api(`/api/jobs/${state.id}/lines/${line.index}/rewrite`,{});await showDocument(await poll(result.id,"rewrite"));}));row.append(button);$("readback-lines").append(row);
  }
}
async function perform(action){
  error("");busy(true);$("progress").value=0;
  try{await action();}catch(exc){error(exc.message);$("status").textContent="The pen paused. Adjust the settings and try again.";}finally{busy(false);}
}
$("generate").addEventListener("click",()=>perform(async()=>{
  const body={text:$("text").value,seed:number("seed"),bias:number("bias"),form_diversity:number("form-diversity"),consistent_style:$("consistent").checked,quality_checks:$("quality").checked,...(tRefinementKnown?{t_refinement:tRefinementSupported&&$("t-refinement").checked}:{}),reference:state.reference,page:pageSettings(),ink:inkSettings()};
  if(cursiveSelectionSupported) body.prefer_cursive_r=$("consistent").checked&&$("quality").checked&&$("prefer-cursive-r").checked;
  const result=await api("/api/generate",body);await showDocument(await poll(result.id,"generate"));
}));
$("rerender").addEventListener("click",()=>perform(async()=>{
  const result=await api(`/api/jobs/${state.id}/render`,{ink:inkSettings(),page:pageSettings()});await showDocument(await poll(result.id,"render"));
}));
$("export").addEventListener("click",()=>perform(async()=>{
  if(state.sourceDirty)throw new Error("Your words or handwriting settings changed. Generate a new take before exporting.");
  if(state.exportDirty&&state.dirty) throw new Error("Apply your ink and paper settings before exporting.");
  if(!state.exportId){const result=await api(`/api/jobs/${state.id}/export`,{});const job=await poll(result.id,"export");state.exportId=job.id;}
  $("downloads").replaceChildren();
  for(const [name,label] of [["inkform.pdf","Download A4 PDF ↗"],["pages.zip","Download 300 dpi PNGs ↗"]]){const a=document.createElement("a");a.href=files(state.exportId,name);a.textContent=label;a.download=name;$("downloads").append(a);}
  $("downloads").hidden=false;$("status").textContent="Print files are ready. Print at 100% / actual size.";
}));
document.querySelectorAll(".tab").forEach(button=>button.addEventListener("click",()=>{
  document.querySelectorAll(".tab").forEach(x=>x.classList.toggle("active",x===button));
  for(const name of ["compose","ink","paper"]) $(`panel-${name}`).hidden=name!==button.dataset.panel;
}));
for(const id of controls) $(id).addEventListener("input",()=>{updateLabels();if(["naturalness","letter-variation","pressure","density-variation","min-word-gap","slant-adjustment","fit-lines-individually",...pageVariationFields].includes(id))$("variation-preset").value="custom";markDirty(["text","seed","bias","form-diversity","consistent","quality","t-refinement","prefer-cursive-r"].includes(id));});
$("load-example").addEventListener("click",()=>{$("text").value=example;updateLabels();markDirty(true);});
$("new-take").addEventListener("click",()=>{$("seed").value=crypto.getRandomValues(new Uint32Array(1))[0]%2147483647;markDirty(true);});
document.querySelectorAll("[data-color]").forEach(button=>button.addEventListener("click",()=>{if(state.busy)return;$("color").value=button.dataset.color;markDirty();}));
$("reference").addEventListener("change",async()=>{
  error("");const file=$("reference").files[0];if(!file)return;
  try{if(file.size>1500000)throw new Error("The reference JSON must be under 1.5 MB.");const reference=JSON.parse(await file.text());if(!reference.text||!Array.isArray(reference.normalized_offsets))throw new Error("Choose an InkForm style reference JSON.");state.reference=reference;$("reference-state").textContent=`Reference: ${reference.text.slice(0,45)}…`;markDirty(true);}catch(exc){error(exc.message);$("reference").value="";state.reference=null;$("reference-state").textContent="";markDirty(true);}
});
$("clear-reference").addEventListener("click",()=>{state.reference=null;$("reference").value="";$("reference-state").textContent="";markDirty(true);});
$("zoom").addEventListener("click",()=>{$("stage").classList.toggle("zoomed");$("zoom").textContent=$("stage").classList.contains("zoomed")?"Fit to page":"Zoom 100%";});
updateLabels();
api("/api/model").then(model=>{cursiveSelectionSupported=Boolean(model.cursive_r_selection_available);updateCursiveSelection();tRefinementKnown=Boolean(model.t_refinement_available);tRefinementSupported=tRefinementKnown&&model.t_refinement_mode==="closed-word-resampling-v1";if(model.t_refinement_default===false)$("t-refinement").checked=false;$("t-refinement").disabled=state.busy||!tRefinementSupported||!$("quality").checked;$("model-state").textContent=model.available?"CPU model ready":"Model not installed";if(!model.available)error("Download the checkpoint with scripts/download_readable_model.py, then reload this page.");}).catch(()=>{$("model-state").textContent="Offline";error("Cannot reach the local server. Start InkForm and reload this page.");});

const presetControlIds = {naturalness:"naturalness",letter_variation:"letter-variation",density_variation:"density-variation",pressure_variation:"pressure",word_gap_min_mm:"min-word-gap",slant_adjustment_deg:"slant-adjustment",fit_lines_individually:"fit-lines-individually",...Object.fromEntries(pageVariationFields.map(id=>[id.replaceAll("-","_"),id]))};
$("variation-preset").addEventListener("change",()=>{
  const preset=variationPresets[$("variation-preset").value];if(!preset)return;
  for(const [key,value] of Object.entries({...preset.page,...preset.ink})){const control=$(presetControlIds[key]);if(control.type==="checkbox")control.checked=value;else control.value=value;}
  updateLabels();markDirty();
});
$("shuffle-variations").addEventListener("click",()=>{$("variation-seed").value=crypto.getRandomValues(new Uint32Array(1))[0]%2147483647;markDirty();});
api("/api/variation-presets").then(presets=>{variationPresets=presets;expandedLayoutSupported=Object.values(presets).some(preset=>Object.hasOwn(preset.page,"fit_lines_individually")&&Object.hasOwn(preset.page,"line_end_variation"));for(const id of ["line-end-variation","fit-lines-individually"])$(id).disabled=state.busy||!expandedLayoutSupported;slantAdjustmentSupported=Object.values(presets).some(preset=>Object.hasOwn(preset.page,"slant_adjustment_deg"));$("slant-adjustment").disabled=state.busy||!slantAdjustmentSupported;$("slant-adjustment").title=slantAdjustmentSupported?"":"Restart InkForm to enable this control.";gapMinimumSupported=Object.values(presets).some(preset=>Object.hasOwn(preset.page,"word_gap_min_mm"));$("min-word-gap").disabled=state.busy||!gapMinimumSupported;$("min-word-gap").title=gapMinimumSupported?"":"Restart InkForm to enable this control.";for(const [key,preset] of Object.entries(presets)){const option=document.createElement("option");option.value=key;option.textContent=preset.label;$("variation-preset").append(option);}$("variation-preset").disabled=state.busy;}).catch(()=>{error("Variation presets could not load. Reload the page; individual sliders still work.");});
