from pathlib import Path
fields=[('width-variation','Letter width',.5),('height-variation','Letter height',.5),('slant-variation','Letter lean',.5),('rhythm-variation','Writing rhythm',.5),('word-spacing-variation','Word spacing',0),('line-start-variation','Line starts',.5),('line-spacing-variation','Line spacing irregularity',.5),('line-tilt-variation','Line tilt',.5),('baseline-variation','Baseline drift',.5)]
p=Path('inkform/static/index.html');s=p.read_text()
needle='          <label class="field-label" for="roughness">'
pos=s.index(needle)
s=s[:pos]+'''          <label class="field-label" for="density-variation">Ink darkness variation <output id="density-variation-value">50%</output></label><input id="density-variation" type="range" min="0" max="1" step="0.05" value="0.5">
'''+s[pos:]
needle='          <label class="field-label" for="naturalness">';pos=s.index(needle)
s=s[:pos]+'''          <label class="field-label" for="variation-preset">Variation preset</label><select id="variation-preset" disabled><option value="custom">Custom</option></select>
'''+s[pos:]
needle='          <p class="fine-note">A shared scale';pos=s.index(needle)
extra='          <details class="variation-group"><summary>Individual variation controls</summary>\n'
for id,label,value in fields:
 extra+=f'          <label class="field-label" for="{id}">{label} <output id="{id}-value">{round(value*100)}%</output></label><input id="{id}" type="range" min="0" max="1" step="0.05" value="{value}">\n'
extra+='''          <label class="field-label" for="variation-seed">Variation seed</label><input id="variation-seed" type="number" min="0" max="2147483647" step="1" value="29">
          <button id="shuffle-variations" class="secondary" type="button">Shuffle variations ↻</button>
          <p class="fine-note">Apply to this take to try another set of variations with the same words and writer. Ink darkness variation is in the Ink tab.</p></details>
'''
s=s[:pos]+extra+s[pos:];p.write_text(s)
p=Path('inkform/static/app.js');s=p.read_text()
start='const controls = '
pos=s.index(start)
s=s[:pos]+'const pageVariationFields = '+str([i for i,_,_ in fields]).replace("'",'"')+';\n'+s[pos:]
s=s.replace('"naturalness","letter-variation"];','"naturalness","letter-variation","density-variation","variation-seed",...pageVariationFields];',1)
s=s.replace('letter_variation:number("letter-variation")','letter_variation:number("letter-variation"),layout_seed:number("variation-seed"),...Object.fromEntries(pageVariationFields.map(id=>[id.replaceAll("-","_"),number(id)]))')
s=s.replace('flow:number("flow"),seed:71','flow:number("flow"),density_variation:number("density-variation"),seed:71')
s=s.replace('"naturalness","letter-variation"])','"naturalness","letter-variation","density-variation",...pageVariationFields])')
s=s.replace('  state.busy=value;','  state.busy=value;\n  $("variation-preset").disabled=value||!Object.keys(variationPresets).length;$("shuffle-variations").disabled=value;')
s=s.replace('const state = {','let variationPresets = {};\nconst state = {',1)
old='for(const id of controls) $(id).addEventListener("input",()=>{updateLabels();markDirty(["text","seed","bias","consistent","quality"].includes(id));});'
new='for(const id of controls) $(id).addEventListener("input",()=>{updateLabels();if(["naturalness","letter-variation","pressure","density-variation",...pageVariationFields].includes(id))$("variation-preset").value="custom";markDirty(["text","seed","bias","consistent","quality"].includes(id));});'
assert old in s;s=s.replace(old,new)
s+='''
const presetControlIds = {naturalness:"naturalness",letter_variation:"letter-variation",density_variation:"density-variation",pressure_variation:"pressure",...Object.fromEntries(pageVariationFields.map(id=>[id.replaceAll("-","_"),id]))};
$("variation-preset").addEventListener("change",()=>{
  const preset=variationPresets[$("variation-preset").value];if(!preset)return;
  for(const [key,value] of Object.entries({...preset.page,...preset.ink}))$(presetControlIds[key]).value=value;
  updateLabels();markDirty();
});
$("shuffle-variations").addEventListener("click",()=>{$("variation-seed").value=crypto.getRandomValues(new Uint32Array(1))[0]%2147483647;markDirty();});
api("/api/variation-presets").then(presets=>{variationPresets=presets;for(const [key,preset] of Object.entries(presets)){const option=document.createElement("option");option.value=key;option.textContent=preset.label;$("variation-preset").append(option);}$("variation-preset").disabled=state.busy;}).catch(()=>{error("Variation presets could not load. Reload the page; individual sliders still work.");});
'''
p.write_text(s)
