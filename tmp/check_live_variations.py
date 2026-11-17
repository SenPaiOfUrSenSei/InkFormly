import httpx
with httpx.Client(base_url='http://127.0.0.1:8000',timeout=30,trust_env=False) as c:
 m=c.get('/api/model').raise_for_status().json();assert m['backend']=='Vasquez · writer 2' and m['available']
 p=c.get('/api/variation-presets').raise_for_status().json();assert len(p)==4
 html=c.get('/').raise_for_status().text;js=c.get('/app.js').raise_for_status().text
 for field in ('width-variation','word-spacing-variation','variation-seed','density-variation','variation-preset'):assert f'id="{field}"' in html
 assert 'pageVariationFields' in js
 assert c.post('/api/generate',json={'text':'Q','quality_checks':False}).status_code==422
 print({'backend':m['backend'],'presets':[v['label'] for v in p.values()],'controls_served':True,'unsupported_character_rejection':True})
