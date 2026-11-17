import json,torch
from pathlib import Path
from inkform.readable_generator import ReadableGenerator
from inkform.word_clarity import word_regions
from inkform.recognition import recognition_image
from torch.nn import functional as F
g=ReadableGenerator();r=g.recognizer_factory();d=json.loads(Path('tmp/t-context/probe-509-My/0.json').read_text());_,labels=g.t_evaluator.character_trace(d['text'],d['offsets'],g.default_reference);off=word_regions(d['offsets'],labels,d['text'])[2]['offsets'];im=recognition_image(off);p=r.processor(images=im,return_tensors='pt').pixel_values;ids=r.processor.tokenizer('stayed',return_tensors='pt').input_ids
with torch.inference_mode():
 e=r.model.encoder(pixel_values=p)
 pred=r.model.generate(pixel_values=p,encoder_outputs=e,max_new_tokens=20)
 print('ids',ids,'generated',pred,'vocab',r.processor.tokenizer.convert_ids_to_tokens(ids[0].tolist()),r.processor.tokenizer.convert_ids_to_tokens(pred[0].tolist()),flush=True)
 for start in [2,0]:
  di=torch.cat((torch.tensor([[start]]),ids[:,:-1]),1);log=r.model(encoder_outputs=e,decoder_input_ids=di).logits
  print('decoder',di,'top',log.argmax(-1),'CE',F.cross_entropy(log.flatten(0,1),ids.flatten()),'shifted',F.cross_entropy(log[:,:-1].flatten(0,1),ids[:,1:].flatten()),flush=True)
