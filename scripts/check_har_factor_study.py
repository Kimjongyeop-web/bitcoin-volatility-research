"""Independent numpy reconstruction and report checks."""
from pathlib import Path
import json,base64
from html.parser import HTMLParser
import numpy as np
from pypdf import PdfReader
W=Path('work/har_factor_study');result=json.loads((W/'results.json').read_text(encoding='utf-8'))
data=np.load(W/'daily_inputs.npz');z=np.load('work/minute_plot_arrays.npz');p=np.load(W/'predictions.npz')
d=z['dates'];v=z['rv_5m'];r=z['returns'].reshape(-1,1440);volume=data['volume_btc']
assert np.array_equal(d,data['dates']) and np.allclose(data['candles']/1440,z['coverage_ratio'])
def features(t,v,volume,r):
 five=r[t-1].reshape(288,5).sum(1)
 return [np.log(v[t-1]),np.log(v[t-7:t]).mean(),np.log(v[t-30:t]).mean(),np.log(volume[t-1]/volume[t-31:t-1].mean()),(five[five<0]**2).sum()/v[t-1]]
x=np.array([features(t,v,volume,r) for t in range(31,len(d))]);dates=d[31:]
# Perturb target day and every later observation; same-day forecast features must not change.
for t in [31,1000,2288,len(d)-1]:
 vv=v.copy();vol=volume.copy();rr=r.copy();vv[t:]*=9;vol[t:]*=7;rr[t:]*=3
 assert np.allclose(features(t,v,volume,r),features(t,vv,vol,rr),rtol=0,atol=0)
cols={'HAR':[0,1,2],'Factors':[3,4],'HAR-X':[0,1,2,3,4]}
for phase,start,end in [('validation','2024-01-01','2026-01-01'),('test','2026-01-01','2026-10-12')]:
 tr=dates<np.datetime64(start);te=(dates>=np.datetime64(start))&(dates<np.datetime64(end));actual=v[31:][te]
 assert dates[tr][-1]<dates[te][0]
 assert np.array_equal(p[phase+'_dates'],dates[te]) and np.array_equal(p[phase+'_actual'],actual)
 for model,idx in cols.items():
  xx=np.column_stack([np.ones(tr.sum()),x[tr][:,idx]]);yy=np.log(v[31:][tr]);beta=np.linalg.lstsq(xx,yy,rcond=None)[0]
  forecast=np.exp(np.column_stack([np.ones(te.sum()),x[te][:,idx]])@beta)*np.exp(yy-xx@beta).mean()
  assert np.allclose(forecast,p[phase+'_'+model],rtol=1e-10,atol=1e-15)
  s=result['phases'][phase]['models'][model]['scores'];ratio=actual/forecast
  assert np.isclose(s['qlike'],np.mean(ratio-np.log(ratio)-1),rtol=1e-10)
  assert np.isclose(s['mse'],np.mean((actual-forecast)**2),rtol=1e-10,atol=1e-18)
class Parser(HTMLParser):
 def __init__(self):super().__init__();self.images=[]
 def handle_starttag(self,tag,attrs):
  if tag=='img':self.images.append(dict(attrs)['src'])
pages=json.loads((W/'report_pages.json').read_text(encoding='utf-8'));pdf=PdfReader('output/pdf/har_factor_study.pdf');h=Parser();h.feed(Path('outputs/har_factor_study.html').read_text(encoding='utf-8'))
assert len(pages)==len(pdf.pages)==len(h.images)
for page,png,src in zip(pages,pdf.pages,h.images):
 assert page['title'] in png.extract_text()
 assert base64.b64decode(src.split(',',1)[1])==Path(page['figure']).read_bytes()
checks={'status':'passed','models_independently_recomputed':6,'causality_checks':4,'coverage_matches_raw_candles':True,'pdf_pages':len(pages),'html_images_match':True}
(W/'checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8');print(json.dumps(checks))
