"""Evaluate saved direction forecasts only. No retraining or portfolio simulation."""
import json,sys
from pathlib import Path
import numpy as np
import btc_slow_overlay as b
from scipy.stats import norm
W=Path('work');dest=W/'btc_direction';dest.mkdir(exist_ok=True)
z=b.base.Z;r=z['returns'];mu=r.mean();sd=r.std(ddof=1)
vol=np.sqrt(z['rv_5m']);vc=vol-vol.mean();den=vc@vc
vol_acf=np.array([vc[k:]@vc[:-k]/den for k in range(1,41)])
old=json.loads((W/'btc_fixed_1min_acf.json').read_text())
for k in [1,2,3,40]:
 centered=r-mu;assert np.isclose(centered[k:]@centered[:-k]/(centered@centered),old['acf'][k-1],atol=1e-12)
tail3=float((abs(r-mu)>3*sd).mean());normal_tail=float(2*norm.sf(3))
meta=json.loads((W/'minute_analysis.json').read_text())
metrics={'source':'Existing Upbit KRW-BTC minute cache; no new data or fitting',
 'data':{'start':str(z['dates'][0]),'end':str(z['dates'][-1]),'days':len(z['dates']),'minutes':len(r),'filled_minutes':int(z['filled_minutes'].sum()),
 'coverage':meta['coverage'],'minute_sd_pct':sd*100,'kurtosis':meta['stats']['minute_return']['kurtosis'],
 'tail_beyond_3sd':tail3,'normal_beyond_3sd':normal_tail,'tail_ratio':tail3/normal_tail,'minute_return_acf':old['acf'][:40],
 'daily_volatility_acf':vol_acf.tolist(),'risk_definition':'daily sqrt(sum of squared 5-minute log returns), not PRV'},'phases':{}}
rows=[]
for phase in ['validation','test']:
 a=np.load(W/f'btc_slow_{phase}_arrays.npz');cut,stop=b.base.phase_bounds(phase);n=stop-cut;entry=a['price'];result={}
 oldresults=json.loads((W/f'btc_slow_{phase}_results.json').read_text())
 for h in [1,7,28]:
  scheduled=np.flatnonzero(a[f'h{h}_review']);complete=scheduled[scheduled+h<=n]
  valid=[]
  for j in complete:
   t=cut+j;end=t+h
   if a[f'h{h}_event'][j] and a[f'h{h}_feature_ok'][j] and b.base.OBS[t] and (b.base.OBS[end] if end<len(b.D) else True):valid.append(j)
  idx=np.array(valid,int);p=a[f'h{h}_p'][idx];ret=entry[idx+h]/entry[idx]-1;y=ret>0;pred=p>.5;active=(p>=.55)|(p<=.45)
  sig=a[f'h{h}_signal'][idx];yy=y[active];ss=sig[active];cm=np.array([[(np.logical_and(ss==s,yy==actual)).sum() for actual in [True,False]] for s in [1,-1]])
  assert int(np.trace(cm))==int(((ss==1)==yy).sum())
  hit=int(((ss==1)==yy).sum());total=int(active.sum());assert total==oldresults['probabilities'][str(h)]['active_full_horizon_signals']
  assert abs(hit/total-oldresults['probabilities'][str(h)]['active_direction_hit_rate'])<1e-12
  out=dict(scheduled_reviews=len(scheduled),complete_horizon_reviews=len(complete),eligible_n=len(idx),signal_n=total,hits=hit,hit_rate=hit/total,
   coverage=total/len(idx),always_up_same_signals=float(yy.mean()),all_cases_accuracy=float((pred==y).mean()),always_up_all_cases=float(y.mean()),
   confusion=cm.tolist(),up_signal_n=int((ss==1).sum()),down_signal_n=int((ss==-1).sum()),zero_returns=int((ret==0).sum()),
   log_loss=oldresults['probabilities'][str(h)]['log_loss'],null_log_loss=oldresults['probabilities'][str(h)]['null_log_loss'],auc=oldresults['probabilities'][str(h)]['auc'])
  for s,name in [(1,'up'),(-1,'down')]:
   mask=ss==s;out[name+'_hit_rate']=float(((ss[mask]==1)==yy[mask]).mean()) if mask.any() else None
  result[str(h)]=out
  for j,prob,actual,rr,ss0,active0 in zip(idx,p,y,ret,sig,active):
   rows.append(dict(phase=phase,horizon_days=h,date_utc=str(a['dates'][j]),end_utc=str(a['dates'][j]+np.timedelta64(h,'D')),probability_up=float(prob),
    actual_up=bool(actual),future_return=float(rr),signal=int(ss0),high_confidence=bool(active0),correct=bool((ss0==1)==actual) if active0 else None))
 metrics['phases'][phase]=result
(dest/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
(dest/'predictions.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(metrics,ensure_ascii=True,indent=2))
