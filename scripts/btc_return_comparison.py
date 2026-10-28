"""Fixed-coefficient, one-day BTC log-return forecasts. No portfolio simulation."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path('work/model_packages').resolve()))
import numpy as np
from arch import arch_model
z=np.load('work/minute_plot_arrays.npz');d=z['dates'];r=z['daily_log_return']*100
assert np.allclose(r,z['returns'].reshape(-1,1440).sum(axis=1)*100)
x=np.array([[1,r[t-1],r[t-7:t].mean(),r[t-30:t].mean()] for t in range(30,len(r))])
y=r[30:];dates=d[30:]
names=['Zero','Mean','MA7','MA30','AR1','HAR_return','GARCH_t_mean','GJR_t_mean']
out={'target':'Next complete UTC day log return in percentage points; forecasts at day start from completed prior days.',
 'protocol':'All days, no prior-down filter; identical targets and training sample from day 30. Fixed coefficients: train through 2023 for validation, refit through 2025 for 2026. Inputs update daily. Eight candidates declared before evaluation. Validation MSE selection, no test tuning.',
 'sources':['https://arch.readthedocs.io/en/stable/univariate/mean.html'], 'phases':{}}
arrays={}
for phase,start,end in [('validation','2024-01-01','2026-01-01'),('test','2026-01-01','2026-10-12')]:
 tr=dates<np.datetime64(start);te=(dates>=np.datetime64(start))&(dates<np.datetime64(end));xt=x[te];yt=y[te]
 assert dates[tr][-1]<dates[te][0]
 b1=np.linalg.lstsq(x[tr,:2],y[tr],rcond=None)[0];bh=np.linalg.lstsq(x[tr],y[tr],rcond=None)[0]
 pred={'Zero':np.zeros(len(yt)),'Mean':np.full(len(yt),y[tr].mean()),'MA7':xt[:,2],'MA30':xt[:,3],'AR1':xt[:,:2]@b1,'HAR_return':xt@bh}
 params={'AR1':b1.tolist(),'HAR_return':bh.tolist()}
 for name,o in [('GARCH_t_mean',0),('GJR_t_mean',1)]:
  model=arch_model(y[tr],mean='Constant',vol='GARCH',p=1,o=o,q=1,dist='t',rescale=False)
  fit=model.fit(disp='off',show_warning=True)
  assert fit.convergence_flag==0
  pred[name]=np.full(len(yt),float(fit.params['mu']))
  params[name]={k:float(v) for k,v in fit.params.items()}
 rows=[]
 for name in names:
  p=pred[name];e=yt-p;mse=float(np.mean(e**2))
  rows.append(dict(model=name,mse=mse,rmse=float(np.sqrt(mse)),mae=float(np.mean(abs(e))),r2_zero=float(1-mse/np.mean(yt**2)),direction_accuracy=None if name=='Zero' else float(np.mean((p>0)==(yt>0))),n=len(yt)))
  arrays[phase+'_'+name]=p
 arrays[phase+'_actual']=yt;arrays[phase+'_dates']=dates[te]
 out['phases'][phase]={'train_n':int(tr.sum()),'n':len(yt),'first':str(dates[te][0]),'last':str(dates[te][-1]),'scores':rows,'parameters':params,'always_up_accuracy':float(np.mean(yt>0))}
out['selected_by_validation_mse']=min(out['phases']['validation']['scores'],key=lambda v:v['mse'])['model']
# Paired moving-block bootstrap: HAR minus zero squared error, preserve local dependence.
diff=(arrays['test_actual']-arrays['test_HAR_return'])**2-arrays['test_actual']**2
rng=np.random.default_rng(20261014);n=len(diff);starts=rng.integers(0,n,size=(3000,int(np.ceil(n/7))));indices=(starts[:,:,None]+np.arange(7))%n
boot=diff[indices.reshape(3000,-1)[:,:n]].mean(axis=1)
out['har_minus_zero_test_mse_ci95']=np.quantile(boot,[.025,.975]).tolist()
out['bootstrap']='3000 circular moving-block replicates, block length 7 days; descriptive, not multiple-testing adjusted.'
p=Path('work/btc_direction');(p/'return_comparison.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
np.savez(p/'return_forecasts.npz',**arrays)
print(json.dumps(out,ensure_ascii=True))
