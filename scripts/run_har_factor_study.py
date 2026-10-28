"""Prespecified HAR vs factors vs HAR-X; retrospective, no portfolio or live trading."""
import os,sys,json,gzip,hashlib,warnings,base64,html
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path('work/model_packages').resolve()))
sys.path.insert(0,str(Path('work/plot_packages').resolve()))
os.environ['MPLCONFIGDIR']=str(Path('work/mpl_config').resolve())
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller,kpss,acf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import acorr_ljungbox,het_arch,breaks_cusumolsresid,linear_reset
from statsmodels.stats.multitest import multipletests
from scipy.stats import jarque_bera
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

W=Path('work/har_factor_study');W.mkdir(exist_ok=True)
def native(v):
 if isinstance(v,dict):return {str(k):native(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [native(x) for x in v]
 if isinstance(v,np.ndarray):return native(v.tolist())
 if isinstance(v,(np.integer,)):return int(v)
 if isinstance(v,(np.bool_,)):return bool(v)
 if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
 return v
def save(name,v): (W/name).write_text(json.dumps(native(v),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
protocol=dict(date='2026-10-15',target='Next UTC-day log RV; 5-minute returns; fixed OLS, training-only Duan smearing.',
 factors=['log(BTC volume yesterday / mean BTC volume over preceding 30 days, excluding yesterday)','negative 5-minute squared-return sum yesterday / yesterday RV'],
 models=['HAR','Factors','HAR-X'],primary_loss='QLIKE',secondary_loss='MSE',
 validation='2024-2025; train through 2023',test='2026-01-01 through 2026-10-11; refit through 2025',
 common_sample_start='31 days after first full UTC day; all models share dates; features strictly before target day',
 inference='HAC 7 lags; sensitivity 1 and 30; Holm correction for two factor coefficients; joint robust Wald',
 bootstrap='5000 circular moving-block samples of paired evaluation losses, blocks 7 and 30; fixed fitted forecasts, descriptive not nested-model p-values',
 sensitivity='99% target/prior-day coverage subset; training Cook influence top 1% removal (diagnostic only); 10-minute RV target rerun',
 stationarity='ADF AIC and KPSS auto on training variables; inspect warnings, no automatic differencing',
 selection='No tuning from diagnostics or test. All three models reported. Annual splits descriptive.',
 limitations='2026 already examined: retrospective. Upbit KRW local market. Conditional associations, not causal effects. RV measurement error; no new untouched test available.')
if (W/'protocol.json').exists():assert json.loads((W/'protocol.json').read_text(encoding='utf-8'))==protocol
else:save('protocol.json',protocol)
z=np.load('work/minute_plot_arrays.npz');D=z['dates'];R=z['returns'].reshape(-1,1440);V=z['rv_5m'];N=len(D)
assert np.allclose(R.reshape(N,288,5).sum(2).__pow__(2).sum(1),V)
assert np.all(np.diff(D)==np.timedelta64(1,'D'))
vol=np.zeros(N);turn=np.zeros(N);counts=np.zeros(N,dtype=int);mapping={str(d):i for i,d in enumerate(D)}
duplicate=0;invalid=0;cache_files=sorted(Path('work/minute_cache').glob('*.json.gz'))
for fi,f in enumerate(cache_files):
 rows=json.load(gzip.open(f,'rt',encoding='utf-8'));seen=set()
 for row in rows:
  stamp=row['candle_date_time_utc'];idx=mapping.get(stamp[:10])
  if idx is None:continue
  if stamp in seen:duplicate+=1;continue
  seen.add(stamp)
  a=float(row['candle_acc_trade_volume']);b=float(row['candle_acc_trade_price'])
  if not (np.isfinite(a) and np.isfinite(b) and a>=0 and b>=0):invalid+=1;continue
  vol[idx]+=a;turn[idx]+=b;counts[idx]+=1
 if fi%25==0:print('cache',fi+1,len(cache_files),flush=True)
assert duplicate==0 and invalid==0 and (vol>0).all()
assert np.all(counts<=1440)
np.savez(W/'daily_inputs.npz',dates=D,volume_btc=vol,turnover_krw=turn,candles=counts,rv=V)
def design(v,step=5):
 five=R.reshape(N,1440//step,step).sum(2);down=(np.where(five<0,five**2,0).sum(1)/v)
 x=np.array([[np.log(v[t-1]),np.log(v[t-7:t]).mean(),np.log(v[t-30:t]).mean(),np.log(vol[t-1]/vol[t-31:t-1].mean()),down[t-1]] for t in range(31,N)])
 return x
X=design(V);dates=D[31:];actual=V[31:];y=np.log(actual)
assert np.isfinite(X).all() and np.all((X[:,-1]>=0)&(X[:,-1]<=1))
columns={'HAR':[0,1,2],'Factors':[3,4],'HAR-X':[0,1,2,3,4]}
names=['logRV_1d','mean_logRV_7d','mean_logRV_30d','volume_shock','downside_share']
def score(a,p):
 ratio=a/p
 return dict(n=len(a),qlike=np.mean(ratio-np.log(ratio)-1),mse=np.mean((a-p)**2),mean_actual=a.mean(),mean_prediction=p.mean(),mean_error=np.mean(p-a),vol_mae_pct=np.mean(abs(np.sqrt(a)-np.sqrt(p)))*100)
def fitmodel(x,yy,idx):
 mu=x[:,idx].mean(0);sd=x[:,idx].std(0);assert (sd>0).all()
 design=sm.add_constant((x[:,idx]-mu)/sd,has_constant='add');fit=sm.OLS(yy,design).fit()
 return fit,mu,sd
def predict(model,x,idx):
 fit,mu,sd=model;design=sm.add_constant((x[:,idx]-mu)/sd,has_constant='add')
 return np.exp(design@fit.params)*np.exp(fit.resid).mean()
def diagnostics(fit):
 robust=fit.get_robustcov_results(cov_type='HAC',maxlags=7)
 infl=fit.get_influence();cook=infl.cooks_distance[0]
 with warnings.catch_warnings(record=True) as ws:
  reset=linear_reset(fit,power=2,use_f=False,cov_type='HAC',cov_kwds={'maxlags':7})
 return dict(coefficients_standardized=fit.params,hac7_se=robust.bse,hac7_p=robust.pvalues,hac7_ci=robust.conf_int(),
  lb_p=acorr_ljungbox(fit.resid,lags=[7,30],return_df=True)['lb_pvalue'].values,
  arch_lm_p=het_arch(fit.resid,nlags=7,ddof=len(fit.params))[1],
  cusum_p=breaks_cusumolsresid(fit.resid,ddof=len(fit.params))[1],reset_hac_p=float(reset.pvalue),
  jarque_bera_p=jarque_bera(fit.resid).pvalue,cook_max=cook.max(),cook_above_4n=int((cook>4/len(cook)).sum()),
  r2=fit.rsquared,hac_sensitivity={str(l):fit.get_robustcov_results(cov_type='HAC',maxlags=l).pvalues for l in [1,30]},warnings=[str(w.message) for w in ws])
def bootstrap(a,b,block):
 diff=a-b;n=len(diff);rng=np.random.default_rng(20261015+block)
 starts=rng.integers(0,n,size=(5000,int(np.ceil(n/block))));idx=((starts[:,:,None]+np.arange(block))%n).reshape(5000,-1)[:,:n]
 return dict(mean=diff.mean(),ci95=np.quantile(diff[idx].mean(1),[.025,.975]))
out={'protocol':protocol,'quality':{'cache_files':len(cache_files),'days':N,'duplicate_rows':duplicate,'invalid_volume_rows':invalid,'missing_minutes':int((1440-counts).sum()),'minimum_daily_coverage':float((counts/1440).min()),'volume_missing_policy':'Sum observed trades only; never forward-fill volume. Missing candles may be no-trade intervals or acquisition gaps; cannot distinguish from cached candles alone.'},'phases':{}}
preds={};fits={}
for phase,start,end in [('validation','2024-01-01','2026-01-01'),('test','2026-01-01','2026-10-12')]:
 train=dates<np.datetime64(start);test=(dates>=np.datetime64(start))&(dates<np.datetime64(end));a=actual[test]
 print('phase',phase,train.sum(),test.sum(),flush=True)
 station={}
 for name,series in [('target_logRV',y[train])]+[(name,X[train,j]) for j,name in enumerate(names)]:
  with warnings.catch_warnings(record=True) as ws:
   ad=adfuller(series,autolag='AIC');kp=kpss(series,regression='c',nlags='auto')
  station[name]={'adf_p':ad[1],'adf_lags':ad[2],'kpss_p':kp[1],'kpss_lags':kp[2],'warnings':[str(w.message) for w in ws]}
 allx=sm.add_constant((X[train]-X[train].mean(0))/X[train].std(0))
 phaseout={'train_n':int(train.sum()),'eval_n':int(test.sum()),'stationarity':station,'correlations':np.corrcoef(X[train],rowvar=False),'vif':{name:variance_inflation_factor(allx,j+1) for j,name in enumerate(names)},'models':{}}
 for name,cols in columns.items():
  model=fitmodel(X[train],y[train],cols);fit=model[0];p=predict(model,X[test],cols);assert np.isfinite(p).all() and (p>0).all()
  diag=diagnostics(fit);diag['terms']=['constant']+[names[j] for j in cols]
  if name=='HAR-X':
   robust=fit.get_robustcov_results(cov_type='HAC',maxlags=7);restriction=np.zeros((2,len(fit.params)));restriction[0,-2]=1;restriction[1,-1]=1
   diag['factor_joint_wald_p']=float(robust.wald_test(restriction,scalar=True).pvalue)
   diag['factor_holm_p']=multipletests(robust.pvalues[-2:],method='holm')[1]
  # Influence sensitivity refits only on training data; evaluate the unchanged holdout.
  cook=fit.get_influence().cooks_distance[0];keep=cook<=np.quantile(cook,.99)
  sens=predict(fitmodel(X[train][keep],y[train][keep],cols),X[test],cols)
  good=(counts[31:][test]/1440>=.99)&(counts[30:-1][test]/1440>=.99)
  calibration=sm.OLS(a*1e4,sm.add_constant(p*1e4)).fit(cov_type='HAC',cov_kwds={'maxlags':7})
  phaseout['models'][name]={'scores':score(a,p),'diagnostics':diag,'smearing':np.exp(fit.resid).mean(),'high_coverage_scores':score(a[good],p[good]),'influence_sensitivity_scores':score(a,sens),'calibration_actual_on_forecast':{'coef':calibration.params,'ci95':calibration.conf_int()},'annual':{year:score(a[np.array([str(d).startswith(year) for d in dates[test]])],p[np.array([str(d).startswith(year) for d in dates[test]])]) for year in sorted(set(str(d)[:4] for d in dates[test]))}}
  preds[phase+'_'+name]=p;fits[phase+'_'+name]=fit
 for other in ['Factors','HAR-X']:
  p=preds[phase+'_HAR'];q=preds[phase+'_'+other];loss1=a/p-np.log(a/p)-1;loss2=a/q-np.log(a/q)-1
  phaseout[other+'_minus_HAR']={metric:{str(block):bootstrap(v,u,block) for block in [7,30]} for metric,u,v in [('QLIKE',loss1,loss2),('MSE',(a-p)**2,(a-q)**2)]}
 out['phases'][phase]=phaseout
 preds[phase+'_actual']=a;preds[phase+'_dates']=dates[test]
save('results.json',out);np.savez(W/'predictions.npz',**preds)
# Measurement sensitivity: identical three models and cutoffs with 10-minute RV.
v10=(R.reshape(N,144,10).sum(2)**2).sum(1);x10=design(v10,10);sens10={}
for phase,start,end in [('validation','2024-01-01','2026-01-01'),('test','2026-01-01','2026-10-12')]:
 tr=dates<np.datetime64(start);te=(dates>=np.datetime64(start))&(dates<np.datetime64(end))
 sens10[phase]={name:score(v10[31:][te],predict(fitmodel(x10[tr],np.log(v10[31:][tr]),cols),x10[te],cols)) for name,cols in columns.items()}
out['rv10_sensitivity']=sens10
# Rolling 3-year coefficients at annual cutoffs: diagnostic only, not model selection.
stability=[]
for year in range(2021,2027):
 tr=(dates>=np.datetime64(f'{year-3}-01-01'))&(dates<np.datetime64(f'{year}-01-01'))
 fit,mu,sd=fitmodel(X[tr],y[tr],columns['HAR-X']);rob=fit.get_robustcov_results(cov_type='HAC',maxlags=7)
 stability.append(dict(cutoff=year,n=int(tr.sum()),factor_coef_original_units=fit.params[-2:]/sd[-2:],ci95_original_units=rob.conf_int()[-2:]/sd[-2:,None]))
out['rolling_coefficient_diagnostic']=stability
save('results.json',out)
print('DONE',json.dumps(native({p:{m:out['phases'][p]['models'][m]['scores'] for m in columns} for p in out['phases']})),flush=True)
