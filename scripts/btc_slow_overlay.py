"""Prespecified low-frequency BTC research, with horizon-matched labels.

Run from project root: python work/btc_slow_overlay.py validation|test
No orders, no derivatives data, no optimization on the new results.
"""
import sys,json,hashlib,math
from pathlib import Path
import numpy as np
import btc_downside_longshort as base
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata

W=Path('work');D=base.D;N=len(D);EP=base.EP
PROTOCOL={
 'revision':'2026-10-14: reference PDF and lower trading frequency',
 'source_pdf':'references/2023010413442181K_02_02.pdf',
 'source_interpretation':'Samsung Securities 2023-01-04: multi-stock benchmark plus long/short tilts, negative weights removed by equal percentage-point reductions; monthly stock lists. BTC adaptation, not replication or a prescribed 130/30 allocation.',
 'data_sha256':base.PROTOCOL['data_sha256'],
 'horizons':[1,7,28], 'primary_design':'7-day core 50% plus volatility-scaled tilt capped at +/-25 percentage points',
 'other_designs':'Pure signed exposure capped at +/-25%; 1-day and 28-day comparisons; fixed 50% BTC at each schedule; original daily strategy as historical comparator.',
 'comparison_status':'All variants reported, no winner selected. 2026 has already been examined; this is retrospective research, not a new blind test.',
 'training':'Last 3 calendar years, monthly coefficient refit, daily overlapping horizon labels; strong L2 lambda 0.1 retained without new tuning.',
 'label':'Y_t = 1[EP[t+h]/EP[t]-1 > 0], EP is the first UTC minute closing price (available at 00:01 UTC).',
 'event':'Sum of the preceding h completed UTC daily log returns < 0. Only event observations train the conditional classifier; trade signal is neutral outside this event.',
 'features':'Same nine baseline features; first shock and large-drop features now refer to preceding h days, normalized by sqrt(h) times the preceding reference daily SD. Other features remain daily 7/30-day trend, drawdown, RV shock, downside share and streak.',
 'threshold':'p >= .55 long tilt; p <= .45 short tilt; otherwise 0. Fixed for all horizons without another threshold search.',
 'training_embargo':'At monthly information cutoff t 00:00 UTC, all training label endpoints must be <= day t-1 at 00:01. Last training start <= t-h-1.',
 'quality':'Preceding h-day mean coverage >=95% for features; future h-day mean coverage >=95% and observed endpoints for training. Test returns are not filtered by future quality.',
 'schedule':'Every h calendar days starting Jan 1 of each evaluation phase, 00:01 UTC = 09:01 KST. 28 days is four weeks, not a calendar month. Final partial holding interval is liquidated and included in PnL, excluded from full-horizon probability scores.',
 'risk':'a_t = .25 * min(1, .01/sqrt(fixed-HAR next-day variance forecast)). This is next-day risk sizing at a review date, not a calibrated h-day variance forecast or a guaranteed realized risk target.',
 'targets':'core: .50 + signal*a_t, so scheduled targets are 25% to 75%; pure: signal*a_t, so scheduled targets -25% to +25%. Short means negative NET BTC position; no simultaneous long and short in the same asset.',
 'no_trade_band':'At review dates only, do not trade if target minus actual weight is less than 5 percentage points. Always close nonzero pure exposure on neutral signal, execute sign reversals, and correct positions outside policy bounds. Opening trades obey the same band except bound corrections.',
 'between_reviews':'Hold BTC quantity unchanged, cash changes only for synthetic short financing; prices cause weight drift. No daily target-weight rebalancing, automatic stop loss or emergency risk rule.',
 'benchmarks':'50% BTC / 50% cash on the same schedule and with the same 5 percentage-point band; BTC buy-and-hold; cash.',
 'fee_one_way':.001,'short_borrow_annual':.10,'cash_interest':0,'initial_capital_krw':10000000,
 'cost_sensitivity':[0,.0005,.001,.0025], 'borrow_sensitivity':[0,.10,.20],
 'validation':['2024-01-01','2025-12-31'],'test':['2026-01-01','2026-10-11'],
 'limitations':'KRW spot-based synthetic signed ledger, not a tradable Upbit short product or a derivatives backtest. No actual funding, basis, borrow availability, margin or liquidation. Overlapping labels have dependent observations; 28-day OOS decisions are especially few.',
 'inference':'56-day circular block bootstrap, 3000 replicates, paired daily net-return differences; descriptive intervals, not a correction for all prior exploratory research.'
}
def save(name,obj):base.save(name,obj)
def lock():
 p=W/'btc_slow_protocol.json'
 if p.exists():assert json.loads(p.read_text(encoding='utf-8'))==PROTOCOL
 else:save(p.name,PROTOCOL)
 return hashlib.sha256(p.read_bytes()).hexdigest()
def horizon_data(h):
 x=base.X.copy();event=np.zeros(N,bool);feature_ok=np.zeros(N,bool);train_ok=np.zeros(N,bool);y=np.full(N,np.nan)
 for t in range(31,N):
  ref=base.V[t-31:t-1].mean();shock=base.RET[t-h:t].sum()/np.sqrt(h*ref)
  x[t,0]=shock;x[t,7]=float(shock<=-1);x[t,8]=x[t,7]*x[t,1]
  event[t]=base.RET[t-h:t].sum()<0
  feature_ok[t]=base.COVER[t-h:t].mean()>=.95 and np.isfinite(x[t]).all()
  if t+h<=N:
   y[t]=float(EP[t+h]>EP[t])
   endpoint_obs=base.OBS[t+h] if t+h<N else True
   train_ok[t]=event[t] and feature_ok[t] and base.COVER[t:t+h].mean()>=.95 and base.OBS[t] and endpoint_obs
 return x,event,feature_ok,train_ok,y
def fit(t,h,data):
 x,event,fok,tok,y=data;idx=np.arange(base.train_start(t),t-h);idx=idx[tok[idx]]
 assert len(idx)>=100 and idx[-1]+h<t
 mu=x[idx].mean(0);sd=x[idx].std(0);sd=np.where(sd<1e-12,1,sd)
 z=np.column_stack([np.ones(len(idx)),(x[idx]-mu)/sd]);yy=y[idx];lam=.1
 def fun(b):
  a=z@b;g=z.T@(expit(a)-yy)/len(idx);g[1:]+=lam*b[1:]
  return np.mean(np.logaddexp(0,a)-yy*a)+lam/2*np.sum(b[1:]**2),g
 p0=(yy.sum()+1)/(len(yy)+2);init=np.zeros(z.shape[1]);init[0]=np.log(p0/(1-p0))
 out=minimize(fun,init,jac=True,method='L-BFGS-B',options={'maxiter':2000,'ftol':1e-13,'gtol':1e-8})
 assert out.success and abs(fun(out.x)[1]).max()<1e-5
 return dict(refit_index=t,refit_date=str(D[t]),horizon=h,indices=idx,training_n=len(idx),training_last=str(D[idx[-1]]),
  last_endpoint=str(D[idx[-1]+h])+'T00:01:00Z',mean=mu,scale=sd,coef=out.x,p_null=p0)
def predict(cut,stop,h,data):
 rows=[];p=[];null=[];ids=[];last=None
 for t in range(cut,stop):
  month=str(D[t])[:7]
  if month!=last:rows.append(fit(t,h,data));last=month
  m=rows[-1];p.append(expit(m['coef'][0]+((data[0][t]-m['mean'])/m['scale'])@m['coef'][1:]));null.append(m['p_null']);ids.append(len(rows)-1)
 return np.array(p),np.array(null),rows,np.array(ids)
def ledger(cut,stop,target,review,kind,fee=.001,borrow=.1,band=.05,intraday=True):
 q=0.;cash=1e7;previous=1e7;peak=1e7;minute_mdd=0.;max_gross=0.;events=[];bounds=(.25,.75) if kind=='core' else ((-.25,.25) if kind=='pure' else (0,1))
 arrays={k:[] for k in ['nav','returns','fees','borrow','gross','weights','q','cash','trade_notional','trade_count','turnover','pre_weight','target','review']}
 skips=0
 for j,t in enumerate(range(cut,stop)):
  price=EP[t];b=q*price;equity=cash+b;assert equity>0;pre=b/equity;w=float(target[j]);traded=0.;paid=0.;count=0;fin=0.
  if review[j]:
   if not base.OBS[t]:skips+=1
   else:
    force=(pre<bounds[0]-1e-10 or pre>bounds[1]+1e-10) if kind in ['core','pure'] else False
    if kind=='pure':force=force or (w==0 and abs(pre)>1e-10) or w*pre<0
    if force or abs(w-pre)>=band-1e-12:
     delta=w*equity-b;traded=delta/(1+fee*w) if delta>=0 else delta/(1-fee*w)
     if abs(traded)>1e-7:
      paid=fee*abs(traded);q+=traded/price;cash-=traded+paid;count=1
      assert abs(q*price/(cash+q*price)-w)<1e-9
      events.append(dict(date_utc=str(D[t]),time_kst='09:01',kind='rebalance',pre_weight=pre,target=w,notional=traded,fee=paid))
  after=cash+q*price;b=q*price;position=b/after
  fin=max(-b,0)*borrow/365;pnl=q*(EP[t+1]-price)
  if intraday:
   rel=np.r_[1,np.exp(np.cumsum(base.R[t,1:])),EP[t+1]/price]
   path=after+b*(rel-1)-fin*np.linspace(0,1,len(rel));assert (path>0).all()
   peaks=np.maximum.accumulate(np.r_[peak,path])[1:];minute_mdd=min(minute_mdd,float(np.min(path/peaks-1)));peak=peaks[-1]
   max_gross=max(max_gross,float(np.max(abs(b*rel)/path)))
  q_during=q;cash_after_trade=cash;cash-=fin;turn=abs(traded)/equity
  exit_notional=0.
  if j==stop-cut-1 and abs(q)>1e-12:
   exit_notional=-q*EP[t+1];before_exit=cash+q*EP[t+1];exit_fee=fee*abs(exit_notional)
   turn+=abs(exit_notional)/before_exit;paid+=exit_fee;cash-=exit_notional+exit_fee;q=0.;count+=1
   events.append(dict(date_utc=str(np.r_[D,np.datetime64('2026-10-12')][t+1]),time_kst='09:01',kind='final_exit',notional=exit_notional,fee=exit_fee))
  end=cash+q*EP[t+1];minute_mdd=min(minute_mdd,end/peak-1)
  values=[end,end/previous-1,paid,fin,pnl,position,q_during,cash_after_trade,traded,count,turn,pre,w,bool(review[j])]
  for k,v in zip(arrays,values):arrays[k].append(v)
  previous=end
 arrays={k:np.array(v) for k,v in arrays.items()};rr=arrays['returns'];nav=np.r_[1e7,arrays['nav']];dd=nav/np.maximum.accumulate(nav)-1;arrays['drawdown']=dd[1:]
 std=rr.std(ddof=1);pos=arrays['weights'];stats=dict(return_net=nav[-1]/1e7-1,sharpe=rr.mean()/std*np.sqrt(365) if std else None,annual_vol=std*np.sqrt(365),mdd=dd.min(),minute_mdd=minute_mdd if intraday else None,
  final_nav=nav[-1],fees_krw=arrays['fees'].sum(),borrow_krw=arrays['borrow'].sum(),gross_pnl_krw=arrays['gross'].sum(),turnover=arrays['turnover'].sum(),
  scheduled_reviews=int(np.sum(review)),rebalance_trades=sum(x['kind']=='rebalance' for x in events),final_exit_trades=sum(x['kind']=='final_exit' for x in events),
  actual_trade_count=int(arrays['trade_count'].sum()),long_days=int((pos>1e-10).sum()),short_days=int((pos<-1e-10).sum()),flat_days=int((abs(pos)<=1e-10).sum()),
  average_abs_weight=abs(pos).mean(),min_weight=pos.min(),max_weight=pos.max(),max_intraday_gross=max_gross if intraday else None,missing_execution_skips=skips)
 assert np.isclose(1e7+stats['gross_pnl_krw']-stats['fees_krw']-stats['borrow_krw'],nav[-1],atol=1e-5,rtol=1e-11)
 return dict(stats=stats,arrays=arrays,trades=events)
def probability_scores(cut,stop,h,data,p,null,review):
 idx=np.arange(cut,stop);mask=review & data[1][cut:stop] & data[2][cut:stop] & (idx+h<=stop)
 mask &= base.OBS[cut:stop]
 for j in np.flatnonzero(mask):
  endpoint=idx[j]+h
  if endpoint<N and not base.OBS[endpoint]:mask[j]=False
 yy=data[4][cut:stop][mask];pp=p[mask];bb=null[mask]
 if not len(yy):return dict(n=0)
 loss=lambda q:float(np.mean(-(yy*np.log(q)+(1-yy)*np.log1p(-q))))
 n1=int(yy.sum());n0=len(yy)-n1
 auc=(rankdata(pp)[yy==1].sum()-n1*(n1+1)/2)/(n1*n0) if n1 and n0 else None
 active=(pp>=.55)|(pp<=.45)
 return dict(n=len(yy),log_loss=loss(pp),null_log_loss=loss(bb),auc=auc,active_full_horizon_signals=int(active.sum()),
  active_direction_hit_rate=float(((pp[active]>.5)==yy[active]).mean()) if active.any() else None,up_rate=yy.mean())
def paired(a,b):
 dif=a-b;n=len(dif);rng=np.random.default_rng(420);block=56
 st=rng.integers(0,n,size=(3000,math.ceil(n/block)));idx=((st[:,:,None]+np.arange(block))%n).reshape(3000,-1)[:,:n];means=dif[idx].mean(1)
 sd=dif.std(ddof=1)
 return dict(mean_daily_active_return=dif.mean(),te_annual=sd*np.sqrt(365),information_ratio=dif.mean()/sd*np.sqrt(365) if sd else None,
  mean_daily_active_ci95=np.quantile(means,[.025,.975]),block_days=block,replicates=3000)
def run(phase):
 sha=lock();cut,stop=base.phase_bounds(phase);n=stop-cut;vf=base.risk_forecast(phase);amplitude=.25*np.minimum(1,.01/np.sqrt(vf))
 arrays=dict(dates=D[cut:stop],price=EP[cut:stop+1],har_variance=vf,amplitude=amplitude);out=dict(protocol_sha256=sha,phase=phase,strategies={},probabilities={},models={},cost_sensitivity=[],comparisons={});trade_rows={};decision_rows=[]
 for h in [1,7,28]:
  data=horizon_data(h)
  # h=1 is an exact reconstruction of the frozen original direction model.
  if h==1:
   assert np.allclose(data[0],base.X,equal_nan=True) and np.array_equal(data[3],base.TRAIN_OK)
  p,null,records,refs=predict(cut,stop,h,data);review=np.arange(n)%h==0
  sig=np.where(p>=.55,1,np.where(p<=.45,-1,0));sig[~(data[1][cut:stop]&data[2][cut:stop])]=0
  arrays.update({f'h{h}_p':p,f'h{h}_signal':sig,f'h{h}_review':review,f'h{h}_refit_id':refs,f'h{h}_event':data[1][cut:stop],f'h{h}_feature_ok':data[2][cut:stop]})
  out['models'][str(h)]=records;out['probabilities'][str(h)]=probability_scores(cut,stop,h,data,p,null,review)
  for j in np.flatnonzero(review):
   t=cut+j;end=min(t+h,stop);decision_rows.append(dict(horizon=h,date=str(D[t]),forecast_up=p[j],event=data[1][t],feature_ok=data[2][t],signal=sig[j],amplitude=amplitude[j],
    target_core=.5+sig[j]*amplitude[j],target_pure=sig[j]*amplitude[j],planned_full_endpoint=str(np.datetime64(D[t])+np.timedelta64(h,'D')),actual_holding_days=end-t,
    realized_holding_return=EP[end]/EP[t]-1,coefficient_refit=records[refs[j]]['refit_date'],training_last_endpoint=records[refs[j]]['last_endpoint']))
  ports={}
  for kind,target in [('pure',sig*amplitude),('core',.5+sig*amplitude),('btc50',np.full(n,.5))]:
   key=f'{kind}_{h}d';port=ledger(cut,stop,target,review,kind);ports[kind]=port
   out['strategies'][key]=port['stats'];trade_rows[key]=port['trades']
   arrays.update({key+'_'+k:v for k,v in port['arrays'].items()})
   if kind in ['core','pure']:
    for c in PROTOCOL['cost_sensitivity']:
     borrowers=PROTOCOL['borrow_sensitivity'] if kind=='pure' else [0]
     for b in borrowers:
      s=ledger(cut,stop,target,review,kind,c,b,intraday=False)['stats'];out['cost_sensitivity'].append(dict(strategy=key,fee=c,borrow=b,**s))
   print(phase,key,json.dumps(base.native(port['stats'])),flush=True)
  out['comparisons'][f'core_{h}d_vs_btc50']=paired(ports['core']['arrays']['returns'],ports['btc50']['arrays']['returns'])
  out['comparisons'][f'pure_{h}d_vs_cash']=paired(ports['pure']['arrays']['returns'],np.zeros(n))
 review=np.zeros(n,bool);review[0]=True
 bh=ledger(cut,stop,np.ones(n),review,'buy_hold');out['strategies']['BTC_buy_hold']=bh['stats'];trade_rows['BTC_buy_hold']=bh['trades'];arrays.update({'BTC_buy_hold_'+k:v for k,v in bh['arrays'].items()})
 # Recompute the old daily policy through this ledger with zero no-trade band.
 old=np.load(W/f'btc_longshort_{phase}_arrays.npz');orig=ledger(cut,stop,old['signal_har_target'],np.ones(n,bool),'original',band=0)
 assert np.allclose(orig['arrays']['nav'],old['signal_har_nav'],rtol=1e-10,atol=1e-4)
 out['strategies']['original_daily']=orig['stats'];trade_rows['original_daily']=orig['trades'];arrays.update({'original_daily_'+k:v for k,v in orig['arrays'].items()})
 np.savez_compressed(W/f'btc_slow_{phase}_arrays.npz',**arrays)
 save(f'btc_slow_{phase}_results.json',out);save(f'btc_slow_{phase}_trades.json',trade_rows);save(f'btc_slow_{phase}_decisions.json',decision_rows)
 print('SAVED',phase,sha,flush=True)
if __name__=='__main__':run(sys.argv[1])
