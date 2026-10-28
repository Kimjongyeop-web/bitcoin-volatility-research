"""Independent accounting, availability and fitted-model checks."""
import json
import numpy as np
import btc_slow_overlay as s
from scipy.optimize import brentq
from scipy.special import expit

errors=[];fit_checks=[];total_days=0;max_error=0.;holding_checks=0;model_count=0
for phase in ['validation','test']:
 cut,stop=s.base.phase_bounds(phase);a=np.load(s.W/f'btc_slow_{phase}_arrays.npz');res=json.loads((s.W/f'btc_slow_{phase}_results.json').read_text());n=stop-cut
 assert res['protocol_sha256']==s.lock()
 old=np.load(s.W/f'btc_longshort_{phase}_arrays.npz')
 assert np.max(abs(a['h1_p']-old['p_up']))<1e-10
 for h in [1,7,28]:
  x,event,fok,tok,y=s.horizon_data(h)
  for mi,m in enumerate(res['models'][str(h)]):
   model_count+=1;t=m['refit_index'];idx=np.array(m['indices']);b=np.array(m['coef']);mean=np.array(m['mean']);scale=np.array(m['scale'])
   assert idx.max()+h<t and idx.min()>=s.base.train_start(t)
   assert np.all(tok[idx]) and np.array_equal(y[idx],(s.EP[idx+h]>s.EP[idx]).astype(float))
   assert np.allclose(mean,x[idx].mean(0))
   assert np.allclose(scale,np.where(x[idx].std(0)<1e-12,1,x[idx].std(0)))
   pred_rows=np.flatnonzero(a[f'h{h}_refit_id']==mi);assert np.all(cut+pred_rows>=t)
   checkp=expit(b[0]+((x[cut+pred_rows]-mean)/scale)@b[1:]);assert np.max(abs(checkp-a[f'h{h}_p'][pred_rows]))<1e-12
   if mi in [0,len(res['models'][str(h)])-1]:
    z=np.column_stack([np.ones(len(idx)),(x[idx]-mean)/scale]);yy=y[idx];reg=np.diag(np.r_[0,np.full(9,.1)])
    bn=np.zeros(10)
    for _ in range(50):
     pp=expit(z@bn);g=z.T@(pp-yy)/len(idx)+reg@bn;H=(z.T*(pp*(1-pp)))@z/len(idx)+reg
     step=np.linalg.solve(H,g);bn-=step
     if abs(step).max()<1e-12:break
    err=float(abs(bn-b).max());assert err<3e-6
    fit_checks.append(dict(phase=phase,h=h,refit=m['refit_date'],max_coefficient_error=err))
  decisions=np.arange(n)%h==0
  assert np.array_equal(decisions,a[f'h{h}_review'])
  sig=np.where(a[f'h{h}_p']>=.55,1,np.where(a[f'h{h}_p']<=.45,-1,0));sig[~(event[cut:stop]&fok[cut:stop])]=0
  assert np.array_equal(sig,a[f'h{h}_signal'])
  amp=.25*np.minimum(1,.01/np.sqrt(s.base.risk_forecast(phase)))
  assert np.allclose(a[f'pure_{h}d_target'],sig*amp) and np.allclose(a[f'core_{h}d_target'],.5+sig*amp)
 for key,stat in res['strategies'].items():
  cash=1e7;q=0.;fee_total=0.;borrow_total=0.;count=0;turn=0.
  review=a[key+'_review'];targets=a[key+'_target'];band=0 if key=='original_daily' else .05
  kind=key.split('_')[0];lower,upper=(.25,.75) if kind=='core' else (-.25,.25)
  for j,t in enumerate(range(cut,stop)):
   price=s.EP[t];equity=cash+q*price;weight=q*price/equity;target=targets[j];trade=0.;fee=0
   must=False
   if kind in ['core','pure']:must=weight<lower-1e-10 or weight>upper+1e-10
   if kind=='pure':must=must or (target==0 and abs(weight)>1e-10) or target*weight<0
   execute=review[j] and s.base.OBS[t] and (must or abs(target-weight)>=band-1e-12)
   if execute:
    # Solve directly for new BTC value after transaction costs, independent of the production algebra.
    old_value=q*price
    value=brentq(lambda v:v-target*(equity-.001*abs(v-old_value)),-2*equity,2*equity,xtol=1e-7)
    trade=value-old_value
    if abs(trade)>1e-7:
     fee=.001*abs(trade);cash-=trade+fee;q=value/price;count+=1;turn+=abs(trade)/equity
   if not review[j] and j>0:
    assert abs(a[key+'_q'][j]-a[key+'_q'][j-1])<1e-10;holding_checks+=1
   assert abs(q-a[key+'_q'][j])<1e-9
   assert abs(trade-a[key+'_trade_notional'][j])<1e-5
   financing=max(-q*price,0)*.10/365;cash-=financing;borrow_total+=financing
   if j==n-1 and abs(q)>1e-12:
    value=q*s.EP[t+1];turn+=abs(value)/(cash+value);fee+=.001*abs(value);cash+=value-.001*abs(value);q=0.;count+=1
   fee_total+=fee;equity=cash+q*s.EP[t+1];err=abs(equity-a[key+'_nav'][j]);max_error=max(max_error,err);assert err<1e-5,(phase,key,j,err)
   total_days+=1
  assert count==stat['actual_trade_count']
  assert abs(turn-stat['turnover'])<1e-9
  assert abs(fee_total-stat['fees_krw'])<1e-5 and abs(borrow_total-stat['borrow_krw'])<1e-5
  assert np.allclose(a[key+'_weights'],a[key+'_q']*s.EP[cut:stop]/(a[key+'_cash']+a[key+'_q']*s.EP[cut:stop]))
s.save('btc_slow_verification.json',dict(status='passed',ledger_day_checks=total_days,max_nav_error_krw=max_error,unchanged_quantity_checks=holding_checks,
 monthly_models_availability_checked=model_count,independent_newton_fits=fit_checks,h1_reproduces_prior_probabilities=True,all_trade_counts_fees_and_financing_reconciled=True))
print('VERIFIED',total_days,'ledger days;',holding_checks,'no-trade holding days;',model_count,'monthly fits; max NAV error',max_error)
