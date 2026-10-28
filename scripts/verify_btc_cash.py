import sys,json,gzip
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path('work/model_packages').resolve()))
from scipy.optimize import brentq
W=Path('work');s=json.loads((W/'btc_cash_results.json').read_text(encoding='utf-8'))
a=np.load(W/'btc_cash_plot_arrays.npz');z=np.load(W/'minute_plot_arrays.npz')
d=z['dates'];start=int(np.searchsorted(d,np.datetime64('2026-01-01')))
r=z['returns'].reshape(-1,1440);capital=s['protocol']['initial_capital_krw'];cost=s['protocol']['base_one_way_cost']
checks=[]
for name in [s['selection']['winner'],'BTC_buy_hold','BTC50_cash50_daily','fixed_validation_mean_weight','cash']:
    quantity=0.;cash=float(capital);nav=[];fees=[];weights=a[name+'_target']
    for j,i in enumerate(range(start,len(d))):
        execution=z['close_krw'][i-1]*np.exp(r[i,0]);B=quantity*execution;wealth=B+cash;weight=weights[j]
        lo=-B;hi=cash/(1+cost)
        fn=lambda amount: (B+amount)-weight*(wealth-cost*abs(amount))
        amount=lo if weight==0 else hi if weight==1 else brentq(fn,lo,hi,xtol=1e-8)
        fee=cost*abs(amount);quantity+=amount/execution;cash-=amount+fee
        endprice=z['close_krw'][i]
        if j==len(weights)-1:
            sale=quantity*endprice;fee+=cost*sale;cash+=sale*(1-cost);quantity=0.
        nav.append(cash+quantity*endprice);fees.append(fee)
    assert np.allclose(nav,a[name+'_nav'],rtol=1e-11,atol=.001),(name,float(np.max(abs(np.array(nav)-a[name+'_nav']))))
    checks.append({'strategy':name,'max_independent_nav_error_krw':float(np.max(abs(np.array(nav)-a[name+'_nav']))),
        'independent_fees_paid_krw':float(sum(fees))})
points={'2025-12-31T23:59:00':None,'2026-01-01T00:00:00':None,'2026-10-11T23:59:00':None}
for month in ['2025-12','2026-01','2026-10']:
    for row in json.load(gzip.open(W/f'minute_cache/{month}.json.gz','rt')):
        key=row['candle_date_time_utc']
        if key in points:points[key]=row['trade_price']
assert points['2025-12-31T23:59:00']==z['close_krw'][start-1]
assert np.isclose(points['2026-01-01T00:00:00'],z['close_krw'][start-1]*np.exp(r[start,0]),rtol=1e-12)
assert points['2026-10-11T23:59:00']==z['close_krw'][-1]
assert np.isclose(a['BTC_buy_hold_nav'][-1]/capital,(1-cost)/(1+cost)*points['2026-10-11T23:59:00']/points['2026-01-01T00:00:00'],rtol=1e-11)
winner=s['selection']['winner'];b=np.array(s['test_parameters'][winner]['coefficients']);smear=s['test_parameters'][winner]['smearing'];v=z['rv_5m']
for j in [0,1,30,150,len(a['dates'])-1]:
    i=start+j;x=np.array([1,np.log(v[i-1]),np.log(v[i-7:i]).mean(),np.log(v[i-30:i]).mean()])
    assert np.isclose(np.exp(x@b)*smear,a[winner+'_forecast'][j],rtol=1e-12)
quality=[];mask=z['coverage_ratio'][start:]>=.95
for k in [winner,'RV_last','GARCH_normal_all','GARCH_t_all','GJR_t_all']:
    ratio=v[start:][mask]/a[k+'_forecast'][mask]
    quality.append({'model':k,'days':int(mask.sum()),'qlike_high_coverage_only':float(np.mean(ratio-np.log(ratio)-1))})
low=[{'date':str(d[i]),'coverage':float(z['coverage_ratio'][i]),'filled_minutes':int(z['filled_minutes'][i]),'max_stale_minutes':int(z['max_stale_minutes'][i])} for i in np.flatnonzero((d>=np.datetime64('2026-01-01'))&(z['coverage_ratio']<.95))]
out=dict(independent_quantity_ledger=checks,raw_source_prices=points,low_coverage_days=low,quality_sensitivity=quality,
    har_independent_forecast_checks=5,buy_hold_price_formula_checked=True)
(W/'btc_cash_verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False))
