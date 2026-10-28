import sys,json,datetime,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path('work/model_packages').resolve()))
from scipy.optimize import brentq
from scipy.linalg import lstsq
W=Path('work');s=json.loads((W/'har_refit_results.json').read_text(encoding='utf-8'))
a=np.load(W/'har_refit_arrays.npz');z=np.load(W/'minute_plot_arrays.npz')
d=z['dates'];v=z['rv_5m'];logv=np.log(v);r=z['returns'].reshape(-1,1440);obs=z['observed_return'].reshape(-1,1440)
methods=s['protocol']['primary_methods']+[s['protocol']['auxiliary_control']]
capital=s['protocol']['execution_protocol']['initial_capital_krw'];cost=s['protocol']['execution_protocol']['base_one_way_cost']
rows=[];fit_checks=[]
assert hashlib.sha256((W/'har_refit_protocol.json').read_bytes()).hexdigest()==s['protocol_sha256']
for phase in ['validation','test']:
    dates=a[phase+'_dates'];cut=int(np.searchsorted(d,dates[0]));stop=cut+len(dates)
    for method in methods:
        prefix=phase+'_'+method+'_'
        lo=a[prefix+'train_start'];hi=a[prefix+'train_end']+1
        assert (hi<=np.arange(cut,stop)).all()
        b=a[prefix+'coef'];smear=a[prefix+'smear']
        xx=np.array([[1,logv[t-1],sum(logv[t-7:t])/7,sum(logv[t-30:t])/30] for t in range(cut,stop)])
        ff=np.exp(np.einsum('ij,ij->i',xx,b))*smear
        assert np.allclose(ff,a[prefix+'forecast'],rtol=1e-12,atol=0)
        for j in sorted(set([0,1,30,len(dates)//2,len(dates)-1])):
            training=np.array([[1,logv[t-1],sum(logv[t-7:t])/7,sum(logv[t-30:t])/30] for t in range(int(lo[j]),int(hi[j]))])
            bb=lstsq(training,logv[lo[j]:hi[j]],lapack_driver='gelsy')[0]
            ss=np.exp(logv[lo[j]:hi[j]]-training@bb).mean()
            assert np.allclose(bb,b[j],rtol=1e-10,atol=1e-11)
            assert np.isclose(ss,smear[j],rtol=1e-12)
            fit_checks.append(dict(phase=phase,method=method,date=str(dates[j]),coefficient_max_error=float(abs(bb-b[j]).max())))
        if method.startswith('fixed'):
            assert np.array_equal(b,np.repeat(b[:1],len(b),axis=0))
        if method=='expanding_all':
            assert (lo==30).all() and np.array_equal(hi,np.arange(cut,stop))
        if method=='rolling_3y':
            assert np.array_equal(hi,np.arange(cut,stop))
            for j,day in enumerate(dates):
                dd=datetime.date.fromisoformat(str(day))
                try: prev=dd.replace(year=dd.year-3)
                except ValueError: prev=dd.replace(year=dd.year-3,day=28)
                assert d[lo[j]]==np.datetime64(prev)
        quantity=0.;cash=float(capital);nav=[];fees=[]
        for j,i in enumerate(range(cut,stop)):
            execution=z['close_krw'][i-1]*np.exp(r[i,0]);B=quantity*execution;wealth=B+cash
            weight=min(1,.01/np.sqrt(a[prefix+'forecast'][j]));fee=0.
            if obs[i,0]:
                low=-B;high=cash/(1+cost)
                fn=lambda amount:(B+amount)-weight*(wealth-cost*abs(amount))
                amount=low if weight==0 else high if weight==1 else brentq(fn,low,high,xtol=1e-8)
                fee=cost*abs(amount);quantity+=amount/execution;cash-=amount+fee
            endprice=z['close_krw'][i]
            if j==len(dates)-1:
                sale=quantity*endprice;fee+=cost*sale;cash+=sale*(1-cost);quantity=0.
            nav.append(cash+quantity*endprice);fees.append(fee)
        assert np.allclose(nav,a[prefix+'nav'],rtol=1e-11,atol=.001)
        assert np.allclose(fees,a[prefix+'fees'],rtol=1e-9,atol=.001)
        rr=np.diff(np.r_[capital,nav])/np.r_[capital,nav[:-1]]
        assert np.allclose(rr,a[prefix+'returns'],rtol=1e-9,atol=1e-12)
        stats=s['phases'][phase]['portfolios'][method]
        assert np.isclose(rr.mean()/rr.std(ddof=1)*np.sqrt(365),stats['sharpe'],rtol=1e-10)
        scores=s['phases'][phase]['forecasts'][method];q=v[cut:stop]/ff
        assert np.isclose(np.mean(q-np.log(q)-1),scores['qlike'],rtol=1e-12)
        assert np.isclose(np.mean((v[cut:stop]-ff)**2),scores['mse'],rtol=1e-12,atol=0)
        rows.append(dict(phase=phase,method=method,forecast_checks=len(dates),ledger_checks=len(dates),
            independent_max_nav_error_krw=float(abs(np.array(nav)-a[prefix+'nav']).max()),fees_total=float(sum(fees))))
out=dict(status='passed',full_period_checks=rows,independent_QR_fit_checks=fit_checks,
    checks=['all forecast dates use only prior training targets and inputs','rolling calendar-year boundaries independently reconstructed',
    '40 representative refits independently match QR least squares and training-only smearing','all forecasts independently reconstructed',
    'quantity/cash ledger using root-solved transaction sizing matches all eight strategy-period runs','forecast scores and Sharpe independently recomputed'])
(W/'har_refit_verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(status=out['status'],forecast_checks=sum(x['forecast_checks'] for x in rows),fit_checks=len(fit_checks),max_nav_error_krw=max(x['independent_max_nav_error_krw'] for x in rows)),indent=2))
