import sys,json,hashlib,gzip,datetime
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path('work/model_packages').resolve()))
from scipy.optimize import brentq
from scipy.special import expit
import btc_downside_longshort as m
W=Path('work');protocol=json.loads((W/'btc_longshort_protocol.json').read_text(encoding='utf-8'))
selection=json.loads((W/'btc_longshort_selection.json').read_text(encoding='utf-8'))
assert hashlib.sha256((W/'btc_longshort_protocol.json').read_bytes()).hexdigest()==selection['protocol_sha256']
checks=[];fitchecks=[];rawchecks=[];probchecks=0
for phase in ['validation','test']:
    a=np.load(W/f'btc_longshort_{phase}_arrays.npz');results=json.loads((W/f'btc_longshort_{phase}_results.json').read_text(encoding='utf-8'))
    models=json.loads((W/f'btc_longshort_{phase}_models.json').read_text(encoding='utf-8'))
    assert results['selection']==selection
    cut,stop=m.phase_bounds(phase)
    assert np.allclose(a['next_return'],a['execution_price'][1:]/a['execution_price'][:-1]-1,rtol=1e-12,atol=1e-14)
    for model in models:
        t=model['refit_index'];idx=np.array(model['training_indices'])
        assert idx.max()<=t-2 and all(m.D[i+1]<m.D[t] for i in idx)
        expected=np.arange(m.train_start(t),t-1);expected=expected[m.TRAIN_OK[expected]]
        assert np.array_equal(idx,expected)
        assert np.allclose(m.X[idx].mean(0),model['mean'],rtol=1e-12,atol=1e-12)
        assert np.allclose(m.X[idx].std(0),model['scale'],rtol=1e-12,atol=1e-12)
    for j,t in enumerate(range(cut,stop)):
        fit=models[int(a['refit_id'][j])];b=np.array(fit['coef']);mean=np.array(fit['mean']);scale=np.array(fit['scale'])
        prob=expit(b[0]+((m.X[t]-mean)/scale)@b[1:])
        assert np.isclose(prob,a['p_up'][j],rtol=1e-13)
        expected=0
        if m.EVENT[t] and m.FEATURE_OK[t]:
            expected=1 if prob>=selection['tau'] else -1 if prob<=1-selection['tau'] else 0
        assert expected==a['signal'][j];probchecks+=1
    # Independent full-Newton optimizer checks representative fitted models.
    for mi in sorted(set([0,len(models)//2,len(models)-1])):
        fit=models[mi];idx=np.array(fit['training_indices']);xx=(m.X[idx]-fit['mean'])/fit['scale'];xx=np.column_stack([np.ones(len(idx)),xx]);yy=m.NEXTUP[idx]
        lam=selection['lambda_value'];pen=np.r_[0,np.full(xx.shape[1]-1,lam)];b=np.zeros(xx.shape[1])
        def loss(beta):
            z=xx@beta;return np.mean(np.logaddexp(0,z)-yy*z)+.5*np.sum(pen*beta**2)
        for step in range(100):
            p=expit(xx@b);g=xx.T@(p-yy)/len(yy)+pen*b
            if np.max(abs(g))<1e-10:break
            h=xx.T@((p*(1-p))[:,None]*xx)/len(yy)+np.diag(pen)
            update=np.linalg.solve(h,g);rate=1.
            while loss(b-rate*update)>loss(b)+1e-15:rate*=.5;assert rate>1e-8
            b-=rate*update
        assert np.max(abs(g))<1e-8
        error=float(abs(b-np.array(fit['coef'])).max());assert error<2e-6
        fitchecks.append(dict(phase=phase,refit_date=fit['refit_date'],coefficient_max_error=error))
    for name,stats in results['portfolios'].items():
        cash=1e7;qty=0.;nav=[];fees=[];borrows=[];prices=a['execution_price'];target=a[name+'_target']
        for j,t in enumerate(range(cut,stop)):
            price=prices[j];B=qty*price;wealth=cash+B;fee=0
            if m.OBS[t]:
                w=target[j];bound=2*(wealth+abs(B))
                fun=lambda amount:(B+amount)-w*(wealth-.001*abs(amount))
                delta=brentq(fun,-bound,bound,xtol=1e-8)
                fee=.001*abs(delta);qty+=delta/price;cash-=delta+fee
            carry=max(-qty*price,0)*.1/365;cash-=carry
            if j==len(target)-1:
                fee+=.001*abs(qty*prices[j+1]);cash+=qty*prices[j+1]-.001*abs(qty*prices[j+1]);qty=0
            nav.append(cash+qty*prices[j+1]);fees.append(fee);borrows.append(carry)
        maxerr=float(abs(np.array(nav)-a[name+'_nav']).max())
        assert np.allclose(nav,a[name+'_nav'],rtol=1e-11,atol=.001)
        assert np.allclose(fees,a[name+'_fees'],rtol=1e-9,atol=1e-6)
        assert np.allclose(borrows,a[name+'_borrow'],rtol=1e-9,atol=1e-6)
        rr=np.diff(np.r_[1e7,nav])/np.r_[1e7,nav[:-1]]
        assert np.allclose(rr,a[name+'_returns'],rtol=1e-9,atol=1e-12)
        assert np.isclose(nav[-1]/1e7-1,stats['return_net'],rtol=1e-10,atol=1e-12)
        checks.append(dict(phase=phase,strategy=name,days=len(nav),max_nav_error_krw=maxerr))
    # Confirm actual cache prices on entry and exit of selected active days.
    active=np.flatnonzero(a['signal'])
    sample=sorted(set([active[0],active[-1],int(np.argmin(a['signal_har_returns']))]))
    needed={str(a['execution_dates'][j])+ 'T00:00:00':float(prices[j]) for i in sample for j in [i,i+1]}
    for month in sorted(set(k[:7] for k in needed)):
        rows=json.load(gzip.open(W/f'minute_cache/{month}.json.gz','rt'))
        cache={row['candle_date_time_utc']:row['trade_price'] for row in rows}
        for date,price in needed.items():
            if date[:7]!=month:continue
            assert date in cache and np.isclose(price,cache[date],rtol=1e-12)
            rawchecks.append(dict(phase=phase,candle=date,price=cache[date]))
old=np.load(W/'btc_cash_plot_arrays.npz');new=np.load(W/'btc_longshort_test_arrays.npz')
for oldname,newname in [('HAR_logRV_all','har_always_long'),('BTC50_cash50_daily','BTC50_cash50'),('BTC_buy_hold','BTC_buy_hold')]:
    assert np.isclose(old[oldname+'_nav'][-1],new[newname+'_nav'][-1],rtol=1e-11)
out=dict(status='passed',probability_and_signal_checks=probchecks,independent_Newton_fit_checks=fitchecks,
    ledger_checks=checks,raw_execution_price_checks=rawchecks,
    max_nav_error_krw=max(c['max_nav_error_krw'] for c in checks),
    notes=['all training labels end before refit information cutoff; previous target embargo confirmed',
    'training-only feature scaling reconstructed','selected model and thresholds unchanged in test',
    'long-only final NAV reproduces prior experiment; daily marks now at execution time',
    'all nine strategies in both phases independently reconciled including signed position, fees and borrow cost'])
m.save('btc_longshort_verification.json',out)
print(json.dumps({k:out[k] for k in ['status','probability_and_signal_checks','max_nav_error_krw']},indent=2))
