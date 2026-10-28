"""Compare predeclared HAR refit policies without changing prior experiments."""
import json, hashlib, datetime
from pathlib import Path
import numpy as np

W = Path('work')
SRC = W / 'backtest_btc_cash.py'
# Reuse the already verified execution/fee functions, without executing its CLI.
base = {'__name__': 'btc_cash_core'}
exec(compile(SRC.read_text(encoding='utf-8').split('\nmode=sys.argv[1]\n')[0], str(SRC), 'exec'), base)
D, V, R, Z, P = (base[k] for k in ['D', 'V', 'R', 'Z', 'P'])
X = base['feature_matrix'](V)
Y = np.log(V)
METHODS = ['fixed_all', 'expanding_all', 'rolling_3y', 'fixed_3y_control']
LABELS = dict(fixed_all='고정 HAR', expanding_all='확장 HAR', rolling_3y='롤링 HAR (3년)', fixed_3y_control='3년 고정 HAR (보조 비교)')
protocol = dict(
    created_local_date='2026-10-14', model='HAR of log 5-minute RV, OLS, 1/7/30 calendar-day lags',
    primary_methods=METHODS[:3], auxiliary_control=METHODS[-1],
    initial_validation_train_end='2023-12-31', validation_start='2024-01-01', validation_end='2025-12-31',
    test_start='2026-01-01', test_end='2026-10-11',
    fixed_all='Fit all available historical targets once at each phase start; retain coefficients and training residual smearing.',
    expanding_all='Refit all historical targets every day, ending at the preceding day.',
    rolling_3y='Refit every day on targets in [forecast date minus 3 calendar years, forecast date). February 29 minus 3 years maps to February 28.',
    fixed_3y_control='Fit the same initial 3-calendar-year target window as rolling once per phase, then keep coefficients fixed. Auxiliary comparison isolates refitting from initial sample choice.',
    rolling_years=3, refit_frequency='daily for expanding and rolling; once per phase for fixed',
    retransformation='exp(predicted log RV) times mean(exp(training residual)); recomputed with each refit',
    feature_availability='For target day t, all features and training targets end at t-1. Window-start targets may use their preceding 30 days as lagged features.',
    decision='Compare every predeclared primary policy. No best-test-return selection or window-length search.',
    evaluation_status='Retrospective additional experiment: 2026 data and investment results were already inspected before this request. Not a fresh blind test.',
    execution_protocol=P, data_sha256=hashlib.sha256((W/'minute_plot_arrays.npz').read_bytes()).hexdigest(),
    core_code_sha256=hashlib.sha256(SRC.read_bytes()).hexdigest())
protocol_path=W/'har_refit_protocol.json'
if protocol_path.exists():
    assert json.loads(protocol_path.read_text(encoding='utf-8')) == protocol
else:
    protocol_path.write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
protocol_hash=hashlib.sha256(protocol_path.read_bytes()).hexdigest()

def three_year_start(i):
    day=datetime.date.fromisoformat(str(D[i])); year=day.year-3
    try: before=day.replace(year=year)
    except ValueError: before=day.replace(year=year,day=28)
    return max(30,int(np.searchsorted(D,np.datetime64(before))))

def fit(lo,hi):
    lo=max(30,lo)
    b,_,rank,_=np.linalg.lstsq(X[lo:hi],Y[lo:hi],rcond=None)
    assert rank==4
    residual=Y[lo:hi]-X[lo:hi]@b
    smear=float(np.exp(residual).mean())
    return b,smear

def forecasts(method,cut,stop):
    f=[]; coeff=[]; smears=[]; starts=[]; ends=[]; ns=[]
    if method.startswith('fixed'):
        lo=30 if method=='fixed_all' else three_year_start(cut)
        b,s=fit(lo,cut)
    for t in range(cut,stop):
        hi=cut if method.startswith('fixed') else t
        if not method.startswith('fixed'):
            lo=30 if method=='expanding_all' else three_year_start(t)
            b,s=fit(lo,hi)
        assert hi<=t and hi>lo
        f.append(float(np.exp(X[t]@b)*s));coeff.append(b.copy());smears.append(s)
        starts.append(lo);ends.append(hi-1);ns.append(hi-lo)
    out=dict(forecast=np.array(f),coef=np.array(coeff),smear=np.array(smears),train_start=np.array(starts),train_end=np.array(ends),train_n=np.array(ns))
    assert np.isfinite(out['forecast']).all() and (out['forecast']>0).all()
    return out

results=dict(protocol=protocol,protocol_sha256=protocol_hash,labels=LABELS,phases={},checks=[])
arrays={};summary=[];daily=[];coef_rows=[]
old=json.loads((W/'btc_cash_results.json').read_text(encoding='utf-8'))
old_test=np.load(W/'btc_cash_plot_arrays.npz')
old_val=np.load(W/'btc_cash_validation_forecasts.npz')
for phase,start,end in [('validation','2024-01-01','2026-01-01'),('test','2026-01-01','2026-10-12')]:
    cut=int(np.searchsorted(D,np.datetime64(start)));stop=int(np.searchsorted(D,np.datetime64(end)))
    n=stop-cut
    phase_result=dict(start=str(D[cut]),end=str(D[stop-1]),days=n,forecasts={},portfolios={},quality={},pairwise=[])
    arrays[phase+'_dates']=D[cut:stop];arrays[phase+'_rv']=V[cut:stop]
    fs={};ports={}
    for method in METHODS:
        trace=forecasts(method,cut,stop);f=trace['forecast'];fs[method]=f
        score=base['scores'](V[cut:stop],f)
        weights=np.minimum(1,P['risk_target_daily_sd']/np.sqrt(f))
        port=base['simulate'](cut,stop,weights,P['base_one_way_cost'],minute_dd=True);ports[method]=port
        phase_result['forecasts'][method]=score
        phase_result['portfolios'][method]=port['stats']
        mask=Z['coverage_ratio'][cut:stop]>=.95
        phase_result['quality'][method]=base['scores'](V[cut:stop][mask],f[mask])
        row=dict(phase=phase,method=method,label=LABELS[method],role='primary' if method in METHODS[:3] else 'control',**score,**port['stats'])
        summary.append(row)
        for name,value in trace.items(): arrays[phase+'_'+method+'_'+name]=value
        for name in ['nav','returns','fees','weights','target','drawdown','turnover']:
            arrays[phase+'_'+method+'_'+name]=port[name]
        for j,t in enumerate(range(cut,stop)):
            coef_rows.append(dict(phase=phase,method=method,date_utc=str(D[t]),
                training_target_first=str(D[trace['train_start'][j]]),training_target_last=str(D[trace['train_end'][j]]),
                training_target_count=int(trace['train_n'][j]),b0=float(trace['coef'][j,0]),b1=float(trace['coef'][j,1]),
                b7=float(trace['coef'][j,2]),b30=float(trace['coef'][j,3]),smearing=float(trace['smear'][j])))
        print(phase,method,'QLIKE',score['qlike'],'MSE',score['mse'],'return',port['stats']['total_return'],'Sharpe',port['stats']['sharpe'],flush=True)
    assert np.isclose(fs['fixed_all'][0],fs['expanding_all'][0],rtol=1e-13)
    assert np.isclose(fs['rolling_3y'][0],fs['fixed_3y_control'][0],rtol=1e-13)
    oldf=old_val['HAR_logRV_all'] if phase=='validation' else old_test['HAR_logRV_all_forecast']
    oldf3=old_val['HAR_logRV_3y'] if phase=='validation' else old_test['HAR_logRV_3y_forecast']
    assert np.allclose(fs['fixed_all'],oldf,rtol=1e-12,atol=0)
    assert np.allclose(fs['fixed_3y_control'],oldf3,rtol=1e-12,atol=0)
    if phase=='test':
        assert np.allclose(ports['fixed_all']['nav'],old_test['HAR_logRV_all_nav'],rtol=1e-12)
    for method,weight in [('BTC50_cash50_daily',.5),('BTC_buy_hold',1.),('cash',0.)]:
        port=base['simulate'](cut,stop,np.full(n,weight),P['base_one_way_cost'],minute_dd=True);ports[method]=port
        phase_result['portfolios'][method]=port['stats']
        summary.append(dict(phase=phase,method=method,label=method,role='benchmark',**port['stats']))
        for name in ['nav','returns','fees','weights','target','drawdown','turnover']:
            arrays[phase+'_'+method+'_'+name]=port[name]
    losses={k:V[cut:stop]/f-np.log(V[cut:stop]/f)-1 for k,f in fs.items()}
    if phase=='test':
        for a,b in [('expanding_all','fixed_all'),('rolling_3y','fixed_all'),('rolling_3y','fixed_3y_control')]:
            phase_result['pairwise'].append(dict(a=a,b=b,metric='qlike',**base['bootstrap_difference'](losses[a],losses[b],a+' minus '+b+' daily QLIKE')))
            phase_result['pairwise'].append(dict(a=a,b=b,metric='net_daily_return',**base['bootstrap_difference'](ports[a]['returns'],ports[b]['returns'],a+' minus '+b+' net daily return')))
        phase_result['cost_sensitivity']=[]
        for cost in P['cost_sensitivity']:
            for method in METHODS+['BTC50_cash50_daily']:
                stats=base['simulate'](cut,stop,ports[method]['target'],cost)['stats']
                phase_result['cost_sensitivity'].append(dict(method=method,cost=cost,**stats))
    phase_result['monthly']=[]
    for month in np.unique(D[cut:stop].astype('datetime64[M]')):
        mask=D[cut:stop].astype('datetime64[M]')==month
        for method in METHODS:
            phase_result['monthly'].append(dict(month=str(month),method=method,days=int(mask.sum()),qlike=float(losses[method][mask].mean()),
                mse=float(np.mean((V[cut:stop][mask]-fs[method][mask])**2)),return_net=float(np.prod(1+ports[method]['returns'][mask])-1)))
    for j,t in enumerate(range(cut,stop)):
        row=dict(phase=phase,date_utc=str(D[t]),information_through_utc=str(D[t-1]),rv_5m=float(V[t]),coverage=float(Z['coverage_ratio'][t]))
        for method in METHODS:
            row[method+'_forecast']=float(fs[method][j]);row[method+'_qlike']=float(losses[method][j])
        for method,p in ports.items():
            for field in ['nav','returns','weights','fees']:row[method+'_'+field]=float(p[field][j])
        daily.append(row)
    phase_result['low_coverage_days']=[dict(date=str(D[i]),coverage=float(Z['coverage_ratio'][i]),filled_minutes=int(Z['filled_minutes'][i])) for i in range(cut,stop) if Z['coverage_ratio'][i]<.95]
    results['phases'][phase]=phase_result
results['checks']=['predeclared protocol hash preserved','fixed-all and fixed-three-year forecasts reproduce previous experiment in both phases',
    'fixed-all portfolio reproduces previous test','first forecasts of each matched-history pair are equal','all training ends precede forecast target',
    'finite positive forecasts and full-rank OLS fits','exact self-financing execution and fees reused']
assert hashlib.sha256(protocol_path.read_bytes()).hexdigest()==protocol_hash
def save(name,obj):
    (W/name).write_text(json.dumps(base['native'](obj),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
save('har_refit_results.json',results);save('har_refit_summary.json',summary);save('har_refit_daily.json',daily);save('har_refit_coefficients.json',coef_rows)
np.savez_compressed(W/'har_refit_arrays.npz',**arrays)
print('DONE',len(daily),'daily rows',len(coef_rows),'coefficient rows',flush=True)
