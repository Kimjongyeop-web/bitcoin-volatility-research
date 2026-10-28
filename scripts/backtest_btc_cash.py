import sys, os, json, math, warnings, hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path('work/model_packages').resolve()))
from arch import arch_model
from scipy.stats import chi2

W=Path('work'); Z=np.load(W/'minute_plot_arrays.npz'); D=Z['dates']
R=Z['returns'].reshape(-1,1440); V=Z['rv_5m']; RET=Z['daily_log_return']
OBS=Z['observed_return'].reshape(-1,1440)
P=json.loads((W/'btc_cash_protocol.json').read_text())
assert np.allclose(R.sum(1),RET,atol=1e-12)
assert np.allclose(np.diff(np.log(Z['close_krw'])),RET[1:],atol=1e-12)
assert np.all(np.diff(D)==np.timedelta64(1,'D')) and (V>0).all()

def native(x):
    if isinstance(x,dict): return {str(k):native(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [native(v) for v in x]
    if isinstance(x,np.ndarray): return native(x.tolist())
    if isinstance(x,np.integer): return int(x)
    if isinstance(x,np.floating): return float(x) if np.isfinite(x) else None
    if isinstance(x,float): return x if math.isfinite(x) else None
    return x

def save(name,x): (W/name).write_text(json.dumps(native(x),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def feature_matrix(v):
    l=np.log(v); x=np.full((len(v),4),np.nan)
    for i in range(30,len(v)): x[i]=[1,l[i-1],l[i-7:i].mean(),l[i-30:i].mean()]
    return x

def scores(y,f):
    assert (y>0).all() and (f>0).all() and np.isfinite(f).all()
    ratio=y/f
    return dict(n=len(y),qlike=float(np.mean(ratio-np.log(ratio)-1)),mse=float(np.mean((f-y)**2)),
        vol_mae_pct=float(100*np.mean(abs(np.sqrt(f)-np.sqrt(y)))),
        mean_forecast_vol_pct=float(100*np.sqrt(f).mean()),mean_measured_vol_pct=float(100*np.sqrt(y).mean()))

def candidate_forecasts(cutoff,end):
    cut=int(np.searchsorted(D,np.datetime64(cutoff))); stop=int(np.searchsorted(D,np.datetime64(end)))
    rows={}; params={}; models={}; x=feature_matrix(V[:stop])
    for name,k in [('RV_last',1),('RV_mean7',7),('RV_mean30',30)]:
        rows[name]=np.array([V[i-k:i].mean() for i in range(cut,stop)])
        params[name]={'family':name,'window':'past_observations','lookback':k}
    for window in P['training_windows']:
        start=0 if window=='all' else int(np.searchsorted(D,np.datetime64(str(int(cutoff[:4])-3)+'-01-01')))
        for family in P['model_families']:
            name=family+'_'+window
            base=dict(family=family,window=window,train_first=str(D[start]),train_last=str(D[cut-1]),train_days=cut-start)
            if family=='HAR_logRV':
                first=max(30,start); xx=x[first:cut]; yy=np.log(V[first:cut]); b=np.linalg.lstsq(xx,yy,rcond=None)[0]
                residual=yy-xx@b; smear=float(np.exp(residual).mean())
                f=np.exp(x[cut:stop]@b)*smear
                rows[name]=f;params[name]=dict(base,coefficients=b.tolist(),smearing=smear,training_target_days=cut-first)
            else:
                asym=int(family=='GJR_t'); dist='normal' if family=='GARCH_normal' else 't'
                model=arch_model(RET[start:cut]*100,mean='Constant',vol='GARCH',p=1,o=asym,q=1,dist=dist,rescale=False)
                starts=[None];sample_var=float(np.var(RET[start:cut]*100));sample_mu=float(np.mean(RET[start:cut]*100))
                for aa,gg,bb in [(0.05,0.05 if asym else 0.,0.90),(0.12,-0.05 if asym else 0.,0.82)]:
                    initial=[sample_mu,sample_var*(1-aa-gg/2-bb),aa]
                    if asym:initial.append(gg)
                    initial.append(bb)
                    if dist=='t':initial.append(8.)
                    starts.append(np.array(initial))
                fits=[];attempts=[];warning_messages=[]
                for initial in starts:
                    with warnings.catch_warnings(record=True) as ww:
                        fit=model.fit(starting_values=initial,disp='off',show_warning=True,options={'maxiter':3000,'ftol':1e-9})
                    attempts.append(dict(flag=int(fit.convergence_flag),loglikelihood=float(fit.loglikelihood)))
                    warning_messages += [str(a.message) for a in ww]
                    if fit.convergence_flag==0 and np.isfinite(fit.loglikelihood):fits.append(fit)
                if not fits:raise RuntimeError((name,attempts))
                fitted=max(fits,key=lambda fit:fit.loglikelihood)
                q=fitted.params.to_dict();omega=q['omega']/10000;alpha=q['alpha[1]'];beta=q['beta[1]'];gamma=q.get('gamma[1]',0.);mu=q['mu']/100
                # arch.forecast recalculates the backcast using fitted residuals.
                # Use that same training-only filtered state, including when beta is near 1.
                libtwo=fitted.forecast(horizon=1,start=len(RET[start:cut])-2,reindex=False).variance.values[:,0]/10000
                h=float(libtwo[0]);last_train_variance=h; f=[]
                for i in range(cut,stop):
                    e=RET[i-1]-mu; h=omega+(alpha+gamma*(e<0))*e*e+beta*h; f.append(h)
                f=np.array(f)
                libfirst=float(libtwo[-1])
                assert np.isclose(f[0],libfirst,rtol=1e-9,atol=1e-13),(name,f[0],libfirst)
                assert omega>0 and alpha>=-1e-9 and beta>=-1e-9 and alpha+gamma>=-1e-9
                rows[name]=f;params[name]=dict(base,parameters_percent_units=q,persistence=alpha+beta+gamma/2,
                    convergence_flag=int(fitted.convergence_flag),loglikelihood=float(fitted.loglikelihood),
                    last_train_variance=last_train_variance,
                    optimizer_attempts=attempts,warning_messages=warning_messages)
                models[name]=dict(mu=mu,omega=omega,alpha=alpha,beta=beta,gamma=gamma)
            print('fit',cutoff,name,flush=True)
    for f in rows.values(): assert len(f)==stop-cut and np.isfinite(f).all() and (f>0).all()
    # Causality of features: changes starting at day t cannot affect t's feature row.
    test_i=cut+min(20,stop-cut-1);alter=V[:stop].copy();alter[test_i:]*=17
    assert np.array_equal(feature_matrix(alter)[test_i],x[test_i])
    return cut,stop,rows,params,models

def trade(btc,cash,w,cost):
    wealth=btc+cash;delta=w*wealth-btc
    amount=delta/(1+cost*w) if delta>=0 else delta/(1-cost*w)
    fee=cost*abs(amount);b=btc+amount;c=cash-amount-fee
    assert b>=-1e-7 and c>=-1e-7 and np.isclose(b+c,wealth-fee,rtol=1e-12,atol=1e-7)
    assert np.isclose(b/(b+c),w,rtol=1e-10,atol=1e-10)
    return max(0.,b),max(0.,c),fee,abs(amount)/wealth

def simulate(start,stop,weights,cost,minute_dd=False):
    assert len(weights)==stop-start and (weights>=0).all() and (weights<=1).all()
    capital=P['initial_capital_krw'];btc=0.;cash=float(capital);nav=[];dayret=[];fees=[];turn=[];actualw=[];closew=[];prew=[]
    peak=capital;mindd=0.
    for j,t in enumerate(range(start,stop)):
        before=btc+cash
        # One minute of causal execution lag: yesterday's holdings still apply.
        btc*=np.exp(R[t,0]);prew.append(btc/(btc+cash))
        if minute_dd:
            peak=max(peak,btc+cash);mindd=min(mindd,(btc+cash)/peak-1)
        if OBS[t,0]:
            btc,cash,fee,turnover=trade(btc,cash,float(weights[j]),cost)
        else:
            # Never trade at a potentially filled endpoint. Preserve existing holdings.
            fee=0.;turnover=0.
        actualw.append(btc/(btc+cash))
        if minute_dd:
            path=np.r_[btc+cash, cash+btc*np.exp(np.cumsum(R[t,1:]))]
            peaks=np.maximum.accumulate(np.r_[peak,path])[1:]
            mindd=min(mindd,float(np.min(path/peaks-1)));peak=float(peaks[-1])
        btc*=np.exp(R[t,1:].sum())
        closew.append(btc/(btc+cash))
        if j==stop-start-1:
            btc,cash,exitfee,exitturn=trade(btc,cash,0.,cost);fee+=exitfee;turnover+=exitturn
            if minute_dd:mindd=min(mindd,(btc+cash)/peak-1)
        ending=btc+cash;nav.append(ending);dayret.append(ending/before-1);fees.append(fee);turn.append(turnover)
    nav=np.array(nav);rr=np.array(dayret);nv=np.r_[capital,nav];dd=nv/np.maximum.accumulate(nv)-1;n=len(rr)
    std=rr.std(ddof=1)
    stats=dict(days=n,total_return=nav[-1]/capital-1,cagr=(nav[-1]/capital)**(365/n)-1,
       annual_vol=std*np.sqrt(365),daily_vol=std,sharpe=rr.mean()/std*np.sqrt(365) if std>0 else None,
       max_drawdown=float(dd.min()),minute_max_drawdown=mindd if minute_dd else None,
       average_btc_weight=float(np.mean(actualw)),average_target_weight=float(weights.mean()),
       full_btc_days=int(np.sum(weights>=1-1e-12)),final_nav=float(nav[-1]),
       fees_paid_krw=float(sum(fees)),fees_pct_initial=float(sum(fees)/capital),
       turnover_total=float(sum(turn)),turnover_annual=float(sum(turn)*365/n),worst_day=float(rr.min()),best_day=float(rr.max()),
       skipped_rebalance_days=int((~OBS[start:stop,0]).sum()))
    return dict(stats=stats,nav=nav,returns=rr,fees=np.array(fees),turnover=np.array(turn),
        weights=np.array(actualw),target=weights,close_weights=np.array(closew),pre_weights=np.array(prew),drawdown=dd[1:])

def validate_ledger():
    for b,c in [(0.,100.),(100.,0.),(43.,57.)]:
        for w in [0.,.25,.5,1.]:
            for fee in [0.,.001,.0025]:trade(b,c,w,fee)
    i=int(np.searchsorted(D,np.datetime64('2026-01-01')));end=len(D)
    cash=simulate(i,end,np.zeros(end-i),.001)
    assert np.array_equal(cash['nav'],np.full(end-i,P['initial_capital_krw']))
    bh=simulate(i,end,np.ones(end-i),0)
    assert np.isclose(bh['nav'][-1]/P['initial_capital_krw'],np.exp(R[i:end].sum()-R[i,0]),rtol=1e-10)
    assert np.isclose(np.prod(1+bh['returns']),bh['nav'][-1]/P['initial_capital_krw'],rtol=1e-10)

def diagnostics(e):
    e=np.asarray(e);n=len(e);ans={}
    for key,xx in [('standardized',e),('squared_standardized',e**2)]:
        z=xx-xx.mean();rho=np.array([np.dot(z[k:],z[:-k])/np.dot(z,z) for k in range(1,11)])
        Q=n*(n+2)*np.sum(rho**2/(n-np.arange(1,11)))
        ans[key]={'acf1':float(rho[0]),'ljung_box_10_Q':float(Q),'approx_p_unadjusted':float(chi2.sf(Q,10))}
    return ans

def bootstrap_difference(a,b,label,seed=213):
    # Paired circular moving-block bootstrap; descriptive CI after prior selection.
    dif=np.asarray(a)-np.asarray(b);n=len(dif);block=14;rep=3000;rng=np.random.default_rng(seed)
    starts=rng.integers(0,n,size=(rep,math.ceil(n/block)))
    idx=((starts[:,:,None]+np.arange(block))%n).reshape(rep,-1)[:,:n]
    means=dif[idx].mean(1)
    return dict(label=label,mean_difference=float(dif.mean()),ci95_low=float(np.quantile(means,.025)),
        ci95_high=float(np.quantile(means,.975)),block_days=block,replications=rep)

mode=sys.argv[1]
if mode=='validation':
    cut,stop,forecasts,parameters,models=candidate_forecasts('2024-01-01','2026-01-01')
    leaderboard=[dict(model=k,**scores(V[cut:stop],v)) for k,v in forecasts.items()]
    leaderboard.sort(key=lambda x:(x['qlike'],x['mse']))
    winner=leaderboard[0]['model'];pred=forecasts[winner]
    fixed_weight=float(np.minimum(1,P['risk_target_daily_sd']/np.sqrt(pred)).mean())
    selection=dict(winner=winner,criterion=P['selection'],validation_days=stop-cut,
        train_end='2023-12-31',validation_end='2025-12-31',fixed_validation_mean_weight=fixed_weight,
        protocol_sha256=hashlib.sha256((W/'btc_cash_protocol.json').read_bytes()).hexdigest(),
        leaderboard=leaderboard,parameters=parameters)
    save('btc_cash_selection.json',selection)
    np.savez_compressed(W/'btc_cash_validation_forecasts.npz',dates=D[cut:stop],**forecasts)
    print(json.dumps(native(selection['leaderboard']),ensure_ascii=False),flush=True)
    print('LOCKED_WINNER',winner,'FIXED_BENCHMARK_WEIGHT',fixed_weight,flush=True)
elif mode=='test':
    sel=json.loads((W/'btc_cash_selection.json').read_text(encoding='utf-8'))
    assert sel['protocol_sha256']==hashlib.sha256((W/'btc_cash_protocol.json').read_bytes()).hexdigest()
    validate_ledger();cut,stop,forecasts,parameters,models=candidate_forecasts('2026-01-01','2026-10-12')
    winner=sel['winner'];allweights={k:np.minimum(1,P['risk_target_daily_sd']/np.sqrt(f)) for k,f in forecasts.items()}
    allweights.update(BTC_buy_hold=np.ones(stop-cut),BTC50_cash50_daily=np.full(stop-cut,.5),
        fixed_validation_mean_weight=np.full(stop-cut,sel['fixed_validation_mean_weight']),cash=np.zeros(stop-cut))
    portfolios={k:simulate(cut,stop,w,P['base_one_way_cost'],minute_dd=True) for k,w in allweights.items()}
    metricrows=[]
    for k,p in portfolios.items():metricrows.append(dict(period='test_2026',strategy=k,selected_primary=k==winner,cost=P['base_one_way_cost'],**p['stats']))
    testscore=[dict(period='test_2026',model=k,selected_primary=k==winner,**scores(V[cut:stop],f)) for k,f in forecasts.items()]
    for row in testscore:row['qlike_improvement_vs_RV_last']=1-row['qlike']/next(q['qlike'] for q in testscore if q['model']=='RV_last')
    sensitivity=[]
    for fee in P['cost_sensitivity']:
        for k in [winner,'RV_last','BTC_buy_hold','BTC50_cash50_daily','fixed_validation_mean_weight']:
            ss=simulate(cut,stop,allweights[k],fee)['stats'];sensitivity.append(dict(strategy=k,cost=fee,**ss))
    valnpz=np.load(W/'btc_cash_validation_forecasts.npz');vcut=int(np.searchsorted(D,np.datetime64('2024-01-01')))
    valweights=np.minimum(1,P['risk_target_daily_sd']/np.sqrt(valnpz[winner]))
    valport={k:simulate(vcut,cut,w,P['base_one_way_cost']) for k,w in {
        winner:valweights,'BTC_buy_hold':np.ones(cut-vcut),'BTC50_cash50_daily':np.full(cut-vcut,.5),
        'fixed_validation_mean_weight':np.full(cut-vcut,sel['fixed_validation_mean_weight'])}.items()}
    for k,p in valport.items():metricrows.append(dict(period='validation_2024_2025_influenced_selection',strategy=k,selected_primary=k==winner,cost=P['base_one_way_cost'],**p['stats']))
    firstcost=portfolios[winner]['fees'][0];lastcost=portfolios[winner]['fees'][-1]
    # Match-risk *diagnostic*: use a fraction of the already constructed fixed benchmark plus cash.
    # This uses realized TEST volatility and is explicitly not an investable ex-ante rule.
    bench=portfolios['fixed_validation_mean_weight'];primary=portfolios[winner]
    ratio=primary['stats']['annual_vol']/bench['stats']['annual_vol']
    diagnostic=None
    if ratio<=1:
        rr=ratio*bench['returns'];nv=P['initial_capital_krw']*np.cumprod(1+rr)
        diagnostic=dict(label='ex_post_risk_matched_fixed_benchmark_diagnostic_only',scaling=ratio,
            total_return=float(nv[-1]/P['initial_capital_krw']-1),annual_vol=float(rr.std(ddof=1)*np.sqrt(365)),
            sharpe=float(rr.mean()/rr.std(ddof=1)*np.sqrt(365)),
            explanation='Uses realized test volatility; not a tradable ex-ante benchmark; scaling net benchmark returns is a diagnostic approximation')
    losses={k:V[cut:stop]/f-np.log(V[cut:stop]/f)-1 for k,f in forecasts.items()}
    bootstrap=[bootstrap_difference(losses[winner],losses['RV_last'],'QLIKE primary minus last-RV; negative is better'),
      bootstrap_difference(primary['returns'],bench['returns'],'net daily return primary minus fixed-validation-weight')]
    diag={k:diagnostics((RET[cut:stop]-models[k]['mu'])/np.sqrt(forecasts[k])) for k in models}
    monthly=[]
    for month in np.unique(D[cut:stop].astype('datetime64[M]')):
        mask=D[cut:stop].astype('datetime64[M]')==month
        for k,p in portfolios.items():
            monthly.append(dict(month=str(month),strategy=k,days=int(mask.sum()),return_net=float(np.prod(1+p['returns'][mask])-1),
                average_weight=float(p['weights'][mask].mean())))
    daily=[]
    for j,t in enumerate(range(cut,stop)):
        row=dict(date_utc=str(D[t]),forecast_information_through=str(D[t-1]),
            rv_5m=float(V[t]),btc_log_return=float(RET[t]),btc_simple_return=float(np.expm1(RET[t])),
            first_minute_return=float(np.expm1(R[t,0])),coverage=float(Z['coverage_ratio'][t]),
            rebalance_endpoint_observed=bool(OBS[t,0]))
        for k,f in forecasts.items():row[k+'_variance_forecast']=float(f[j])
        for k,p in portfolios.items():
            for field in ['target','weights','pre_weights','returns','nav','fees','turnover']:row[k+'_'+field]=float(p[field][j])
        daily.append(row)
    report=dict(protocol=P,selection=sel,test_parameters=parameters,test_forecast_scores=testscore,portfolio_metrics=metricrows,
       cost_sensitivity=sensitivity,monthly=monthly,bootstrap=bootstrap,residual_diagnostics=diag,
       ex_post_risk_matched_diagnostic=diagnostic,quality=dict(full_days=len(D),test_days=stop-cut,
           test_start=str(D[cut]),test_end=str(D[stop-1]),test_low_coverage_days=int((Z['coverage_ratio'][cut:stop]<.95).sum()),
           test_filled_minutes=int(Z['filled_minutes'][cut:stop].sum()),validation_skipped_entries=int((~OBS[vcut:cut,0]).sum()),
           test_skipped_entries=int((~OBS[cut:stop,0]).sum())),
       checks=['minute returns sum to daily log returns','daily returns match consecutive daily closes',
       'arch first forecast independently matches manual recursion for all GARCH fits',
       'feature rows use only prior observations','converged GARCH fits and admissible coefficients',
       'self-financing fees and target weights checked for buys and sells','cash NAV invariant',
       'zero-cost buy-and-hold matches lagged source price ratio','test selection locked before test forecasts'],
       data_sha256=hashlib.sha256((W/'minute_plot_arrays.npz').read_bytes()).hexdigest())
    save('btc_cash_results.json',report);save('btc_cash_daily.json',daily);save('btc_cash_summary.json',metricrows)
    save('btc_cash_forecast_scores.json', [dict(period='validation_2024_2025',selected_primary=r['model']==winner,**r) for r in sel['leaderboard']]+testscore)
    arrays={'dates':D[cut:stop],'validation_dates':D[vcut:cut],'realized_rv':V[cut:stop]}
    for k,p in portfolios.items():
        for f in ['nav','returns','weights','target','drawdown']:arrays[k+'_'+f]=p[f]
    for k,p in valport.items():arrays['validation_'+k+'_nav']=p['nav']
    for k,v in forecasts.items():arrays[k+'_forecast']=v
    np.savez_compressed(W/'btc_cash_plot_arrays.npz',**arrays)
    print('SELECTED',winner,flush=True)
    print(json.dumps(native(testscore),ensure_ascii=False),flush=True)
    print(json.dumps(native(metricrows[:len(portfolios)]),ensure_ascii=False),flush=True)
else:raise ValueError(mode)
