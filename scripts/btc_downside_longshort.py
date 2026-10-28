"""Predeclared downside-event directional model and synthetic BTC/KRW ledger."""
import sys,json,hashlib,datetime,gzip,math
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path('work/model_packages').resolve()))
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata

W=Path('work');Z=np.load(W/'minute_plot_arrays.npz');D=Z['dates'];R=Z['returns'].reshape(-1,1440)
V=Z['rv_5m'];RET=Z['daily_log_return'];CLOSE=Z['close_krw'];COVER=Z['coverage_ratio']
OBS=Z['observed_return'].reshape(-1,1440)[:,0]
last_rows=json.load(gzip.open(W/'minute_cache/2026-10.json.gz','rt'))
last_row=next(x for x in last_rows if x['candle_date_time_utc']=='2026-10-12T00:00:00')
EP=np.r_[CLOSE*np.exp(-R[:,1:].sum(axis=1)),float(last_row['trade_price'])]
NEXTRET=EP[1:]/EP[:-1]-1
NEXTUP=(NEXTRET>0).astype(float)
NEXT_OBS=np.r_[OBS[1:],True]
FIVE=R.reshape(-1,288,5).sum(axis=2)
assert np.allclose((FIVE**2).sum(axis=1),V,rtol=1e-10,atol=1e-15)
NAMES=['shock_1d','trend_7d','trend_30d','drawdown_30d','rv_shock','downside_variance_share','negative_streak','large_drop','large_drop_x_trend7']
X=np.full((len(D),len(NAMES)),np.nan)
for t in range(31,len(D)):
    variance_ref=V[t-31:t-1].mean();sigma=np.sqrt(variance_ref)
    shock=RET[t-1]/sigma;trend7=RET[t-7:t].sum()/(np.sqrt(7)*sigma)
    trend30=RET[t-30:t].sum()/(np.sqrt(30)*sigma)
    dd=np.log(CLOSE[t-1]/CLOSE[t-30:t].max())/sigma
    rvshock=np.log(V[t-1]/variance_ref)
    downside=np.sum(FIVE[t-1][FIVE[t-1]<0]**2)/V[t-1]
    streak=0
    for j in range(t-1,max(-1,t-11),-1):
        if RET[j]>=0:break
        streak+=1
    large=float(shock<=-1)
    X[t]=[shock,trend7,trend30,dd,rvshock,downside,streak,large,large*trend7]
EVENT=np.r_[False,RET[:-1]<0]
FEATURE_OK=np.r_[False,COVER[:-1]>=.95]&np.isfinite(X).all(axis=1)
TRAIN_OK=EVENT&FEATURE_OK&(COVER>=.95)&OBS&NEXT_OBS
PROTOCOL=dict(created_local_date='2026-10-14',purpose='Learn next-day direction after a negative prior UTC day; size synthetic BTC long/short exposure with HAR risk forecasts.',
    data_source='Existing Upbit BTC/KRW minute-price cache only; no derivatives series or new account connection.',
    event='Prior completed UTC daily log return < 0. A large-drop diagnostic is prior return / preceding 30-day RV-based daily SD <= -1.',
    label='1 if next execution-to-execution SIMPLE return > 0, else 0; not a label of whether the preceding day fell.',
    information_time='UTC 00:00 / KST 09:00. Execute at close of UTC 00:00 minute, observable at UTC 00:01.',
    label_horizon='Execution at day t 00:01 through day t+1 00:01. Last test exit 2026-10-12 00:01 UTC.',
    embargo='At refit time t 00:00, include training targets only through t-2: the t-1 label ends at t 00:01 and is not yet available.',
    direction_model='L2-regularized binary logistic regression on prior-down-day examples only, standardized using each training window only.',
    features=NAMES,training_window='3 calendar years before each monthly refit, plus one-day label availability embargo',refit='first prediction day of each UTC calendar month',
    objective='mean(log(1+exp(intercept+X*b)) - y*(intercept+X*b)) + lambda/2 * sum(b**2); intercept not penalized',
    lambda_candidates=[.001,.01,.1],lambda_selection='minimum validation log loss on feature-eligible prior-down days with observed endpoints; tie favors stronger regularization',
    confidence_candidates=[.50,.55,.60],threshold_selection='maximum validation net daily Sharpe among thresholds with >=30 active days; tie favors fewer traded-notional turnovers',
    confidence_rule='On a feature-eligible prior-down day, long if p_up >= tau, short if p_up <= 1-tau, otherwise flat. On other days flat. If tau=0.5, exact p=0.5 is flat.',
    selection_order='Choose lambda from validation probability loss, then threshold from validation strategy net Sharpe. Freeze both before test. Report all validation alternatives.',
    validation=['2024-01-01','2025-12-31'],test=['2026-01-01','2026-10-11'],
    volatility='Existing fixed-all HAR forecasts from each prior phase: validation parameters through 2023; test parameters through 2025. Daily UTC 5-minute RV is a risk proxy for the one-minute-shifted holding window.',
    size_rule='signed weight = signal * min(1, .01/sqrt(HAR variance forecast)); cap applies at rebalancing',
    initial_capital_krw=10000000,fee_one_way=.001,short_borrow_annual=.10,cash_interest=0,
    fee_sensitivity=[0,.0005,.001,.0025],borrow_sensitivity=[0,.10,.20],
    ledger='Synthetic signed BTC quantity plus KRW cash including short-sale proceeds. Financing assumed available; short borrow charged daily at annual rate/365 on entry short notional. Exact post-fee target sizing, final liquidation included.',
    boundaries='Not executable on Upbit spot as a short strategy. No actual derivatives basis, funding, borrow availability, exchange margin, liquidation or order-book fills modeled. No real orders.',
    quality='Training excludes examples with prior-day or target-day candle coverage below 95%, or missing execution endpoints. Live signals can only use prior-day quality; all test dates remain in PnL. Missing execution candle retains prior position.',
    annualization_days=365,scope='Single preregistered prototype, nine directional features, three regularization strengths and three thresholds. No test-driven retuning.',
    evaluation_status='2026 was already explored and backtested in this conversation. Results are retrospective additional research, not fresh blind evidence.',
    data_sha256=hashlib.sha256((W/'minute_plot_arrays.npz').read_bytes()).hexdigest(),last_execution_price=EP[-1].item(),
    last_execution_source_candle=last_row)

def native(x):
    if isinstance(x,dict):return {k:native(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [native(v) for v in x]
    if isinstance(x,np.ndarray):return native(x.tolist())
    if isinstance(x,np.integer):return int(x)
    if isinstance(x,np.bool_):return bool(x)
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    return x
def save(name,obj):
    (W/name).write_text(json.dumps(native(obj),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def lock_protocol():
    path=W/'btc_longshort_protocol.json'
    if path.exists():assert json.loads(path.read_text(encoding='utf-8'))==native(PROTOCOL)
    else:save(path.name,PROTOCOL)
    return hashlib.sha256(path.read_bytes()).hexdigest()
def train_start(t):
    day=datetime.date.fromisoformat(str(D[t]))
    try:day=day.replace(year=day.year-3)
    except ValueError:day=day.replace(year=day.year-3,day=28)
    return max(31,int(np.searchsorted(D,np.datetime64(day))))
def fit_at(t,lam):
    # Excludes label t-1, which finishes a minute AFTER the information cutoff.
    idx=np.arange(train_start(t),t-1);idx=idx[TRAIN_OK[idx]]
    assert len(idx)>=150 and idx.max()<=t-2
    xx=X[idx];yy=NEXTUP[idx];mean=xx.mean(axis=0);scale=xx.std(axis=0)
    scale=np.where(scale<1e-12,1,scale);xx=(xx-mean)/scale
    design=np.column_stack([np.ones(len(idx)),xx])
    def objective(b):
        linear=design@b
        loss=np.mean(np.logaddexp(0,linear)-yy*linear)+lam*.5*np.dot(b[1:],b[1:])
        grad=design.T@(expit(linear)-yy)/len(yy);grad[1:]+=lam*b[1:]
        return loss,grad
    init=np.zeros(design.shape[1]);p0=(yy.sum()+1)/(len(yy)+2);init[0]=np.log(p0/(1-p0))
    fit=minimize(objective,init,jac=True,method='L-BFGS-B',options={'maxiter':2000,'ftol':1e-13,'gtol':1e-8})
    assert fit.success and np.max(abs(objective(fit.x)[1]))<1e-5,(t,lam,fit.message)
    return dict(refit_date=str(D[t]),refit_index=t,training_first=str(D[idx[0]]),training_last=str(D[idx[-1]]),
        training_last_label_endpoint=str(D[idx[-1]+1])+'T00:01:00Z',training_indices=idx,n=len(idx),
        positive=int(yy.sum()),lambda_value=lam,mean=mean,scale=scale,coef=fit.x,baseline_probability=p0,
        optimizer_gradient_max=float(np.max(abs(objective(fit.x)[1]))))
def predict_period(cut,stop,lam):
    p=[];null=[];records=[];refit_ids=[];prior_month=None
    for t in range(cut,stop):
        month=str(D[t])[:7]
        if month!=prior_month:
            model=fit_at(t,lam);records.append(model);prior_month=month
        prob=float(expit(model['coef'][0]+((X[t]-model['mean'])/model['scale'])@model['coef'][1:]))
        assert np.isfinite(prob)
        p.append(prob);null.append(model['baseline_probability']);refit_ids.append(len(records)-1)
    return np.array(p),np.array(null),records,np.array(refit_ids)
def direction_scores(cut,stop,p,null):
    mask=EVENT[cut:stop]&FEATURE_OK[cut:stop]&OBS[cut:stop]&NEXT_OBS[cut:stop]
    yy=NEXTUP[cut:stop][mask];pp=p[mask];bb=null[mask]
    def ll(q):return float(np.mean(np.logaddexp(0,np.log(q/(1-q)))-yy*np.log(q/(1-q))))
    ranks=rankdata(pp);pos=yy==1;n1=int(pos.sum());n0=len(yy)-n1
    auc=float((ranks[pos].sum()-n1*(n1+1)/2)/(n1*n0)) if n1 and n0 else None
    return dict(n=len(yy),up_rate=float(yy.mean()),log_loss=ll(pp),null_log_loss=ll(bb),brier=float(np.mean((pp-yy)**2)),
        null_brier=float(np.mean((bb-yy)**2)),auc=auc,accuracy=float(np.mean((pp>.5)==yy)),
        constant_up_accuracy=float(yy.mean()),mean_p=float(pp.mean()),min_p=float(pp.min()),max_p=float(pp.max()))
def signals(cut,stop,p,tau):
    sig=np.where(p>=tau,1,np.where(p<=1-tau,-1,0))
    if tau==.5:sig[p==.5]=0
    sig[~(EVENT[cut:stop]&FEATURE_OK[cut:stop])]=0
    return sig.astype(int)
def simulate(cut,stop,weights,cost=.001,borrow=.1,intraday=False):
    cash=float(PROTOCOL['initial_capital_krw']);q=0.;nav=[];fees=[];borrow_paid=[];gross=[];rets=[];turn=[];positions=[];skips=0
    peak=cash;minute_dd=0.;max_leverage=0.;positive_equity=True;previous=cash
    for j,t in enumerate(range(cut,stop)):
        price=EP[t];B=q*price;equity=cash+B
        assert equity>0
        fee=0.;traded=0.
        if OBS[t]:
            w=float(weights[j]);assert -1<=w<=1
            delta=w*equity-B
            traded=delta/(1+cost*w) if delta>=0 else delta/(1-cost*w)
            fee=cost*abs(traded);cash-=traded+fee;q+=traded/price
            assert np.isclose(q*price/(cash+q*price),w,rtol=1e-9,atol=1e-10)
        else:skips+=1
        after=cash+q*price;B=q*price
        positions.append(B/after);turn.append(abs(traded)/equity)
        financing=max(-B,0)*borrow/365
        pnl=q*(EP[t+1]-price)
        if intraday:
            rel=np.r_[1,np.exp(np.cumsum(R[t,1:])),EP[t+1]/price]
            path=after+B*(rel-1)-financing*np.linspace(0,1,len(rel))
            positive_equity=positive_equity and bool((path>0).all())
            assert (path>0).all(),'Synthetic account insolvent; trading protocol requires explicit default handling.'
            peaks=np.maximum.accumulate(np.r_[peak,path])[1:]
            minute_dd=min(minute_dd,float(np.min(path/peaks-1)));peak=float(peaks[-1])
            max_leverage=max(max_leverage,float(np.max(abs(B*rel)/path)))
        cash-=financing
        if j==stop-cut-1:
            fee_exit=cost*abs(q*EP[t+1]);cash+=q*EP[t+1]-fee_exit;q=0.;fee+=fee_exit
        end=cash+q*EP[t+1]
        nav.append(end);rets.append(end/previous-1);previous=end;fees.append(fee);borrow_paid.append(financing);gross.append(pnl)
        if intraday:minute_dd=min(minute_dd,end/peak-1)
    nav=np.array(nav);rets=np.array(rets);positions=np.array(positions);nv=np.r_[PROTOCOL['initial_capital_krw'],nav]
    dd=nv/np.maximum.accumulate(nv)-1;std=rets.std(ddof=1);active=abs(positions)>1e-10
    stats=dict(days=len(nav),return_net=float(nav[-1]/nv[0]-1),sharpe=float(rets.mean()/std*np.sqrt(365)) if std>0 else None,
        annual_vol=float(std*np.sqrt(365)),mdd=float(dd.min()),minute_mdd=minute_dd if intraday else None,
        long_days=int((positions>1e-10).sum()),short_days=int((positions<-1e-10).sum()),flat_days=int((~active).sum()),
        avg_abs_weight=float(abs(positions).mean()),fees_krw=float(sum(fees)),borrow_krw=float(sum(borrow_paid)),
        gross_pnl_krw=float(sum(gross)),final_nav=float(nav[-1]),turnover=float(sum(turn)),skipped_rebalances=skips,
        max_intraday_gross_exposure_to_equity=max_leverage if intraday else None,positive_equity_on_observed_path=positive_equity if intraday else None,
        gross_direction_hit_rate=float(np.mean(positions[active]*NEXTRET[cut:stop][active]>0)) if active.any() else None,
        long_gross_pnl_krw=float(np.array(gross)[positions>1e-10].sum()),short_gross_pnl_krw=float(np.array(gross)[positions<-1e-10].sum()))
    assert np.isclose(nv[0]+sum(gross)-sum(fees)-sum(borrow_paid),nav[-1],rtol=1e-11,atol=1e-5)
    return dict(stats=stats,nav=nav,returns=rets,fees=np.array(fees),borrow=np.array(borrow_paid),gross=np.array(gross),weights=positions,target=weights,drawdown=dd[1:])
def phase_bounds(phase):
    dates=PROTOCOL[phase];cut=int(np.searchsorted(D,np.datetime64(dates[0])));stop=int(np.searchsorted(D,np.datetime64(dates[1])))+1
    return cut,stop
def risk_forecast(phase):
    if phase=='validation':return np.load(W/'btc_cash_validation_forecasts.npz')['HAR_logRV_all']
    return np.load(W/'btc_cash_plot_arrays.npz')['HAR_logRV_all_forecast']
def block_difference(x,y,seed=331):
    diff=np.array(x)-np.array(y);n=len(diff);rng=np.random.default_rng(seed);block=14;rep=3000
    starts=rng.integers(0,n,size=(rep,math.ceil(n/block)));idx=((starts[:,:,None]+np.arange(block))%n).reshape(rep,-1)[:,:n]
    means=diff[idx].mean(axis=1)
    return dict(mean=float(diff.mean()),ci95=[float(np.quantile(means,.025)),float(np.quantile(means,.975))],block=block,replications=rep)
def event_table(cut,stop):
    out=[];y=NEXTRET[cut:stop]
    masks={'all':np.ones(stop-cut,dtype=bool),'prior_down':EVENT[cut:stop],'large_prior_drop':X[cut:stop,0]<=-1,
        'prior_nonnegative':~EVENT[cut:stop]}
    for name,mask in masks.items():
        yy=y[mask]
        out.append(dict(group=name,n=int(mask.sum()),up_rate=float((yy>0).mean()) if len(yy) else None,
            mean_next_return=float(yy.mean()) if len(yy) else None,median_next_return=float(np.median(yy)) if len(yy) else None))
    return out
def save_phase(phase,cut,stop,p,null,records,refs,sig,ports,other):
    arr=dict(dates=D[cut:stop],execution_dates=np.r_[D,np.datetime64('2026-10-12')][cut:stop+1],
        execution_price=EP[cut:stop+1],next_return=NEXTRET[cut:stop],prior_return=RET[cut-1:stop-1],
        features=X[cut:stop],p_up=p,p_null=null,event=EVENT[cut:stop],feature_ok=FEATURE_OK[cut:stop],signal=sig,refit_id=refs)
    for k,port in ports.items():
        for field in ['nav','returns','fees','borrow','gross','weights','target','drawdown']:arr[k+'_'+field]=port[field]
    np.savez_compressed(W/f'btc_longshort_{phase}_arrays.npz',**arr)
    daily=[]
    for j,t in enumerate(range(cut,stop)):
        row=dict(date_utc=str(D[t]),information_cutoff=str(D[t])+'T00:00:00Z',execute_at=str(D[t])+'T00:01:00Z',
            exit_mark_at=str(np.r_[D,np.datetime64('2026-10-12')][t+1])+'T00:01:00Z',execution_price=EP[t],next_execution_price=EP[t+1],
            prior_day_return=RET[t-1],prior_down=EVENT[t],feature_ok=FEATURE_OK[t],p_up=p[j],p_null=null[j],signal=int(sig[j]),
            next_return_realized=NEXTRET[t],refit_date=records[refs[j]]['refit_date'],training_last_label_endpoint=records[refs[j]]['training_last_label_endpoint'],
            har_forecast=risk_forecast(phase)[j])
        for k,port in ports.items():
            for f in ['nav','returns','fees','borrow','weights']:row[k+'_'+f]=float(port[f][j])
        daily.append(row)
    save(f'btc_longshort_{phase}_daily.json',daily)
    save(f'btc_longshort_{phase}_models.json',records)
    save(f'btc_longshort_{phase}_results.json',dict(phase=phase,start=str(D[cut]),end=str(D[stop-1]),
        final_exit=str(np.r_[D,np.datetime64('2026-10-12')][stop])+'T00:01:00Z',
        direction=direction_scores(cut,stop,p,null),portfolios={k:x['stats'] for k,x in ports.items()},events=event_table(cut,stop),**other))
def run(phase):
    protocol_hash=lock_protocol();cut,stop=phase_bounds(phase);vf=risk_forecast(phase);size=np.minimum(1,.01/np.sqrt(vf))
    if phase=='validation':
        candidates=[];predictions={}
        for lam in PROTOCOL['lambda_candidates']:
            pred=predict_period(cut,stop,lam);predictions[str(lam)]=pred
            scores=direction_scores(cut,stop,pred[0],pred[1]);candidates.append(dict(lambda_value=lam,**scores))
            print('VALIDATION_PROB',lam,json.dumps(scores),flush=True)
        chosen=min(candidates,key=lambda r:(r['log_loss'],-r['lambda_value']))['lambda_value']
        p,null,records,refs=predictions[str(chosen)]
        thresholds=[]
        for tau in PROTOCOL['confidence_candidates']:
            sig=signals(cut,stop,p,tau);port=simulate(cut,stop,sig*size)
            thresholds.append(dict(tau=tau,**port['stats']))
            print('VALIDATION_STRATEGY',tau,json.dumps(port['stats']),flush=True)
        eligible=[x for x in thresholds if x['long_days']+x['short_days']>=30 and x['sharpe'] is not None]
        assert eligible
        choice=max(eligible,key=lambda x:(x['sharpe'],-x['turnover']))
        tau=choice['tau'];selection=dict(protocol_sha256=protocol_hash,lambda_value=chosen,tau=tau,
            probability_candidates=candidates,threshold_candidates=thresholds,
            selected_before_test=True,validation_probability_beats_null=min(candidates,key=lambda r:(r['log_loss'],-r['lambda_value']))['log_loss']<candidates[0]['null_log_loss'],
            validation_return_positive=choice['return_net']>0,
            validation_has_at_least_10_long_and_short_days=min(choice['long_days'],choice['short_days'])>=10)
        save('btc_longshort_selection.json',selection)
    elif phase=='test':
        selection=json.loads((W/'btc_longshort_selection.json').read_text(encoding='utf-8'))
        assert selection['protocol_sha256']==protocol_hash
        chosen=selection['lambda_value'];tau=selection['tau'];p,null,records,refs=predict_period(cut,stop,chosen)
    else:raise ValueError(phase)
    sig=signals(cut,stop,p,tau);event=EVENT[cut:stop]&FEATURE_OK[cut:stop]
    weights={'signal_har':sig*size,'signal_fixed50':sig*.5,'signal_long_only':np.maximum(sig,0)*size,
        'after_down_always_long':event.astype(float)*size,'after_down_always_short':-event.astype(float)*size,
        'har_always_long':size,'BTC50_cash50':np.full(stop-cut,.5),'BTC_buy_hold':np.ones(stop-cut),'cash':np.zeros(stop-cut)}
    ports={k:simulate(cut,stop,w,intraday=True) for k,w in weights.items()}
    extra=dict(protocol_sha256=protocol_hash,selection=selection,
        low_coverage_target_days=[dict(date=str(D[t]),coverage=COVER[t]) for t in range(cut,stop) if COVER[t]<.95])
    if phase=='test':
        extra['cost_sensitivity']=[dict(fee=fee,borrow=borrow,**simulate(cut,stop,weights['signal_har'],fee,borrow)['stats']) for fee in PROTOCOL['fee_sensitivity'] for borrow in PROTOCOL['borrow_sensitivity']]
        extra['bootstrap']={k:block_difference(ports['signal_har']['returns'],ports[k]['returns']) for k in ['cash','after_down_always_long','after_down_always_short','signal_fixed50','har_always_long']}
        mask=EVENT[cut:stop]&FEATURE_OK[cut:stop]&OBS[cut:stop]&NEXT_OBS[cut:stop];y=NEXTUP[cut:stop]
        losses=-(y*np.log(p)+(1-y)*np.log(1-p));null_losses=-(y*np.log(null)+(1-y)*np.log(1-null))
        # Full-calendar blocks preserve gaps between prior-down events.
        rng=np.random.default_rng(225);n=len(p);st=rng.integers(0,n,size=(3000,math.ceil(n/14)))
        idx=((st[:,:,None]+np.arange(14))%n).reshape(3000,-1)[:,:n]
        dif=np.where(mask,losses-null_losses,0);est=dif[idx].sum(1)/mask[idx].sum(1)
        extra['probability_loss_difference']=dict(mean=float((losses-null_losses)[mask].mean()),ci95=np.quantile(est,[.025,.975]).tolist())
        extra['calibration']=[]
        for low,high in [(0,.4),(.4,.45),(.45,.5),(.5,.55),(.55,.6),(.6,1.0000001)]:
            sub=mask&(p>=low)&(p<high)
            extra['calibration'].append(dict(lower=low,upper=min(1,high),n=int(sub.sum()),mean_probability=float(p[sub].mean()) if sub.any() else None,up_rate=float(y[sub].mean()) if sub.any() else None))
    extra['monthly']=[]
    for month in np.unique(D[cut:stop].astype('datetime64[M]')):
        mask=D[cut:stop].astype('datetime64[M]')==month
        for k,port in ports.items():
            extra['monthly'].append(dict(month=str(month),strategy=k,return_net=float(np.prod(1+port['returns'][mask])-1),
                long_days=int((port['weights'][mask]>1e-10).sum()),short_days=int((port['weights'][mask]<-1e-10).sum())))
    save_phase(phase,cut,stop,p,null,records,refs,sig,ports,extra)
    print(phase,'FROZEN',chosen,tau,flush=True)
    for k,x in ports.items():print(k,json.dumps(x['stats']),flush=True)

if __name__=='__main__':run(sys.argv[1])
