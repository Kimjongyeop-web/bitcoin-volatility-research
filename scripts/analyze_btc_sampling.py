"""BTC-only sampling sensitivity; common daily endpoints and source grid."""
import json, math
from pathlib import Path
import numpy as np

Z=np.load('work/minute_plot_arrays.npz')
dates=Z['dates']; n=len(dates); R=Z['returns'].reshape(n,1440)
assert np.isfinite(R).all() and np.all(np.diff(dates)==np.timedelta64(1,'D'))
freqs=[1,2,3,5,10,15,30,60]
V={m:np.square(R.reshape(n,1440//m,m).sum(axis=2)).sum(axis=1) for m in freqs}
P=Z['prv_scalar']
assert np.allclose(V[1],Z['rv_1m'],rtol=1e-12,atol=1e-15)
assert np.allclose(V[5],Z['rv_5m'],rtol=1e-12,atol=1e-15)
for m in freqs:
    assert np.allclose(R.reshape(n,1440//m,m).sum(axis=(1,2)),Z['daily_log_return'],rtol=1e-10,atol=1e-12)
    assert np.isfinite(V[m]).all() and (V[m]>0).all()
assert (P>0).all()
qmask=Z['coverage_ratio']>=.95
years=np.array([str(d)[:4] for d in dates])

def ranks(x):
    _,inv,cnt=np.unique(x,return_inverse=True,return_counts=True)
    return (np.cumsum(cnt)-(cnt-1)/2)[inv]

def comparison(x,ref,mask=None):
    if mask is not None: x=x[mask];ref=ref[mask]
    N=len(x); k=math.ceil(.1*N)
    # Report cutoff ties explicitly if present; our observations have none.
    a=np.argsort(x,kind='stable')[-k:];b=np.argsort(ref,kind='stable')[-k:]
    assert np.unique(x).size==N and np.unique(ref).size==N
    overlap=len(np.intersect1d(a,b))
    rat=x/ref
    return dict(days=N,mean_rv=float(x.mean()),median_rv=float(np.median(x)),
      mean_rv_ratio_to_5m=float(x.mean()/ref.mean()),
      daily_ratio_median=float(np.median(rat)),daily_ratio_p05=float(np.quantile(rat,.05)),daily_ratio_p95=float(np.quantile(rat,.95)),
      pearson_rv_to_5m=float(np.corrcoef(x,ref)[0,1]),
      spearman_to_5m=float(np.corrcoef(ranks(x),ranks(ref))[0,1]),
      top10_count=k,top10_overlap_count=overlap,top10_overlap_fraction=overlap/k,
      median_vol_pct=float(np.median(100*np.sqrt(x))))

summary=[]
for m in freqs:
    summary.append(dict(measure=f'RV_{m}m',interval_minutes=m,returns_per_day=1440//m,**comparison(V[m],V[5])))
summary.append(dict(measure='PRV',interval_minutes=None,returns_per_day=None,**comparison(P,V[5])))
quality=[dict(measure=f'RV_{m}m',**comparison(V[m],V[5],qmask)) for m in freqs]
quality.append(dict(measure='PRV',**comparison(P,V[5],qmask)))
annual=[]
for year in np.unique(years):
    mask=years==year
    annual.append(dict(year=year,days=int(mask.sum()),
      low_coverage_days=int((mask & ~qmask).sum()),
      mean_ratio_rv1_to5=float(V[1][mask].mean()/V[5][mask].mean()),
      median_ratio_rv1_to5=float(np.median(V[1][mask]/V[5][mask])),
      mean_ratio_prv_to5=float(P[mask].mean()/V[5][mask].mean()),
      signature_mean=[float(V[m][mask].mean()/V[5][mask].mean()) for m in freqs],
      signature_median=[float(np.median(V[m][mask])/np.median(V[5][mask])) for m in freqs],
      spearman_1_5=float(np.corrcoef(ranks(V[1][mask]),ranks(V[5][mask]))[0,1])))

# Same 24h endpoints for every phase. Nonzero offsets have two partial edge
# blocks; all interior blocks retain the requested sampling interval.
C=np.c_[np.zeros(n),np.cumsum(R,axis=1)]
phase={};phase_summary=[]
for m in [5,15,60]:
    vals=[]
    for shift in range(m):
        bounds=np.unique(np.r_[0,np.arange(shift,1440,m),1440])
        increments=np.diff(C[:,bounds],axis=1)
        assert np.allclose(increments.sum(axis=1),Z['daily_log_return'],rtol=1e-10,atol=1e-12)
        vals.append(np.square(increments).sum(axis=1))
    vals=np.stack(vals,axis=1)
    assert np.allclose(vals[:,0],V[m],rtol=1e-9,atol=1e-14)
    phase[m]=dict(mean=vals.mean(axis=1),minimum=vals.min(axis=1),maximum=vals.max(axis=1))
    spread=(vals.max(axis=1)-vals.min(axis=1))/np.median(vals,axis=1)
    phase_summary.append(dict(interval_minutes=m,phases=m,
       median_daily_range_over_phase_median=float(np.median(spread)),
       p95_daily_range_over_phase_median=float(np.quantile(spread,.95)),
       min_phase_mean_ratio_to_base=float(vals.mean(axis=0).min()/V[m].mean()),
       max_phase_mean_ratio_to_base=float(vals.mean(axis=0).max()/V[m].mean()),
       phase_average_mean_ratio_to_base=float(vals.mean()/V[m].mean())))

def event(i,reason):
    r=R[i]; j=int(np.argmax(abs(r)))
    return dict(index=int(i),date=str(dates[i]),reason=reason,coverage_pct=float(100*Z['coverage_ratio'][i]),
       filled_minutes=int(Z['filled_minutes'][i]),max_stale_minutes=int(Z['max_stale_minutes'][i]),
       rv1_to5=float(V[1][i]/V[5][i]),rv1_to_prv=float(V[1][i]/P[i]),
       vol_pct={str(m):float(100*np.sqrt(V[m][i])) for m in [1,5,15,60]},
       prv_vol_pct=float(100*np.sqrt(P[i])),
       max_abs_minute_index=j,max_abs_minute_logreturn_pct=float(r[j]*100),
       max_minute_share_of_rv1=float(r[j]*r[j]/V[1][i]),
       max_minute_start_utc=str(dates[i].astype('datetime64[m]')+np.timedelta64(j,'m')),
       max_minute_start_kst=str(dates[i].astype('datetime64[m]')+np.timedelta64(j+540,'m')),
       phase5_min=float(phase[5]['minimum'][i]),phase5_max=float(phase[5]['maximum'][i]))

ratio=V[1]/V[5]
eligible=np.flatnonzero(qmask)
imax=eligible[np.argmax(ratio[eligible])];imin=eligible[np.argmin(ratio[eligible])]
iprv=eligible[np.argmax((V[1]/P)[eligible])]
events=[event(imax,'관측률 95% 이상 중 RV1/RV5 최대'),event(imin,'관측률 95% 이상 중 RV1/RV5 최소'),event(iprv,'관측률 95% 이상 중 RV1/PRV 최대')]
full_extremes=[event(np.argmax(ratio),'전체 자료 RV1/RV5 최대'),event(np.argmin(ratio),'전체 자료 RV1/RV5 최소')]
daily=[]
for i,d in enumerate(dates):
    row=dict(date_utc=str(d),observed_minutes=int(Z['observed_minutes'][i]),filled_minutes=int(Z['filled_minutes'][i]),
       coverage_ratio=float(Z['coverage_ratio'][i]),max_stale_minutes=int(Z['max_stale_minutes'][i]))
    row.update({f'rv_{m}m':float(V[m][i]) for m in freqs})
    row['prv_scalar']=float(P[i])
    row.update({f'vol_{m}m_pct':float(100*np.sqrt(V[m][i])) for m in freqs})
    row['prv_vol_pct']=float(100*np.sqrt(P[i]))
    row['ratio_rv1_to5']=float(ratio[i]);row['ratio_rv1_to_prv']=float(V[1][i]/P[i])
    for m in phase:
        row.update({f'rv_{m}m_phase_{key}':float(val[i]) for key,val in phase[m].items()})
    daily.append(row)

decomp=[]
for m in freqs:
    # Algebraic identity: RV(m)-RV(1) equals within-block cross products.
    b=R.reshape(n,1440//m,m)
    s=float((np.square(b.sum(axis=2))-np.square(b).sum(axis=2)).sum())
    assert np.isclose(s,float((V[m]-V[1]).sum()),rtol=1e-10,atol=1e-12)
    decomp.append(dict(minutes=m,total_cross_contribution=s))
result=dict(period=[str(dates[0]),str(dates[-1])],days=n,frequencies=freqs,
    minute_count=int(R.size),filled_minutes=int(Z['filled_minutes'].sum()),
    quality_days=int(qmask.sum()),excluded_low_coverage_days=int((~qmask).sum()),
    summary=summary,quality_summary=quality,annual=annual,phase_summary=phase_summary,
    events=events,full_extremes=full_extremes,cross_products=decomp,
    formula='RV_d(m)=sum_j (sum of m consecutive 1-minute log returns in block j)^2',
    reference='5-minute RV is a comparison reference, not a ground-truth measure')
for name,obj in [('analysis',result),('daily',daily),('summary',summary)]:
    Path(f'work/btc_sampling_{name}.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
np.savez_compressed('work/btc_sampling_arrays.npz',dates=dates,rv=np.stack([V[m] for m in freqs],axis=1),prv=P,
   quality=qmask,rv5_phase_min=phase[5]['minimum'],rv5_phase_max=phase[5]['maximum'])
print(json.dumps(result,ensure_ascii=False),flush=True)
