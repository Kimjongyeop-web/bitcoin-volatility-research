"""Compare non-overlapping BTC return intervals on the identical minute grid."""
import json
from pathlib import Path
import numpy as np

Z = np.load('work/minute_plot_arrays.npz')
R = Z['returns']
assert len(R) == len(Z['dates']) * 1440 and np.isfinite(R).all()
old = json.loads(Path('work/minute_analysis.json').read_text(encoding='utf-8'))
out = {'start_date_utc': str(Z['dates'][0]), 'end_date_utc': str(Z['dates'][-1]),
       'days':len(Z['dates']), 'filled_minutes':int(Z['filled_minutes'].sum()),
       'intervals': []}
for m in [1, 5, 15, 60]:
    x = R.reshape(-1,m).sum(axis=1)
    # Log-return aggregation must telescope to the identical daily returns.
    assert np.allclose(x.reshape(-1,1440//m).sum(axis=1), Z['daily_log_return'], rtol=1e-10, atol=1e-12)
    z = x - x.mean()
    den = np.dot(z,z)
    maxlag = max(40, 360//m)
    # Padding by maxlag suffices for only the leading autocovariances.
    nfft = 1 << (len(z)+maxlag-1).bit_length()
    f = np.fft.rfft(z,n=nfft)
    cov = np.fft.irfft(f*f.conj(),n=nfft)[:maxlag+1]
    acf = cov[1:] / den
    for k in sorted(set([1,2,3,40,maxlag])):
        assert np.isclose(acf[k-1], np.dot(z[k:],z[:-k])/den, rtol=1e-10,atol=1e-13)
    if m == 1:
        assert np.allclose(acf[:40],old['acf']['minute_return'][:40],rtol=1e-10,atol=1e-13)
    # Sensitivity: preserve original clock positions, accept pairs only when
    # every underlying 1m return has two actually observed price endpoints.
    clean = Z['observed_return'].reshape(-1,m).all(axis=1)
    pair = clean[1:] & clean[:-1]
    observed_pair_corr = float(np.corrcoef(x[1:][pair],x[:-1][pair])[0,1])
    item = dict(minutes=m,n=len(x),acf=acf.tolist(),sd_percent=float(x.std(ddof=1)*100),
                acf1=float(acf[0]), common_60min=float(acf[60//m-1]),
                complete_observed_pairs_lag1=int(pair.sum()),
                observed_pairs_lag1_pearson=observed_pair_corr)
    out['intervals'].append(item)
    print(json.dumps({k:v for k,v in item.items() if k!='acf'}),flush=True)
Path('work/btc_interval_acf.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
