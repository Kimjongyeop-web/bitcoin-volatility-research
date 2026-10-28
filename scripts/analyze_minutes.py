"""Validate minute source, export source by year, and calculate daily variance diagnostics."""
import csv, gzip, io, json, zipfile, math
from pathlib import Path
from datetime import datetime, timezone, timedelta
import numpy as np

ROOT=Path('work/minute_cache'); OUT=Path('outputs'); WORK=Path('work')
progress=json.loads((ROOT/'progress.json').read_text())
assert progress.get('complete') and progress['months_done']==progress['total_months'], 'Download is incomplete'
meta=json.loads((ROOT/'download_meta.json').read_text())
files=sorted(ROOT.glob('????-??.json.gz'));assert len(files)==progress['total_months']
arrays=[]; last=None; source_rows=0
fields=['candle_start_utc','candle_start_kst','market','open_krw','high_krw','low_krw','close_krw','volume_btc','turnover_krw','last_trade_timestamp_ms']
zip_path=OUT/'upbit_btc_1min_all_yearly_csv.zip'
archive=zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5)
current_year=None; stream=None; writer=None
for path in files:
 with gzip.open(path,'rt',encoding='utf-8') as f: rows=json.load(f)
 if not rows:continue
 year=path.name[:4]
 if year!=current_year:
  if stream:stream.close()
  stream=io.TextIOWrapper(archive.open(f'upbit_btc_1min_{year}.csv','w'),encoding='utf-8-sig',newline='')
  writer=csv.writer(stream);writer.writerow(fields);current_year=year
 t=np.array([r['candle_date_time_utc'] for r in rows],dtype='datetime64[m]').astype(np.int64)
 a=np.array([[r[k] for k in ['opening_price','high_price','low_price','trade_price','candle_acc_trade_volume','candle_acc_trade_price']] for r in rows],dtype=float)
 assert np.all(np.diff(t)>0) and (last is None or t[0]>last)
 assert np.all(np.isfinite(a)) and np.all(a[:,:4]>0) and np.all(a[:,4:]>=0)
 assert np.all(a[:,1]>=np.maximum(a[:,0],a[:,3])) and np.all(a[:,2]<=np.minimum(a[:,0],a[:,3]))
 assert np.all(a[:,1]>=a[:,2]);last=t[-1]
 for r in rows:
  assert r['market']=='KRW-BTC' and r['unit']==1
  writer.writerow([r['candle_date_time_utc']+'Z',r['candle_date_time_kst']+'+09:00',r['market'],r['opening_price'],r['high_price'],r['low_price'],r['trade_price'],r['candle_acc_trade_volume'],r['candle_acc_trade_price'],r['timestamp']])
 arrays.append((t,a));source_rows+=len(rows)
 print('Validated',path.name,len(rows),flush=True)
if stream:stream.close()
archive.writestr('README.txt',('Upbit KRW-BTC one-minute source candles. CSVs split by year for Excel row limits.\n'
 'Rows are only API-returned candles, without artificial OHLC rows.\n'
 'UTC and KST columns are candle START times; the close is the last trade in that minute.\n'
 'Volume is BTC, turnover and all prices are KRW. Only closed minutes are included.\n'
 'An absent candle is not proof of a data loss: Upbit omits intervals with no trades.\n'
 'Sampling cutoff (exclusive UTC): '+meta['cutoff_exclusive_utc']+'Z\n'
 'Source: https://api.upbit.com/v1/candles/minutes/1\n').encode('utf-8'))
archive.close()
t=np.concatenate([x[0] for x in arrays]);a=np.concatenate([x[1] for x in arrays]);del arrays
assert source_rows==progress['candles']
start=int(t[0]//1440*1440); cutoff=int(np.datetime64(meta['cutoff_exclusive_utc'],'m').astype(np.int64))
grid=np.arange(start,cutoff,dtype=np.int64); positions=t-start
observed=np.zeros(len(grid),dtype=bool);observed[positions]=True
lookup=np.full(len(grid),-1,dtype=np.int32);lookup[positions]=np.arange(len(t),dtype=np.int32)
np.maximum.accumulate(lookup,out=lookup)
close=np.full(len(grid),np.nan);valid=lookup>=0;close[valid]=a[lookup[valid],3]
minute_r=np.full(len(grid),np.nan);minute_r[1:]=np.diff(np.log(close))
stale=grid-np.maximum.accumulate(np.where(observed,grid,start-1))

def acfs(x,maxlag):
 x=np.asarray(x);assert np.all(np.isfinite(x));z=x-x.mean();den=np.dot(z,z)
 return np.array([np.dot(z[k:],z[:-k])/den for k in range(1,maxlag+1)]) if den else np.full(maxlag,np.nan)
def stat(x):
 x=np.asarray(x);x=x[np.isfinite(x)];z=x-x.mean();m2=np.mean(z*z)
 return {'n':len(x),'mean':float(x.mean()),'std':float(x.std(ddof=1)),
  'median':float(np.median(x)),'skew':float(np.mean(z**3)/m2**1.5),
  'kurtosis':float(np.mean(z**4)/m2**2),'min':float(x.min()),'max':float(x.max())}

def prv(x, band=None):
 """Scalar diagonal of paper Appendix A.6. x contains m in-day log increments."""
 m=len(x);K=int(np.sqrt(m)) if band is None else band;psi=1/12
 g=np.minimum(np.arange(K+1)/K,1-np.arange(K+1)/K)
 # A.6 k=1..m-K+1: preaverage uses increments k+1..k+K-1,
 # while correction uses increments k..k+K-1 (1-based increments).
 bar=np.correlate(x,np.r_[0.,g[1:K]],mode='valid')
 correction=np.correlate(x*x,np.diff(g)**2,mode='valid')
 L=m-2*K+1
 T=(m/L)*(np.pi/(2*K))*np.sum(np.abs(bar[:L])*np.abs(bar[K:K+L]))
 u=4*np.sqrt(max(T,0))*(K/m)**.47
 retained=np.abs(bar)<=u
 raw=np.sum((bar*bar-.5*correction)*retained)/(psi*K)
 return float(max(raw,0)),float(raw),float(u),float(1-retained.mean())

# Independent direct-loop check protects the endpoint and correction indexing.
test=np.random.default_rng(123).normal(0,.001,90);K=9;g=np.minimum(np.arange(K+1)/K,1-np.arange(K+1)/K)
b=[];c=[]
for k in range(1,len(test)-K+2):
 b.append(sum(g[l]*test[k+l-1] for l in range(1,K)))
 c.append(sum((g[l]-g[l-1])**2*test[k+l-2]**2 for l in range(1,K+1)))
b=np.array(b);c=np.array(c);L=len(test)-2*K+1
T=len(test)/L*np.pi/(2*K)*sum(abs(b[k])*abs(b[k+K]) for k in range(L))
u=4*np.sqrt(T)*(K/len(test))**.47
direct=sum((b[k]**2-.5*c[k]) for k in range(len(b)) if abs(b[k])<=u)/(K/12)
assert np.isclose(prv(test,9)[1],direct,rtol=1e-12,atol=1e-15)

daily=[];days=[];full_n=(cutoff-start)//1440
raw_day=t//1440;daily_volume=np.bincount((t-start)//1440,weights=a[:,4],minlength=full_n+1)
daily_turnover=np.bincount((t-start)//1440,weights=a[:,5],minlength=full_n+1)
for d in range(full_n):
 s=d*1440;e=s+1440;x=minute_r[s:e]
 if not np.all(np.isfinite(x)):continue # first partial day has no prior price
 date=str(np.datetime64(start//1440+d,'D'));rv=float(np.dot(x,x));pr,pr_raw,u,excluded=prv(x)
 logday=float(x.sum());assert np.isclose(logday,np.log(close[e-1]/close[s-1]),atol=1e-12)
 # 5-minute sensitivity on the identical day, based on five successive 1m returns.
 rv5=float(np.sum(x.reshape(288,5).sum(axis=1)**2))
 count=int(observed[s:e].sum())
 daily.append({'date':date,'observed_minutes':count,'filled_minutes':1440-count,'coverage_ratio':count/1440,
  'max_stale_minutes':int(stale[s:e].max()),'close_krw':float(close[e-1]),'daily_log_return':logday,
  'daily_return_squared':logday**2,'rv_1m':rv,'realized_volatility_1m_pct':100*np.sqrt(rv),
  'rv_5m':rv5,'prv_scalar':pr,'prv_raw_before_psd':pr_raw,'prv_volatility_pct':100*np.sqrt(pr),
  'prv_threshold':u,'prv_excluded_window_fraction':excluded,'volume_btc':float(daily_volume[d]),'turnover_krw':float(daily_turnover[d])})
 days.append(start//1440+d)
assert np.all(np.diff(days)==1)
del a, rows
D={k:np.array([r[k] for r in daily]) for k in daily[0] if k!='date'}
dt=np.array([r['date'] for r in daily],dtype='datetime64[D]')
analysis_start=(days[0]*1440-start);analysis_end=(days[-1]+1)*1440-start
R=minute_r[analysis_start:analysis_end];assert np.all(np.isfinite(R))
obs_ret=observed[analysis_start:analysis_end]&observed[analysis_start-1:analysis_end-1]
year_labels=np.array([r['date'][:4] for r in daily]);annual={}
for year in sorted(set(year_labels)):
 mask=year_labels==year;mi=np.repeat(mask,1440);annual[year]={
  'days':int(mask.sum()),'mean_coverage':float(D['coverage_ratio'][mask].mean()),
  'minute_return':stat(R[mi]),'rv_1m':stat(D['rv_1m'][mask]),'prv':stat(D['prv_scalar'][mask]),
  'realized_vol_median_pct':float(np.median(D['realized_volatility_1m_pct'][mask]))}

# Reconcile minute aggregation to the previously downloaded independent daily endpoint.
with open('outputs/upbit_btc_krw_daily_all.csv',encoding='utf-8-sig',newline='') as f: old={r['date']:r for r in csv.DictReader(f)}
close_mismatches=[];volume_mismatches=[];turnover_mismatches=[]
for row in daily:
 o=old.get(row['date'])
 if not o:continue
 if row['close_krw']!=float(o['close_krw']):close_mismatches.append(row['date'])
 if not np.isclose(row['volume_btc'],float(o['volume_btc']),rtol=1e-8,atol=1e-6):volume_mismatches.append(row['date'])
 if not np.isclose(row['turnover_krw'],float(o['turnover_krw']),rtol=1e-8,atol=.1):turnover_mismatches.append(row['date'])

series={'minute_return':R,'minute_squared_return':R*R,'minute_absolute_return':abs(R),
        'daily_return':D['daily_log_return'],'daily_squared_return':D['daily_return_squared'],
        'rv_1m':D['rv_1m'],'prv':D['prv_scalar'],'rv_5m':D['rv_5m']}
S={k:stat(v) for k,v in series.items()};ACF={k:acfs(v,60).tolist() for k,v in series.items()}
largest=np.argsort(abs(R))[-10:][::-1]
top=[{'candle_start_utc':str(np.datetime64(int(start+analysis_start+i),'m'))+'Z','log_return_pct':float(R[i]*100),'simple_return_pct':float(np.expm1(R[i])*100),'both_minutes_observed':bool(obs_ret[i])} for i in largest]
summary={'source_meta':meta,'source_rows':source_rows,'source_first_utc':str(np.datetime64(int(t[0]),'m'))+'Z',
 'source_last_utc':str(np.datetime64(int(t[-1]),'m'))+'Z','days':len(daily),'analysis_period':[str(dt[0]),str(dt[-1])],
 'grid_minutes':len(R),'filled_minutes':int(D['filled_minutes'].sum()),'coverage':float(D['coverage_ratio'].mean()),
 'days_below_95pct_coverage':int(np.sum(D['coverage_ratio']<.95)),
 'longest_stale_minutes':int(D['max_stale_minutes'].max()),'negative_prv_raw_days':int(np.sum(D['prv_raw_before_psd']<0)),
 'stats':S,'acf':ACF,'annual':annual,'top_minute_returns':top,
 'observed_adjacent_minute_return_stats':stat(R[obs_ret]),
 'prv_stats_coverage_ge95pct':stat(D['prv_scalar'][D['coverage_ratio']>=.95]),
 'reconciliation':{'close_mismatches':close_mismatches,'volume_mismatches':volume_mismatches,'turnover_mismatches':turnover_mismatches},
 'method':{'day':'UTC 00:00 to 24:00 = KST 09:00 to next 09:00','missing':'Previous available close carried forward; counts retained; no invented raw candles.',
 'rv':'Sum of 1440 one-minute log return squares. Includes jumps and microstructure effects.',
 'prv':'Scalar Appendix A.6 adaptation, m=1440, K=37, g(x)=min(x,1-x), psi=1/12, alpha=.47, c0=4; T estimated with lag-K preaveraged products; project to nonnegative.',
 'scope':'Descriptive diagnostics, not factor decomposition, FIVAR estimation, structural-break test or forecast.'}}
(WORK/'minute_analysis.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
(WORK/'minute_daily_rows.json').write_text(json.dumps(daily,separators=(',',':')),encoding='utf-8')
np.savez_compressed(WORK/'minute_plot_arrays.npz',dates=dt,returns=R,observed_return=obs_ret,**D)
print(json.dumps({k:summary[k] for k in ['source_rows','source_first_utc','source_last_utc','days','analysis_period','filled_minutes','coverage','days_below_95pct_coverage','longest_stale_minutes','negative_prv_raw_days','reconciliation']},indent=2))
