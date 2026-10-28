"""Keep 1-minute returns fixed; vary only the lag, through one day."""
import json, os, sys
from pathlib import Path
import numpy as np

Z=np.load('work/minute_plot_arrays.npz')
r=Z['returns']
assert np.isfinite(r).all() and len(r)==1440*len(Z['dates'])
z=r-r.mean(); den=np.dot(z,z); maxlag=1440
nfft=1<<(len(z)+maxlag-1).bit_length()
f=np.fft.rfft(z,nfft)
a=np.fft.irfft(f*f.conj(),nfft)[1:maxlag+1]/den
selected=[1,2,3,5,10,15,30,40,60,120,360,720,1440]
for k in selected:
    assert np.isclose(a[k-1],np.dot(z[k:],z[:-k])/den,rtol=1e-10,atol=1e-13)
old=json.loads(Path('work/btc_interval_acf.json').read_text(encoding='utf-8'))['intervals'][0]['acf']
assert np.allclose(a[:len(old)],old,rtol=1e-10,atol=1e-13)
summary={'period_utc':[str(Z['dates'][0]),str(Z['dates'][-1])], 'n':len(r),
         'filled_minutes':int(Z['filled_minutes'].sum()),
         'selected_lags':{str(k):float(a[k-1]) for k in selected},
         'acf':a.tolist()}
Path('work/btc_fixed_1min_acf.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k!='acf'}),flush=True)

os.environ['MPLCONFIGDIR']=str(Path('work/mpl_config').resolve())
sys.path.insert(0,str(Path('work/plot_packages').resolve()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FormatStrFormatter
for weight in ['Regular','Bold']:
    font_manager.fontManager.addfont(f'work/NotoSansKR-{weight}.ttf')
plt.rcParams.update({'font.family':font_manager.FontProperties(fname='work/NotoSansKR-Regular.ttf').get_name(),
 'font.size':11,'axes.titlesize':13,'axes.titleweight':'bold','axes.unicode_minus':False,
 'text.color':'#18253A','axes.labelcolor':'#364152','axes.edgecolor':'#CBD5E1',
 'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#F8FAFC',
 'axes.facecolor':'white','savefig.facecolor':'#F8FAFC'})
blue='#2355A4'
fig,axes=plt.subplots(2,2,figsize=(14,10.4),sharey=True)
fig.subplots_adjust(left=.09,right=.965,top=.81,bottom=.20,hspace=.48,wspace=.20)
fig.text(.09,.963,'1분 수익률을 고정하고, 비교 시차만 늘리기',fontsize=22,weight='bold',va='top')
fig.text(.09,.913,'모든 점은 두 1분 수익률의 자기상관입니다. 가로축 60 = 60분 떨어진 두 1분 수익률.',fontsize=11.5,color='#475569')
fig.text(.09,.875,f'{summary["period_utc"][0]}–{summary["period_utc"][1]} (UTC) · BTC/KRW · 수익률 {len(r):,}개 · 네 패널의 세로축 동일',fontsize=10.5,color='#64748B')
panels=[(40,'01  1~40분 차이',[1,5,10,20,30,40]),
        (60,'02  1~60분 차이 (1시간까지)',[1,10,20,30,40,50,60]),
        (360,'03  1~360분 차이 (6시간까지)',[1,60,120,180,240,300,360]),
        (1440,'04  1~1,440분 차이 (24시간까지)',[1,240,480,720,960,1200,1440])]
for ax,(limit,title,ticks) in zip(axes.flat,panels):
    k=np.arange(1,limit+1)
    if limit<=60:
        ax.vlines(k,0,a[:limit],color=blue,lw=1.25,alpha=.80)
        ax.scatter(k,a[:limit],color=blue,s=12,zorder=3)
    else:
        ax.plot(k,a[:limit],color=blue,lw=.85)
    ax.axhline(0,color='#475569',lw=.9)
    ax.set_title(title,loc='left',pad=12)
    ax.set_xlim(.4,limit+.6); ax.set_ylim(-.086,.018)
    ax.set_xticks(ticks); ax.set_yticks([-.08,-.06,-.04,-.02,0])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_xlabel('두 1분 수익률 사이의 시간 차이 (분)')
    ax.set_ylabel('1분 수익률 ACF')
    ax.grid(axis='y',color='#E3E8EF',lw=.7);ax.set_axisbelow(True)
    ax.text(.97,.06,f'{limit:,}분 차이: {a[limit-1]:+.5f}',transform=ax.transAxes,ha='right',fontsize=10,color='#475569')
axes[0,0].text(.13,.07,'1분: −0.0741\n2분: −0.0377\n3분: −0.0084',transform=axes[0,0].transAxes,color=blue,fontsize=10,linespacing=1.5)
fig.text(.09,.108,'수익률을 5분·1시간으로 합산하지 않았습니다. 시차마다 전체 기간의 가능한 모든 수익률 쌍을 사용했습니다.',fontsize=10.5,color='#475569')
fig.text(.09,.050,'네 패널은 같은 ACF의 가로축 범위를 넓혀 보여줍니다. 이전 범위의 점들은 반복됩니다.\n출처: 업비트 1분봉. 미관측 분은 직전 가격으로 보완했고 극단값은 유지했습니다.\n0에 가까운 점만으로 독립성이나 효과 소멸을 확정할 수 없습니다. 통계적 유의성 검정은 포함하지 않았습니다.',fontsize=10,color='#64748B',linespacing=1.6)
fig.savefig('outputs/btc_fixed_1min_acf_24h.png',dpi=160)
plt.close(fig)

notes=['BTC 1분 수익률 고정 ACF: 1~1,440분 시차','',
 f'기존 자료: 업비트 BTC/KRW, UTC {summary["period_utc"][0]}~{summary["period_utc"][1]}, {len(r):,}개 1분 로그수익률.',
 f'미관측 분 {summary["filled_minutes"]:,}개는 기존 분석처럼 직전 가격 유지. 극단값 제거 없음.',
 'r_t = ln(C_t/C_(t-1)); C_t는 매 1분봉 종가.',
 'ACF(k) = sum[t=k+1..N]((r_t-mean)*(r_(t-k)-mean)) / sum[t=1..N]((r_t-mean)^2).',
 '전체 기간 평균을 사용. 시차에 따라 양쪽 벡터의 평균과 분모를 따로 구하는 Pearson 상관계수와 미세하게 다름.',
 '각 k에 사용되는 쌍 개수는 N-k. 일정 간격으로 일부 관측치만 추출하지 않으며 수익률을 합산하지 않음.',
 'FFT 계산을 선택된 시차의 직접 내적 계산과 대조. 기존 1~360분 결과와 일치 확인.',
 '그림의 모든 패널은 동일한 ACF의 가로축 범위만 변경. 네 패널의 세로축은 동일.',
 '통계적 유의성 검정, 예측력 평가, 매매전략 검증은 수행하지 않음.',
 '', '시차(분)\t수익률_쌍수\tACF']
notes += [f'{k}\t{len(r)-k}\t{a[k-1]:.12f}' for k in range(1,1441)]
Path('outputs/btc_fixed_1min_acf_values.txt').write_text('\n'.join(notes),encoding='utf-8')
print('Saved plot and calculation notes.')
