import os,sys,json,base64,html
from pathlib import Path
os.environ['MPLCONFIGDIR']=str(Path('work/mpl_config').resolve())
sys.path.insert(0,str(Path('work/plot_packages').resolve()))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager,colors

S=json.loads(Path('work/btc_sampling_analysis.json').read_text(encoding='utf-8'))
checks=json.loads(Path('work/btc_sampling_source_checks.json').read_text(encoding='utf-8'))
A=np.load('work/btc_sampling_arrays.npz'); Z=np.load('work/minute_plot_arrays.npz')
D=A['dates']; V=A['rv']; P=A['prv']; R=Z['returns'].reshape(-1,1440)
F=S['frequencies']; X=np.arange(len(F)); summary=S['summary']; quality=S['quality_summary']
for weight in ['Regular','Bold']:font_manager.fontManager.addfont(f'work/NotoSansKR-{weight}.ttf')
plt.rcParams.update({'font.family':font_manager.FontProperties(fname='work/NotoSansKR-Regular.ttf').get_name(),
 'font.size':10.5,'axes.titlesize':13,'axes.titleweight':'bold','axes.unicode_minus':False,
 'text.color':'#18253A','axes.labelcolor':'#364152','axes.edgecolor':'#CBD5E1',
 'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#F8FAFC',
 'axes.facecolor':'white','savefig.facecolor':'#F8FAFC'})
blue='#2355A4';teal='#14857E';orange='#C9742F';purple='#8851A3';gray='#64748B'
period=f'{S["period"][0]}–{S["period"][1]} (UTC) · 업비트 BTC/KRW · {S["days"]:,}일'
def grid(ax):
    ax.grid(axis='y',color='#E3E8EF',lw=.7);ax.set_axisbelow(True)
def heading(fig,title,subtitle=period):
    fig.text(.085,.965,title,fontsize=21,weight='bold',va='top')
    fig.text(.085,.965-.60/fig.get_figheight(),subtitle,fontsize=10.5,color=gray)
def save(fig,name):
    fig.savefig('outputs/'+name+'.png',dpi=155);plt.close(fig)

# 1: signature plots; retain RV units instead of confusing them with volatility.
fig,axs=plt.subplots(1,2,figsize=(13.8,7))
fig.subplots_adjust(left=.085,right=.965,top=.79,bottom=.245,wspace=.25)
heading(fig,'측정 간격에 따라 일별 RV의 수준이 달라진다')
for ax,key,title in zip(axs,['mean_rv','median_rv'],['전체 날짜의 평균 RV','전체 날짜의 중앙값 RV']):
    y=np.array([s[key] for s in summary[:-1]])*1e4
    ax.plot(X,y,color=blue,marker='o',lw=2,ms=5,label='RV')
    ax.axhline(summary[-1][key]*1e4,color=purple,ls='--',lw=1.5,label='PRV (비교용)')
    ax.set_xticks(X,[str(m) for m in F]);ax.set_xlabel('수익률 측정 간격 (분)')
    ax.set_ylabel('일별 분산 추정값 × 10,000');ax.set_ylim(0,y.max()*1.16)
    ax.set_title(title,loc='left');ax.legend(frameon=False);grid(ax)
    for j in [0,3,7]:ax.annotate(f'{y[j]:.2f}',(j,y[j]),xytext=(0,9),textcoords='offset points',ha='center',fontsize=10,color=blue)
fig.text(.085,.142,'평균 RV: 1분 / 5분 = 1.200배  ·  15분 / 5분 = 0.902배  ·  60분 / 5분 = 0.799배',fontsize=11,weight='bold')
fig.text(.085,.059,'RV는 분산 척도입니다. 20%의 RV 차이는 변동성(제곱근)의 20% 차이를 뜻하지 않습니다.\n가로축은 간격별 항목을 동일한 간격으로 배치했습니다. PRV 점선은 정답선이 아닙니다.\n잡음·점프에 강건한 PRV와 일반 RV는 추정 목표가 달라, 서로 같아야 하는 것은 아닙니다.',fontsize=10,color=gray,linespacing=1.6)
save(fig,'btc_sampling_signature')

# 2: daily level ratios versus overlap of high-variation dates.
fig,axs=plt.subplots(1,2,figsize=(13.8,7.7))
fig.subplots_adjust(left=.085,right=.965,top=.80,bottom=.24,wspace=.27)
heading(fig,'수치의 크기와 고변동 날짜의 일치도를 따로 확인')
vals=[V[:,j]/V[:,3] for j in range(len(F))]+[P/V[:,3]]
labels=[f'{m}분' for m in F]+['PRV']; stats=[]
for label,v in zip(labels,vals):
    p=np.quantile(v,[.05,.25,.5,.75,.95]);stats.append(dict(label=label,whislo=p[0],q1=p[1],med=p[2],q3=p[3],whishi=p[4],fliers=[]))
bp=axs[0].bxp(stats,showfliers=False,patch_artist=True,widths=.6)
for patch in bp['boxes']:patch.set_facecolor('#DFEAF7');patch.set_edgecolor(blue)
for line in bp['medians']:line.set_color(blue)
axs[0].axhline(1,color=gray,ls='--',lw=1)
axs[0].set_ylim(0,2.55);axs[0].set_ylabel('해당 날짜의 추정값 / 같은 날의 5분 RV')
axs[0].set_title('날짜마다 측정값의 비율이 달라진다',loc='left')
axs[0].tick_params(axis='x',labelrotation=35);grid(axs[0])
overlap=np.array([s['top10_overlap_fraction'] for s in summary])*100
bars=axs[1].bar(np.arange(9),overlap,color=[blue]*8+[purple],width=.68)
axs[1].set_xticks(np.arange(9),labels,rotation=35);axs[1].set_ylim(0,112)
axs[1].set_ylabel('5분 RV 상위 10% 날짜와 겹치는 비율 (%)')
axs[1].set_title('고변동 날짜는 대체로 비슷하게 포착',loc='left');grid(axs[1])
for b,y in zip(bars,overlap):axs[1].text(b.get_x()+b.get_width()/2,y+2,f'{y:.1f}',ha='center',fontsize=9)
fig.text(.085,.145,'1분 대 5분: 날짜별 RV 비율의 중앙값 1.170배 · 순위상관 0.965 · 상위 날짜 331개 중 303개 공통',fontsize=10.5,weight='bold')
fig.text(.085,.062,'왼쪽 상자: 25~75백분위, 가운데 선: 중앙값, 수염: 5~95백분위 (범위 밖 값은 그림에서만 생략).\n오른쪽: 각 방법에서 값이 큰 331일(전체의 약 10%)을 골라 5분 RV의 331일과 비교했습니다.\n5분 RV는 비교 기준이며 정답이 아닙니다. PRV의 순위 차이에는 추정 목표 차이도 반영됩니다.',fontsize=10,color=gray,linespacing=1.6)
save(fig,'btc_sampling_agreement')

# 3: stability across years and source coverage.
fig,axs=plt.subplots(1,2,figsize=(13.8,8.3),gridspec_kw={'width_ratios':[1.18,1]})
fig.subplots_adjust(left=.085,right=.95,top=.81,bottom=.22,wspace=.33)
heading(fig,'전체 평균의 결과가 모든 연도에 그대로 적용되지는 않는다')
matrix=np.array([a['signature_mean'] for a in S['annual']])
im=axs[0].imshow(matrix,cmap='RdBu_r',norm=colors.TwoSlopeNorm(vmin=.55,vcenter=1,vmax=1.65),aspect='auto')
axs[0].set_xticks(X,[f'{m}' for m in F]);axs[0].set_yticks(np.arange(10),[a['year']+('*' if a['year'] in ['2017','2026'] else '') for a in S['annual']])
axs[0].set_xlabel('수익률 측정 간격 (분)');axs[0].set_title('해당 연도의 평균 5분 RV = 1',loc='left')
for i in range(10):
    for j in range(8):axs[0].text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=9,color='white' if matrix[i,j]>1.44 or matrix[i,j]<.68 else '#18253A')
fig.colorbar(im,ax=axs[0],fraction=.045,pad=.035).set_label('평균 RV의 비율')
axs[1].plot(X,[s['mean_rv_ratio_to_5m'] for s in summary[:-1]],color=blue,marker='o',label='전체 3,303일')
axs[1].plot(X,[s['mean_rv_ratio_to_5m'] for s in quality[:-1]],color=teal,marker='s',label='관측률 ≥ 95%: 3,172일')
axs[1].axhline(1,color=gray,lw=.8,ls='--');axs[1].set_ylim(.72,1.28)
axs[1].set_xticks(X,[str(m) for m in F]);axs[1].set_xlabel('수익률 측정 간격 (분)')
axs[1].set_ylabel('각 표본의 평균 5분 RV 대비 비율')
axs[1].set_title('관측률이 낮은 날을 제외한 민감도',loc='left');axs[1].legend(frameon=False,fontsize=9,loc='upper right');grid(axs[1])
fig.text(.085,.131,'1분/5분 평균 RV 비율: 2017년 1.641배 → 2025년 1.044배 · 2026년 1.047배',fontsize=10.5,weight='bold')
fig.text(.085,.057,'* 2017년은 9/26부터, 2026년은 10/11까지의 부분 연도입니다.\n관측률 기준으로 131일을 제외하면 1분/5분 평균 RV 비율은 1.200배에서 1.104배로 줄어듭니다.\n제외된 날의 시장 상태와 연도 구성도 달라지므로, 이 차이를 결측 처리의 순수한 효과로 해석할 수는 없습니다.',fontsize=10,color=gray,linespacing=1.6)
save(fig,'btc_sampling_stability')

# 4: actual daily volatility levels (square roots).
fig,axs=plt.subplots(2,1,figsize=(13.8,9.3))
fig.subplots_adjust(left=.09,right=.965,top=.83,bottom=.17,hspace=.45)
heading(fig,'변동성이 크게 움직이는 날짜는 대부분 함께 나타난다')
series=[(100*np.sqrt(V[:,0]),blue,'1분 RV'),(100*np.sqrt(V[:,3]),teal,'5분 RV'),(100*np.sqrt(V[:,5]),orange,'15분 RV'),(100*np.sqrt(P),purple,'PRV')]
for y,c,l in series:
    axs[0].plot(D,y,color=c,lw=.85,alpha=.8,label=l)
    axs[1].plot(D[-90:],y[-90:],color=c,lw=1.3,label=l)
axs[0].set_yscale('log');axs[0].set_title('전체 기간 · 세로축 로그 눈금',loc='left')
axs[0].set_yticks([.5,1,2,5,10,20,50]);axs[0].set_yticklabels(['0.5','1','2','5','10','20','50'])
axs[0].xaxis.set_major_locator(mdates.YearLocator());axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axs[0].set_xlim(D[0],D[-1])
axs[0].legend(frameon=False,ncol=4,loc='upper right')
axs[1].set_title(f'최근 90일: {D[-90]}–{D[-1]} · 세로축 일반 눈금',loc='left')
axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'));axs[1].set_ylim(bottom=0)
axs[1].set_xlim(D[-90],D[-1])
for ax in axs:ax.set_ylabel('일별 변동성 (%)');grid(ax)
fig.text(.09,.071,'표시값 = 100 × √(일별 분산 추정값). 앞의 RV 곡선과 단위가 다릅니다.\nPRV는 일반 RV보다 대체로 낮지만, 이것만으로 더 정확하다고 판정할 수 없습니다. 날짜 구분은 UTC입니다.',fontsize=10.5,color=gray,linespacing=1.7)
save(fig,'btc_sampling_daily_volatility')

# 5: inspect data-selected dates, without inventing an event narrative.
fig,axs=plt.subplots(3,2,figsize=(14.3,13.5),gridspec_kw={'width_ratios':[1.2,1]})
fig.subplots_adjust(left=.085,right=.97,top=.85,bottom=.18,hspace=.60,wspace=.25)
heading(fig,'측정 방법의 차이가 큰 날짜: 원본 가격 경로 확인',
 '관측률 95% 이상인 날에서 RV1/RV5 최대·최소, RV1/PRV 최대인 날을 선택했습니다.')
reasons=['1분 RV가 5분 RV보다 특히 큰 날','5분 RV가 1분 RV보다 큰 날','RV와 PRV의 차이가 특히 큰 날']
for row,(event,reason) in enumerate(zip(S['events'],reasons)):
    i=event['index'];j=event['max_abs_minute_index'];path=100*np.exp(np.r_[0,np.cumsum(R[i])]);k=np.arange(1441)
    x=(k-1)/60
    ax=axs[row,0];ax.plot(x,path,color=blue,lw=.8);ax.axhline(100,color=gray,ls='--',lw=.6)
    ax.set_xlim(0,24);ax.set_xticks([0,3,6,9,12,15,18,21,24],['09','12','15','18','21','00','03','06','09'])
    ax.set_xlabel('한국시간 (가운데 00시부터 다음 날)');ax.set_ylabel('직전 일 마지막 가격 = 100')
    ax.set_title(f'{event["date"]} (UTC) · {reason}',loc='left',fontsize=11.5,pad=12)
    ax.text(.02,.035,f'RV1/RV5 {event["rv1_to5"]:.2f}배 · RV1/PRV {event["rv1_to_prv"]:.2f}배\n관측률 {event["coverage_pct"]:.2f}% · 보완 {event["filled_minutes"]}분',transform=ax.transAxes,fontsize=9,bbox=dict(facecolor='white',alpha=.88,edgecolor='none'),linespacing=1.5)
    ax=axs[row,1];lo=max(0,j+1-15);hi=min(1441,j+1+16)
    ax.plot(k[lo:hi]-1-j,path[lo:hi],color=blue,lw=1.25,marker='.',ms=4,label='1분 종가')
    for shift,c in [(0,teal),(2,orange)]:
        b=np.unique(np.r_[0,np.arange(shift,1440,5),1440]);b=b[(b>=lo)&(b<hi)]
        ax.scatter(b-1-j,path[b],color=c,s=34,marker='s' if shift==0 else '^',zorder=5,label=f'5분 경계: 기준+{shift}분')
    ax.axvline(0,color=gray,ls=':',lw=.8)
    ax.set_title(event['max_minute_start_kst'].replace('T',' ')+' KST 주변',loc='left',fontsize=11.5,pad=12)
    ax.set_xlabel('가장 큰 1분 변화 시각 기준 (분)');ax.set_ylabel('직전 일 마지막 가격 = 100')
    if row==0:ax.legend(frameon=False,fontsize=8,loc='best')
    for ax in axs[row]:grid(ax)
fig.text(.085,.045,'모든 가격 경로를 저장된 업비트 원본 캔들과 대조했습니다. 확대 그림의 점은 5분 수익률을 만들 때 선택되는 종가입니다.\n왕복 움직임은 같은 블록 안에서 상쇄될 수 있고, 같은 방향의 움직임은 합산 후 제곱할 때 커질 수 있습니다.\n이 그림만으로 실제 움직임과 미시구조 잡음을 구분하거나 점프의 크기를 확정할 수는 없습니다.',fontsize=10.5,color=gray,linespacing=1.7)
save(fig,'btc_sampling_event_paths')

# 6: phase sensitivity in RV units; concrete Dec 3 example in volatility units.
fig,axs=plt.subplots(1,2,figsize=(13.8,7.8))
fig.subplots_adjust(left=.085,right=.965,top=.80,bottom=.255,wspace=.29)
heading(fig,'같은 간격도 묶기 시작하는 분에 따라 결과가 달라진다')
ps=S['phase_summary'];x=np.arange(3)
y1=np.array([p['median_daily_range_over_phase_median'] for p in ps])*100
y2=np.array([p['p95_daily_range_over_phase_median'] for p in ps])*100
axs[0].bar(x-.18,y1,width=.34,color=blue,label='날짜별 민감도의 중앙값')
axs[0].bar(x+.18,y2,width=.34,color='#9EBADB',label='날짜별 민감도의 95백분위')
axs[0].set_xticks(x,['5분','15분','60분']);axs[0].set_ylim(0,215)
axs[0].set_ylabel('(시작분별 RV 최대−최소) / 중앙값 × 100')
axs[0].set_title('간격이 길수록 시작분에 더 민감했다',loc='left');axs[0].legend(frameon=False,fontsize=9);grid(axs[0])
for xs,ys in [(x-.18,y1),(x+.18,y2)]:
    for xx,yy in zip(xs,ys):axs[0].text(xx,yy+3,f'{yy:.1f}%',ha='center',fontsize=9)
ec=next(e for e in checks if e['date']=='2024-12-03');e=next(e for e in S['events'] if e['date']=='2024-12-03')
vv=[p['vol_pct'] for p in ec['phase5']]
axs[1].bar(np.arange(5),vv,color=teal,width=.64)
axs[1].set_xticks(np.arange(5),['기준','+1분','+2분','+3분','+4분'])
axs[1].set_ylim(0,46);axs[1].set_ylabel('5분 RV로 계산한 일별 변동성 (%)')
axs[1].set_title('2024-12-03: 같은 5분 간격의 차이',loc='left');grid(axs[1])
for j,y in enumerate(vv):axs[1].text(j,y+1,f'{y:.2f}%',ha='center',fontsize=10)
fig.text(.085,.153,'2024-12-03의 5분 기준 변동성: 16.45%~38.91% · 기준 경계에서는 19.89%',fontsize=11,weight='bold')
fig.text(.085,.059,'왼쪽은 RV(분산)의 민감도, 오른쪽은 √RV를 %로 표시한 값입니다. 서로 다른 단위입니다.\n하루의 시작·끝은 고정하고 내부 경계만 이동했습니다. 이동한 경우 양 끝에는 짧은 부분 구간이 포함됩니다.\n전체 기간의 평균 차이는 작아도 특정 날짜의 차이는 클 수 있습니다. 낮은 값이 더 정확하다는 뜻은 아닙니다.',fontsize=10,color=gray,linespacing=1.6)
save(fig,'btc_sampling_phase_sensitivity')

print('Saved six figures.')
