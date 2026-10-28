import os,sys,json
from pathlib import Path
os.environ['MPLCONFIGDIR']=str(Path('work/mpl_config').resolve())
sys.path.insert(0,str(Path('work/plot_packages').resolve()))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
for weight in ['Regular','Bold']:font_manager.fontManager.addfont(f'work/NotoSansKR-{weight}.ttf')
plt.rcParams.update({'font.family':font_manager.FontProperties(fname='work/NotoSansKR-Regular.ttf').get_name(),
 'font.size':11,'axes.titlesize':13,'axes.titleweight':'bold','axes.unicode_minus':False,
 'text.color':'#17263C','axes.labelcolor':'#39465A','axes.edgecolor':'#CCD6E1',
 'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#F7F9FC',
 'axes.facecolor':'white','savefig.facecolor':'#F7F9FC'})
S=json.loads(Path('work/btc_cash_results.json').read_text(encoding='utf-8'))
A=np.load('work/btc_cash_plot_arrays.npz');D=A['dates'];winner=S['selection']['winner'];C=S['protocol']['initial_capital_krw']
B='#2355A4';O='#C87936';G='#67768A';T='#17847A';P='#895AA5'
labels={'HAR_logRV_all':'HAR · 전체 학습','HAR_logRV_3y':'HAR · 최근 3년',
 'GARCH_normal_all':'GARCH 정규 · 전체','GARCH_normal_3y':'GARCH 정규 · 3년',
 'GARCH_t_all':'GARCH t · 전체','GARCH_t_3y':'GARCH t · 3년','GJR_t_all':'GJR t · 전체','GJR_t_3y':'GJR t · 3년',
 'RV_last':'어제 RV','RV_mean7':'최근 7일 평균 RV','RV_mean30':'최근 30일 평균 RV',
 'BTC_buy_hold':'BTC 100% 보유','BTC50_cash50_daily':'BTC 50% 고정',
 'fixed_validation_mean_weight':'BTC 46.1% 고정','cash':'현금 100%'}
period='업비트 BTC/KRW · 2026-01-01–10-11 UTC · 284일 · 편도 비용 0.10% · 현금 이자 0%'
def head(fig,title,sub=period):
    fig.text(.09,.966,title,fontsize=21,weight='bold',va='top')
    fig.text(.09,.966-.53/fig.get_figheight(),sub,fontsize=10.5,color=G,va='top')
def grid(ax):ax.grid(axis='y',color='#E2E8F0',lw=.7);ax.set_axisbelow(True)
def dateaxis(ax):ax.xaxis.set_major_locator(mdates.MonthLocator());ax.xaxis.set_major_formatter(mdates.DateFormatter('%m월'))
def save(fig,name):fig.savefig('outputs/'+name+'.png',dpi=150);plt.close(fig)

fig,axs=plt.subplots(2,1,figsize=(13.6,10),sharex=True,gridspec_kw={'height_ratios':[1.25,1]})
fig.subplots_adjust(left=.09,right=.965,top=.83,bottom=.135,hspace=.22)
head(fig,'BTC + 현금: 예측 기반 비중 조절의 실제 성과')
colors={winner:B,'fixed_validation_mean_weight':T,'BTC50_cash50_daily':G,'BTC_buy_hold':O,'cash':'#9AA5B5'}
start=D[0]-np.timedelta64(1,'D');dates=np.r_[start,D]
for k in [winner,'fixed_validation_mean_weight','BTC50_cash50_daily','BTC_buy_hold','cash']:
    nav=np.r_[C,A[k+'_nav']];lw=2.5 if k==winner else 1.6
    axs[0].plot(dates,nav/10000,color=colors[k],lw=lw,label=labels[k],ls='--' if k=='cash' else '-')
    axs[1].plot(dates,100*(nav/np.maximum.accumulate(nav)-1),color=colors[k],lw=lw)
axs[0].set_ylabel('계좌 평가액 (만원)');axs[1].set_ylabel('직전 최고점 대비 낙폭 (%)')
axs[0].legend(loc='upper center',bbox_to_anchor=(.5,1.13),ncol=3,frameon=False,fontsize=10)
for ax in axs:grid(ax);dateaxis(ax)
fig.text(.09,.067,'HAR: 누적 −6.32%  ·  연환산 변동성 17.39%  ·  일별 종가 기준 최대낙폭 −23.32%',weight='bold',fontsize=11)
fig.text(.09,.031,'평가 구간을 보고 모형을 바꾸지 않았습니다. 2026년은 앞선 데이터 탐색에서 본 기간이므로 완전한 블라인드 검증은 아닙니다.',color=G,fontsize=10)
save(fig,'btc_cash_equity')

fig,axs=plt.subplots(2,1,figsize=(13.6,9.7),sharex=True,gridspec_kw={'height_ratios':[1.15,1]})
fig.subplots_adjust(left=.09,right=.965,top=.83,bottom=.14,hspace=.28)
head(fig,'내일의 위험 예측이 BTC 비중으로 바뀌는 과정')
axs[0].plot(D,100*np.sqrt(A['realized_rv']),color='#A2ACBA',lw=1.2,label='해당 날짜가 끝난 뒤 측정한 변동성 (5분 RV의 제곱근)')
axs[0].plot(D,100*np.sqrt(A[winner+'_forecast']),color=B,lw=2,label='전날까지의 정보로 만든 HAR 예측')
axs[0].set_ylabel('일별 변동성 (%)');axs[0].legend(frameon=False,fontsize=10);axs[0].set_ylim(bottom=0)
axs[1].fill_between(D,0,100*A[winner+'_weights'],color=B,alpha=.15)
axs[1].plot(D,100*A[winner+'_weights'],color=B,lw=1.6,label='HAR 기반 BTC 비중')
axs[1].axhline(100*S['selection']['fixed_validation_mean_weight'],color=T,ls='--',label='검증 구간에서 고정한 비교 비중 46.1%')
axs[1].set_ylim(0,105);axs[1].set_ylabel('재조정 직후 BTC 비중 (%)');axs[1].legend(frameon=False,fontsize=10,loc='upper right')
for ax in axs:grid(ax);dateaxis(ax)
fig.text(.09,.072,'비중 = min(100%, 목표 일별 변동성 1% ÷ 예측 일별 변동성)',fontsize=12,weight='bold')
fig.text(.09,.03,'한국시간 09:00에 전날 정보가 완성된 뒤, 09:01 가격으로 재조정합니다. 나머지는 현금이며 공매도·레버리지는 없습니다.',fontsize=10,color=G)
save(fig,'btc_cash_forecast_weights')

order=[x['model'] for x in S['selection']['leaderboard']];y=np.arange(len(order))
val={x['model']:x['qlike'] for x in S['selection']['leaderboard']};test={x['model']:x['qlike'] for x in S['test_forecast_scores']}
fig,axs=plt.subplots(1,2,figsize=(14,8.7),sharey=True)
fig.subplots_adjust(left=.205,right=.95,top=.80,bottom=.17,wspace=.22)
head(fig,'예측 모형 선택과 최종 평가를 시간 순서로 분리', 'QLIKE: 낮을수록 좋음 · 동일한 날짜의 5분 RV로 평가 · 파란색은 검증 구간에서 선정한 주 모형')
for ax,vals,title in [(axs[0],val,'모형 선택: 2024–2025년 (731일)'),(axs[1],test,'최종 평가: 2026년 (284일)')]:
    v=np.array([vals[k] for k in order]);ax.barh(y,v,color=[B if k==winner else '#B4C4D9' for k in order],height=.64)
    ax.set_yticks(y,[labels[k]+(' ★' if k==winner else '') for k in order]);ax.set_xlim(0,v.max()*1.18)
    ax.set_title(title,loc='left',pad=13);ax.set_xlabel('정규화 QLIKE');ax.grid(axis='x',color='#E4E9F0');ax.set_axisbelow(True)
    for j,t in enumerate(v):ax.text(t+.018*v.max(),j,f'{t:.3f}',va='center',fontsize=10)
axs[0].invert_yaxis()
fig.text(.09,.092,'주 모형 HAR의 2026년 QLIKE는 어제 RV 대비 32.4% 감소했지만, MSE는 3.3% 증가했습니다.',fontsize=11,weight='bold')
fig.text(.09,.035,'2026년에는 3년 학습 HAR의 QLIKE가 소폭 낮았으나, 이를 보고 주 모형을 교체하지 않았습니다.\n모형 선택은 예측 손실만 사용했으며 투자 수익률로 선택하지 않았습니다. 두 패널의 가로축 범위는 다릅니다.',fontsize=10,color=G,linespacing=1.5)
save(fig,'btc_cash_model_comparison')

fig,axs=plt.subplots(1,2,figsize=(13.6,7.6))
fig.subplots_adjust(left=.09,right=.96,top=.79,bottom=.245,wspace=.29)
head(fig,'비중을 자주 바꾸면 예측 개선이 비용으로 소진될 수 있다')
for k,c in [(winner,B),('fixed_validation_mean_weight',T),('BTC_buy_hold',O)]:
    vals=[q for q in S['cost_sensitivity'] if q['strategy']==k]
    axs[0].plot([q['cost']*10000 for q in vals],[q['total_return']*100 for q in vals],marker='o',color=c,lw=2,label=labels[k])
axs[0].set_xlabel('매매금액 대비 편도 총비용 (bp, 10bp = 0.10%)');axs[0].set_ylabel('평가 기간 누적 수익률 (%)')
axs[0].set_xticks([0,5,10,25]);axs[0].legend(frameon=False,fontsize=10);grid(axs[0])
keys=[winner,'GARCH_normal_all','RV_last','fixed_validation_mean_weight'];stats={q['strategy']:q for q in S['portfolio_metrics'] if q['period']=='test_2026'}
v=[stats[k]['fees_paid_krw']/10000 for k in keys]
axs[1].bar(np.arange(len(keys)),v,color=[B,G,P,T],width=.6)
axs[1].set_xticks(np.arange(len(keys)),['HAR','GARCH\n정규','어제 RV','46.1%\n고정']);axs[1].set_ylabel('누적 지급 거래비용 (만원)');axs[1].set_ylim(0,max(v)*1.2);grid(axs[1])
for j,x in enumerate(v):axs[1].text(j,x+.8,f'{x:.1f}',ha='center',fontsize=11,weight='bold')
fig.text(.09,.139,'HAR: 비용 0%일 때 −4.08% → 편도 비용 0.10%일 때 −6.32%',fontsize=12,weight='bold')
fig.text(.09,.051,'비용은 수수료와 슬리피지를 합한 가정이며 과거 업비트 수수료를 복원한 값이 아닙니다.\n매일 전액에 비용을 부과하지 않고, 가격 변화로 비중이 달라진 실제 보유액과 목표 보유액의 차이에만 부과했습니다.\n오른쪽 지급 비용과 왼쪽 수익률 차이는 복리 및 후속 투자금액 변화 때문에 정확히 같지 않습니다.',fontsize=10,color=G,linespacing=1.5)
save(fig,'btc_cash_cost_sensitivity')
print('saved 4 figures')
