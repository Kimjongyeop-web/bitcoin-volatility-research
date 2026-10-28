import os,sys,json,base64,html,zipfile
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
    'text.color':'#17263C','axes.labelcolor':'#39465A','axes.edgecolor':'#CBD5E1',
    'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#F7F9FC',
    'axes.facecolor':'white','savefig.facecolor':'#F7F9FC'})
W=Path('work');O=Path('outputs')
s=json.loads((W/'har_refit_results.json').read_text(encoding='utf-8'))
verified=json.loads((W/'har_refit_verification.json').read_text(encoding='utf-8'));assert verified['status']=='passed'
a=np.load(W/'har_refit_arrays.npz');d=a['test_dates'];t=s['phases']['test'];v=s['phases']['validation']
methods=s['protocol']['primary_methods'];allmethods=methods+[s['protocol']['auxiliary_control']]
labels={'fixed_all':'고정 HAR','expanding_all':'확장 HAR','rolling_3y':'롤링 HAR · 3년','fixed_3y_control':'3년 고정 HAR · 보조',
    'BTC50_cash50_daily':'BTC 50%·현금 50%','BTC_buy_hold':'BTC 100% 보유','cash':'현금 100%'}
colors={'fixed_all':'#2254AA','expanding_all':'#DA8432','rolling_3y':'#168675','fixed_3y_control':'#8791A1','BTC50_cash50_daily':'#81559B'}
styles={'fixed_all':'-','expanding_all':'--','rolling_3y':'-','fixed_3y_control':':','BTC50_cash50_daily':'--'}
period='2026-01-01–10-11 UTC, 284일 · 목표 일별 변동성 1% · 편도 비용 0.10%'
def header(fig,title,subtitle=period):
    fig.text(.085,.967,title,fontsize=21,weight='bold',va='top')
    fig.text(.085,.967-.52/fig.get_figheight(),subtitle,fontsize=10.5,color='#637286',va='top')
def grid(ax):ax.grid(axis='y',color='#E2E8F0',lw=.7);ax.set_axisbelow(True)
def dates(ax):ax.xaxis.set_major_locator(mdates.MonthLocator());ax.xaxis.set_major_formatter(mdates.DateFormatter('%m월'))
def save(fig,name):fig.savefig(O/(name+'.png'),dpi=150);plt.close(fig)

fig,axs=plt.subplots(2,2,figsize=(14.4,10.4),gridspec_kw={'height_ratios':[1.3,1]})
fig.subplots_adjust(left=.085,right=.97,top=.80,bottom=.13,hspace=.47,wspace=.30)
header(fig,'HAR 재학습 비교: 차이는 작고, 롤링의 우위는 확정되지 않았다')
dd=np.r_[d[0]-np.timedelta64(1,'D'),d]
for k in allmethods+['BTC50_cash50_daily']:
    nav=np.r_[1e7,a['test_'+k+'_nav']]
    axs[0,0].plot(dd,(nav/1e7-1)*100,color=colors[k],ls=styles[k],lw=1.7,label=labels[k])
    axs[0,1].plot(dd,(nav/np.maximum.accumulate(nav)-1)*100,color=colors[k],ls=styles[k],lw=1.7)
axs[0,0].set_title('거래비용 반영 누적 수익률',loc='left');axs[0,0].set_ylabel('수익률 (%)')
axs[0,1].set_title('일별 종가 기준 낙폭',loc='left');axs[0,1].set_ylabel('직전 최고점 대비 (%)')
for ax in axs[0]:dates(ax);grid(ax)
handles,leglabels=axs[0,0].get_legend_handles_labels()
fig.legend(handles,leglabels,loc='upper center',bbox_to_anchor=(.525,.86),ncol=3,frameon=False,fontsize=10)
short=['고정','확장','롤링 3년','3년 고정\n(보조)']
for ax,metric,mult,title in [(axs[1,0],'qlike',1,'분산 예측 오차: QLIKE ↓'),(axs[1,1],'mse',1e7,'분산 예측 오차: MSE ↓')]:
    vals=[t['forecasts'][k][metric]*mult for k in allmethods]
    ax.bar(range(4),vals,color=[colors[k] for k in allmethods],width=.6)
    ax.set_xticks(range(4),short);ax.set_ylim(0,max(vals)*1.23);ax.set_title(title,loc='left');grid(ax)
    ax.set_ylabel('QLIKE' if metric=='qlike' else r'MSE ($\times 10^{-7}$)')
    for j,num in enumerate(vals):ax.text(j,num+max(vals)*.035,f'{num:.5f}' if metric=='qlike' else f'{num:.4f}',ha='center',fontsize=10)
fig.text(.085,.065,'롤링 −5.67% · 확장 −6.29% · 고정 −6.32%   |   보조: 3년 고정 −5.58% · BTC 50% 유지 −4.56%',fontsize=11,weight='bold')
fig.text(.085,.028,'이미 살펴본 2026년의 추가 비교 실험입니다. 그래프가 겹치는 것은 세 전략의 예측과 투자 비중이 매우 비슷하기 때문입니다.',fontsize=10,color='#637286')
save(fig,'har_refit_comparison')

fig,axs=plt.subplots(2,2,figsize=(14.4,9.8),sharex=True)
fig.subplots_adjust(left=.085,right=.97,top=.80,bottom=.14,hspace=.34,wspace=.29)
header(fig,'계수를 고정하거나, 매일 다시 추정하거나', '같은 HAR 식의 계수 변화 · 고정·확장은 전체 과거 자료, 롤링은 최근 3년 사용')
for j,ax in enumerate(axs.flat):
    for k in allmethods:
        ax.plot(d,a['test_'+k+'_coef'][:,j],color=colors[k],ls=styles[k],lw=1.8,label=labels[k])
    ax.set_title([r'상수항 $b_0$',r'어제 로그분산 계수 $b_1$',r'최근 7일 평균 로그분산 계수 $b_7$',r'최근 30일 평균 로그분산 계수 $b_{30}$'][j],loc='left')
    ax.set_ylabel('회귀계수');grid(ax);dates(ax)
handles,leglabels=axs[0,0].get_legend_handles_labels()
fig.legend(handles,leglabels,loc='upper center',bbox_to_anchor=(.525,.87),ncol=4,frameon=False,fontsize=10)
fig.text(.085,.065,'고정은 수평선, 확장·롤링은 매일 재추정한 계수입니다. 같은 초기 3년 구간의 고정선을 함께 비교했습니다.',fontsize=11)
fig.text(.085,.028,'계수가 더 많이 변한다고 예측이 더 정확한 것은 아닙니다. 예측 분산으로 되돌릴 때의 잔차 보정값도 각 학습 구간에서 다시 구했습니다.',fontsize=10,color='#637286')
save(fig,'har_refit_coefficients')

fig,axs=plt.subplots(2,1,figsize=(14.4,9.8),sharex=True)
fig.subplots_adjust(left=.085,right=.97,top=.80,bottom=.14,hspace=.30)
header(fig,'계수 갱신이 예측 변동성과 BTC 비중에 미친 영향')
axs[0].plot(d,100*np.sqrt(a['test_rv']),color='#BDC5CF',lw=1,label='사후 측정 변동성')
for k in methods:
    axs[0].plot(d,100*np.sqrt(a['test_'+k+'_forecast']),color=colors[k],ls=styles[k],lw=1.7,label=labels[k])
    axs[1].plot(d,100*a['test_'+k+'_weights'],color=colors[k],ls=styles[k],lw=1.7)
axs[0].set_ylabel('일별 변동성 (%)');axs[0].set_ylim(bottom=0)
axs[1].set_ylabel('리밸런싱 직후 BTC 비중 (%)');axs[1].set_ylim(0,105)
for ax in axs:grid(ax);dates(ax)
handles,leglabels=axs[0].get_legend_handles_labels()
fig.legend(handles,leglabels,loc='upper center',bbox_to_anchor=(.525,.87),ncol=4,frameon=False,fontsize=10)
fig.text(.085,.065,'BTC 비중 = min(100%, 목표 일별 변동성 1% ÷ 예측 일별 변동성). 나머지는 현금입니다.',fontsize=11,weight='bold')
fig.text(.085,.028,'모든 방식이 매일 최신 입력값으로 예측합니다. 고정 HAR도 예측값과 비중은 매일 바뀌며, 계수만 유지됩니다.',fontsize=10,color='#637286')
save(fig,'har_refit_forecasts_weights')

def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def image(name,caption):
    data=base64.b64encode((O/(name+'.png')).read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{data}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'
def perf(phase,keys):
    p=s['phases'][phase]
    rows=[]
    for k in keys:
        f=p['forecasts'].get(k);st=p['portfolios'][k]
        rows.append([labels[k],f"{f['qlike']:.5f}" if f else '—',f"{f['mse']*1e7:.4f}" if f else '—',
            f"{st['total_return']*100:+.2f}%",'정의되지 않음' if st['sharpe'] is None else f"{st['sharpe']:.3f}",
            f"{st['annual_vol']*100:.2f}%",f"{st['max_drawdown']*100:.2f}%",f"{st['fees_paid_krw']:,.0f}"])
    return table(['방식','QLIKE ↓','MSE ×10⁻⁷ ↓','누적수익률','샤프비율','연환산 변동성','최대낙폭','총 거래비용(원)'],rows)
policytable=table(['방식','계수 학습 범위','계수 재학습','예측 입력 갱신'],[
    ['고정 HAR','각 평가 구간 시작 전까지의 전체 과거 데이터','평가 구간 시작 시 한 번','매일'],
    ['확장 HAR','시작일 유지, 전날까지의 데이터를 계속 추가','매일','매일'],
    ['롤링 HAR','예측일에서 3년 전부터 전날까지','매일','매일'],
    ['3년 고정 HAR (보조)','롤링과 같은 초기 3년 학습 구간','평가 구간 시작 시 한 번','매일']])
cirows=[]
for q in t['pairwise']:
    scale=100 if q['metric']=='net_daily_return' else 1
    cirows.append([labels[q['a']]+' − '+labels[q['b']], '일평균 순수익률 차이(%p)' if scale==100 else '평균 QLIKE 차이',
        f"{q['mean_difference']*scale:+.6f}",f"[{q['ci95_low']*scale:+.6f}, {q['ci95_high']*scale:+.6f}]"])
costrows=[]
for k in allmethods+['BTC50_cash50_daily']:
    vals=[x for x in t['cost_sensitivity'] if x['method']==k]
    costrows.append([labels[k]]+[f"{x['total_return']*100:+.2f}%" for x in vals])
qualityrows=[[labels[k],t['quality'][k]['n'],f"{t['quality'][k]['qlike']:.5f}",f"{t['quality'][k]['mse']*1e7:.4f}"] for k in allmethods]
garch_url='https://arch.readthedocs.io/en/latest/univariate/univariate_volatility_modeling.html'
forecast_url='https://arch.readthedocs.io/en/latest/univariate/univariate_volatility_forecasting.html'
body=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BTC HAR 고정·확장·롤링 비교</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f1f5f9;color:#17263c;font-family:"Malgun Gothic","Noto Sans KR",sans-serif;line-height:1.75}}main{{max-width:1200px;margin:auto;background:white;padding:48px 44px 64px}}h1{{font-size:32px;line-height:1.3}}h2{{margin-top:48px;font-size:23px;border-top:1px solid #d8e1eb;padding-top:24px}}h3{{font-size:18px;margin-top:28px}}p,li{{font-size:15px}}.eyebrow{{color:#557087;font-size:13px}}.lead{{font-size:20px;font-weight:700}}.note{{color:#596b80;font-size:14px}}.box{{background:#edf4fa;border-left:4px solid #2254aa;padding:14px 20px;margin:24px 0}}.scroll{{overflow:auto}}table{{width:100%;border-collapse:collapse;font-size:13px;margin:18px 0}}th{{background:#eaf0f6;text-align:right;padding:12px 10px;white-space:nowrap}}td{{padding:11px 10px;text-align:right;border-bottom:1px solid #e2e8f0;white-space:nowrap}}td:first-child,th:first-child{{text-align:left}}figure{{margin:26px 0}}img{{width:100%;height:auto}}figcaption{{font-size:14px;color:#52677d;padding:8px 5px}}.eq{{padding:16px 20px;background:#f5f7fa;font-family:Consolas,"Malgun Gothic",monospace;overflow:auto;white-space:nowrap}}a{{color:#2254aa}}@media(max-width:700px){{main{{padding:24px 16px}}h1{{font-size:26px}}}}@media print{{main{{padding:0}}figure,table,.eq{{break-inside:avoid}}}}
</style><main>
<div class="eyebrow">업비트 BTC/KRW · 일별 분산 예측 · 2026-10-14 작성</div>
<h1>HAR 고정·확장·롤링 학습 비교</h1>
<p class="lead">세 방식의 차이는 작았다. 매일 재학습하는 것의 우위는 확인되지 않았다.</p>
<p>2026년 평가에서 롤링 HAR의 누적수익률은 −5.67%로 기존 고정 HAR의 −6.32%보다 0.65%p 높았다. 그러나 같은 최근 3년 자료로 한 번만 학습한 고정 HAR는 −5.58%였고, 예측 오차도 롤링보다 작았다. 학습 기간의 차이와 재학습 효과를 구분해야 한다.</p>
<div class="box"><b>해석 범위</b><br>2026년은 이미 데이터 탐색과 투자 성과를 확인한 기간이다. 이번 비교는 추가적인 사후 연구이며 새로운 블라인드 검증이 아니다. 세 가지 방식과 3년 창, 매일 재학습 규칙을 계산 전에 정했고, 이번 테스트 수익률을 보고 창 길이나 매매 규칙을 조정하지 않았다.</div>
<h2>1. 실험 조건</h2>
{policytable}
<p>자료는 기존에 수집한 2017-09-26~2026-10-11의 완결된 UTC 일자 3,303일이다. 새로운 데이터를 추가 수집하지 않았다. 1분 종가를 정규 격자로 정리하고, 5분 로그수익률 제곱을 합산한 일별 실현분산(RV)을 HAR의 학습·평가 대상으로 사용했다. 처음 30일은 입력 이력을 만드는 데 사용한다.</p>
<p><b>2024~2025년 비교:</b> 초기 학습은 2023년 말까지. <b>2026년 비교:</b> 2025년 말까지 다시 학습한 상태에서 시작한다. 확장·롤링은 각 예측일 전날까지 관측된 값을 추가해 매일 다시 학습한다. 롤링은 달력 기준 최근 3년이므로 표본 수는 윤년에 따라 1,095~1,096일이다. 창 시작 부근 학습 표적의 7·30일 입력에는 창 이전의 관측값도 사용된다.</p>
<div class="eq">log V̂ₜ의 선형 예측 = b₀ + b₁ log Vₜ₋₁ + b₇ 평균(log Vₜ₋₇,…,log Vₜ₋₁) + b₃₀ 평균(log Vₜ₋₃₀,…,log Vₜ₋₁)<br>분산 예측 = exp(선형 예측) × 학습 잔차 exp의 평균</div>
<p>OLS 회귀식과 1·7·30일 입력, 학습 잔차만 사용한 로그 역변환 보정은 모두 동일하다. 고정은 계수와 보정값을 유지하고, 확장·롤링은 둘 다 다시 추정한다. ‘3년 고정’은 롤링과 초기 학습 자료를 맞추기 위한 보조 비교다.</p>
<p>모든 전략의 BTC 비중은 min(1, 0.01/√예측분산). 나머지는 원화 현금이고 현금 이자는 0%이다. 한국시간 09:01에 매일 리밸런싱하며 직전 보유분은 첫 1분 수익률을 먼저 받는다. 실제 매매금액에 편도 0.10%의 수수료·슬리피지 가정을 적용하고 최초 진입과 최종 청산 비용도 포함한다. 초기 자금 1,000만원, 공매도·레버리지 없음, 실제 주문 없음.</p>
<h2>2. 2026년 결과</h2>
<p class="note">2026-01-01~10-11 UTC, 284일. MSE는 원래 분산 단위의 값에 10⁷을 곱해 표시했다. 샤프비율과 변동성은 365일로 연율화했다. 최대낙폭은 일별 종가 계좌가치 기준이다.</p>
{perf('test',allmethods+['BTC50_cash50_daily','BTC_buy_hold','cash'])}
<p>확장 HAR는 고정 HAR와 거의 같은 결과였다. 롤링은 기존 전체 기간 고정보다 QLIKE가 약 0.80% 작았지만, 초기 3년 고정보다 QLIKE가 약 0.48% 컸다. 재학습 자체가 도움이 되었다고 단정할 수 없다. 세 주 전략 모두 50:50 고정 비중보다 누적수익률·샤프비율·최대낙폭이 불리했다. 다만 연환산 변동성은 약 17.4%로 50:50의 약 20.5%보다 낮았다.</p>
{image('har_refit_comparison','위: 비용을 반영한 계좌 수익률과 낙폭. 아래: 같은 날짜의 실현분산에 대한 예측 오차. 막대는 모두 0에서 시작한다.')}
<h2>3. 계수와 투자 비중은 어떻게 달라졌나</h2>
{image('har_refit_coefficients','전체 과거 자료를 사용하는 확장 방식은 고정 계수에서 조금만 변했다. 롤링은 오래된 자료를 제외하면서 계수가 더 많이 변했으나 이것만으로 성능 개선을 뜻하지 않는다.')}
{image('har_refit_forecasts_weights','세 방식 모두 매일 새로운 입력값을 사용하므로 예측값과 비중은 매일 바뀐다. 회색 변동성은 해당 날짜가 끝난 뒤 측정한 값이며 투자 결정 시 미리 사용하지 않았다.')}
<h2>4. 다른 기간·비용·데이터 품질에서도 확인</h2>
<h3>2024~2025년 비교</h3>{perf('validation',allmethods)}
<p>이 구간에서는 기존 고정 HAR의 QLIKE가 가장 낮고, 롤링 HAR는 가장 높았다. 반면 투자 샤프비율은 롤링이 더 높았다. 예측 오차 순위와 투자 성과 순위는 일치하지 않으며, 2026년 순위가 다른 기간에도 유지된다고 볼 수 없다.</p>
<h3>편도 거래비용별 2026년 누적수익률</h3>{table(['방식','0%','0.05%','0.10%','0.25%'],costrows)}
<p>재학습 자체에는 거래비용을 부과하지 않는다. 재학습으로 목표 비중이 바뀌면 실제 매매량과 비용이 달라진다.</p>
<h3>예측·수익률 차이의 불확실성</h3>{table(['비교','지표','평균 차이','근사 95% 구간'],cirows)}
<p>동일 날짜를 묶은 14일 원형 블록 부트스트랩을 3,000회 적용했다. 표시한 모든 구간이 0을 포함한다. 작은 성과 차이가 반복될 것이라고 주장할 근거는 부족하다. 이는 기존에 본 기간에 대한 설명적 불확실성 평가이며, 새로운 독립 검증을 대체하지 않는다.</p>
<h3>분봉 관측 비율 95% 이상인 날짜만 예측 오차 평가</h3>{table(['방식','날짜 수','QLIKE','MSE ×10⁻⁷'],qualityrows)}
<p>2026-04-12와 2026-07-05는 관측 비율이 각각 약 76.8%, 79.9%인 날이다. 두 날짜를 오차 평가에서 제외해도 순위는 같았다. 투자 경로에서는 해당 날짜를 삭제하지 않았다. 관측되지 않은 분봉은 기존 방식대로 직전 종가로 채웠다. 이 민감도 검사는 학습 데이터를 정제해 다시 학습한 실험은 아니다.</p>
<h2>5. GARCH-t와 GJR-t의 의미</h2>
<p>이번에 재학습한 세 방식은 모두 HAR이다. GARCH-t와 GJR-t는 앞선 보고서에 있던 별도의 변동성 모형 이름이며 이번 비교에서 새로 재학습하지 않았다.</p>
<p><b>공통의 -t:</b> 변동성으로 나눈 수익률 충격이 표준화 Student-t 분포를 따른다는 뜻이다. 시간 첨자 t나 t검정을 뜻하지 않는다. 같은 분산의 정규분포보다 두꺼운 꼬리를 표현하고, 자유도 ν도 학습한다. 분산이 1이 되도록 표준화하며 ν &gt; 2를 사용한다.</p>
<div class="eq">rₜ = μ + εₜ,　εₜ = √hₜ · zₜ,　zₜ ∼ 표준화 Student-t(ν)</div>
<h3>GARCH-t: 충격의 크기와 이전 위험을 반영</h3>
<div class="eq">hₜ₊₁ = ω + α εₜ² + β hₜ</div>
<p>hₜ₊₁은 다음 날 조건부 분산, εₜ는 당일 수익률에서 추정 평균 μ를 뺀 충격이다. ω는 분산 식의 상수항, α는 새 충격의 제곱에 대한 반응, β는 직전 조건부 분산의 지속 정도다. 같은 크기의 양·음 충격은 제곱값이 같으므로 같은 영향을 준다.</p>
<h3>GJR-t: 충격의 부호에 따른 반응 차이를 추가</h3>
<div class="eq">hₜ₊₁ = ω + α εₜ² + γ εₜ² 1(εₜ &lt; 0) + β hₜ</div>
<p>GJR은 Glosten·Jagannathan·Runkle의 이름에서 왔다. 음의 충격에는 α+γ, 양의 충격에는 α가 적용된다. γ가 양수이면 같은 크기의 음의 충격 뒤에 분산이 더 커진다. γ의 부호와 크기는 자료에서 추정하므로 BTC에서 항상 이 패턴이 있다고 가정하지 않는다. 여기서 부호는 평균 대비 충격의 부호이며, 다음 날 가격 방향을 예측하는 매수·매도 신호가 아니다.</p>
<p class="note">모형 식과 분포 정의: <a href="{garch_url}">arch 공식 문서</a>. 학습 구간 방식: <a href="{forecast_url}">arch 예측 문서</a>. 데이터: <a href="https://docs.upbit.com/kr/reference/list-candles-minutes">업비트 분봉 API</a>.</p>
<h2>6. 계산 검증</h2>
<p>기존 전체 기간 고정 HAR와 기존 3년 고정 HAR의 예측을 두 평가 구간 모두 재현했다. 모든 4,060개 예측의 학습 종료일과 입력 시점을 점검했고, 40개 대표 재학습 결과를 별도의 QR 최소제곱 계산으로 확인했다. BTC 수량·현금 장부를 따로 계산해 8개 전략·기간 조합의 일별 계좌가치를 대조했으며 최대 차이는 0.000001원 미만이었다.</p>
<p class="note">재현에 필요한 프로토콜, 일별 예측·계좌가치, 계수 이력과 계산 코드는 함께 제공한 분석 묶음에 들어 있다. 원본 분봉 자료는 기존 다운로드 파일을 사용한다. FIVAR의 다자산 공분산 모형을 재현한 실험은 아니다.</p>
</main></html>'''
(O/'har_refit_report.html').write_text(body,encoding='utf-8')
readme='''HAR fixed / expanding / rolling comparison
Read har_refit_report.html for methods, results, model definitions and limitations.
All dates are UTC. Existing BTC/KRW data end at 2026-10-11.
The 2026 experiment is retrospective, not a new blind holdout.
JSON files preserve full precision; displayed tables are rounded.
har_refit_daily.json: 1,015 dates in two separately initialized phases.
har_refit_coefficients.json: 4,060 daily coefficient records including the 3-year fixed control.
summary: phase, method, forecast loss and portfolio performance.
Variance is expressed in squared decimal log-return units. Returns/weights are decimal fractions.
Run Python from the original workspace: work/compare_har_refits.py then work/verify_har_refits.py.
The comparison reuses work/backtest_btc_cash.py and the existing work/minute_plot_arrays.npz (not duplicated here).
Required Python libraries: numpy, scipy, arch. Plots additionally need matplotlib and the existing Korean fonts.
No CSV/XLSX or external account was created. No orders were submitted.
'''
# This is a report plus reproducibility archive, not a spreadsheet deliverable.
with zipfile.ZipFile(O/'har_refit_bundle.zip','w',compression=zipfile.ZIP_DEFLATED) as zz:
    for name in ['har_refit_report.html','har_refit_comparison.png','har_refit_coefficients.png','har_refit_forecasts_weights.png']:zz.write(O/name,name)
    for name in ['har_refit_protocol.json','har_refit_results.json','har_refit_summary.json','har_refit_daily.json','har_refit_coefficients.json','har_refit_verification.json']:
        zz.write(W/name,'data/'+name)
    for name in ['compare_har_refits.py','verify_har_refits.py','report_har_refits.py','backtest_btc_cash.py']:
        zz.write(W/name,'code/'+name)
    zz.writestr('README.txt',readme)
print('Created report, 3 figures and reproducibility bundle.')
