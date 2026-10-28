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
plt.rcParams.update({'font.family':font_manager.FontProperties(fname='work/NotoSansKR-Regular.ttf').get_name(),'font.size':11,
 'axes.titlesize':13,'axes.titleweight':'bold','axes.unicode_minus':False,'text.color':'#17263C','axes.labelcolor':'#39465A',
 'axes.edgecolor':'#CBD5E1','axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#F5F8FC','axes.facecolor':'white','savefig.facecolor':'#F5F8FC'})
W=Path('work');O=Path('outputs');O.mkdir(exist_ok=True)
res={p:json.loads((W/f'btc_slow_{p}_results.json').read_text()) for p in ['validation','test']}
a=np.load(W/'btc_slow_test_arrays.npz');d=a['dates'];t=res['test'];v=res['validation'];verification=json.loads((W/'btc_slow_verification.json').read_text());assert verification['status']=='passed'
COL=['#8290A4','#8470B5','#176BCE','#169B8B'];hs=[1,7,28];hlabels=['매일','7일마다','28일마다']
def header(fig,title,subtitle):
 fig.text(.075,.965,title,fontsize=21,weight='bold',va='top');fig.text(.075,.908,subtitle,fontsize=10.5,color='#637286',va='top')
def grid(ax):ax.grid(axis='y',color='#E0E6EF',lw=.7);ax.set_axisbelow(True)
def dateaxis(ax):ax.xaxis.set_major_locator(mdates.MonthLocator());ax.xaxis.set_major_formatter(mdates.DateFormatter('%m월'))
def savefig(fig,name):fig.savefig(O/(name+'.png'),dpi=150);plt.close(fig)

fig,axs=plt.subplots(2,2,figsize=(14.4,10.4));fig.subplots_adjust(left=.075,right=.97,top=.81,bottom=.13,hspace=.43,wspace=.29)
header(fig,'거래 횟수는 줄었다. 수익성은 아직 확인되지 않았다.','2026-01-01 ~ 10-11, 284일 · 순롱숏 정책 비교 · 초기 1,000만원 · 편도 비용 0.10%')
keys=['original_daily','pure_1d','pure_7d','pure_28d'];labs=['기존 매일\n한도 ±100%','수정 매일\n한도 ±25%','수정 7일\n한도 ±25%','수정 28일\n한도 ±25%']
for ax,field,mult,title,unit in [(axs[0,0],'actual_trade_count',1,'실제 거래 횟수','회'),(axs[0,1],'fees_krw',1/10000,'거래비용 합계','만원'),(axs[1,0],'return_net',100,'비용 차감 후 누적 수익률','%')]:
 vals=[t['strategies'][k][field]*mult for k in keys];ax.bar(range(4),vals,color=COL,width=.59);ax.set_xticks(range(4),labs,fontsize=10);ax.set_title(title,loc='left');ax.set_ylabel(unit);grid(ax)
 span=max(vals)-min(vals);ax.set_ylim(min(0,min(vals))-max(span*.16,.1),max(0,max(vals))+max(span*.25,.16))
 for i,n in enumerate(vals):ax.text(i,n+(span*.045 if n>=0 else -span*.08),f'{n:.0f}' if field=='actual_trade_count' else f'{n:+.2f}' if field=='return_net' else f'{n:.2f}',ha='center',va='bottom' if n>=0 else 'top',fontsize=11,weight='bold')
 ax.axhline(0,color='#758297',lw=.7)
xx=np.arange(3);val=[v['strategies'][f'pure_{h}d']['return_net']*100 for h in hs];test=[t['strategies'][f'pure_{h}d']['return_net']*100 for h in hs]
axs[1,1].bar(xx-.17,val,.34,color='#B4C7E1',label='검증 2024~2025 · 2년');axs[1,1].bar(xx+.17,test,.34,color='#176BCE',label='평가 2026 · 284일');axs[1,1].set_xticks(xx,hlabels);axs[1,1].set_title('수정 정책의 기간별 성과',loc='left');axs[1,1].set_ylabel('누적 수익률 (%)');grid(axs[1,1]);axs[1,1].axhline(0,color='#758297',lw=.7);axs[1,1].legend(fontsize=9,frameon=False,loc='upper right');axs[1,1].set_ylim(-3.2,7)
for i in range(3):
 for shift,nums in [(-.17,val),(.17,test)]:axs[1,1].text(i+shift,nums[i]+(.15 if nums[i]>=0 else -.15),f'{nums[i]:+.2f}',ha='center',va='bottom' if nums[i]>=0 else 'top',fontsize=10)
fig.text(.075,.057,'수정안은 비중 한도·거래 허용폭도 바뀌었습니다. 주기가 길어지면 학습할 수익률의 기간과 하락 조건도 달라집니다.',fontsize=10.5)
fig.text(.075,.026,'따라서 성과 차이를 거래 빈도 하나의 효과로 해석할 수 없습니다. 7일·28일 정책은 평가 기간에 순숏 보유일이 0일이었습니다.',fontsize=10.5,color='#9C4E37')
savefig(fig,'btc_longshort_frequency')

fig,axs=plt.subplots(2,2,figsize=(14.4,10.4));fig.subplots_adjust(left=.075,right=.97,top=.81,bottom=.12,hspace=.35,wspace=.25)
header(fig,'자료의 아이디어를 BTC 50% + 비중 조절로 적용','기본 BTC 50%·현금 50%에 최대 ±25%p 신호를 반영 · 실제 매매는 5%p 차이부터 · 같은 주기의 기준 포트폴리오와 비교')
navdates=np.r_[d,d[-1]+np.timedelta64(1,'D')]
for col,h in enumerate([7,28]):
 for key,label,color,ls in [(f'core_{h}d','신호 반영','#176BCE','-'),(f'btc50_{h}d','BTC 50% 기준','#8997AA','--')]:
  nav=np.r_[1e7,a[key+'_nav']];axs[0,col].plot(navdates,(nav/1e7-1)*100,color=color,ls=ls,lw=2,label=f"{label}  {t['strategies'][key]['return_net']*100:+.2f}%")
  axs[1,col].plot(d,a[key+'_drawdown']*100,color=color,ls=ls,lw=2)
 axs[0,col].set_title(f'{h}일 간격 · 누적 수익률',loc='left');axs[0,col].set_ylabel('%');axs[0,col].legend(frameon=False,fontsize=10,loc='upper right')
 axs[1,col].set_title(f'{h}일 간격 · 일별 평가 낙폭',loc='left');axs[1,col].set_ylabel('직전 최고점 대비 (%)')
 for row in range(2):grid(axs[row,col]);dateaxis(axs[row,col])
fig.text(.075,.055,'7일: 신호 반영 −5.17%, 기준 −5.02%   |   28일: 신호 반영 −4.35%, 기준 −4.89%',fontsize=11,weight='bold')
fig.text(.075,.025,'신호 반영안은 실제 순숏이 없는 비중 조절형입니다. 2026년에는 두 주기 모두 기준 포트폴리오보다 최대낙폭이 컸습니다.',fontsize=10.5,color='#9C4E37')
savefig(fig,'btc_longshort_core_comparison')

fig,axs=plt.subplots(3,2,figsize=(14.4,12.8));fig.subplots_adjust(left=.075,right=.97,top=.84,bottom=.09,hspace=.36,wspace=.25)
header(fig,'신호와 실제 보유 비중: 2026년에는 숏이 나오지 않았다','동그라미·삼각형은 정해진 검토일 · 가중치 선은 가격 변동에 따른 비중 변화도 포함 · 28일은 달력상 월 1회와 다름')
for col,h in enumerate([7,28]):
 mask=a[f'h{h}_review'];sig=a[f'h{h}_signal'];event=a[f'h{h}_event']&a[f'h{h}_feature_ok'];p=a[f'h{h}_p'];price=a['price'][:-1]/1e6
 axs[0,col].plot(d,price,color='#77869C',lw=1.2);axs[0,col].scatter(d[mask&(sig==0)],price[mask&(sig==0)],s=28,color='#C3CBD7',label='중립 결정',zorder=3)
 axs[0,col].scatter(d[mask&(sig==1)],price[mask&(sig==1)],s=70,color='#169B8B',marker='^',label='롱 방향 결정',zorder=4)
 axs[0,col].set_title(f'{h}일 간격 · 결정 당시 BTC 가격',loc='left');axs[0,col].set_ylabel('백만원 / BTC');axs[0,col].legend(frameon=False,fontsize=9,loc='upper right')
 axs[1,col].scatter(d[mask&~event],p[mask&~event],color='#CAD1DC',s=25,label='하락 조건 미충족·사용 안 함')
 axs[1,col].scatter(d[mask&event],p[mask&event],color='#176BCE',s=34,label='하락 조건 충족')
 for y,c in [(.55,'#169B8B'),(.45,'#B05A47')]:axs[1,col].axhline(y,color=c,ls='--',lw=1)
 axs[1,col].axhspan(.45,.55,color='#DCE5EE',alpha=.35);axs[1,col].set_ylim(.2,.85);axs[1,col].set_ylabel('향후 상승 확률 예측');axs[1,col].set_title(f'모델이 예측한 앞으로 {h}일의 방향',loc='left');axs[1,col].legend(frameon=False,fontsize=8,loc='lower left')
 axs[2,col].plot(d,100*a[f'core_{h}d_weights'],color='#176BCE',lw=1.7,label='BTC 50% + 비중 조절')
 axs[2,col].plot(d,100*a[f'pure_{h}d_weights'],color='#169B8B',lw=1.7,label='순롱숏 정책의 실제 노출')
 axs[2,col].axhline(0,color='#758297',lw=.8);axs[2,col].axhline(50,color='#AAB4C4',ls=':',lw=.8);axs[2,col].set_ylim(-30,85);axs[2,col].set_ylabel('순 BTC 비중 (%)');axs[2,col].set_title('검토일 사이에는 BTC 수량을 유지',loc='left');axs[2,col].legend(frameon=False,fontsize=9,loc='lower left')
 for row in range(3):grid(axs[row,col]);dateaxis(axs[row,col])
fig.text(.075,.039,'롱은 상승확률 55% 이상, 숏은 45% 이하에서만 허용했습니다. 롱·숏 횟수를 맞추려고 문턱을 사후 변경하지 않았습니다.',fontsize=10.5)
fig.text(.075,.016,'거래 사이의 비중 변동은 추가 매매가 아닙니다. 중립 신호는 순롱숏 정책에서 청산, 비중 조절형에서는 50% 복귀 목표입니다.',fontsize=10.5,color='#637286')
savefig(fig,'btc_longshort_positions')

def fmt(x,digits=2):return '—' if x is None else f'{x:.{digits}f}'
def pc(x):return fmt(x*100)+'%'
def table(headers,rows):return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(c))+'</th>' for c in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(c)+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def img(name,caption):return '<figure><img src="data:image/png;base64,'+base64.b64encode((O/(name+'.png')).read_bytes()).decode()+'"><figcaption>'+caption+'</figcaption></figure>'
def section(k,title,body):return f'<section><div class="eyebrow">{k}</div><h2>{title}</h2>{body}</section>'
parts=[]
parts.append(section('01 / 결론','거래를 줄이는 수정은 가능했다. 헤지 성과는 아직 입증되지 않았다.',
 '<div class="cards"><div><b>53 → 10 → 5회</b><span>수정 순롱숏의 실제 거래 횟수<br>매일 → 7일 → 28일</span></div><div><b>−1.22% / +0.01%</b><span>7일 / 28일 순롱숏 누적 수익률<br>비용 차감 후 · 2026년 평가</span></div><div><b>순숏 보유 0일</b><span>7일·28일 정책 모두<br>낮은 빈도만으로 헤지가 완성되지 않음</span></div></div>'
 '<p>첨부 자료의 “기준 포트폴리오에 롱·숏 의견을 비중 증감으로 반영한다”는 아이디어를 적용했다. 기본 비중은 BTC 50%·현금 50%, 신호에 따른 추가·감축 한도는 ±25%p다. 실제 음수 BTC 포지션을 허용하는 순롱숏 정책도 별도로 계산했다.</p>'
 '<p><strong>이번 연구의 핵심 결과:</strong> 거래 횟수와 비용은 줄었지만, 주간·4주 방향 모델의 예측력은 평가 구간에서 유지되지 않았다. 4주 정책의 +0.01%를 수익성의 증거로 삼을 수 없으며, 두 정책 모두 이 구간에서 숏에 진입하지 않았다.</p>'
 '<p class="note">계산일 2026-10-14. 기존 업비트 BTC/KRW 분봉 자료만 사용. 학습 자료는 2017-09-26부터, 검증은 2024-01-01~2025-12-31, 평가는 2026-01-01~10-11의 284개 일별 보유 구간. 최종 청산은 2026-10-12 09:01 KST. 이미 살펴본 2026년을 다시 분석한 후속 연구이므로 새로운 블라인드 검증으로 보지 않는다.</p>'))
parts.append(section('02 / 참고 자료 반영','130/30을 복사하지 않고, 기본 비중과 신호 비중을 나눴다.',
 '<p>참고: 삼성증권 김동영, 「포트 전략 – 롱/숏 시그널로 롱 포트폴리오 구성하기」, 2023-01-04. 3~7쪽은 주식 벤치마크에 롱·숏 비중을 더한 뒤, 음수 종목 비중을 0으로 만들고 양수 비중을 같은 %p만큼 차감해 합계 100%로 조정한다. 8쪽의 실증 사례는 월별로 제시된 롱·숏 종목 목록을 사용한다.</p>'+
 table(['자료의 구성','이번 BTC 적용'],[
 ['여러 주식의 상대적 롱·숏 의견','단일 BTC의 시간에 따른 방향 신호'],['주식 벤치마크 + 비중 증감','BTC 50%·현금 50% + 최대 ±25%p'],['월별 종목 목록·보유 구간','7일·28일마다만 매매를 검토'],['여러 종목의 음수 비중 제거·동일 차감','단일 위험자산이라 해당 다종목 알고리즘은 재현하지 않음'],['130% 롱 / 30% 숏 형태가 생기는 예','같은 BTC라면 순노출 100%. 이를 분산된 롱숏 전략으로 취급하지 않음']])+
 '<p class="note">50% 기본 비중, ±25%p 한도, 5%p 거래 허용폭은 이번 비교를 위해 정한 연구용 설정이며 원문이 권한 비율이나 최적화 결과가 아니다. 원문과 FIVAR는 별개이며, 이번 실험은 FIVAR 모형 재현도 아니다.</p>'))
parts.append(section('03 / 운용 규칙','모델 갱신, 신호 평가, 실제 거래를 구분했다.',
 table(['항목','적용 규칙'],[
 ['하락 조건','직전 1일 / 7일 / 28일의 누적 로그수익률이 음수일 때만 방향 신호 사용'],['예측할 결과','각각 앞으로 1일 / 7일 / 28일의 체결시점 간 수익률이 양수인지'],['계수 재학습','매월 처음 예측하는 날, 최근 3년의 완료된 학습 표본으로 로지스틱 회귀 재추정'],['신호','상승확률 ≥55%: 롱 방향 / ≤45%: 숏 방향 / 그 사이 또는 조건 미충족: 중립'],['비중 검토','각 검증·평가 구간의 1월 1일을 시작점으로 1일 / 7일 / 28일마다, 한국시간 09:01'],['실제 매매','목표와 현재 비중 차이가 5%p 이상일 때. 순롱숏의 중립 청산·방향 전환·허용 범위 이탈은 예외적으로 실행'],['검토일 사이','BTC 수량 유지. 일별 가격 변화로 비중은 자연스럽게 변동. 자동 손절·비상 축소 규칙은 이번 실험에 없음'],['비용','편도 거래대금의 0.10%. 합성 순숏에는 연 10% 차입비 가정. 현금 이자 0%. 마지막 청산 비용 포함']])+
 '<div class="formula">aₜ = 25% × min[1, 1% ÷ HAR의 다음 날 예측 표준편차]<br>비중 조절형: wₜ = 50% + sₜ × aₜ<br>순롱숏: wₜ = sₜ × aₜ　(sₜ = +1, 0, −1)</div>'
 '<p>예측 일별 변동성이 2%이면 aₜ는 12.5%p다. 비중 조절형의 롱·중립·숏 의견은 BTC <strong>62.5%·50%·37.5%</strong> 목표로, 순롱숏에서는 <strong>+12.5%·0%·−12.5%</strong>로 바뀐다. “숏 의견으로 37.5% 보유”는 BTC를 적게 사는 것이고, −12.5%는 실제 순숏이다. 둘은 다르다.</p>'
 '<p class="note">HAR는 기존 고정 계수 모형이며 최신 입력값으로 다음 날 분산을 예측한다. 이 수치를 검토일의 비중 조절에 사용했으며, 보유 기간 전체의 변동성을 1%로 보장하지 않는다. 7·28일 미래 수익률에 맞춰 방향 모델은 다시 학습했지만, 7·28일 분산 예측 모형을 별도로 검증한 것은 아니다. 목표 한도는 거래 시점에 적용되며 이후 가격 변화로 비중이 벗어날 수 있다.</p>'))
parts.append(section('04 / 거래 빈도와 비용','실제 거래 횟수와 보유일 수는 다르다.',img('btc_longshort_frequency','거래 횟수는 실제 비중 조정 거래와 마지막 청산을 센다. 여러 날 포지션을 보유해도 수량을 바꾸지 않으면 추가 거래가 아니다. 비용 감소에는 한도 축소·거래 허용폭·신호 및 예측 기간 변경이 함께 영향을 미친다.')+
 table(['순롱숏 정책','검증 수익률<br>2024~2025','평가 수익률<br>2026','평가 거래 수','거래비용','차입비','평가 롱/숏/현금 일수'],[[('기존 매일 ±100%' if k=='original_daily' else {1:'수정 매일 ±25%',7:'수정 7일 ±25%',28:'수정 28일 ±25%'}[int(k.split('_')[1][:-1])]),pc(v['strategies'][k]['return_net']),pc(t['strategies'][k]['return_net']),t['strategies'][k]['actual_trade_count'],f"{t['strategies'][k]['fees_krw']:,.0f}원",f"{t['strategies'][k]['borrow_krw']:,.0f}원",f"{t['strategies'][k]['long_days']} / {t['strategies'][k]['short_days']} / {t['strategies'][k]['flat_days']}"] for k in keys])+
 '<p>수정 매일 정책은 기존 전략을 단순히 1/4로 줄인 결과가 아니다. 5%p 거래 허용폭 때문에 일부 진입과 조정도 달라진다. 28일 정책은 거래가 5번뿐이고 숏은 없었다. 이 작은 표본의 소폭 플러스 수익을 근거로 28일이 최적이라고 선택하지 않는다.</p>'))
parts.append(section('05 / 기준 포트폴리오 비교','비중 조절은 기본 BTC 보유보다 나았는가?',img('btc_longshort_core_comparison','선의 수익률은 복리 누적 수익률, 아래 패널은 매일 평가한 최고점 대비 낙폭이다. 비교 기준에도 같은 검토 주기와 5%p 거래 허용폭을 적용했다.')+
 table(['검토 주기','전략','검증 누적 수익률','평가 누적 수익률','평가 Sharpe','평가 최대낙폭','거래 수'],[[f'{h}일',kind,pc(v['strategies'][f'{key}_{h}d']['return_net']),pc(t['strategies'][f'{key}_{h}d']['return_net']),fmt(t['strategies'][f'{key}_{h}d']['sharpe']),pc(t['strategies'][f'{key}_{h}d']['mdd']),t['strategies'][f'{key}_{h}d']['actual_trade_count']] for h in hs for key,kind in [('core','50% + 신호'),('btc50','50% 기준')]])+
 '<p>주간 비중 조절형은 검증 기간에 기준보다 높은 누적 성과를 보였지만, 2026년에는 −5.17%로 기준 −5.02%보다 소폭 낮았다. 4주 정책은 기준 대비 누적 성과가 0.54%p 높았으나 최대낙폭은 더 컸다. 따라서 “비중을 덜 자주 조절하면 하락 위험이 줄어든다”는 결론은 이번 결과에서 나오지 않는다.</p>'))
parts.append(section('06 / 신호 해석','주간·4주 정책은 이 평가 기간에 매수와 관망만 했다.',img('btc_longshort_positions','회색 확률 점은 하락 조건을 충족하지 않아 매매에 쓰지 않은 모델 출력이다. 삼각형은 롱 방향 결정이지 매번 새로 매수했다는 표시가 아니다. 실제 주문 여부는 거래 허용폭과 기존 보유량에 따라 달라진다.')+
 table(['기간','예측·보유 기간','완료된 조건 충족 표본','확신 신호 수','방향 적중률','Log loss','과거 비율만 예측한 Log loss','AUC'],[[('검증 2024~2025' if phase=='validation' else '평가 2026'),f'{h}일',r['n'],r['active_full_horizon_signals'],pc(r['active_direction_hit_rate']) if r['active_direction_hit_rate'] is not None else '—',fmt(r['log_loss'],4),fmt(r['null_log_loss'],4),fmt(r['auc'],3)] for phase in ['validation','test'] for h in hs for r in [res[phase]['probabilities'][str(h)]]])+
 '<p>Log loss는 실제 결과에 부여한 예측 확률의 −log 값 평균으로, 작을수록 좋다. 기준 예측은 각 학습 구간에서 관측한 상승 비율이다. AUC는 상승 사례를 하락 사례보다 높은 확률로 순위 매기는 능력이며 0.5가 무작위 수준이다. 평가 기간의 주간·4주 모델은 이 기준 예측보다 Log loss가 나빴다.</p>'
 '<p class="note">확률 평가 표본은 검토일 중 하락·자료 품질 조건을 충족하고 보유 기간이 완전히 끝난 사례다. 주간은 22개, 4주는 6개로 적다. 확신 신호 수와 실제 거래 수가 다른 이유는 진입·청산·연속 보유·거래 허용폭 때문이다. 학습에는 서로 겹치는 일별 7·28일 라벨을 사용해 표본끼리 독립적이지 않다.</p>'))
costrows=[]
for h in hs:
 for c in [0,.0005,.001,.0025]:
  r=next(x for x in t['cost_sensitivity'] if x['strategy']==f'pure_{h}d' and x['fee']==c and x['borrow']==.10)
  costrows.append([f'{h}일',f'{c*100:.2f}%',pc(r['return_net']),r['actual_trade_count']])
parts.append(section('07 / 비용 민감도와 불확실성','비용을 낮춰도 방향 예측 문제는 남는다.',table(['순롱숏 검토 주기','편도 비용 가정','평가 누적 수익률','실제 거래 수'],costrows)+
 '<p>각 비용 가정에서 현금·보유량·거래 허용폭을 다시 계산했다. 차입비는 연 10%로 고정했다. 주간·4주 정책은 평가 구간에 숏이 없어 이 가정이 손익에 영향을 주지 않는다. 실제 선물 펀딩은 이 고정 차입비와 다른 비용 구조다.</p>'+
 table(['비중 조절형 vs 같은 주기의 50% 기준','연율 TE','연율 IR','일평균 초과수익 95% 구간'],[[f'{h}일',pc(r['te_annual']),fmt(r['information_ratio'],3),f"[{r['mean_daily_active_ci95'][0]*10000:+.2f}, {r['mean_daily_active_ci95'][1]*10000:+.2f}] bp/일"] for h in hs for r in [t['comparisons'][f'core_{h}d_vs_btc50']]])+
 '<div class="formula">초과수익 eₜ = 전략 일수익률 − 기준 일수익률<br>TE = sd(eₜ) × √365<br>IR = mean(eₜ) ÷ sd(eₜ) × √365</div>'
 '<p>95% 구간은 56일 달력 블록으로 묶은 순환 부트스트랩 3,000회다. 모든 구간이 0을 포함한다. IR는 일별 산술 초과수익으로 구했으므로 두 복리 누적 수익률 차이와 부호가 반드시 같지는 않다. 원문의 비용 없는 고정 오버레이 성질을 이번 비용·비중 제한·변동 크기 조절에 그대로 적용할 수 없다.</p>'))
parts.append(section('08 / 재현·검증','계산에서 확인한 것과 아직 확인하지 못한 것',
 '<p>λ=0.1, 확률 문턱 55/45, 기본 BTC 50%, 추가 한도 25%p, 거래 허용폭 5%p, 검토 간격 1·7·28일을 새 결과 계산 전에 파일로 고정했다. 기간별 최고 수익을 보고 설정을 다시 고르지 않았으며 비교 정책을 모두 제시했다. 1일 모델은 앞서 고정한 방향 확률을 정확히 재현했다.</p>'
 '<p>월별 재학습 시점 t의 학습 라벨은 종료 시점이 t−1일 00:01 UTC 이전까지인 것만 사용했다. 즉 h일 라벨의 마지막 시작일은 t−h−1이다. 평균·표준편차는 학습 표본만으로 계산했다. HAR 계수는 검증 시 2023년 말, 평가 시 2025년 말까지만 사용한 기존 고정 모형이다.</p>'
 f'<p><strong>검증 통과:</strong> {verification["monthly_models_availability_checked"]}개 월별 모델의 정보 사용 시점, 독립 Newton 방식의 대표 12개 계수 재계산, 독립 현금·BTC 장부 {verification["ledger_day_checks"]:,}일 대조, 검토일 사이 수량 유지 {verification["unchanged_quantity_checks"]:,}건을 확인했다. 독립 장부의 최대 평가금 차이는 {verification["max_nav_error_krw"]:.2e}원이다. 거래·청산·차입 비용과 거래 횟수도 일치했다.</p>'
 '<p><strong>남은 한계:</strong> 업비트 현물 분봉으로 만든 합성 순숏 장부다. 실제 선물의 펀딩·베이시스·담보·강제청산·체결 가능성은 검증하지 않았다. 원자료의 무거래·누락 구간은 기존 전방 채움 방식을 유지했다. 학습은 품질 조건을 적용했지만 평가 손익에서 사후 저품질 날짜를 지우지 않았다. 2026년 자료는 이미 여러 번 살펴봤으므로 실전 채택을 뒷받침하는 새로운 표본 검증은 남아 있다.</p>'
 '<p><strong>현재 판단:</strong> 적은 거래 빈도와 제한된 비중은 운용 규칙으로 구현했지만, 하락을 포착해 수익을 내는 숏 신호는 아직 확보하지 못했다. 다음 연구는 거래를 더 줄이는 최적화보다, 새로운 기간에서 하락 방향 신호의 예측력과 실제 숏 상품의 비용을 검증하는 데 초점을 두는 편이 타당하다.</p>'
 '<p class="note">참고 방법 문서: <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html">scikit-learn 시계열 분할·gap</a>, <a href="https://support.kraken.com/articles/4844359082772-linear-multi-collateral-derivatives-contract-specifications">실제 파생상품의 펀딩·계약 구조 예시</a>. 이들 서비스를 통해 실제 거래를 실행하지 않았다.</p>'))
css='''body{margin:0;background:#eaf0f7;color:#17263c;font:16px/1.8 "Malgun Gothic","Noto Sans KR",sans-serif}main{max-width:1180px;margin:28px auto}header,section{background:white;padding:46px 52px;margin:0 0 22px;border-radius:12px}header{background:#102843;color:white}h1{font-size:35px;line-height:1.45;margin:15px 0}h2{font-size:26px;line-height:1.5;margin:12px 0 25px}.eyebrow{font-size:12px;font-weight:700;letter-spacing:1.4px;color:#6582a6}header .eyebrow{color:#9cc3ec}p{margin:18px 0}.note,figcaption{color:#607084;font-size:13px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.cards>div{background:#f1f6fc;border-radius:10px;padding:19px}.cards b{display:block;font-size:25px;color:#176bce}.cards span{display:block;font-size:13px;margin-top:10px}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;margin:20px 0;font-size:13px}th,td{text-align:left;padding:11px 12px;border-bottom:1px solid #dce5ef;vertical-align:top}th{background:#edf3fa;color:#23496e}tr:nth-child(even){background:#f8fafd}figure{margin:26px -24px}img{display:block;width:100%;height:auto}figcaption{padding:8px 24px}.formula{background:#edf4fb;border-left:4px solid #176bce;padding:20px 25px;line-height:2;margin:24px 0;font-size:17px}a{color:#176bce}footer{padding:20px;text-align:center;color:#607084;font-size:12px}@media(max-width:700px){main{margin:0}header,section{padding:26px 20px;border-radius:0}.cards{grid-template-columns:1fr}h1{font-size:28px}h2{font-size:23px}figure{margin:20px -8px}}@media print{body{background:white}main{max-width:none;margin:0}header,section{border-radius:0;margin:0;padding:22px;break-before:page}header{break-before:auto}figure,table,.cards,.formula{break-inside:avoid}h2{break-after:avoid}footer{display:none}a{color:inherit;text-decoration:none}}'''
report='<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>BTC 롱숏 전략 수정 - 비중과 거래 빈도</title><style>'+css+'</style></head><body><main><header><div class="eyebrow">BTC RESEARCH / PORTFOLIO REVISION</div><h1>비중은 제한하고,<br>매매는 덜 자주 한다.</h1><p>삼성증권 포트폴리오 구성 자료를 참고한 BTC·현금 전략 수정<br>7일·28일 방향 학습 / 기준 비중 + 신호 / 순롱숏 비교</p><p>2026.10.14 · 업비트 BTC/KRW · 연구용 과거 검증</p></header>'+''.join(parts)+'</main><footer>숏 비율이나 거래 주기의 최적값을 확정한 결과가 아닙니다. 실제 주문은 실행하지 않았습니다.</footer></body></html>'
(O/'btc_longshort_revised_report.html').write_text(report,encoding='utf-8')
readme='''BTC 비중·거래 빈도 수정 결과 (2026-10-14)

outputs/btc_longshort_revised_report.html : 모든 그림이 내장된 보고서
outputs/*.png : 원본 그래프
work/btc_slow_protocol.json : 새 비교 규칙과 고정 시점
work/btc_slow_*_results.json : 모델·성과·비용 민감도·불확실성
work/btc_slow_*_arrays.npz : 일별 가격·예측·순보유량·목표·현금·매매·비용·평가금
work/btc_slow_*_decisions.json : 검토일별 결정과 실제 보유 구간 수익률
work/btc_slow_*_trades.json : 실제 비중 조정·최종 청산 내역
work/btc_slow_verification.json : 독립 검증 결과
work/btc_slow_overlay.py / verify_btc_slow.py / report_btc_slow.py : 계산·검증·보고서 코드
work/btc_downside_longshort.py : 기존 일별 전략과 데이터 로더

코드 실행 위치는 원래 프로젝트 루트다.
python work/btc_slow_overlay.py validation
python work/btc_slow_overlay.py test
python work/verify_btc_slow.py
python work/report_btc_slow.py

필수 패키지: numpy, scipy, matplotlib. 기존 work/model_packages, work/plot_packages 경로를 사용한다.
원자료·폰트는 용량 때문에 이 묶음에 다시 넣지 않았다. 전체 재계산에는 원래 프로젝트의
work/minute_plot_arrays.npz, work/minute_cache/2026-10.json.gz,
work/btc_cash_validation_forecasts.npz, work/btc_cash_plot_arrays.npz,
work/btc_longshort_validation_arrays.npz, work/btc_longshort_test_arrays.npz,
work/NotoSansKR-Regular.ttf 및 Bold.ttf가 필요하다.
첨부 삼성증권 PDF 원문도 이 묶음에 포함하지 않았다. 기존 첨부 원문을 참고한다.

비용은 가정값이며 순숏은 현물 가격 기반 합성 장부다. 실제 파생상품 성과가 아니다.
공매도 진입일 0일인 주간·4주 평가 결과를 숏 헤지 성공으로 해석해서는 안 된다.
'''
(W/'btc_slow_README.txt').write_text(readme,encoding='utf-8')
files=[O/'btc_longshort_revised_report.html']+list(O.glob('btc_longshort_*.png'))+list(W.glob('btc_slow_*.json'))+list(W.glob('btc_slow_*_arrays.npz'))+[W/'btc_slow_overlay.py',W/'verify_btc_slow.py',W/'report_btc_slow.py',W/'btc_downside_longshort.py',W/'btc_slow_README.txt']
with zipfile.ZipFile(O/'btc_longshort_revised_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
 for f in sorted(set(files)):z.write(f,str(f).replace('\\','/'))
print('WROTE report, three figures and bundle',len(set(files)))
