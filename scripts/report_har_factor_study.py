import os,sys,json,base64,html
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path('work/plot_packages').resolve()));sys.path.insert(0,str(Path('work/model_packages').resolve()))
os.environ['MPLCONFIGDIR']=str(Path('work/mpl_config').resolve())
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from statsmodels.tsa.stattools import acf
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
from pypdf import PdfReader
W=Path('work/har_factor_study');o=json.loads((W/'results.json').read_text(encoding='utf-8'));a=np.load(W/'daily_inputs.npz');pr=np.load(W/'predictions.npz')
z=np.load('work/minute_plot_arrays.npz');d=z['dates'];r=z['returns'].reshape(-1,1440);v=z['rv_5m'];N=len(d)
x=np.array([[np.log(v[t-1]),np.log(v[t-7:t]).mean(),np.log(v[t-30:t]).mean(),np.log(a['volume_btc'][t-1]/a['volume_btc'][t-31:t-1].mean()),np.where(r[t-1].reshape(288,5).sum(1)<0,r[t-1].reshape(288,5).sum(1)**2,0).sum()/v[t-1]] for t in range(31,N)])
for w in ['Regular','Bold']:font_manager.fontManager.addfont(f'work/NotoSansKR-{w}.ttf');pdfmetrics.registerFont(TTFont('KR'+w,f'work/NotoSansKR-{w}.ttf'))
plt.rcParams.update({'font.family':font_manager.FontProperties(fname='work/NotoSansKR-Regular.ttf').get_name(),'font.size':11,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'text.color':'#17304f','axes.labelcolor':'#17304f','figure.facecolor':'white','savefig.facecolor':'white'})
blue='#2467c8';teal='#148a7b';orange='#d3773e';colors=[blue,orange,teal];models=['HAR','Factors','HAR-X'];labels=['HAR','팩터만','HAR-X'];terms=['1일 로그 RV','7일 평균 로그 RV','30일 평균 로그 RV','거래량 충격','하락 비중'];pages=[]
def add(title,sub,fig,notes,foot):
 path=W/f'figure_{len(pages)+1:02d}.png';fig.savefig(path,dpi=150);plt.close(fig);pages.append(dict(title=title,sub=sub,figure=str(path),notes=notes,foot=foot))
def figblank():
 fig,ax=plt.subplots(figsize=(12,4));fig.subplots_adjust(left=.02,right=.98,bottom=.04,top=.96);ax.axis('off');return fig,ax
def table(ax,rows,heads,widths=None):
 t=ax.table(cellText=rows,colLabels=heads,colWidths=widths,bbox=[0,.03,1,.94],cellLoc='left');t.auto_set_font_size(False);t.set_fontsize(11)
 for (i,j),c in t.get_celld().items():c.set_edgecolor('#e1e8f1');c.set_facecolor('#dfeaf9' if i==0 else '#f2f6fb' if i%2 else 'white')
 return t
def pval(p):return '<0.0001' if p<.0001 else f'{p:.4f}'
fig,ax=figblank();table(ax,[['1. 입력','거래량 충격 + 하락 변동성 비중','오늘까지 관측 → 내일 로그 RV'],['2. 진단','정상성 · 상관 · VIF','학습 자료에서만 확인'],['3. 추정','HAR / 팩터만 / HAR-X','같은 표본 · OLS · 고정 계수'],['4. 오류 점검','HAC · 잔차 · 구조 안정성','문제가 있으면 결과 해석에 반영'],['5. 평가','QLIKE 우선 · MSE 보조','2024-25 검증 / 2026 사후 평가'],['6. 민감도','기간 · 결측 · 영향점 · RV 간격','진단 후 설정을 최적화하지 않음']],['과정','실행 항목','원칙'],[.16,.42,.42])
add('HAR에 팩터를 추가할 가치가 있는가?','한 번의 사전 지정 비교 | 경제적 설명과 실제 예측 개선을 분리',fig,['팩터 두 개와 모형 세 개를 고정한 뒤 분석했다. 단일 팩터 최적화·포트폴리오 구성은 하지 않았다.','검증은 2023년까지 학습, 평가는 2025년까지 재학습. 2026년은 이미 본 자료이므로 사후 연구다.'],'공통 시작: 원자료 첫날 +31일. 기존 HAR의 +30일보다 하루 늦춰 거래량 기준창을 맞춤. 계수 고정, 입력은 매일 갱신.')
fig,axs=plt.subplots(2,1,figsize=(12,4),sharex=True);fig.subplots_adjust(left=.09,right=.98,top=.94,bottom=.12,hspace=.22)
for ax,idx,c,lab in [(axs[0],3,blue,'거래량 충격'),(axs[1],4,teal,'하락 변동성 비중')]:ax.plot(d[31:],x[:,idx],lw=.6,color=c);ax.set_ylabel(lab);ax.grid(alpha=.15)
add('거래활동과 하락 방향의 구성 정보를 추가한다','업비트 BTC/KRW 완결 UTC 일자 2017-09-26~2026-10-11 | 5분 RV',fig,['거래량 충격 = log(당일 BTC 거래량 / 그 이전 30일 평균). 하락 비중 = 음수 5분 수익률 제곱합 / RV.','가격 결측은 기존 처리 유지, 거래량은 관측 거래만 합산했다. 무거래와 수집 누락은 캐시만으로 구분 불가.'],f"캐시 {o['quality']['cache_files']}개, 중복 {o['quality']['duplicate_rows']}건, 잘못된 거래량 {o['quality']['invalid_volume_rows']}건. 미관측 분 {o['quality']['missing_minutes']:,}개. 팩터는 목표일 이전 값만 사용.")
fig,ax=figblank();rows=[]
for key,lab in [('target_logRV','목표 로그 RV')]+list(zip(['logRV_1d','mean_logRV_7d','mean_logRV_30d','volume_shock','downside_share'],terms)):
 v1=o['phases']['validation']['stationarity'][key];v2=o['phases']['test']['stationarity'][key]
 rows.append([lab,pval(v1['adf_p']),pval(v1['kpss_p']),pval(v2['adf_p']),pval(v2['kpss_p'])])
table(ax,rows,['변수','ADF ~2023','KPSS ~2023','ADF ~2025','KPSS ~2025'],[.28,.18,.18,.18,.18])
add('정상성은 두 검정과 시간 흐름을 함께 본다','학습 구간에서만 검사 | ADF: 단위근 가설 / KPSS: 수준 정상성 가설',fig,['ADF p가 작으면 단위근 가설을 기각, KPSS p가 작으면 수준 정상성 가설을 기각한다.','두 결과가 엇갈릴 수 있다. 구조 변화·지속성 가능성을 함께 보며, 자동 차분·변수 삭제는 하지 않았다.'],'ADF 시차 AIC, KPSS auto. KPSS p=0.01 또는 0.10은 표의 경계값일 수 있음. 원 경고는 상세 결과 JSON에 보존.')
fig,axs=plt.subplots(1,2,figsize=(12,4));fig.subplots_adjust(left=.15,right=.97,top=.94,bottom=.22,wspace=.45)
corr=np.array(o['phases']['test']['correlations']);im=axs[0].imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r');axs[0].set_xticks(range(5),['1일','7일','30일','거래량','하락']);axs[0].set_yticks(range(5),terms)
for i in range(5):
 for j in range(5):axs[0].text(j,i,f'{corr[i,j]:.2f}',ha='center',va='center',fontsize=9,color='white' if abs(corr[i,j])>.65 else '#17304f')
vals=list(o['phases']['test']['vif'].values());axs[1].barh(terms,vals,color=blue);axs[1].invert_yaxis();axs[1].set_xlim(0,max(vals)*1.23);axs[1].set_xlabel('VIF · 임계값만으로 삭제하지 않음')
for i,n in enumerate(vals):axs[1].text(n+.05,i,f'{n:.2f}',va='center')
add('변수 중복: HAR의 겹치는 평균과 팩터를 구분','2025년까지 학습 구간 | 상수 포함 VIF, 표준화된 설계행렬',fig,[f'추가 팩터 VIF: 거래량 {vals[3]:.2f}, 하락 비중 {vals[4]:.2f}. HAR 평균창은 서로 겹친다.','VIF는 개별 계수 추정의 불안정성 지표다. 낮은 VIF 자체가 유의성이나 예측력을 뜻하지 않는다.'],'표준화 평균·표준편차는 각 학습 구간에서만 추정. 검증 학습 구간의 VIF·상관행렬도 상세 결과에 저장.')
fig,ax=figblank();diag=o['phases']['test']['models']['HAR-X']['diagnostics'];rows=[]
for i,name in enumerate(['상수']+terms):
 lo,hi=diag['hac7_ci'][i];rows.append([name,f"{diag['coefficients_standardized'][i]:.4f}",f'[{lo:.4f}, {hi:.4f}]',pval(diag['hac7_p'][i])])
table(ax,rows,['변수','계수','HAC 95% 신뢰구간','p-value'],[.25,.2,.35,.2])
add('계수 유의성과 예측 개선은 별개의 질문이다','HAR-X · 2025년까지 학습 | 설명변수 1표준편차 증가당 로그 RV 변화',fig,[f"두 팩터 공동 Wald p={pval(diag['factor_joint_wald_p'])}. 개별 Holm 보정 p: 거래량 {pval(diag['factor_holm_p'][0])}, 하락 {pval(diag['factor_holm_p'][1])}.",'HAC는 추론의 표준오차를 보정한다. 잘못된 함수 형태·내생성·구조 변화를 고치거나 인과성을 증명하지 않는다.'],'HAC 7일 시차, 1·30일 시차 민감도도 저장. 개별 팩터 2개에만 Holm 적용; 나머지 진단 p는 탐색적이다.')
fig,ax=figblank();rows=[]
for model,label in zip(models,labels):
 dg=o['phases']['test']['models'][model]['diagnostics']
 rows.append([label,pval(dg['lb_p'][0]),pval(dg['lb_p'][1]),pval(dg['arch_lm_p']),pval(dg['reset_hac_p']),pval(dg['cusum_p'])])
table(ax,rows,['모형','LB 7일','LB 30일','ARCH-LM','RESET','CUSUM'],[.15,.17,.17,.17,.17,.17])
add('잔차와 구조 안정성: 작은 p값은 경고이지 자동 탈락은 아니다','2025년까지 학습 잔차 | 각 열은 서로 다른 귀무가설의 p값',fig,['LB는 잔차 자기상관, ARCH-LM은 조건부 이분산성, RESET은 함수 형태 누락 가능성을 살핀다.','CUSUM은 계수 안정성 진단이다. 강한 의존·이분산 하에서는 명목 p값에 한계가 있어 롤링 계수도 확인한다.'],'잔차 정규성(Jarque-Bera)은 참고용으로 JSON 저장; 정규성은 OLS 필수 조건이 아님. 로그 예측의 분산 복원은 학습 잔차 smearing 사용.')
fig,axs=plt.subplots(1,2,figsize=(12,4));fig.subplots_adjust(left=.07,right=.98,top=.88,bottom=.16,wspace=.3)
for ax,metric,mult,title in [(axs[0],'qlike',1,'QLIKE ↓'),(axs[1],'mse',1e6,'MSE × 백만 ↓')]:
 for j,(phase,lab,c) in enumerate([('validation','검증 2024-25','#aab7c8'),('test','평가 2026',blue)]):
  vals=[o['phases'][phase]['models'][m]['scores'][metric]*mult for m in models];ax.bar(np.arange(3)+(j-.5)*.32,vals,width=.30,label=lab,color=c)
  for k,n in enumerate(vals):ax.text(k+(j-.5)*.32,n+max(vals)*.02,f'{n:.4f}',ha='center',fontsize=9)
 ax.set_xticks(range(3),labels);ax.set_title(title);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.set_ylim(0,ax.get_ylim()[1]*1.13);ax.legend(fontsize=9)
t=o['phases']['test'];h=t['models']['HAR']['scores']['qlike'];hx=t['models']['HAR-X']['scores']['qlike'];ci=t['HAR-X_minus_HAR']['QLIKE']['7']['ci95']
add('최종 판단은 동일 날짜의 예측 오차로 한다','같은 학습 표본과 평가 날짜 | 검증 731일 · 평가 284일',fig,[f'2026 HAR-X의 QLIKE 변화: HAR 대비 {(hx/h-1)*100:+.2f}%. 음수일수록 개선이다.',f'HAR-X − HAR 손실차 95% 구간(7일 블록): [{ci[0]:.4f}, {ci[1]:.4f}].'],'5,000회 순환 블록 부트스트랩. 고정된 예측 손실의 기술적 신뢰구간; 재학습·모형 선택 불확실성을 반영한 포함모형 검정은 아님. 30일 블록도 저장.')
fig,ax=figblank();rows=[]
for yr in ['2024','2025','2026']:
 phase='test' if yr=='2026' else 'validation';rows.append([yr]+[f"{o['phases'][phase]['models'][m]['annual'][yr]['qlike']:.4f}" for m in models])
for lab,key in [('2026 고품질 날짜','high_coverage_scores'),('2026 영향점 민감도','influence_sensitivity_scores')]:rows.append([lab]+[f"{t['models'][m][key]['qlike']:.4f}" for m in models])
rows.append(['2026 10분 RV']+[f"{o['rv10_sensitivity']['test'][m]['qlike']:.4f}" for m in models])
table(ax,rows,['조건 / QLIKE ↓']+labels,[.40,.20,.20,.20])
add('결론이 기간·데이터 처리에 따라 바뀌는가?','민감도 분석은 견고성 확인용 | 가장 좋은 설정을 사후 채택하지 않음',fig,['고품질: 목표일·직전일 관측률 99% 이상. 영향점: 학습 Cook 거리 상위 1% 제외 후 재학습한 보조 결과.','10분 RV는 목표 변수와 하락 비중을 함께 다시 계산했다. 행 사이의 손실을 직접 비교하지 않는다.'],'급락일은 본 분석에서 제거하지 않음. 영향점 민감도는 각 모형별 학습 Cook 거리로 계산. 전체 자료 결과가 주 분석.')
fig,axs=plt.subplots(1,2,figsize=(12,4));fig.subplots_adjust(left=.10,right=.98,bottom=.17,top=.9,wspace=.3)
s=o['rolling_coefficient_diagnostic']
for j,(ax,label) in enumerate(zip(axs,['거래량 충격','하락 비중'])):
 co=np.array([r['factor_coef_original_units'][j] for r in s]);interval=np.array([r['ci95_original_units'][j] for r in s]);ax.errorbar([r['cutoff'] for r in s],co,yerr=np.array([co-interval[:,0],interval[:,1]-co]),fmt='o-',color=colors[j],capsize=4);ax.axhline(0,color='#777',lw=.8);ax.set_title(label);ax.set_xlabel('학습 종료 다음 해');ax.set_ylabel('원 단위 계수 · HAC 95% 구간');ax.grid(alpha=.15)
add('팩터 관계가 시기에 따라 유지되는지도 확인한다','매년 직전 3년으로 HAR-X 재추정 | 예측 모형 선택에 사용하지 않은 보조 진단',fig,['계수의 부호·크기·불확실성이 기간에 따라 어떻게 바뀌는지 확인한다. 공통 효과로 단정하지 않는다.','본문의 고정 모형 평가와 이 롤링 계수 진단은 구분된다. 롤링 예측 성능 우위를 검증한 것은 아니다.'],'추가 분석 전용 보고서. 결론은 기존 HAR와의 추가 정보 비교이며, 경제적 인과관계나 투자 수익을 입증하지 않음.')
fig,ax=figblank()
table(ax,[['팩터만','HAR보다 검증·평가 오차가 큼','이 두 팩터만으로 HAR를 대체할 근거 없음'],['HAR-X','2026 QLIKE 1.13% · MSE 5.55% 개선','QLIKE 손실차 신뢰구간은 0 포함'],['거래량 충격','양수 계수 · Holm 보정 p < 0.0001','추가 정보 후보; 단독 추가의 효과는 미분리'],['하락 비중','Holm 보정 p = 0.420','현재 식에서는 추가 설명력이 뚜렷하지 않음'],['잔차 / 정상성','자기상관·이분산 잔존 / 로그 RV 검사 상충','HAC가 모형 자체의 누락을 해결한 것은 아님'],['예측 수준','2026 HAR-X 평균 분산 예측 약 15.9% 높음','분산 복원 보정과 기간별 편향도 추가 점검 대상']],['항목','관측 결과','해석'],[.17,.39,.44])
add('HAR를 기준으로 유지하고 거래량 확장을 후속 검증한다','가정이 모두 만족된 완성 모형이라는 결론은 아님',fig,['팩터만보다 HAR가 유리했고, HAR-X의 개선 폭은 작았다. 거래량은 후보로 남기되 최종 채택은 보류한다.','다음 실험은 거래량만 추가한 모형과 잔차 구조 보완을 별도 검증한다. 이번 결과로 설정을 바꾸지는 않았다.'],'95% 손실차 구간은 기술적 분석. MSE 구간도 블록 7일에서는 0 미포함, 30일에서는 포함. 2026은 새 블라인드 테스트가 아님.')
P=Path('output/pdf/har_factor_study.pdf');c=Canvas(str(P),pagesize=(960,540));c.setTitle('BTC HAR와 팩터 모형 - 통계 진단과 예측 비교')
def txt(x,y,s,size=12,bold=False):
 font='KRBold' if bold else 'KRRegular'
 while pdfmetrics.stringWidth(s,font,size)>868:size-=.2
 assert size>=7.5
 c.setFont(font,size);c.setFillColor(HexColor('#17304f'));c.drawString(x,y,s)
for i,p in enumerate(pages,1):
 c.setFillColor(HexColor('#f6f9fc'));c.rect(0,0,960,540,fill=1,stroke=0);c.setFillColor(HexColor(blue));c.rect(0,531,960,9,fill=1,stroke=0)
 txt(46,501,f'BTC RESEARCH / HAR-X DIAGNOSTICS                                      {i:02d} / {len(pages):02d}',10)
 txt(46,462,p['title'],25,True);txt(46,435,p['sub'],11)
 c.drawImage(p['figure'],48,139,width=864,height=288)
 for j,line in enumerate(p['notes']):txt(46,110-j*28,line,12.8,j==0)
 txt(46,42,p['foot'],8);txt(46,19,'실행 2026-10-15 | 업비트 BTC/KRW 기존 캐시 | 결과·프로토콜·진단 상세는 동봉 JSON',8)
 c.showPage()
c.save();pdf=PdfReader(P);assert len(pdf.pages)==len(pages)
for page,p in zip(pdf.pages,pages):assert p['title'] in page.extract_text()
css='body{background:#e9eff7;color:#17304f;font:17px/1.7 "Malgun Gothic",sans-serif;margin:0}main{max-width:1240px;margin:24px auto}section{background:#f6f9fc;border-top:8px solid #2467c8;padding:32px 42px;margin:24px 0}h1{font-size:29px}img{width:100%}.sub,.foot{color:#60758f}.sub{font-size:14px}.foot{font-size:12px;border-top:1px solid #dae3ef;padding-top:12px}'
body=[]
for p in pages:
 img=base64.b64encode(Path(p['figure']).read_bytes()).decode();body.append('<section><h1>'+html.escape(p['title'])+'</h1><p class="sub">'+html.escape(p['sub'])+'</p><img src="data:image/png;base64,'+img+'" alt="'+html.escape(p['title'])+'">'+''.join('<p>'+html.escape(n)+'</p>' for n in p['notes'])+'<p class="foot">'+html.escape(p['foot'])+'</p></section>')
Path('outputs/har_factor_study.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>BTC HAR-X 통계 진단</title><style>'+css+'</style><main>'+''.join(body)+'</main></html>',encoding='utf-8')
(W/'report_pages.json').write_text(json.dumps(pages,ensure_ascii=False,indent=2),encoding='utf-8')
print('Report ready',len(pages),P)
