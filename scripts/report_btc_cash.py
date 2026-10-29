from pathlib import Path
import json,base64,html,csv,math,zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'; WORK=ROOT/'work'
S=json.loads((WORK/'btc_cash_results.json').read_text(encoding='utf-8'))
V=json.loads((WORK/'btc_cash_verification.json').read_text(encoding='utf-8'))
PRIMARY=S['selection']['winner']
P={r['strategy']:r for r in S['portfolio_metrics'] if r['period']=='test_2026'}
F={r['model']:r for r in S['test_forecast_scores']}
VAL={r['model']:r for r in S['selection']['leaderboard']}
N={'HAR_logRV_all':'HAR · 전체 학습 (주 모형)','HAR_logRV_3y':'HAR · 최근 3년',
   'GARCH_normal_all':'GARCH 정규 · 전체','GARCH_normal_3y':'GARCH 정규 · 최근 3년',
   'GARCH_t_all':'GARCH t · 전체','GARCH_t_3y':'GARCH t · 최근 3년',
   'GJR_t_all':'GJR t · 전체','GJR_t_3y':'GJR t · 최근 3년',
   'RV_last':'어제 RV','RV_mean7':'최근 7일 평균 RV','RV_mean30':'최근 30일 평균 RV',
   'BTC_buy_hold':'BTC 100% 보유','BTC50_cash50_daily':'BTC 50% · 현금 50% 고정',
   'fixed_validation_mean_weight':'BTC 46.1% · 현금 53.9% 고정','cash':'현금 100%'}
def esc(x): return html.escape(str(x))
def pct(x):return f'{100*x:+.2f}%'
def ppos(x):return f'{100*x:.2f}%'
def table(headers,rows):
    return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def img(name,alt):
    b=base64.b64encode((OUT/(name+'.png')).read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{b}" alt="{esc(alt)}"></figure>'

main_keys=[PRIMARY,'fixed_validation_mean_weight','BTC50_cash50_daily','BTC_buy_hold','cash']
main_table=table(['전략','누적 수익률','연환산 변동성','최대낙폭 (일 종가)','평균 BTC 비중','마지막 평가액','지급 비용'],[
    [N[k],pct(P[k]['total_return']),ppos(P[k]['annual_vol']),pct(P[k]['max_drawdown']),ppos(P[k]['average_btc_weight']),f"{P[k]['final_nav']:,.0f}원",f"{P[k]['fees_paid_krw']:,.0f}원"] for k in main_keys])
model_order=[r['model'] for r in S['selection']['leaderboard']]
forecast_table=table(['모형','2024–25 QLIKE','2026 QLIKE','2026 분산 MSE','2026 변동성 MAE (％p)'],[
    [N[k],f"{VAL[k]['qlike']:.4f}",f"{F[k]['qlike']:.4f}",f"{F[k]['mse']:.4e}",f"{F[k]['vol_mae_pct']:.4f}"] for k in model_order])
all_portfolio=table(['전략','누적 수익률','연환산 변동성','일별 최대낙폭','평균 BTC 비중','지급 비용'],[
    [N[k],pct(P[k]['total_return']),ppos(P[k]['annual_vol']),pct(P[k]['max_drawdown']),ppos(P[k]['average_btc_weight']),f"{P[k]['fees_paid_krw']:,.0f}원"] for k in model_order])
cost_table=table(['편도 비용','HAR 수익률','46.1% 고정 수익률','BTC 100% 수익률'],[
    [f'{c*100:.2f}%']+[pct(next(r['total_return'] for r in S['cost_sensitivity'] if r['strategy']==k and r['cost']==c)) for k in [PRIMARY,'fixed_validation_mean_weight','BTC_buy_hold']] for c in [0,.0005,.001,.0025]])
params=[]
for k in ['GARCH_normal_all','GARCH_t_all','GJR_t_all','GARCH_normal_3y','GARCH_t_3y','GJR_t_3y']:
    q=S['test_parameters'][k];p=q['parameters_percent_units']
    params.append([N[k],f"{p['omega']:.4f}",f"{p['alpha[1]']:.4f}",f"{p['beta[1]']:.4f}",f"{p.get('gamma[1]',0):.4f}",f"{q['persistence']:.6f}",f"{p['nu']:.3f}" if 'nu' in p else '정규'])
param_table=table(['모형','ω','α','β','γ','지속성','자유도 ν'],params)
quality_table=table(['UTC 날짜','실제 캔들 비율','전 가격으로 채운 분','가장 긴 연속 구간'],[[x['date'],ppos(x['coverage']),str(x['filled_minutes']),str(x['max_stale_minutes'])+'분'] for x in V['low_coverage_days']])

body=r'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BTC와 현금 — 변동성 예측 투자 실험과 상세 해설</title><style>
:root{--ink:#17263c;--muted:#536477;--blue:#2355a4;--border:#dce4ed;--teal:#17847a}
*{box-sizing:border-box}body{margin:0;background:#f2f5fa;color:var(--ink);font:17px/1.85 "Malgun Gothic","Noto Sans KR",Arial,sans-serif}
main{max-width:1140px;margin:auto;padding:50px 46px 70px;background:#fff}header{border-bottom:3px solid var(--ink);padding-bottom:24px;margin-bottom:26px}
h1{font-size:36px;line-height:1.35;letter-spacing:-1px;margin:8px 0 20px}h2{font-size:25px;margin:50px 0 15px;line-height:1.5}h3{font-size:20px;margin:25px 0 12px}p{margin:12px 0}strong{color:#123960}
a{color:var(--blue);text-decoration:underline;text-underline-offset:3px}small,.muted{color:var(--muted);font-size:14px}
.eyebrow{font-weight:bold;color:var(--teal);font-size:13px;letter-spacing:1px}.lead{font-size:20px;line-height:1.8}.callout{background:#edf3fb;border-left:4px solid var(--blue);padding:18px 23px;margin:25px 0}
.note{background:#f7f9fc;padding:17px 22px;border:1px solid var(--border);border-radius:7px}.formula{background:#f3f6fa;border:1px solid var(--border);padding:16px 19px;font-family:Consolas,"Malgun Gothic",monospace;overflow-x:auto;white-space:pre-wrap;font-size:16px;line-height:1.8}
figure{margin:22px -15px 10px}figure img{display:block;width:100%;height:auto}.caption{border-left:3px solid #b2c4db;padding:2px 0 2px 20px;margin:10px 0 30px}
.table-wrap{overflow-x:auto;margin:20px 0}table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.55}th{background:#eaf0f8;color:#173a62;text-align:left;white-space:nowrap}td,th{padding:12px 11px;border-bottom:1px solid var(--border);vertical-align:top}tr:nth-child(even) td{background:#fafbfd}td:first-child{min-width:170px}td:not(:first-child){font-variant-numeric:tabular-nums}
nav{display:flex;gap:12px 25px;flex-wrap:wrap;font-size:14px;margin-top:22px}ul,ol{padding-left:25px}li{margin:8px 0}details{border:1px solid var(--border);border-radius:6px;padding:16px 20px;margin:20px 0}summary{font-weight:bold;cursor:pointer;color:var(--blue)}code{font-family:Consolas,monospace;overflow-wrap:anywhere;font-size:.9em}footer{border-top:1px solid var(--border);margin-top:50px;padding-top:20px}
@media(max-width:700px){main{padding:25px 20px}h1{font-size:28px}h2{font-size:23px}figure{margin-left:-15px;margin-right:-15px}body{font-size:16px}table{font-size:13px}}
@media print{body{background:white;font-size:11pt}main{max-width:none;padding:0}nav{display:none}h2{break-after:avoid}figure,table,.formula{break-inside:avoid}.table-wrap{overflow:visible}details{display:block}a{color:inherit}}
</style></head><body><main>
<header><div class="eyebrow">BTC / KRW · CASH · RESEARCH NOTE · 2026-10-13</div>
<h1>변동성을 예측하면<br>BTC와 현금을 더 잘 배분할 수 있을까?</h1>
<p class="lead"><strong>HAR의 QLIKE는 단순 예측보다 좋아졌지만, 이번 비중 조절 전략은 고정 비중 전략보다 수익률과 최대낙폭이 나빴다.</strong> 예측 평가와 투자 평가를 별도로 확인해야 한다는 결과다.</p>
<p class="muted">업비트 KRW-BTC 1분 자료에서 일별 위험을 측정 · 평가: 2026-01-01–10-11 UTC, 284일 · 초기 1,000만원 · 편도 비용 0.10% · 현금 이자 0%</p>
<nav><a href="#design">계산 순서</a><a href="#models">모형의 의미</a><a href="#equity">투자 성과</a><a href="#weights">예측과 비중</a><a href="#scores">QLIKE·MSE</a><a href="#costs">거래비용</a><a href="#metrics">지표 사전</a><a href="#longshort">롱숏 확장</a><a href="#data">검증·데이터</a></nav></header>

<div class="callout"><strong>이 보고서에서 확인한 것</strong><br>2024–2025년 예측 손실로 주 모형을 먼저 선택하고, 2026년에는 그 선택과 투자 규칙을 바꾸지 않았다. 다만 2026년 자료는 앞선 탐색 분석에서 이미 살펴봤으므로, 완전히 처음 보는 자료를 이용한 실시간 검증으로 부를 수는 없다.</div>
@@MAIN_TABLE@@
<p>HAR는 ‘어제의 RV를 그대로 내일 예측으로 사용하는 방법’보다 QLIKE가 <strong>32.4% 낮았다</strong>. 반면 분산의 MSE는 <strong>3.3% 높았다</strong>. 따라서 모든 오차 기준에서 좋아졌다고 말할 수 없다.</p>
<p>투자에서는 HAR의 누적 수익률이 <strong>−6.32%</strong>, BTC 46.1% 고정이 <strong>−4.11%</strong>, BTC 50% 고정이 <strong>−4.56%</strong>였다. HAR의 연환산 변동성은 더 낮았지만, 일별 최대낙폭은 <strong>−23.32%</strong>로 두 고정 비중 전략보다 컸다. ‘평소 덜 흔들림’과 ‘최고점에서 덜 떨어짐’은 서로 다른 특성이다.</p>

<h2 id="design">1. 어떤 자료로, 어떤 순서로 계산했나</h2>
<p>사용 자산은 <strong>업비트 원화 비트코인과 원화 현금</strong>이다. 기존에 수집한 1분 캔들 중 하루가 완성된 2017-09-26–2026-10-11 UTC의 3,303일을 사용했다. S&amp;P 500, 채권, 달러 환율은 입력하지 않았다. ‘전체 자료를 사용한다’는 말은 미래의 2026년 자료까지 미리 학습한다는 뜻이 아니다.</p>
<ol><li><strong>최초 학습: 2017-09-26–2023-12-31.</strong> 모형 계수를 추정했다. 최근 3년 모형은 2021–2023년으로 추정했다.</li>
<li><strong>모형 선택: 2024–2025년, 731일.</strong> 계수를 고정하고 매일 전날까지 관측한 값으로 다음 날 위험을 예측했다. 평균 QLIKE가 가장 낮은 <strong>전체 기간 학습 HAR</strong>를 주 모형으로 선택했다. 투자 수익률로 모형을 고르지 않았다.</li>
<li><strong>최종 학습: 2025-12-31까지.</strong> 선택한 HAR를 2017년부터 다시 추정했다. 최근 3년 비교 모형은 2023–2025년으로 추정했다.</li>
<li><strong>최종 평가: 2026-01-01–10-11, 284일.</strong> 계수는 고정하고, 매일 관측되는 과거 수익률과 RV만 갱신했다. 평가 결과를 본 뒤 계수·목표 변동성·비용·주 모형을 바꾸지 않았다.</li></ol>
<p>UTC 하루는 한국시간 오전 9시부터 다음 날 오전 9시까지다. 전날 자료가 09:00에 완성된다고 보고, <strong>09:01의 1분 캔들 종가로 매매</strong>했다. 첫 1분 동안은 기존 BTC 수량이 수익을 얻고, 재조정 후에는 새 수량이 나머지 시간의 가격 변화를 받는다. 같은 종가를 관측하고 동시에 그 가격으로 미리 거래하는 오류를 피하기 위한 지연이다. 다만 캔들 종가 체결은 호가·주문 대기열까지 재현한 체결 모형은 아니다.</p>

<h3>1분 자료인데 왜 ‘5분 RV’를 쓰나?</h3>
<p>데이터는 1분 간격으로 보유하고 있다. 여기서 5개씩 겹치지 않게 묶은 로그수익률 288개로 하루 위험을 측정했다. 5분은 위험 <strong>측정 간격</strong>이고, 예측 대상과 비중 조절 주기는 <strong>하루</strong>다.</p>
<div class="formula">1분 로그수익률: r = log(현재 가격 / 직전 가격)
5분 로그수익률: r⁽⁵⁾ = 해당 5개의 1분 로그수익률 합
하루 실현분산: Vₜ = Σⱼ₌₁²⁸⁸ (r⁽⁵⁾ₜ,ⱼ)²
하루 측정 변동성: σₜ = √Vₜ</div>
<p>예를 들어 V가 0.0004이면 √V는 0.02이므로 측정 변동성은 2%다. 이것은 그날 수익률이 +2%였다는 뜻도, 다음 날 ±2% 안에 반드시 들어온다는 뜻도 아니다. 하루 안에서 일어난 가격 움직임의 크기를 요약한 값이다.</p>
<p>5분 RV는 매우 촘촘한 1분 RV에 비해 미시구조 잡음의 영향을 줄이려는 간단한 선택이며, 실제 잠재 분산 그 자체는 아니다. 이번에는 급격한 점프의 위험도 투자 손익에 반영해야 하므로, 점프를 제거하는 PRV 대신 점프가 포함된 RV를 평가 기준으로 삼았다.</p>

<h2 id="models">2. HAR, GARCH, GJR는 각각 무엇을 예측하나</h2>
<p>여기서 모형은 모두 <strong>다음 날 위험의 크기</strong>를 예측한다. GARCH의 평균 수익률 항도 추정하지만, 이번 전략은 그 항으로 매수·매도 방향을 정하지 않았다.</p>
<h3>HAR: 단기·중기·장기 변동성 정보를 합치는 회귀</h3>
<div class="formula">xₜ = log(Vₜ)
예측 로그분산 = b₀ + b₁xₜ + b₇·평균(xₜ,…,xₜ₋₆)
                         + b₃₀·평균(xₜ,…,xₜ₋₂₉)
다음 날 분산 예측 fₜ₊₁ = exp(예측 로그분산) × smearing 계수</div>
<p><strong>어제 하루가 얼마나 위험했는지, 최근 일주일이 얼마나 위험했는지, 최근 한 달이 얼마나 위험했는지</strong>를 합쳐 내일을 예측한다. BTC는 주말도 거래하므로 이번에는 1·7·30 <strong>달력일</strong>을 사용했다. 전통적인 주식 HAR의 거래일 설정을 그대로 복제한 것은 아니다. 또한 ‘RV 평균의 로그’가 아니라 ‘RV 로그의 평균’을 사용했다.</p>
<p>로그로 예측한 값을 단순히 지수 변환하면 원래 분산 척도의 평균 예측과 차이가 생긴다. 이를 보정하기 위해 <strong>학습 구간 회귀 잔차에 지수를 취한 값의 평균</strong>을 곱했다. 이를 Duan smearing이라고 한다. 테스트 구간의 오차로 이 보정값을 고르지 않았다.</p>
<p>최종 HAR 계수는 b₀ = −0.51045, b₁ = 0.48597, b₇ = 0.31343, b₃₀ = 0.13218, smearing = 1.35672다. 회귀 입력끼리 겹치므로 이 숫자를 단순히 ‘하루 정보가 48.6% 기여한다’고 해석하면 안 된다. <a href="https://statmath.wu.ac.at/~hauser/LVs/FinEtricsQF/References/Corsi2009JFinEtrics_LMmodelRealizedVola.pdf">HAR 원 논문</a>의 다중 기간 아이디어를 BTC의 달력일 자료에 적용한 비교 모형이다.</p>

<h3>GARCH: 어제 충격과 어제까지의 위험 상태를 이어받는 모형</h3>
<div class="formula">일별 로그수익률 rₜ = μ + εₜ
εₜ = √hₜ · zₜ
hₜ₊₁ = ω + α·εₜ² + β·hₜ</div>
<p>h는 해당 날짜의 <strong>조건부 분산</strong>이다. ω는 기본 분산 항, α는 새 충격의 제곱에 반응하는 정도, β는 이전 위험 상태를 유지하는 정도다. α·ε² 항은 수익률의 부호를 제곱으로 지우므로 큰 상승과 큰 하락 모두 위험을 올릴 수 있다.</p>
<p>학습에서 정한 α·β 등의 계수로 테스트 구간의 h를 하루씩 전진시킨다. 2026년의 실제 수익률을 하나 관측하면 그 다음 날 예측에만 사용한다. 테스트 기간 전체를 한꺼번에 보고 미리 h를 만들지 않는다.</p>
<p><strong>GARCH 정규</strong>는 표준화 충격 z의 분포를 정규분포로 두고, <strong>GARCH t</strong>는 꼬리가 두꺼운 Student t 분포로 둔다. t의 자유도 ν가 작을수록 극단적인 충격에 더 많은 확률을 둔다. t 모형을 쓴다고 상승·하락의 방향까지 예측해 주지는 않는다. 모형 정의는 <a href="https://econ.duke.edu/~boller/Published_Papers/joe_86.pdf">GARCH 원 논문</a>과 <a href="https://arch.readthedocs.io/en/latest/univariate/univariate_volatility_modeling.html">arch 공식 모형 설명</a>을 참고했다.</p>

<h3>GJR-GARCH t: 충격의 부호에 따라 위험 반응을 달리 허용</h3>
<div class="formula">hₜ₊₁ = ω + α·εₜ² + γ·1(εₜ &lt; 0)·εₜ² + β·hₜ</div>
<p>γ 항은 음의 충격일 때만 추가된다. γ가 양수이면 같은 크기의 음의 충격이 양의 충격보다 미래 분산을 더 올린다. 그러나 <strong>γ가 언제나 양수라고 가정하지 않았다</strong>. 이번 전체 학습에서는 −0.00887, 최근 3년에서는 +0.11748로 추정됐다. 이 결과만으로 BTC의 비대칭성이 안정적이라고 단정할 수 없다.</p>
<details><summary>추정 계수와 모형 한계 자세히 보기</summary>@@PARAM_TABLE@@
<p>추정을 위해 일별 로그수익률에 100을 곱했다. 표의 ω는 그 백분율 수익률의 제곱 단위이며, 최종 예측 분산은 다시 10,000으로 나눠 소수 수익률 단위로 환산했다. α·β·γ는 무차원이다.</p>
<p>GARCH 지속성은 α+β, 대칭 혁신분포를 사용하는 GJR는 α+β+γ/2로 계산했다. t 모형들은 약 1의 경계에 걸렸다. 최적화는 수렴했고 하루 앞 분산은 양수지만, 이를 안정적인 유한 장기 평균 분산이 확인된 것으로 해석하면 안 된다. ω/(1−지속성)로 장기 분산을 계산하지 않았다. 자유도도 약 2.6–3.0으로 낮아 극단값에 대한 결과 민감성이 크다.</p>
<p>각 모형은 기본값을 포함한 3개의 정해진 시작값으로 적합해, 수렴한 결과 중 우도가 가장 높은 것을 사용했다. 학습 자료에서 우도를 비교했으며 테스트 투자 수익률로 추정값을 선택하지 않았다.</p></details>
<h3>‘어제 RV’, ‘7일 평균’, ‘30일 평균’은 무엇인가?</h3>
<p>복잡한 모형 없이 fₜ₊₁ = Vₜ, 또는 최근 7·30일 V의 산술평균을 사용하는 기준선이다. HAR의 평균 로그 입력과는 다르다. 복잡한 모형이 단순한 기준선보다 얼마나 나은지를 확인하기 위해 포함했다.</p>

<h2 id="equity">3. 그래프 1 — 계좌 평가액과 최고점 대비 낙폭</h2>
@@FIG_EQUITY@@
<div class="caption"><p><strong>위 패널:</strong> 가로축은 2026년 날짜, 세로축은 계좌에 남은 원화 현금과 BTC 평가액의 합이다. 모든 선이 1,000만원에서 출발한다. 예를 들어 950만원이면 누적 −5%다. 현금 선은 이자를 0%로 두었으므로 1,000만원에서 수평이다.</p>
<p><strong>파란 HAR 선:</strong> 위험 예측에 따라 BTC 비중을 매일 바꾼 전략이다. <strong>초록·회색 선:</strong> 각각 BTC 46.1%와 50%가 되도록 매일 재조정한다. <strong>주황 선:</strong> 시작할 때 전액 BTC를 사고 끝까지 보유한 전략이다. 첫 진입과 마지막 청산 비용도 반영했다.</p>
<p><strong>아래 패널:</strong> 각 전략이 자기 자신의 과거 최고 평가액에서 얼마나 내려왔는지 표시한다. Dₜ = Wₜ / max(W₀,…,Wₜ) − 1로 계산한다. 선이 0이면 최고점을 기록 중이고, −20%이면 이전 최고점보다 20% 낮다. 처음 투자한 원금 대비 손실률과는 다르다.</p>
<p><strong>이 그래프의 결론:</strong> HAR는 BTC 전액 보유보다 손실과 낙폭이 작았다. 그러나 그중 상당 부분은 BTC를 덜 보유해서 생길 수 있는 효과다. 평균 BTC 비중이 50.5%였던 HAR를 50% 고정과 비교하면, 단순히 현금을 함께 보유하는 것보다 더 좋은 투자 결과가 확인되지 않았다. HAR의 일별 변동성은 낮았지만 누적 손실과 최대낙폭은 더 컸다.</p></div>
<p>46.1% 고정 비중은 <strong>2024–2025년 HAR 목표 비중의 평균 46.0707%</strong>로 미리 정했다. 2026년의 평균을 보고 맞춘 비중이 아니다. 50%는 별도의 단순 기준선이다. 고정 비중 전략도 가격 변화 때문에 비중이 어긋나므로 매일 소량을 사고팔며, 그 비용까지 부담했다.</p>

<h2 id="weights">4. 그래프 2 — 위험 예측이 투자 비중으로 바뀌는 과정</h2>
@@FIG_WEIGHTS@@
<div class="caption"><p><strong>위 패널 회색:</strong> 그날이 끝난 뒤 5분 RV의 제곱근으로 측정한 실제 관측 위험이다. 예측 시점에는 아직 알 수 없다. <strong>파란색:</strong> 전날까지의 정보로 미리 계산한 그날의 위험 예측이다. 두 선이 가까우면 그 날짜의 예측 오차가 작다. 회색이 갑자기 튀는데 파란색이 못 따라가면 예상하지 못한 위험 상승이다.</p>
<p><strong>아래 패널:</strong> 매일 재조정한 직후의 BTC 비중이다. 예측 위험이 커지면 비중이 내려가고, 작아지면 올라간다. 70%면 그 시점 계좌의 70%를 BTC에, 30%를 현금에 배분한 것이다. 하루 중 가격이 바뀌면 실제 비중도 다시 달라진다.</p>
<p><strong>비중 선이 하락 신호인 것은 아니다.</strong> BTC 가격이 내려갈 것이라고 판단해서 비중을 줄인 것이 아니라, 어느 방향이든 크게 움직일 것으로 예측했기 때문에 투자 규모를 줄였다. 이 구조에서는 급락 직후 위험 예측이 높아져 비중을 줄이고, 뒤따르는 반등에 적게 참여할 수도 있다. 이는 가능한 작동 원리이며, 특정 손실 구간의 인과관계를 이 그림만으로 확정한 것은 아니다.</p></div>
<div class="formula">목표 일별 변동성 = 1% = 0.01
BTC 목표 비중 w = min(1, 0.01 / √f)
현금 비중 = 1 − w</div>
<p>예측 BTC 변동성이 2%라면 BTC 50%·현금 50%, 4%라면 BTC 25%·현금 75%가 된다. 0.5%이면 공식상 200%가 필요하지만 레버리지를 금지했으므로 BTC 100%에서 멈춘다. 따라서 1%는 <strong>목표</strong>이며 매일 보장되는 손실 한도나 실제 표준편차가 아니다.</p>
<p>정확한 비중 공식의 직관은 현금의 변동성이 0이고 하루 동안 w가 고정되어 있을 때 σ<sub>계좌</sub> ≈ w·σ<sub>BTC</sub>라는 관계다. 실제 계산에서는 매매비용, 첫 1분의 기존 비중, 가격에 따른 비중 변화가 있으므로 근사 관계를 성과 계산에 그대로 대입하지 않았다. 실제 계좌 일별 표준편차는 0.9102%였다.</p>
<h3>실제 첫날의 숫자로 따라가기</h3>
<p>2025-12-31까지의 정보로 만든 2026-01-01의 HAR 분산 예측은 <strong>0.0001914118</strong>이었다. 제곱근은 약 <strong>1.3835%</strong>, 따라서 목표 비중은 1%/1.3835% = <strong>72.2796%</strong>다. 첫 매수에 약 7,223원의 비용이 들고, 그날 종료 평가액은 약 10,037,909원이었다. 그날이 끝난 뒤 측정한 RV는 0.0000464835로, 그 제곱근은 약 0.6818%였다. <strong>예측 1.3835%와 사후 관측 0.6818%를 뒤섞지 않는 것이 핵심</strong>이다.</p>

<h2 id="scores">5. 그래프 3 — QLIKE와 MSE는 무엇을 평가하나</h2>
@@FIG_MODELS@@
<div class="caption"><p><strong>왼쪽:</strong> 2024–2025년 예측 오차로 모형을 고른 결과다. 막대가 짧을수록 평균 QLIKE가 작다. 전체 학습 HAR가 0.3381로 가장 낮아 주 모형이 됐다.</p>
<p><strong>오른쪽:</strong> 정해진 모형들을 2026년에 평가한 결과다. 주 모형 HAR는 0.2140, 어제 RV 기준선은 0.3163이다. (0.3163−0.2140)/0.3163 ≈ 32.4%만큼 손실이 줄었다. 이 숫자는 수익률이 32.4% 높아졌다는 뜻도, 예측 적중률이 32.4%라는 뜻도 아니다.</p>
<p>오른쪽에서 최근 3년 HAR가 0.2113으로 약간 더 낮지만, 이를 본 뒤 주 모형을 바꾸지 않았다. 두 패널은 가로축 범위가 다르므로 막대의 화면상 길이보다 표시된 숫자를 비교해야 한다.</p></div>
<div class="formula">그날 관측한 분산 대용치 = Vₜ, 미리 예측한 분산 = fₜ
정규화 QLIKE = 평균[Vₜ/fₜ − log(Vₜ/fₜ) − 1]
분산 MSE (예측 문맥의 MSPE) = 평균[(Vₜ − fₜ)²]
변동성 MAE = 평균[|√Vₜ − √fₜ|] × 100  (단위: %포인트)</div>
<p><strong>QLIKE:</strong> 예측 분산이 관측 분산에 비해 얼마나 비례적으로 어긋났는지를 평가한다. f = V이면 해당 날짜 손실은 0이다. 실제보다 분산을 절반으로 예측하면 손실은 약 0.3069, 두 배로 예측하면 약 0.1931이다. 과소예측과 과대예측을 대칭으로 벌점 주는 함수는 아니다. 평균 QLIKE는 확률이 아니며, 낮을수록 좋다.</p>
<p>논문에서 쓰는 log(f)+V/f 형태와 이 보고서의 정규화 QLIKE는 <strong>동일 날짜·동일 V로 비교할 때 모형에 무관한 항</strong>만큼 차이 난다. 따라서 모형 순위는 같지만 수치의 절대 수준을 논문 표와 직접 비교하면 안 된다. 분산 대용치의 측정 오차와 손실함수 선택에 관한 배경은 <a href="https://public.econ.duke.edu/~ap172/Patton_vol_proxies_JoE_2011.pdf">Patton의 변동성 예측 평가 연구</a>를 참고할 수 있다.</p>
<p><strong>MSE:</strong> 분산 예측의 절대 오차를 제곱해 평균한다. 큰 분산이 나타난 날의 큰 오차가 결과를 많이 좌우할 수 있다. 수익률을 소수로 입력했으므로 MSE의 단위는 소수 로그수익률의 4제곱이다. 10⁻⁷처럼 작게 보이는 숫자 자체가 예측이 거의 완벽하다는 뜻은 아니다.</p>
<p>주 모형 MSE는 1.5483×10⁻⁷, 어제 RV는 1.4992×10⁻⁷이었다. HAR가 약 3.3% 나빴다. 변동성 MAE도 0.5222%포인트 대 0.4952%포인트로 더 컸다. <strong>이번에 확실하게 말할 수 있는 개선은 선택 기준인 QLIKE에서의 개선</strong>이다. 평균 예측 변동성도 2.1970%로 평균 측정 변동성 1.9382%보다 높았다.</p>
@@FORECAST_TABLE@@
<p class="note"><strong>FIVAR와의 관계:</strong> 신민석 교수님 등의 FIVAR는 여러 자산의 변동성·공분산 행렬을 예측한다. 이번 것은 BTC 하나의 분산을 HAR·GARCH로 예측하는 실험이다. POETPRVM의 행렬 추정 절차나 FIVAR를 재현한 것이 아니다. 논문의 MSPE·QLIKE는 예측 평가이고, 예측한 행렬로 구성한 포트폴리오의 성과는 별도 평가다. 여기서도 그 두 평가 단계를 분리했다. <a href="https://arxiv.org/html/2109.05227v3">FIVAR 논문</a></p>

<h2 id="costs">6. 그래프 4 — 매매비용이 결과를 얼마나 바꾸나</h2>
@@FIG_COSTS@@
<div class="caption"><p><strong>왼쪽:</strong> 매매금액 1원당 부담하는 편도 총비용을 바꿨을 때의 누적 수익률이다. 10bp는 0.10%다. HAR는 비용이 0일 때 −4.08%였지만, 0.10%에서는 −6.32%, 0.25%에서는 −9.57%였다. 같은 예측과 목표 비중을 사용하고 비용만 바꿨다.</p>
<p><strong>오른쪽:</strong> 0.10% 비용 가정 아래 실제로 지급한 총비용이다. HAR 약 22.0만원, 46.1% 고정 약 1.8만원이었다. 위험 예측이 자주 바뀌면 매매 규모도 커져 비용이 증가한다. 어제 RV를 그대로 쓰는 방식은 특히 비중 변화가 커 약 38.3만원이 들었다.</p>
<p>HAR의 무비용 수익률도 46.1% 고정의 무비용 수익률을 넘지 못했다. 따라서 ‘거래비용만 없애면 우수한 전략’이라고 결론 내릴 수 없다. 거래비용은 손실 차이의 일부를 더 키웠다.</p></div>
@@COST_TABLE@@
<p>비용 0.10%는 <strong>수수료와 슬리피지를 합한 연구 가정</strong>이며 과거 업비트의 실제 수수료 일정을 복원한 수치는 아니다. 매일 계좌 전체에 0.10%를 부과하지 않았다. 가격 변화 후 현재 BTC 보유액과 새 목표 보유액 사이에서 <strong>실제로 사고판 금액</strong>에만 부과했다.</p>
<p>예를 들어 계좌가 1,000만원, BTC가 500만원일 때 목표 비중을 40%로 내리면 대략 100만원을 매도한다. 비용이 0.10%이면 약 1,000원이 들며, 정확한 계산에서는 이 비용으로 줄어든 계좌 총액까지 반영해 목표 40%를 맞춘다. 첫 진입 매수와 마지막 전량 매도 비용도 포함했다.</p>
<details><summary>수량과 현금 잔고로 계산한 정확한 회계식</summary>
<p>매매 직전 총자산 W = BTC 평가액 B + 현금 C, 비용률 c, 목표 BTC 비중 w라 두면 매매금액 x는 매수일 때 (wW−B)/(1+cw), 매도일 때 (wW−B)/(1−cw)다. 매도 x는 음수다. 매매 후 B′ = B+x, C′ = C−x−c|x|이며 B′/(B′+C′) = w가 된다.</p>
<p>수익률은 이 잔고를 실제 가격으로 평가해 계산했다. 소수 수익률을 합산해 계좌를 만든 것이 아니다. 별도로 BTC 수량·현금을 추적하고 수치적으로 목표 비중을 푸는 계산과 대조했으며, 주요 5개 전략의 모든 284일 평가액이 0.00000004원 이내로 일치했다.</p></details>

<h2 id="metrics">7. 성과표의 각 숫자를 정확히 읽는 법</h2>
<h3>누적 수익률과 최종 평가액</h3>
<p>누적 수익률 = 마지막 평가액 / 시작 평가액 − 1이다. HAR의 −6.3151%는 1,000만원이 약 <strong>936만8,490원</strong>이 되었다는 뜻이다. 2026년 284일의 결과이며 1년 전체 수익률이 아니다. 일별 수익률을 더한 값도 아니다.</p>
<h3>일별·연환산 변동성</h3>
<p>매일의 <strong>계좌 순수익률</strong>을 모아 표본 표준편차를 계산했다. 연환산은 일별 표준편차 × √365다. BTC는 주말도 거래하므로 252를 쓰지 않았다. HAR의 17.39%는 이 관례적 연환산 값이며, 내년 손실이 최대 17.39%라는 뜻이 아니다. 수익률 자기상관이나 시장 환경 변화가 있으면 실제 1년 위험과 달라질 수 있다.</p>
<p>RV는 한 날짜 안의 5분 움직임으로 측정한 위험이고, 이 성과표의 변동성은 284개의 <strong>일별 계좌 수익률</strong> 사이의 표준편차다. 같은 ‘변동성’이라는 단어를 쓰지만 자료를 묶는 방향과 대상이 다르다.</p>
<h3>최대낙폭 MDD</h3>
<p>각 날짜의 ‘직전 최고점 대비 하락률’을 구한 다음 가장 나쁜 값을 고른다. 이번 표의 −23.32%는 평가 기간에 HAR 계좌가 과거 최고점에서 가장 깊게 빠졌을 때의 비율이다. 마지막 날 손실 −6.32%와 달라도 모순이 아니다. 시간의 <strong>진행 순서</strong>가 중요한 지표다.</p>
<p>표는 일별 종가 기준이다. 1분 종가까지 확인하면 HAR 최대낙폭은 <strong>−24.05%</strong>, 46.1% 고정은 <strong>−19.27%</strong>, BTC 전액은 <strong>−38.23%</strong>였다. 1분 안의 체결가 극단값까지 측정한 최대 손실은 아니다.</p>
<h3>평균 BTC 비중</h3>
<p>매일 재조정 직후의 BTC 비중을 산술평균한 값이다. HAR 50.50%는 매일 50.50%를 유지했다는 뜻이 아니다. 높을 때와 낮을 때를 평균한 결과다. 비중이 가격 상승일에 높았는지 하락일에 높았는지도 성과에 영향을 준다.</p>
<h3>샤프 비율 Sharpe ratio</h3>
<p>여기서는 현금 수익률 0%를 기준으로 <strong>일별 계좌 수익률 평균 / 표준편차 × √365</strong>로 계산했다. 평균 보상에 비해 얼마나 흔들렸는지를 본다. HAR는 약 −0.396이었다. 평균 수익이 음수일 때에는 위험이 작아지는 것이 음의 샤프 수치를 더 나쁘게 만들 수도 있으므로, 샤프 하나로 전략의 우열을 판단하지 않는다. 현금 100%는 분자·분모가 모두 0이어서 샤프를 0으로 꾸미지 않고 비어 있게 뒀다.</p>
<h3>CAGR와 매매 회전율</h3>
<p>CSV의 CAGR은 (마지막 평가액/시작 평가액)<sup>365/284</sup>−1이다. 284일 성과를 같은 복리 속도로 1년으로 환산한 수치이며 실제 관측 1년 수익률이나 미래 전망이 아니다.</p>
<p>회전율은 매매할 때마다 |매매금액|/직전 계좌 평가액을 더한 <strong>편도 회전율</strong>이다. HAR는 기간 합계 약 23.59배, 연환산 약 30.31배였다. 레버리지가 23.59배라는 뜻이 아니라, 여러 날 사고판 금액의 비중을 누적한 것이다. 마지막 청산도 포함했다.</p>

<h2>8. 예측 개선과 투자 우수성의 증거는 어느 정도인가</h2>
<p>같은 날짜의 차이를 묶어 14일 연속 블록으로 3,000번 재표집했다. 날짜를 모두 독립으로 취급하지 않고 일정한 시간 의존성을 유지하려는 원형 블록 부트스트랩이다.</p>
<p>HAR−어제 RV의 평균 QLIKE 차이는 −0.1024이고, 95% 구간은 [−0.1546, −0.0560]이었다. 이 재표집 방법 아래에서는 HAR의 QLIKE가 더 낮다는 결과가 유지됐다.</p>
<p>HAR−46.1% 고정의 평균 일별 순수익률 차이는 <strong>−0.00896%포인트</strong>, 95% 구간은 <strong>[−0.04290, +0.02569]%포인트</strong>였다. 구간에 0이 들어가므로 초과수익을 확인했다고 보기 어렵다. 이것은 누적 수익률 차이의 신뢰구간이 아니다. 또한 14일이라는 블록 선택, 짧은 평가 기간, 과거 탐색에 따른 편향을 이 계산이 없애 주지는 않는다.</p>
<details><summary>비교한 모든 변동성 기반 전략의 2026년 투자 결과</summary>@@ALL_PORTFOLIO@@
<p>GARCH나 단순 평균 RV 중 HAR보다 투자 수익률이 높은 경우도 있었다. 그러나 2026년 투자 결과를 본 뒤 가장 좋은 모형을 주 전략으로 바꾸면 테스트 자료로 전략을 선택하게 된다. 이 표는 비교 정보이며 새로 선정한 우승 전략 표가 아니다. 예측 QLIKE가 더 좋다고 투자 수익률까지 반드시 좋아지는 것은 아니라는 점도 확인할 수 있다.</p></details>

<h2 id="longshort">9. 상승·하락 신호를 따로 표시하고 롱숏으로 확장하려면</h2>
<p><strong>가능하다. 다만 ‘방향을 정하는 신호’와 ‘투자 금액을 정하는 위험 예측’을 분리해야 한다.</strong> 지금 계산한 HAR·GARCH는 주로 두 번째 역할이다. 위험 증가만으로 하락을 예측한 것은 아니다.</p>
<div class="formula">과거 정보 → 방향 모형 → 예상 순수익 또는 상승·하락 확률
                        ↓
                 롱 / 관망 / 숏 신호
                        ↓
위험 모형(HAR·GARCH) → 투자 크기 + 비중 상한 → 주문 규칙
                        ↓
             비용·펀딩·증거금을 포함한 사후 평가</div>
<h3>신호와 정답을 구분하기</h3>
<p>다음 날 실제 수익률이 양수였는지 음수였는지는 학습용 <strong>정답</strong>으로 쓸 수 있다. 예를 들어 실제 다음 날 수익률이 +a를 넘으면 상승, −a보다 작으면 하락, 그 사이면 중립으로 라벨을 붙인다. a는 비용과 작은 잡음 움직임을 고려한 임계값이며 학습·검증 구간에서 정해야 한다. 테스트 결과를 보고 유리한 값으로 바꾸면 안 된다.</p>
<p>실제 거래 차트의 화살표는 <strong>그 날짜 이전까지의 정보로 예측한 신호</strong>여야 한다. 미래 가격을 본 뒤 그날에 상승·하락 화살표를 붙이는 것은 설명용 정답 표시일 뿐, 매매 가능했던 신호의 증거가 아니다. 차트를 만들 때 예측 화살표와 사후 정답 표시는 색이나 패널을 분리한다.</p>
<h3>반드시 정할 내용</h3>
<ol><li><strong>보유 기간:</strong> 1시간 뒤인가, 하루 뒤인가. 신호의 목표 수익률·실제 체결 시점·청산 시점이 같은 기간을 가리켜야 한다.</li>
<li><strong>입력 정보:</strong> 과거 수익률, 추세, 거래량 변화, 과거 RV 등 신호 시점까지 알 수 있는 정보. 이것들이 실제로 방향 예측력을 갖는지는 별도 검증이 필요하다.</li>
<li><strong>판단 기준:</strong> 예측 기대수익이 거래비용과 여유폭을 넘을 때 롱, 음의 기대수익이 숏 비용과 여유폭을 넘을 때 숏, 나머지는 관망. 상승 확률 51%만으로 거래를 결정하면 상승·하락 크기가 다른 문제를 놓칠 수 있다.</li>
<li><strong>비중:</strong> 방향을 s ∈ {−1,0,+1}로 표시하고, 예시로 w = s × min(비중 상한, 목표 변동성/예측 변동성)을 사용할 수 있다. +는 롱, −는 숏, 0은 현금이다. 이것은 향후 설계 예시이며 이번에 검증한 롱숏 전략은 아니다.</li>
<li><strong>진입·유지·전환·청산:</strong> 이미 롱인데 약한 롱 신호가 다시 나왔을 때 유지할지, 신호가 약해지면 현금으로 갈지, 반대 신호에서 즉시 전환할지 정해야 한다. 롱 50%에서 숏 50%로 바꾸면 순노출 100%포인트만큼 거래해 비용이 증가한다.</li>
<li><strong>거래 수단:</strong> 현물 보유분을 팔아 현금으로 바꾸는 것과 숏을 여는 것은 다르다. 선물·무기한선물이라면 해당 거래소 계약 가격, 기준·마크 가격, 펀딩비, 수수료, 증거금·강제청산 조건을 넣어야 한다. 코인 차입 숏이라면 차입 가능 여부와 이자도 필요하다.</li>
<li><strong>평가:</strong> 상승·하락별 정밀도와 혼동행렬, 예측 확률의 보정, 비용 차감 수익률·최대낙폭·회전율·롱/숏별 손익을 따로 본다. 동일 기간의 매수 보유·현금·롱만 허용하는 전략과도 비교한다.</li></ol>
<p>예시로 ‘예상 다음 날 수익 +0.8%, 왕복 예상 비용 0.2%, 위험 예측 2%, 목표 위험 1%’라면 롱 50%를 검토할 수 있다. 기대수익이 −0.8%라면 숏 50%, +0.1%처럼 비용보다 작으면 관망하는 식이다. <strong>숫자는 구조 설명용</strong>이며 이번 BTC 자료에서 발견한 수익 신호가 아니다. 실제 수익은 예측과 다를 수 있다.</p>
<p>‘상승·하락 확률’을 쓰는 경우에도 기대수익 ≈ 상승확률×상승 시 평균 이익 − 하락확률×하락 시 평균 손실을 함께 고려해야 한다. 맞힌 횟수가 많아도 몇 번의 큰 손실 때문에 전체 수익이 음수가 될 수 있다.</p>
<h3>개인도 비트코인을 숏할 수 있나?</h3>
<p>개인 대상의 코인 차입 마진과 선물·무기한선물 숏 상품은 존재한다. 다만 거주지와 거래소별 이용 자격을 확인해야 한다. 차입 숏은 빌린 BTC를 먼저 팔고 나중에 되사서 갚는 구조다. 파생상품 숏은 계약 가격 하락에 이익이 나는 포지션이다. 가격 상승은 손실이며, 증거금이 부족하면 강제청산될 수 있다. <a href="https://www.kraken.com/learn/what-is-crypto-margin-trading">공식 마진 설명</a>, <a href="https://www.kraken.com/features/futures/bitcoin">공식 BTC 선물 설명</a></p>
<p>여기서 분석한 업비트 KRW-BTC 현물 가격에 수익률 부호만 반대로 붙인 결과를 실제 선물 숏 성과라고 부르면 안 된다. 원화 현물과 달러·USDT 파생상품 사이의 가격 차이와 펀딩 등이 다르기 때문이다. <strong>이번 완료된 백테스트는 0≤BTC 비중≤100%인 현금·현물 전략이며, 롱숏은 위와 같은 별도의 다음 연구 단계다.</strong></p>

<h2 id="data">10. 자료 품질과 검산</h2>
<p>업비트 문서에 따르면 체결이 없는 분에는 캔들이 생성되지 않는다. 따라서 API에 없는 캔들을 모두 다운로드 오류라고 간주하지 않았다. 연속 시간 격자를 만들 때는 직전 가격으로 채웠고, 해당 분의 수익률은 0으로 처리했다. 거래 재개 때의 가격 변화는 이후 수익률에 반영된다. <a href="https://docs.upbit.com/kr/reference/list-candles-minutes">업비트 분 캔들 문서</a></p>
<p>2026년 평가 구간에서 총 686분을 채웠다. 실제 캔들 비율 95% 미만인 날은 다음 두 날이다. 거래 중단·거래 부재 등의 구체적 원인은 이번 계산에서 확정하지 않았다.</p>
@@QUALITY_TABLE@@
<p>이 두 날만 <strong>예측 손실 평가에서 제외</strong>한 282일의 QLIKE도 HAR 0.2148, 어제 RV 0.3174로 개선 방향이 유지됐다. 수익률 시계열에서 해당 날을 지우거나 이어 붙이지 않았으며 실제 투자 성과표는 284일 전체다. 이런 필터가 나머지 학습 자료나 위험 측정에 미친 모든 영향을 제거한 것은 아니다.</p>
<p>주요 5개 전략에 대해 독립적으로 BTC 수량과 현금 잔고를 재계산했고, 원시 월별 캐시와 진입·종료 가격을 대조했다. 진입 가격은 2026-01-01 UTC 00:00 캔들 종가 128,030,000원, 종료 가격은 2026-10-11 UTC 23:59 캔들 종가 113,364,000원이다. BTC 100% 전략의 최종 자산은 비용을 포함한 (1−c)/(1+c)×종료가/진입가 공식과 일치했다.</p>
<p>재조정에 쓰인 첫 1분 양 끝 가격은 모든 평가일에 관측되어 있었다. HAR의 일부 예측은 원본 RV의 과거 값으로 별도 재계산했고, GARCH 첫 예측도 추정 패키지와 직접 재귀식 사이에 일치함을 확인했다.</p>
<p class="note"><strong>결과의 범위:</strong> 단일 거래소의 원화 BTC, 짧은 2026년 284일, 이전에 탐색한 평가 자료, 가정한 비례 매매비용, 1분 종가 체결, 이자 0% 현금이라는 조건의 후향적 실험이다. 세금·최소 주문금액·수량 반올림·호가별 시장충격은 따로 모형화하지 않았다. 실거래 주문을 실행하지 않았다. 이 결과만으로 실전 초과수익이나 다른 시기의 성과를 입증하지 못한다.</p>

<h2>11. 내려받은 CSV를 읽는 방법</h2>
<p><a href="btc_cash_daily.csv">일별 예측·비중·손익</a>는 284행, <a href="btc_cash_summary.csv">전략 성과 요약</a>은 19행, <a href="btc_cash_forecast_scores.csv">예측 모형 점수</a>는 22행이다. HTML과 같은 폴더에 CSV를 두면 이 링크가 작동한다. 별도 압축 파일에는 이 보고서와 CSV, 그림, 아래 설명을 포함했다.</p>
<details><summary>일별 데이터의 필드와 단위</summary>
<ul><li><code>date_utc</code>: 손익·위험을 측정한 UTC 날짜. <code>forecast_information_through</code>: 예측에 사용할 수 있었던 마지막 날짜.</li>
<li><code>rv_5m</code>: 당일 5분 로그수익률 제곱합. 분산 단위이며 제곱근 ×100을 하면 변동성 %다.</li>
<li><code>btc_log_return</code>: UTC 하루 BTC 로그수익률. <code>btc_simple_return</code>: exp(로그수익률)−1. 0.01이 +1%다.</li>
<li><code>first_minute_return</code>: 당일 첫 1분의 단순수익률. 이 시간에는 기존 보유 수량이 수익을 얻는다.</li>
<li><code>coverage</code>: 1,440분 중 실제 캔들이 있는 비율. <code>rebalance_endpoint_observed</code>: 재조정 첫 1분 양 끝 가격을 관측했는지.</li>
<li><code>모형명_variance_forecast</code>: 사전에 계산한 그날 분산 예측. 원자료 수익률은 소수 단위.</li>
<li><code>전략명_target</code>: 목표 BTC 비중. <code>전략명_weights</code>: 아침 재조정 직후 실제 비중. <code>전략명_pre_weights</code>: 그 매매 직전 가격 변화까지 반영한 비중. 0.5가 50%다. 마지막 날의 weights도 아침 비중이며, 종료 청산 후 BTC 비중은 0이다.</li>
<li><code>전략명_returns</code>: 비용 차감한 계좌 일별 단순수익률. <code>전략명_nav</code>: 하루 종료 계좌 평가액(원).</li>
<li><code>전략명_fees</code>: 실제 매매금액에 부과한 비용(원). 마지막 날은 종료 청산 비용 포함. <code>전략명_turnover</code>: 당일 편도 거래금액/매매 직전 평가액 비율의 합.</li></ul></details>
<details><summary>요약·점수 데이터의 필드와 단위</summary>
<p><code>period</code>가 <code>test_2026</code>인 행이 최종 평가다. <code>validation_2024_2025</code> 행은 모형 선택에 사용한 구간이므로 독립 검증 성과가 아니다. <code>selected_primary</code>는 사전에 선택한 주 모형인지 여부다.</p>
<p><code>total_return</code>, <code>cagr</code>, <code>annual_vol</code>, <code>daily_vol</code>, <code>max_drawdown</code>, <code>minute_max_drawdown</code>, <code>average_btc_weight</code>, <code>average_target_weight</code>, <code>fees_pct_initial</code>, <code>worst_day</code>, <code>best_day</code>, <code>cost</code>는 모두 0.01=1%인 소수 비율이다. <code>fees_pct_initial</code>는 지급 비용/초기 자본이다.</p>
<p><code>days</code>는 일수, <code>final_nav</code>와 <code>fees_paid_krw</code>는 원, <code>turnover_total</code>과 <code>turnover_annual</code>은 누적·연환산 편도 회전배수다. <code>full_btc_days</code>는 목표 비중이 100%인 날 수, <code>skipped_rebalance_days</code>는 체결 가격 미관측으로 재조정하지 못한 날 수다.</p>
<p>점수표의 <code>n</code>은 예측 평가 일수다. <code>qlike</code>는 정규화 QLIKE, <code>mse</code>는 분산 예측 제곱오차 평균, <code>vol_mae_pct</code>는 변동성 오차 %포인트다. <code>mean_forecast_vol_pct</code>와 <code>mean_measured_vol_pct</code>는 각각 예측·관측 변동성 평균을 이미 %로 바꾼 값이므로 100을 다시 곱하지 않는다. <code>qlike_improvement_vs_RV_last</code>는 1−모형 QLIKE/어제 RV QLIKE로, 0.3235가 약 32.35% 개선이다.</p>
<p>빈 칸은 계산하지 않았거나 정의되지 않는 값이다. 예를 들어 현금의 샤프는 정의되지 않고, 일부 검증 구간의 분 단위 최대낙폭은 계산하지 않았다. 빈 칸을 0으로 해석하지 않는다.</p></details>
<footer><p><strong>판단:</strong> BTC의 다음 날 위험을 추정해 현금 비중을 조절하는 것은 구현할 수 있다. 이번 실험은 QLIKE 개선을 확인했지만, 거래비용을 반영한 우수한 투자 전략을 확인하지는 못했다. 롱숏을 연구하려면 이 위험 모형에 별도의 방향 신호와 실제 숏 거래 조건을 추가해야 한다.</p><p class="muted">분석일 2026-10-13 · 모든 표와 그림은 저장된 동일 계산 결과에서 생성했다. HTML 그림은 내장되어 오프라인에서도 보인다.</p></footer>
</main></body></html>'''

replacements={'MAIN_TABLE':main_table,'PARAM_TABLE':param_table,'FORECAST_TABLE':forecast_table,
 'ALL_PORTFOLIO':all_portfolio,'COST_TABLE':cost_table,'QUALITY_TABLE':quality_table,
 'FIG_EQUITY':img('btc_cash_equity','BTC·현금 전략의 계좌 평가액 및 낙폭'),
 'FIG_WEIGHTS':img('btc_cash_forecast_weights','예측 변동성과 측정 변동성, BTC 비중'),
 'FIG_MODELS':img('btc_cash_model_comparison','검증 구간 및 평가 구간의 QLIKE 비교'),
 'FIG_COSTS':img('btc_cash_cost_sensitivity','거래비용 변화에 따른 성과와 지급 비용')}
for k,v in replacements.items():body=body.replace('@@'+k+'@@',v)
assert '@@' not in body and body.count('data:image/png;base64,')==4
(OUT/'btc_cash_backtest_report.html').write_text(body,encoding='utf-8')

readme='''BTC와 현금: 변동성 예측 투자 연구 / 2026-10-13

btc_cash_backtest_report.html을 웹 브라우저에서 열면 4개 그래프 아래의 상세 해설,
HAR·GARCH·GJR의 의미, QLIKE·MSE·수익률·최대낙폭의 계산 방법,
롱숏 방향 신호와 위험 예측을 분리하는 설계, 데이터 품질을 읽을 수 있습니다.
그림은 HTML 안에 포함되어 인터넷 없이 볼 수 있습니다.

기본 설정
- 업비트 BTC/KRW 현물과 원화 현금만. 현금 이자 0%.
- 최초 학습 2017-09-26~2023-12-31, 검증 2024~2025.
- 검증 QLIKE로 전체 학습 HAR 선택 후 2025-12-31까지 재학습.
- 평가 2026-01-01~2026-10-11 UTC, 284일. 계수 고정, 과거 관측만 갱신.
- 하루 경계 KST 09:00. 09:01의 첫 1분 종가로 비중 변경.
- 목표 일별 변동성 1%, BTC 비중 min(1, .01/sqrt(분산 예측)).
- 초기 1,000만원, 비용 편도 0.10%, 첫 진입과 마지막 청산 포함.
- 비용은 연구용 수수료·슬리피지 가정이며 실제 과거 수수료 복원값 아님.
- 2026년은 이전 탐색에 사용했던 자료이므로 완전한 블라인드 테스트 아님.

기본 결과
HAR 누적 -6.32%, 연환산 변동성 17.39%, 일 종가 최대낙폭 -23.32%.
46.1% BTC 고정 누적 -4.11%, BTC 전액 보유 -11.63%.
HAR QLIKE는 어제 RV 예측보다 약 32.4% 개선, 분산 MSE는 약 3.3% 악화.
예측 오차 개선만으로 수익성이나 롱숏 방향 신호를 입증하지 못했습니다.

데이터 파일
- btc_cash_daily.csv: 284일 × 124필드. 일별 사전 예측·목표 비중·실제 비중·손익.
- btc_cash_summary.csv: 19행. 평가 전략 15개와 검증 구간 성과 4개.
- btc_cash_forecast_scores.csv: 22행. 11개 모형의 검증·평가 점수.
CSV의 단위·필드 설명은 HTML 마지막 부분에 있습니다.
수익률·비중은 원칙적으로 소수 단위(0.01=1%). 단, 이름이 _pct인 예측 점수
필드는 이미 % 또는 %포인트입니다. 빈 칸은 0이 아닙니다.

이번에는 현물·현금 전략만 실행했습니다. 롱숏 부분은 향후 설계 설명이며,
실제 선물 데이터나 펀딩비·강제청산을 반영한 숏 백테스트가 아닙니다.
'''
(OUT/'btc_cash_README.txt').write_text(readme,encoding='utf-8-sig')

# Verify every CSV field against the authoritative calculation records.
checks=[]
for name in ['daily','summary','forecast_scores']:
    expected=json.loads((WORK/f'btc_cash_{name}.json').read_text(encoding='utf-8'))
    with (OUT/f'btc_cash_{name}.csv').open(encoding='utf-8-sig',newline='') as f:actual=list(csv.DictReader(f))
    assert len(actual)==len(expected)
    for row,ref in zip(actual,expected):
        for k,val in row.items():
            e=ref.get(k)
            if e is None:assert val==''
            elif isinstance(e,bool):assert val==str(e).lower()
            elif isinstance(e,(int,float)):assert math.isclose(float(val),e,rel_tol=1e-14,abs_tol=1e-14)
            else:assert val==str(e)
    checks.append({'file':f'btc_cash_{name}.csv','rows':len(actual),'all_fields_match':True})
(WORK/'btc_cash_deliverable_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
bundle=['btc_cash_backtest_report.html','btc_cash_README.txt']+[f'btc_cash_{n}.csv' for n in ['daily','summary','forecast_scores']]+[f'btc_cash_{n}.png' for n in ['equity','forecast_weights','model_comparison','cost_sensitivity']]
with zipfile.ZipFile(OUT/'btc_cash_backtest_bundle.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
    for name in bundle:z.write(OUT/name,name)
print(json.dumps({'html_bytes':(OUT/'btc_cash_backtest_report.html').stat().st_size,'embedded_figures':4,'csv_checks':checks,'zip_files':len(bundle)},ensure_ascii=False))
