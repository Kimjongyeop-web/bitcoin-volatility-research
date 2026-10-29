# Bitcoin Volatility Research

**BTC/KRW 실현변동성 · HAR / HAR-X · 거래비용 반영 전략**

[보고서 미리보기](REPORT.md) · [PDF 보고서](reports/har_factor_study.pdf) · [상세 통계](reports/har_factor_study/results.json) · [코드](scripts/)

[![HAR와 팩터 비교 보고서](assets/report-cover.png)](REPORT.md)

## 연구 질문

BTC 변동성 예측에서 거래량 충격과 하락 변동성 비중이 HAR 모형에 추가 정보를 제공하는지 살펴본다. 계수의 통계적 관계와 실제 예측 오차 개선을 구분한다.

| 항목 | 설정 |
|---|---|
| 원자료 | Upbit BTC/KRW 1분 시세 |
| 관측 기간 | 완결 UTC 일자 2017-09-26–2026-10-11 |
| 실현변동성 | 5분 수익률 제곱합 |
| 모형 비교 | HAR · 팩터 모형 · HAR-X |
| 검증 / 평가 | 2024–2025년 / 2026-01-01–2026-10-11 |

## 주요 결과

2026년 HAR-X의 QLIKE는 HAR보다 **약 1.13% 낮았다**. 다만 손실차의 95% 구간 **[−0.0069, 0.0019]**이 0을 포함해 개선을 확정하기 어렵다. 2026년은 이미 관찰한 표본에 대한 사후 평가다.

![모형별 예측 오차 비교 · 보고서 7쪽](assets/report-07.png)

## 연구 구성

| 분야 | 내용 |
|---|---|
| 데이터 진단 | 관측률·결측, 표본 간격별 RV, 자기상관 |
| 예측 | HAR / HAR-X, 고정 계수·재추정 비교 |
| 전략 | BTC–현금, 하락 비중 롱숏, 저빈도 오버레이 |
| 수집 | 별도 [Market Data API](https://github.com/Kimjongyeop-web/market-data-api) |

<details>
<summary>입력과 재현 경로</summary>

월별 원응답과 `download_meta.json`, `progress.json`이 `work/minute_cache/`의 입력이다.

`analyze_minutes.py` → `run_har_factor_study.py` → `report_har_factor_study.py` → `check_har_factor_study.py`

전략별 스크립트에는 별도의 선행 입력이 필요하다. 보고서 생성에는 외부 렌더러와 글꼴 의존성이 있으며 전체 실행 경로의 새 환경 재현은 미검증 상태다. 실행별 소스 복사본과 빈 작업 폴더는 `prepare_run.py --run-id research_001`로 구성된다.

</details>

[데이터 범위](DATA_POLICY.md)
