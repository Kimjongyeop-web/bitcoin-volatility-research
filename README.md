# Bitcoin Volatility Research

업비트 BTC/KRW 1분 원자료 기반 실현변동성, HAR·HAR-X 예측과 비용 포함 전략 연구입니다.

## 연구 구분

- 데이터 진단: 원자료 QC, 표본 간격별 RV·자기상관.
- 예측: HAR / Factors / HAR-X, 고정·재추정 비교.
- 전략: BTC-현금, 하락 비중 롱숏, 저빈도 오버레이.
- API 수집: 별도 `market-data-api`의 Upbit 코드.

## 보고서 결과

[HAR 팩터 보고서](reports/har_factor_study.pdf) · [상세 통계](reports/har_factor_study/results.json)

완결 UTC 일자 2017-09-26~2026-10-11, 5분 수익률 제곱합을 RV로 사용했습니다. 검증은 2024~2025년, 평가는 2026-01-01~2026-10-11입니다.
2026 HAR-X의 QLIKE는 HAR 대비 약 1.13% 낮았지만 손실차 95% 구간 [-0.0069, 0.0019]는 0을 포함합니다. 거래량 충격과 하락 비중의 관계를 조사한 것으로, 경제적 인과나 투자 수익을 입증하지 않습니다. 이미 본 2026년 표본의 사후 연구입니다.

## 재현 순서와 제한

1. 새 `runs/<id>/`에 `work/`, `outputs/`를 만들고 해당 실행 폴더를 현재 디렉터리로 사용합니다.
2. 동일 cutoff의 보존한 월별 원응답과 `download_meta.json`, `progress.json`을 `work/minute_cache/`에 복사합니다.
3. `analyze_minutes.py`로 일별 배열을 만듭니다.
4. `run_har_factor_study.py`로 모형·진단을 계산합니다.
5. `report_har_factor_study.py`와 `check_har_factor_study.py`의 렌더러·폰트·보고서 선행 입력을 준비하고 보고서까지 생성·검증합니다.

그 밖의 전략 스크립트는 개별 연구 스냅샷으로 각각의 선행 입력이 필요합니다. 모든 경로를 새 환경에서 실행 검증하지 않았습니다. 글꼴·외부 렌더러와 환경 이식성 보완이 필요합니다. 원본 데이터·모델·예측 캐시는 포함하지 않습니다.

## 격리 실행 폴더 만들기

먼저 python prepare_run.py --run-id research_001 을 실행하면 소스 복사본과 빈 work/outputs 폴더만 생성됩니다. 연구·다운로드·삭제는 실행하지 않습니다. 이후 위 재현 순서에 맞춰 입력과 작업 디렉터리를 준비하세요.

