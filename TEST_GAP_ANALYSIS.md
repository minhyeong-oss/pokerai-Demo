# PRD v1.3 테스트 분류

- 작성일: 2026-11-22
- 기준 커밋: `d60ceeb`
- 기준 문서: `PRD.md`, `CODE_CHANGE_SPEC.md`
- 현재 기준선: 테스트 파일 16개, 테스트 88개 전부 통과

## 1. 분류 원칙

- **기존 충족**: v1.3 변경과 충돌하지 않아 그대로 유지할 수 있다.
- **수정**: 검증 목적은 유효하지만 필드명, 단위, 계산식 또는 기대값을 v1.3에 맞춰 바꿔야 한다.
- **신규**: 현재 테스트에 대응 검증이 없어 새 테스트가 필요하다.

기존 테스트는 삭제하지 않는다. 계약이 바뀐 경우에도 원래 검증하던 성질은 남기고 입력과 기대값만 새 계약에 맞춘다. 구현은 변경요구사항별로 실패 테스트를 먼저 추가하거나 수정한 뒤 진행한다.

## 2. 테스트 파일별 분류

| 테스트 파일 | 현재 테스트 수 | 분류 | 유지할 검증 | v1.3에서 바꿀 내용 |
|---|---:|---|---|---|
| `test_cards.py` | 16 | 기존 충족 | 카드 표기, 중복 카드, 잘못된 입력 거부 | 변경 없음 |
| `test_train.py` | 3 | 기존 충족 | Kuhn CFR 수렴값, 결정성, 초기화 | MVP 범위 밖이므로 회귀 테스트만 유지 |
| `test_observation.py` | 7 | 수정 | 불변성, 카드·보드·스트리트·인덱스 검증 | PRD 필드명, 정수 칩, 식별자, 헤즈업 고정, 타입이 있는 공개 이벤트 |
| `test_environment.py` | 9 | 수정 | 덱 재현성, 비공개 정보 차단, 불법 행동 원자성, 정산·보존, PHH 재생 | 정수 칩, 새 이벤트, 식별자, 비대칭 시작 스택, 미콜 초과액 반환 |
| `test_actions.py` | 5 | 수정 | 후보 불변성, 수동 행동, 음수 비율 거부 | 정수 칩, 범위 밖 후보 제거, floor, 상황별 후보 집합, 중복 제거 |
| `test_agent.py` | 3 | 수정 | 재현성, 중복 이벤트 멱등성, 충돌 이벤트 거부 | 이벤트 자동 처리, 새 식별자, 분리 시드, 양자화 레인지와 스냅샷 연동 |
| `test_audit_logging.py` | 2 | 수정 | JSON 직렬화, JSONL 한 줄 한 레코드 | 감사 스키마 2, 스냅샷 선저장, 필수 메타데이터, 비공개 정보 차단, 마지막 불완전 줄 복구 |
| `test_baselines.py` | 2 | 수정 | 합법 후보 선택, 고정 시드 재현성 | 실제 상대와 우도 모델이 같은 분포를 사용하도록 통합, random 균등 분포 |
| `test_cli.py` | 3 | 수정 | 명령별 JSON 출력과 기본 실행 | 칩 단위 입력, 세 마스터 시드, 프로파일, 검증 모드, v1.3 기본 표본 수 |
| `test_equity.py` | 13 | 수정 | 충돌·블로커·가중 레인지·재현성·다인 팟 검증 | 최소 표본 수, Wilson/Hoeffding 구간, 리버 전수 계산, 고정 패 검증 API |
| `test_evaluate.py` | 5 | 수정 | 좌석 교환, 쌍 평균, 단일 표본 처리, 홀수 핸드 거부 | paired Student t, Bonferroni 보정, 우위 판정, 고정 시나리오 기대값 |
| `test_opponents.py` | 2 | 수정 | 행동 확률 합이 1인지 검증 | 새 패 강도, 점수·크기 softmax, ε 혼합, random 모델, 합법 행동 한정 |
| `test_policy.py` | 6 | 수정 | 폴드 손익, 롤아웃 결과 사용, 숏스택 상황 | 모든 비폴드 행동 MC, 동률 규칙, 구간 겹침 진단, Decision 확장, 기존 프록시 제거 |
| `test_ranges.py` | 7 | 수정 | 블로커, 조합 정규화, 베이즈 갱신, 오류 처리 | `10^15` 정수 질량, 재정규화 금지, 누적 질량 샘플링, 양자화 경계 |
| `test_rollout.py` | 3 | 수정 | 재현성, 유한 EV, 폴드의 정확값 | 표본별 레인지·반응·쇼다운, 체크·콜 MC, 정산식, t 구간, 표본 스트림 분리 |
| `test_simulation.py` | 2 | 수정 | 종단 간 실행, 제로섬, 결과 보고 | 세 마스터 시드, 같은 덱 좌석 교환, 실제/가정 모델 분리, 프로파일·실패 정보 |

파일 기준으로 `test_cards.py`와 `test_train.py`는 그대로 유지할 수 있다. 나머지 14개는 기존 검증을 살리면서 계약 변경과 새 합격 기준을 보강한다.

## 3. 변경요구사항별 분류

아래 분류는 해당 CR 전체가 현재 테스트로 충족되는지를 기준으로 한다. 일부 동작이 이미 검증되더라도 v1.3 계약이 추가되면 `수정`으로 표시했다.

| CR | 내용 | 분류 | 주 테스트 파일 |
|---|---|---|---|
| CR-01 | Observation 필드명 통일 | 수정 | `test_observation.py`, `test_agent.py`, `test_cli.py` |
| CR-02 | 금액을 정수 칩으로 통일 | 수정 | `test_observation.py`, `test_environment.py`, `test_actions.py` |
| CR-03 | 식별자 추가 | 신규 | `test_observation.py`, `test_environment.py`, `test_audit_logging.py` |
| CR-04 | 헤즈업 고정 | 수정 | `test_observation.py`, `test_agent.py`, `test_environment.py` |
| CR-05 | 공개 이벤트 구조 고정 | 신규 | `test_observation.py`, `test_environment.py` |
| CR-06 | 비대칭 시작 스택 허용 | 신규 | `test_environment.py`, `test_rollout.py` |
| CR-07 | 덱 시드와 정책 시드 분리 | 수정 | `test_agent.py`, `test_simulation.py`, `test_cli.py` |
| CR-08 | 범위 밖 후보 제거 | 수정 | `test_actions.py` |
| CR-09 | floor 정수화 | 수정 | `test_actions.py` |
| CR-10 | 상황별 후보 집합 | 수정 | `test_actions.py`, `test_environment.py` |
| CR-11 | 점수 모델 교체 | 수정 | `test_opponents.py`, `test_baselines.py` |
| CR-12 | 패 강도 h 교체 | 수정 | `test_opponents.py` |
| CR-13 | 크기 분포와 ε 혼합 추가 | 신규 | `test_opponents.py` |
| CR-14 | 우도 입력 확장 | 수정 | `test_opponents.py`, `test_ranges.py` |
| CR-15 | random 모델 추가 | 수정 | `test_opponents.py`, `test_baselines.py` |
| CR-16 | 기준선 상대를 같은 모델로 교체 | 수정 | `test_baselines.py`, `test_simulation.py` |
| CR-17 | 이벤트 처리 방식 | 수정 | `test_agent.py`, `test_observation.py` |
| CR-18 | 표본 단위 롤아웃 | 수정 | `test_rollout.py` |
| CR-19 | 체크·콜도 MC 표본으로 계산 | 수정 | `test_policy.py`, `test_rollout.py` |
| CR-20 | 정산식 교체 | 수정 | `test_environment.py`, `test_rollout.py`, `test_policy.py` |
| CR-21 | EV 신뢰구간 | 수정 | `test_rollout.py`, `test_policy.py` |
| CR-22 | 동률 규칙 | 수정 | `test_policy.py` |
| CR-23 | PRD에 없는 경로 제거와 Decision 확장 | 수정 | `test_policy.py`, `test_audit_logging.py` |
| CR-24 | 에쿼티 신뢰구간 | 수정 | `test_equity.py` |
| CR-25 | 리버 전수 계산 | 신규 | `test_equity.py`, `test_cli.py` |
| CR-26 | 완전 열거 검증 모드 | 신규 | `test_equity.py`, `test_cli.py` |
| CR-27 | 레인지 스냅샷 저장소 | 신규 | `test_ranges.py`, `test_audit_logging.py`, `test_agent.py` |
| CR-28 | 판단 로그 항목 추가 | 수정 | `test_audit_logging.py` |
| CR-29 | 평가 통계 | 수정 | `test_evaluate.py` |
| CR-30 | 가정 모델과 실제 상대 분리 | 수정 | `test_baselines.py`, `test_simulation.py` |
| CR-31 | 평가 프로파일과 비용 측정 | 수정 | `test_simulation.py`, `test_cli.py` |
| CR-32 | CLI 정리 | 수정 | `test_cli.py` |
| CR-33 | 문서 갱신 | 수정 | 문서 검토 및 CLI 도움말 스모크 테스트 |
| CR-34 | 버전과 의존성 계약 | 수정 | 패키지 메타데이터 및 감사 로그 테스트 |

CR 기준 집계는 `신규 7개`, `수정 27개`, `기존 전체 충족 0개`다. 이는 현재 코드가 쓸 수 없다는 뜻이 아니라, v1.3에서 CR 하나가 여러 계약을 함께 묶고 있어 전체를 그대로 통과하는 CR이 아직 없다는 뜻이다.

## 4. 구현 순서

서로 의존하는 계약을 고려해 다음 순서로 진행한다.

1. **공통 계약**: CR-01~07, CR-34
2. **행동 후보**: CR-08~10
3. **상대 모델과 레인지**: CR-11~17, CR-27
4. **롤아웃과 정책**: CR-18~23
5. **에쿼티**: CR-24~26
6. **감사 로그**: CR-28
7. **평가와 CLI**: CR-29~32
8. **문서 마감**: CR-33

각 묶음이 끝날 때 전체 회귀 테스트를 실행한다. 어느 단계에서도 덱 재현성, 카드 충돌 검사, 비공개 정보 차단, 칩 보존, 불법 행동의 원자성은 깨지면 안 된다.

## 5. 다음 구현 단계의 완료 조건

첫 구현 묶음인 공통 계약은 아래 조건을 모두 만족하면 끝난 것으로 본다.

- Observation이 PRD 필드명과 정수 칩만 사용한다.
- `run_id`, `hand_id`, `decision_id`와 안정적인 순번 좌표가 분리된다.
- 플레이어 수는 정확히 2명이고 상태 값이 고정 열거형으로 검증된다.
- 공개 기록은 `ActionEvent`와 `BoardEvent`만 받는다.
- 시작 스택은 좌석별로 다르게 줄 수 있고 정산 뒤 전체 칩이 보존된다.
- 환경, hero 정책, villain 정책 시드가 별도 입력이며 파생 규칙이 고정된다.
- 패키지 버전과 스키마 버전이 명세 값과 일치한다.
- 기존 88개 검증 의도와 새 계약 테스트가 모두 통과한다.
