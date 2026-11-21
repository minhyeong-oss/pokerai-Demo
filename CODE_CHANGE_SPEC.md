# 헤즈업 홀덤 의사결정 시뮬레이터 코드 수정 명세

- 문서 버전 1.1
- 작성일 2026-11-22
- 기준 문서 PRD v1.3
- 대상 저장소 `minhyeong-oss/pokerai-Demo`, 커밋 `4252698`
- 상태 PRD v1.3 기준본과 수정 명세 보완 완료. Python 코드 구현은 시작하지 않음

## 1. 개요

현재 코드를 PRD v1.3과 대조해 수정이 필요한 위치와 내용을 정의한다. 각 항목은 CR 번호, 대상 파일, 현재 동작, 요구 동작과 PRD 근거를 가진다. 기존 테스트의 세부 분류는 이 문서에서 미리 단정하지 않고 다음 단계에서 `기존 충족`, `수정`, `신규`로 나눈다.

## 2. 결론

- 사용자가 결정한 4개 항목(칩 단위, 필드명, 스냅샷, 리버 전수 계산)만 반영해서는 PRD를 만족하지 못한다.
- `opponents.py`, `rollout.py`, `baselines.py`는 수정 규모가 재작성급이다. 공통 상대 행동 모델을 먼저 추가한 뒤 기존 호출부를 점진적으로 교체한다.
- 엔진 쪽 칩 단위(`environment.py`)는 이미 1칩 = 0.01BB이므로 수정이 없다.
- 정책 시드에서 덱 시드를 역산할 수 있다. PRD의 정보 경계 위반이며 가장 먼저 고쳐야 한다.

## 3. 파일별 수정 요약

| 파일 | 수정 규모 | 관련 CR | 단계 |
|---|---|---|---|
| `observation.py` | 큼 | CR-01~05 | 1 |
| `environment.py` | 중간 | CR-02, CR-05, CR-06 | 1 |
| `simulation.py` | 중간 | CR-07, CR-30, CR-31 | 1, 6 |
| `actions.py` | 중간 | CR-02, CR-08~10 | 2 |
| `opponents.py` | 재작성 | CR-11~15 | 3 |
| `baselines.py` | 재작성 | CR-16 | 3 |
| `agent.py` | 중간 | CR-14, CR-17, CR-27 | 4, 5 |
| `rollout.py` | 재작성 | CR-18~21 | 4 |
| `policy.py` | 큼 | CR-22, CR-23 | 4 |
| `equity.py` | 큼 | CR-24~26 | 5 |
| `ranges.py` | 작음 | CR-27 | 5 |
| `logging.py` | 중간 | CR-28 | 5 |
| 신규 스냅샷 저장소 모듈 | 신규 | CR-27 | 5 |
| `evaluate.py` | 중간 | CR-29 | 6 |
| `cli.py` | 중간 | CR-32 | 6 |
| `pyproject.toml` | 작음 | CR-21, CR-29, CR-34 | 4, 6 |
| `PRD.md`, README | 작음 | CR-33 | 1 |
| `cards.py`, `train.py`, `screen_adapter.py` | 없음 | 없음 | 없음 |

## 4. 수정 요구사항

### 4.1 단계 1. Observation, 환경, 시드

#### CR-01. Observation 필드명 통일

- 대상: `observation.py`와 이 필드를 읽는 `actions.py`, `policy.py`, `baselines.py`, `rollout.py`, `agent.py`, `environment.py`, `cli.py`
- 현재: `hero_cards`, `to_call`, `action_history`, `player_statuses`
- 요구: `hole_cards`, `call_cost`, `public_history`, `statuses`
- 근거: PRD §5.1
- 비고: `to_call` 값은 이미 남은 스택으로 제한되어 있어 값은 바꾸지 않는다. `PracticeHandResult` 같은 별도 결과 DTO의 필드명은 Observation 계약과 자동으로 묶지 않고 기존 공개 API 호환성을 검토해 별도로 결정한다.

#### CR-02. 금액을 정수 칩으로 통일

- 대상: `observation.py`, `environment.py`, `actions.py`, `policy.py`, `rollout.py`, `baselines.py`
- 현재: 엔진은 정수 칩인데 Observation을 만들 때 BB 실수로 변환한다. `pot`, `to_call`, `stacks`, 기여액, `min_raise_to`, `max_raise_to`, `raise_to`가 모두 `float`다.
- 요구: Observation, `LegalActions`, `CandidateAction`의 모든 금액을 `int` 칩으로 바꾼다. BB 변환(칩 / 100)은 CLI 출력과 평가 보고서에서만 한다.
- 근거: PRD §3.1, §5.1
- 비고: `actions.py`의 `_AMOUNT_TOLERANCE` 기반 중복 제거는 정수 동등 비교로 바꾼다.

#### CR-03. 식별자 추가

- 대상: `observation.py`, `environment.py`
- 현재: `hand_id`, `decision_id` 필드가 없다.
- 요구: `run_id`는 사용자 지정 식별자 또는 UUIDv4로 만들고 시드에서 파생하지 않는다. `hand_id=<run_id>:h<순번>`, `decision_id=<hand_id>:d<순번>`, `event_id=<hand_id>:e<순번>`으로 만든다. 순번은 0부터 단조 증가하고 재사용하지 않는다. 자동 생성 run_id는 계산 결과의 재현 비교에서 제외한다.
- 근거: PRD §5.1

#### CR-04. 헤즈업 고정

- 대상: `observation.py`, `policy.py`
- 현재: 플레이어 2~10명을 허용한다. `policy.py`는 `"out"` 상태를 참조한다.
- 요구: 플레이어 수를 2명으로 고정하고 직렬화 상태를 정확히 `active`, `folded`, `all-in`으로 제한한다. 내부 Enum을 쓰더라도 JSON 값은 이 세 문자열과 같아야 한다.
- 근거: PRD §3.1, §5.1

#### CR-05. 공개 이벤트 구조 고정

- 대상: `observation.py`, `environment.py`
- 현재: 기록에 `sequence`, 스트리트, 행동자, 종류, 금액만 있다. Observation의 이력은 자유 형식이고 구조 검증이 없다. 보드 이벤트가 없다.
- 요구: `public_history`를 `ActionEvent | BoardEvent`로 고정한다. 행동 이벤트는 `type=action`, `event_id`, `actor_index`, `street`, `action`, 정수 `paid_amount`, 정수 또는 null `raise_to`, `pre_state`를 가진다. `pre_state`에는 당시 board, pot, stacks, street·hand 기여액, call_cost, 엔진 범위 `legal_actions`, 정책의 중복 제거 후보 `candidate_actions`를 넣는다. 보드 이벤트는 `type=board`, `event_id`, `street`, 새 카드와 전체 board를 가진다. 두 구조를 타입과 값으로 검증한다.
- 근거: PRD §5.1, FR-2

#### CR-06. 비대칭 시작 스택 허용

- 대상: `environment.py`
- 현재: 시작 스택이 상수로 고정되어 있다.
- 요구: 시작 스택을 좌석별 인자로 받는다. 기본값은 각 10,000칩이다. 칩 보존 기대값, 종료 스택 합, 문서화된 `starting_stack` 속성과 관련 오류 메시지도 인자 합계를 사용하게 바꾼다.
- 근거: PRD FR-5, §8.1 숏스택 콜

#### CR-07. 덱 시드와 정책 시드 분리

- 대상: `simulation.py`, `agent.py`, `rollout.py`, `baselines.py`
- 현재: 정책 시드가 `덱 시드 × 10,000 + 좌석 × 1,000`이다. 10,000으로 나누면 덱 시드가 나오고 이 값이 판단 로그에 기록된다. 이벤트 ID가 `"{덱 시드}:{순번}"` 형태다.
- 요구: 입력을 `environment_master_seed`, `hero_policy_master_seed`, `villain_policy_master_seed`로 분리한다. 음수가 아닌 마스터 정수의 키는 `SHA-256(UTF-8("pokerai-master-v1:" + 10진수 값))`으로 만든다. 각 도메인 필드는 `4바이트 big-endian 길이 + UTF-8 값`으로 인코딩하고 HMAC-SHA-256 결과 앞 8바이트를 big-endian 비부호 정수 시드로 쓴다. 단순 덧셈·곱셈 파생은 금지한다.
- 먼저 정책별 핸드 시드를 파생하고 분석 시드는 그 핸드 시드에서 `(purpose, decision_index, candidate_id, sample_index)`로 파생한다. purpose는 `equity`, `rollout_response`, `rollout_runout`, `actual_action`을 분리한다. 후보 ID는 인덱스가 아니라 `fold`, `check`, `call`, `raise:<정수 raise_to>`다.
- 평가 쌍의 덱 시드는 `(environment_master_seed, deck, pair_index)`로 한 번만 만들어 좌석 교환 두 게임에서 동일하게 쓴다. 정책별 핸드 시드는 각 정책 마스터에서 `(pair_index, leg_index, physical_seat, policy_id)`로 파생한다.
- 단일 핸드는 덱 시드를 `(deck, single, hand_sequence)`, 정책별 핸드 시드를 `(single, hand_sequence, physical_seat, policy_id)`로 파생한다.
- UUID run_id가 재현 결과를 바꾸지 않도록 전체 decision_id와 event_id는 시드 키에 넣지 않는다. 핸드 내 `decision_index`를 안정적인 계산 좌표로 사용하고 로그에 별도로 남긴다.
- 정책, 실행 중 판단 로그와 ID에는 덱 시드나 환경 마스터를 넣지 않는다. 덱 시드는 종료 후 핸드 결과에만 저장한다. 표본 수·후보 수 변경이 덱과 실제 기준선 행동 스트림을 바꾸지 않는 테스트를 추가한다.
- 근거: PRD FR-6 시드와 접근 분리, §8.1 정보 경계

### 4.2 단계 2. 합법 후보와 사이징

#### CR-08. 범위 밖 후보 제거

- 대상: `actions.py`의 `generate_candidate_actions`
- 현재: `_clamp`로 범위를 벗어난 크기를 최소·최대값에 맞춘다.
- 요구: 범위를 벗어난 일반 사이징 후보는 제거한다. `_clamp`를 삭제한다.
- 근거: PRD FR-3

#### CR-09. floor 정수화

- 대상: `actions.py`
- 현재: `base_after_call + fraction × pot_after_call`을 실수 그대로 쓴다.
- 요구: `raise_to(f) = B + C + floor(f × (P + C))`로 계산한 뒤 범위를 검사한다.
- 근거: PRD FR-3

#### CR-10. 상황별 후보 집합

- 대상: `actions.py`
- 현재: 콜 금액 유무와 관계없이 최소 레이즈를 항상 후보에 넣는다.
- 요구: 콜 금액이 없으면 체크, 0.5팟, 1팟, 올인만 둔다. 콜 금액이 있으면 폴드, 콜, 최소 레이즈, 0.5팟, 1팟, 올인을 둔다.
- 근거: PRD FR-3 후보 표

### 4.3 단계 3. 상대 모델과 기준선

#### CR-11. 점수 모델 교체

- 대상: `opponents.py`
- 현재: `fold_bias`, `call_bias`, `aggression`, `bluff_frequency`를 비율로 정규화한다.
- 요구: f0, c0, r0, b 파라미터로 `fold_score`, `check_score`, `call_score`, `raise_score`를 계산하고 합법 계열에 대해 softmax를 적용한다. 네 모델의 파라미터는 PRD 표를 따른다.
- 근거: PRD FR-2

#### CR-12. 패 강도 h 교체

- 대상: `opponents.py`
- 현재: 프리플롭 h가 다른 식이다(AA=0.97, AKs=0.68, 22=0.55). 포스트플롭은 보드 매치·플러시·스트레이트 가산점이다.
- 요구: 프리플롭은 PRD 식을 쓴다(AA=1, AKs≈0.80617, 22≈0.25926, JTo≈0.56173, 72o≈0.15432). 포스트플롭은 최선 5장 패의 등급 c와 타이브레이크 벡터로 `h = (c + k) / 9`를 계산한다.
- 근거: PRD FR-2

#### CR-13. 크기 분포와 ε 혼합 추가

- 대상: `opponents.py`
- 현재: 콜 부담 q, 크기 z, ε 혼합이 없다. 레이즈 크기와 무관하게 우도가 같다.
- 요구: q와 z를 계산한다. raise 계열 확률을 `size_score`의 softmax로 크기별로 나눈다. 최종 후보별 확률에 ε=0.01 균등 혼합을 적용한다.
- 근거: PRD FR-2

#### CR-14. 우도 입력 확장

- 대상: `opponents.py`, `agent.py`
- 현재: 우도 계산에 보드와 `facing_bet` 여부만 넘긴다.
- 요구: 행동 직전의 보드, 팟, 콜 비용, 행동자의 스트리트 기여액, `raise_to`, 합법 후보 집합을 넘긴다. 값은 CR-05의 이벤트에서 가져온다.
- 근거: PRD FR-2

#### CR-15. random 모델 추가

- 대상: `opponents.py`
- 현재: 스타일 목록에 random이 없다. `simulation.py`가 random 상대에 balanced를 가정한다.
- 요구: 우도는 합법 후보 균등분포, 롤아웃 응답은 폴드·콜 균등분포인 random 모델을 추가한다.
- 근거: PRD FR-7

#### CR-16. 기준선 상대를 같은 모델로 교체

- 대상: `baselines.py`
- 현재: `HeuristicBaseline`이 자체 MC 에쿼티와 스타일별 임계값, 블러프 확률로 행동한다. 우도 계산 모델과 다른 모델이다.
- 요구: CR-11~13의 모델에서 자신의 패로 후보별 확률을 계산해 그 분포에서 행동을 뽑는다. `simulations` 인자와 에쿼티 호출을 제거한다. 파라미터는 설정과 로그에 저장한다.
- 근거: PRD FR-2 "같은 모델과 설정을 실제 기준선 정책과 레인지 갱신 우도 계산에서 재사용"
- 비고: random의 균등 선택은 현재도 PRD와 같다.
- 구현 방식: 점수·softmax·크기 분포를 한 번만 구현한 공통 `OpponentActionModel`을 먼저 추가한다. 우도 계산과 실제 기준선은 같은 객체를 호출하게 하고 기존 구현은 새 모델의 계약 테스트가 통과한 뒤 제거한다.

### 4.4 단계 4. 레인지 갱신과 근사 EV

#### CR-17. 이벤트 처리 방식

- 대상: `agent.py`
- 현재: 호출자가 `observe_opponent_action`을 불러 줘야 레인지가 갱신된다. `event_id`가 선택 사항이라 생략하면 같은 행동이 두 번 반영될 수 있다.
- 요구: 에이전트가 Observation의 `public_history`에서 아직 처리하지 않은 이벤트를 직접 처리한다. `event_id`를 필수로 한다.
- 근거: PRD §4 4번, FR-2

#### CR-18. 표본 단위 롤아웃

- 대상: `rollout.py`
- 현재: 레인지 평균 폴드율을 구한 뒤 콜 분기에 무조건부 에쿼티를 쓴다.
- 요구: 표본마다 가중 레인지에서 상대 패를 뽑고, 그 패로 폴드·콜 응답을 뽑고, 콜이면 같은 패로 런아웃과 쇼다운을 계산한다.
- 근거: PRD FR-4 상대 응답과 조건부 레인지

#### CR-19. 체크·콜도 MC 표본으로 계산

- 대상: `rollout.py`
- 현재: 체크·콜은 `에쿼티 × 팟` 닫힌 식이고 표본 수에 에쿼티 표본 수를 적는다.
- 요구: 체크·콜도 후보당 EV 표본으로 계산한다. EV 표본 수는 에쿼티 표본 수와 독립이며 기본 2,000이고 2 미만은 거부한다.
- 근거: PRD FR-4 표본 수와 선택

#### CR-20. 정산식 교체

- 대상: `rollout.py`
- 현재: 매칭된 일반 콜에서는 기존 식이 맞지만, 상대의 미콜 초과 베팅이 존재하는 비대칭 상태에서도 `에쿼티 × (팟 + 콜) - 콜`을 그대로 써 초과액까지 승리 대상으로 계산한다.
- 요구: `contestable_pot = 2 × min(H, V)`, `hero_uncalled_refund = max(H - V, 0)`, `sample_EV = share × contestable_pot + refund - ΔH`를 쓴다. 폴드 분기는 현재 팟이다.
- 근거: PRD FR-4 정산식
- 비고: 폴드 EV 0과 폴드 분기 EV는 현재도 PRD와 같다.

#### CR-21. EV 신뢰구간

- 대상: `rollout.py`
- 현재: 1.96 정규근사 구간이다.
- 요구: 양측 t 기반 95% 구간으로 바꾸고 `ci_method=student_t_mc`를 기록한다. 폴드는 `exact=true`, `sample_count=0`, 구간 [EV, EV]다.
- 구현: `scipy==1.17.1`의 `scipy.stats.t.ppf`를 사용하고 SciPy와 전이 의존성의 실제 버전을 로그에 기록한다.
- 근거: PRD FR-4, FR-1

#### CR-22. 동률 규칙

- 대상: `policy.py`의 `_tie_break_key`
- 현재: 허용 오차 없이 점수, EV, 구간 하한, 종류(체크, 폴드, 콜, 레이즈), 투자액 순으로 비교한다.
- 요구: 최대 평균 EV와 차이가 1e-9칩 이내인 후보를 먼저 그룹으로 만든다. 그 그룹 안에서 추가 투자액이 작은 행동, 이후 체크, 콜, 폴드, `raise_to` 오름차순으로 고른다. 비추이적일 수 있는 `math.isclose`를 정렬 키에 직접 넣지 않는다. 신뢰구간 겹침 여부를 진단 정보에 남긴다.
- 근거: PRD FR-4

#### CR-23. PRD에 없는 경로 제거와 Decision 확장

- 대상: `policy.py`
- 현재: `risk_aversion` 옵션, `SimpleResponseModel`, `_showdown_proxy`(무조건부 에쿼티 사용)가 있다. `ActionEvaluation.samples`는 양수만 허용한다. 구간은 `NormalDist` z 점수로 만든다. `Decision`에는 에쿼티, 구간, 신뢰수준, 선택 규칙만 있다.
- 요구: `risk_aversion`과 프록시 경로를 제거하고 모든 후보를 CR-18 롤아웃으로 평가한다. 표본 수 0과 `exact` 표시를 허용한다. `Decision`에 승·무·패 확률, 구간별 `ci_method`·`confidence_level`, 상대 모델 식별자·파라미터, 레인지 버전과 스냅샷 해시 ID, 시드, 계산 시간, 가정 식별자를 추가한다.
- 근거: PRD §5.2, FR-4

### 4.5 단계 5. 에쿼티, 스냅샷, 로그

#### CR-24. 에쿼티 신뢰구간

- 대상: `equity.py`
- 현재: 에쿼티 구간이 정규근사다. 승·무·패 구간과 `ci_method` 기록이 없다. 표본 수 1을 허용한다.
- 요구: 에쿼티는 Hoeffding 95% 구간(반폭 `sqrt(ln(2/0.05) / (2n))`, [0, 1]로 제한), 승·무·패는 Wilson 95% 구간을 쓴다. 모든 구간에 `ci_method`와 `confidence_level`을 기록한다. 표본 수 2 미만은 거부하고 기본값은 10,000이다.
- 근거: PRD FR-1

#### CR-25. 리버 전수 계산

- 대상: `equity.py`
- 현재: 모든 스트리트가 MC다.
- 요구: 보드 5장이고 상대가 1명이면 레인지 전체를 열거한다. `ci_method=exact`, `confidence_level=null`, 구간 [값, 값]과 열거 수를 기록한다. 에쿼티 시드는 쓰지 않는다. 상대 2~9명 리버는 MC를 유지한다.
- 근거: PRD FR-1 리버 전수 계산

#### CR-26. 완전 열거 검증 모드

- 대상: 별도 검증 모듈, `equity.py`, 테스트 자원
- 현재: 열거 모드가 없다.
- 요구: 실제 상대 패를 입력받지 않는 정책용 에쿼티 API와 별개로, 고정된 양쪽 홀카드와 선택적 보드를 받는 검증 전용 완전 열거 API를 추가한다. 정책과 에이전트 모듈은 이 API를 호출할 수 없다. `AsAh` 대 `KcKd` 보드 없음의 기준값을 테스트 자원에 고정하고 산출 방법, 하드웨어, 코드 버전, 실측 시간을 함께 저장한다.
- 근거: PRD FR-1, §8.2

#### CR-27. 레인지 스냅샷 저장소

- 대상: 신규 모듈, `ranges.py`, `agent.py`, `logging.py`
- 현재: 로그에 레인지 분포가 없고 저장소도 없다. `range_version`의 해시는 이벤트 이력의 해시다.
- 요구: 갱신 후 확률을 `mass = ROUND_HALF_EVEN(Decimal(str(p)) × 10^15)`의 비음수 정수로 양자화하고, 저장 전에 이 mass 분포를 에쿼티와 롤아웃 계산에도 적용한다. 0 mass 조합은 제거하고 전부 0이면 오류다. 부동소수점으로 다시 정규화해 저장하지 않고 샘플링은 `[0, total_mass)` 정수 누적합으로 한다.
- 스냅샷 스키마 1은 `schema_version`, `quantization_scale=10^15`, 고정 덱 순서의 `(card1, card2, mass)` 목록과 `total_mass`를 가진다. 공백 없는 정렬 키 UTF-8 JSON의 압축 전 바이트를 SHA-256으로 해시한다.
- 파일은 `<policy_id>/<hash[0:2]>/<hash>.json.gz`이며 gzip `mtime=0`을 사용한다. 같은 디렉터리의 임시 파일을 flush·fsync한 뒤 원자적 rename으로 확정한다. 스냅샷을 먼저 확정하고 판단 레코드를 쓴다.
- 복원 시 파일·스키마·total_mass·내용 해시를 검증한다. 누락이나 불일치면 오류로 중단한다. 같은 해시 보장 범위는 같은 코드·의존성·운영 환경이다. 저장소는 정책별로 분리한다.
- 근거: PRD FR-6 레인지 스냅샷 저장소

#### CR-28. 판단 로그 항목 추가

- 대상: `logging.py`, `agent.py`
- 현재: 시드, 표본 수, 스타일 이름은 기록한다. 아래 항목은 없다.
- 요구: 상대 모델 전체 파라미터, 스냅샷 해시 ID, 처리한 이벤트 ID, 가정 식별자 `fold_or_call_then_checkdown_v1`, 의존성 버전, 구간별 `ci_method`·`confidence_level`, 다중 비교 보정 여부, 난수 생성기 종류를 추가한다. 개별 판단의 다중 비교 보정 값은 `not_applicable`로 기록한다. 리버에서는 에쿼티 시드를 null로 기록한다.
- 근거: PRD FR-6 판단 로그

### 4.6 단계 6. 평가와 CLI

#### CR-29. 평가 통계

- 대상: `evaluate.py`
- 현재: 1.96 정규근사 구간이다. Bonferroni 보정이 없다.
- 요구: `scipy==1.17.1`의 `scipy.stats.t.ppf`로 `t(0.975, n-1)` 구간을 계산하고 `ci_method=paired_student_t`를 기록한다. 사전에 선언한 비교 수 m으로 `t(1-0.05/(2m), n-1)` Bonferroni 구간을 함께 출력한다. 보정 구간이 0을 포함하면 동시 우위로 표시하지 않는다.
- 근거: PRD FR-7, §8.4
- 비고: 좌석 교환 쌍 평균을 통계 단위로 쓰는 것과 한 쌍일 때 구간 null은 현재도 PRD와 같다.

#### CR-30. 가정 모델과 실제 상대 분리

- 대상: `simulation.py`
- 현재: 가정 모델이 항상 실제 상대 스타일과 같다.
- 요구: 두 값을 별도 설정으로 받는다. 보고서에 파라미터 일치 여부를 출력한다.
- 근거: PRD FR-7

#### CR-31. 평가 프로파일과 비용 측정

- 대상: `simulation.py`
- 현재: 프로파일이 없고 기본값이 20핸드, 에쿼티 500, EV 200이다. 계산 시간을 출력하지 않는다.
- 요구: smoke(상대별 10쌍, 에쿼티 1,000, EV 200)와 최종 목표(1,000쌍, 10,000, 2,000) 프로파일을 명시 선택하게 한다. 시드 쌍 수, 핸드 수, 계산 시간, 핸드당 의사결정 수, 후보 수와 하드웨어를 기록한다. 실행 중 표본 수를 바꾸지 않는다.
- 좌석 교환: 쌍별 덱 시드는 pair_index만으로 한 번 파생해 두 leg에서 같게 유지한다. 정책별 시드만 pair_index, leg_index, physical_seat, policy_id로 분리한다.
- 근거: PRD FR-7

#### CR-32. CLI 정리

- 대상: `cli.py`
- 현재: `advise`가 금액을 BB 실수로 받는다. 기본 표본 수가 `play`는 2,000과 500, `advise`는 EV 500이다. `--risk-aversion`이 있다.
- 요구: 금액 입력을 정수 칩으로 직접 받고 소수 입력은 거부한다. 기존 BB 입력과 조용히 혼동되지 않도록 도움말과 플래그 이름에 chip 단위를 명시하거나 명시적 `--amount-unit chips`를 요구한다. 기본 표본 수를 10,000과 2,000으로 맞춘다. `--risk-aversion`을 제거한다. 가정 모델 선택, 평가 프로파일 선택, 검증 전용 완전 열거 옵션을 추가한다.
- 근거: PRD FR-1, FR-4, FR-7

#### CR-33. 문서 갱신

- 대상: `PRD.md`, README
- 현재: 기준 커밋 `4252698`의 `PRD.md`는 v1.2였고 이번 문서 작업에서 v1.3 기준본으로 교체했다. README는 여전히 금액을 BB로 설명한다.
- 요구: PRD v1.3과 이 수정 명세를 문서 커밋으로 먼저 고정한다. 코드 반영 커밋에서 README의 금액 설명을 CR-02, CR-32에 맞춘다.

#### CR-34. 버전과 의존성 계약

- 대상: `pyproject.toml`, `logging.py`, 스냅샷 모듈, 모델 설정
- 현재: 패키지 0.1.0, 판단 로그 스키마 1이며 SciPy 의존성이 없다.
- 요구: 패키지 버전을 0.2.0, 판단 로그 `schema_version=2`, 레인지 스냅샷 `schema_version=1`, 정책 모델 버전을 `baseline-ev-v2`로 올린다. `scipy==1.17.1`을 런타임 의존성에 추가한다. 스키마 1 판단 로그를 스키마 2로 조용히 해석하지 않고 지원하지 않는 스키마는 명시적으로 거부한다.
- 근거: PRD FR-4, FR-6, FR-7

## 5. 테스트 수정

기존 회귀 테스트는 최대한 유지한다. 계약이 바뀐 테스트만 수정하고 PRD v1.3 합격 테스트를 추가한다. 다음 단계에서 각 항목을 `기존 충족`, `수정`, `신규`로 분류하며, 분류 전에는 기존 테스트를 삭제하거나 전면 재작성한다고 가정하지 않는다.

| 테스트 파일 | 추가·수정할 검증 | PRD |
|---|---|---|
| `test_observation.py` | PRD 필드명, 정수 칩, 2명 고정, 이벤트 구조 | §5.1 |
| `test_environment.py` | 비대칭 스택의 초과 베팅 반환과 경쟁 팟, 칩 보존 | §8.1 |
| `test_actions.py` | floor 정수화, 범위 밖 제거, 상황별 후보, 숏 올인 | §8.1 |
| `test_opponents.py` | h 검증값 5개, ε 혼합 후 확률 합 1과 양수, 크기 선호 방향 | §8.2 |
| `test_baselines.py` | 기준선 행동 분포가 우도 모델과 일치 | FR-2 |
| `test_ranges.py` | 보드 카드 제거, 중복 이벤트 무효, 고정 `10^15` mass 양자화·total_mass·정수 누적 샘플링 | §8.2, §8.1 |
| `test_rollout.py` | 올인 콜 EV 기준식, 강한 패만 콜하는 합성 모델, 숏스택 정산의 PokerKit 대조 | §8.2 |
| `test_policy.py` | 확정 승리·패배 상태의 선택, 동률 규칙 | §8.3 |
| `test_equity.py` | 완전 열거 기준값 대비 MC 오차 0.01, 리버 전수 계산, 표본 수 변경 무영향 | §8.2 |
| `test_audit_logging.py` | 로그와 스냅샷에 상대 패·덱 시드 없음, 정규 JSON·SHA-256 고정값, 스냅샷 누락·스키마·total_mass·해시 불일치 오류, 복원 | §8.1 |
| `test_simulation.py` | 숨겨진 상대 패를 바꿔도 Decision 동일, 세 마스터 시드 분리, 표본·후보 수 변경 독립성, 쌍 내 동일 덱·정책 좌석 교환, 재현, PHH 재생 | §8.1, §8.2, FR-7 |
| `test_evaluate.py` | 항상 폴드 시나리오 -75bb/100, t 구간, Bonferroni | §8.3, §8.4 |
| `test_cli.py` | 칩 입력 변환, 프로파일 선택 | FR-7 |

## 6. 수정하지 않는 부분

- `environment.py`: 1BB = 100칩, SB 50, BB 100, 스택 10,000. 좌석 0이 BB, 좌석 1이 SB·버튼. 매 행동 후와 종료 시 칩 보존 검사. 불법 행동은 상태 변경 전에 거부.
- 레인지 갱신 경로: 유효 조합이 없으면 오류로 중단. 같은 이벤트 ID의 동일 이벤트 재처리는 무시하고, 같은 ID에 다른 내용이 오면 오류로 중단.
- `logging.py`: NaN, Infinity 직렬화 거부.
- `cards.py`: 엄격한 표기 검사, 중복 카드와 잘못된 장수 거부.
- `screen_adapter.py`: 인터페이스 정의만 있고 화면 캡처·클릭 기능 없음.
- `train.py`: Kuhn CFR. PRD에서 후속 범위이므로 그대로 두고 MVP 검증 대상에서 제외한다.
- `pyproject.toml`: `pokerkit==0.7.7` 고정은 유지하되 CR-34에 따라 `scipy==1.17.1`과 패키지 0.2.0을 반영한다.

## 7. 작업 순서와 종료 기준

PRD §10의 단계와 같은 순서를 따르되, 각 CR은 `실패하는 합격 테스트 추가 또는 기존 계약 테스트 수정 → 구현 → 관련 테스트 → 전체 회귀 테스트` 순서로 완료한다. 다음 단계의 테스트 작성을 앞 단계 완료까지 미루지 않는다.

| 단계 | CR | 종료 기준 |
|---|---|---|
| 1 | CR-01~07, CR-33 | Observation이 §5.1과 일치, 정책 시드와 이벤트 ID에서 덱 시드를 유도할 수 없음 |
| 2 | CR-08~10 | §8.1 합법 행동·레이즈 검증 통과 |
| 3 | CR-11~16 | h 검증값과 ε 정규화 통과, 기준선과 우도가 같은 모델 사용 |
| 4 | CR-17~23 | §8.2 레인지·조건부 응답·EV 정산과 §8.3 행동 선택 통과 |
| 5 | CR-24~28 | §8.2 에쿼티·리버 전수 계산과 §8.1 정보 경계·스냅샷 무결성·재현 통과 |
| 6 | CR-29~34 | §8.3 -75bb/100과 통계 검증 통과, 버전·스키마 계약과 프로파일 비용 측정 완료 |

## 8. 다음 단계에서 확인·분류할 사항

- 기존 테스트 16개를 CR별로 `기존 충족`, `수정`, `신규`로 분류한다.
- `hand_contributions`가 미콜 반환을 엔진 처리 시점에 한 번만 차감하는지(PRD §5.1).
- JSONL의 마지막 불완전 줄을 복구 시 보고하는지(PRD FR-6).
- 평가 도중 실패한 핸드를 실패로 표시하고 시드와 원인을 남기는지(PRD §7).
- 좌석 인덱스와 버튼·블라인드 매핑을 검증하는 테스트가 있는지(PRD FR-5).
- 고정 mass 양자화가 반복 갱신에서 모든 조합을 0으로 만들 가능성과 오류 경로를 검증한다.
- HMAC 파생 입력의 길이 구분 인코딩, 64비트 변환과 후보 ID 정규 문자열의 고정 기대값을 테스트 자원으로 만든다.
