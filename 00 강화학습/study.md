# 강화학습 공부 
출처 : https://www.edwith.org/move37/lecture/59774?isDesc=false

### 지도 학습
- 어떤 값을 예측
- 값이 이미 데이터 속에 존재 - 레이블, 타겟 변수, 종속 변수
- 이외의 특성 - 독립 변수
- 선형 회귀, 로지스텍 회귀, 신경망, 결정트리 - 입력값을 가지고 레이블을 알아내는 함수

### 비지도 학습
- 사용할 수 있는 레이블이 없는 경우
- K-means, 혼합 모델
- 데이터를 그룹으로 묶어 시각화하고 연관성 파악
- 데이터 축약해서 표현 - 오토 인코더
- 데이터 전처리에 주로 사용

### 마르코프 체인
- 어떠한 상태에서 다른 상태로 연속적으로 이동할 수 있는 상태들의 집합이 있다고 가정
- 각각의 이동은 한단계로만 되어있고 'T'라는 전이 모델을 기반으로 함
- 'T'는 어떻게 하나의 상태에서 다른 상태로 이동할 수 있는지를 정의
- 현재 상태가 주어졌을 때 미래는 과거와 조건부 독립이라는 것으로 현재 진행중인 상태는 딱 한 단계전의 상태에만 의존한다
- 가장 일반적으로 어떤 환경 내 에이전트의 학습을 강화학습의 문제로 표현하는 방식이 마르코프 결정 과정

#### 마르코프 결정 과정의 요소
- 상태 집합
- 초기 상태
- 행동 집합
- 전이 모델 - 다음 상태에 도달할 수 있는 확률
- 보상 함수 - R(s,a)

# 벨만 방정식
- State - 에이전트가 환경의 특정 시점에서 관찰하고 있는 것을 숫자로 표현한 것
- Action - 현재 상태에 정책을 적용하여 계산된 에이전트가 환경에 제공하는 입력
- Reward - 에이전트가 게임의 목표를 얼마나 잘 수행하고 있는지를 반영하는 환경의 피드백 신호

#### 강화학습의 목표
- 주어진 현재 상태에서 환경이 제공하는 장기적 기대 보상을 극대화할 최적의 조치를 선택합니다.

#### 동적 계획법
- 알고리즘의 종류
- 복잡한 문제를 하위문제로 나누어 분해함으로써 단순화 함
- 하위 문제를 재귀적으로 해결

#### 벨만 방정식은 어떤 질문에 대답하나?
- 현재 상태에서 최선의 조치를 취하고 각 후속 단계에서 기대할 수 있는 장기적인 보상은 무엇인가?
- State의 가치는 무엇인가?
- 각 상태의 장점 또는 단점과 관련된 예상 보상을 평가하는 데 도움이 됩니다.

---------------------
- 마르코프 state란?
    
    State가 마르코프 하다라고 말할 수 있다는것은
    
    State10의 상태는 오직 State9에 의해서만 결정되고, State9의 상태는 State8에 의해서 결정됨. 
    
    이런 재귀를 통해서 State0까지 내려감. 즉 현재의 상태는 과거의 모든 상태를 포함하고 있다는 뜻.
    
    - 환경 State는 마르코프하고
    - history  또한 마르코프 하다
    - ‘*’ 이 붙으면 최적화를 의미함
- Observable 환경
    - MDP (마르코프 decision process) → agent state == environment state
    - POMDP(Partial observable MDP) → agent state ≠ environment state
    
- RL agent의 구성요소
    - Policy
        - agent의 행동을 규정
        - state를 넣으면 action을 뱉어줌
        - 파이 로 표시함  a = 파이(s)
        - Deterministic policy : 액션을 뱉어줌
        - Stochastic policy: 액션의 확률들을 뱉어줌
    - Value Function
        - 상황이 얼마나 좋은지를 나타냄
        - future reward의 합산을 예측함
        - 여러가지 경우의수를 따라갔을때의 총 기대값
        - v파이(s) → S(t)에서 파이(Policy)를 따라갔을때의 S총 기대값
        - r → decrease factor? State에서 멀어질수록 점차 작게 영향주게끔      
    - Model
        - 환경이 어떻게 될지 예측하는 애
        - 리워드를 예측하는것,  State transition을 예측하는것 을 수행해야함
        - 모델이 있으면 model base, 없으면 model free
        
- RL agent의 분류 방법
    - Value Based : No Policy , Value Function
    - Policy Based : Policy, No Value Function
    - Actor Critic : Policy, value Function
    - Model Free / Model Based
        - Policy, value function, No Model → model free
        - policy, value function, Model → model based
- 문제의 분류 방법
    - 러닝 문제 → 환경에 던져주고 상호작용하며 학습
    - Planning 문제 → environment를 안다고 표현 (서치 라고도 함)
        - 리워드, 트랜지션이 어떻게 되는지 암
        - environment를 아니까 내부 시뮬레이션이 가능함
        - 몬테카를로 서치 같은 경우
        
- Exploration, Exploitation
    - Exploration → 환경에 대한 정보를 모으는것
    - Exploitation → 정보들을 이용해 reward를 극대화 하는것
- Prediction , Control 문제
    - Prediction : 미래를 평가하는 문제 → Given a Policy → value function을 학습시키는 문제다
    - Control : 미래를 최적화하는것 → Find the best Policy → policy를 찾는 문제다
