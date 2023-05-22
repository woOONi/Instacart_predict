# Instacart 예측 프로젝트
### 1. 프로젝트 개요

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2ca48dc7-fdcc-49ab-8b7a-71dd4e42d021/Untitled.png)
![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b4c1cf00-3285-405b-a1be-7210a7cfa3bd/Untitled.png)

1-1. 주제

- Instacart의 이전에 구매한 제품이 사용자의 다음 주문에 포함될지 예측하는 분류 모델 만들기

1-2. 주제 선정의 배경

- Instacart 소개
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c5dac99b-d619-4216-b20a-bc07e3bf1330/Untitled.png)
    
    - 쇼퍼(Shopper)가 대신 장을 봐주는 서비스로, 스토어마다 상품을 전시하고 고객이 전시된 상품들을 주문하면 해당 내역을 쇼퍼가 대신 장을봐서 1-2시간 내 배송을 수행
    - 온라인 장보기의 우버로 불림
    - 월마트, 코스트코, 알디 등 미국 주요 Grosery 스토어와 제휴를 맺고 있음
- 목적
    - 재주문할 제품을 예측하는 모델을 만드는 과정을 통해, Classifier 알고리즘마다 차이를 이해하고, Feature Engeering과 하이퍼 파라미터 튜닝을 함으로써 예측력을 높이는 것
    - Feature 를 만들기까지 고객, 제품, 주문 내역에 대한 EDA를 심도있게 진행함으로써 
    Feature Engeering 한 과정에 대해 설득력을 높이는 것
- 필요성
    - 다음 주문에 재주문이 될 제품을 예측하는 것을 통해
    홈페이지 메인에 어떤 제품 위주로 노출을 시켜야 할지,
    쇼퍼가 어느 식료품에서 대기해야 할지, 
    재주문 될 제품을 전시하는 식료품에 어느 정도 제품 준비가 필요할지 등의 정보를 확보할 수 있다.
    그 결과 고객이 주문하는 타이밍에 맞춰서 서비스가 원활하게 제공될 수 있다.

1-3. 본 프로젝트의 활용 방안 제시

이전에 구매한 제품이 사용자의 다음 주문에 포함될지 예측하도록 모델을 만들어서, 

Instacart의 데이터 과학팀이 사용자가 다시 구매하거나 처음으로 시도하거나 세션 중에 장바구니에 추가할 제품을 예측하는 모델을 서비스화 시키는 데 활용할 수 있다.

### 2. 프로젝트 팀 구성 및 역할

| 팀원 | 역할 |
| --- | --- |
| 김미리 | EDA, 모델링, PPT |
| 김상민 | 팀장, EDA, 모델링, PPT |
| 김지원 | EDA, 모델링, PPT |
| 양희권 | EDA, 모델링, PPT |
| 이호연 | EDA, 모델링, PPT, 발표 |

### 3. 프로젝트 수행 절차 및 방법

3-1. 데이터 설명 (데이터 출처, 데이터 개요)

- 데이터 출처

[Instacart Market Basket Analysis](https://www.kaggle.com/competitions/instacart-market-basket-analysis/data)

- 데이터 개요
    
    대회에서 주어진 데이터 셋은 제출 데이터셋까지 포함하여 총 7개
    
    - **aisles** , **departments** , **products** 데이터셋
        - 위 3개의 데이터들은 **products** 설명해주는 데이터로 아래와 같은 관계이다.
            
            ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f5dbf566-56ef-415b-a8f7-6e287ef6ffb2/Untitled.png)
            
    - **orders** 와 **order_products_train** , **order_products_prior** 의 관계
        - **orders** 데이터는 주문번호, user의 번호가 포함되어 있으며 몇 번째 구매인지, 무슨 요일, 몇 시에 구매했는지 이전 주문일자와 몇일 차이나는지에 관한 데이터가 포함되어 있다.
            
            ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/616f9f38-591f-4e80-abe7-a2f2be02bb8f/Untitled.png)
            
        - **orders** 데이터의 `eval_set`
            
            > **prior, train, test** 로 나눠지며, user가 구매한 가장 최근의 데이터를 **train**과 **test**로, user의 이전 데이터를 **test** 로 두었다.
            > 
            
        - **order_products_prior** 데이터는 주문번호에 따른 구매 제품번호, 장바구니 순서, 재구매 여부에 관한 데이터가 들어있으며 `eval_set = prior` 인 경우의 데이터이다.
            
            ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/faf7444b-778b-43a6-8a0a-8753cab3088b/Untitled.png)
            
        - 마찬가지로 **order_products_train** 데이터도 주문번호에 따른 제품번호, 장바구니 순서, 재구매 여부에 관한 데이터가 들어 있으며 `eval_set = train` 인 경우의 데이터이다
            
            ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/eb59b80b-554e-41f6-87c5-49c432a6a135/Untitled.png)
            

3-2. 데이터 샘플

```python
# 캐글 데이터 불러오기
orders = pd.read_csv('data/orders.csv')
order_products_train = pd.read_csv('data/order_products__train.csv')
order_products_prior = pd.read_csv('data/order_products__prior.csv')
products = pd.read_csv('data/products.csv')
aisles = pd.read_csv('data/aisles.csv')
departments = pd.read_csv('data/departments.csv')
```

| orders | (3421083, 7) |
| --- | --- |
| order_products_train | (1384617, 4) |
| order_products_prior | (32434489, 4) |
- 예시
    
    ![스크린샷 2022-10-12 오후 4.42.12.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ec3eaab8-6da7-445f-80b3-64d24f81fece/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-10-12_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.42.12.png)
    

3-3. 데이터 수집 및 전처리

- 데이터 수집
    - 캐글 경진대회에서 데이터셋 다운로드
- 데이터 전처리
    - 결측치 확인
    - 중복값 확인
    - EDA하면서 파생변수 만들기

3-4. 활용 라이브러리 등 기술적 요소

- [x]  pandas
- [x]  numpy
- [x]  matplotlib
- [x]  seaborn
- [x]  sklearn

3-7. 프로젝트에서 분석한 내용

- [x]  결측치 확인
- [x]  중복값 확인
- [x]  데이터 타입 확인
- [x]  이상치 확인
- [x]  전체 수치 변수의 히스토그램 그리기
- [x]  수치 데이터 기술 통계 구하기
- [ ]  범주 데이터 기술 통계 구하기
- [x]  파생변수 만들기
- [x]  데이터프레임 병합
- [x]  상관계수 구하기
- [x]  빈도수 구하기
- [x]  groupby, pivot_tabe 등을 통한 데이터 집계
- [ ]  기타:

3-6. 결과물

- Best Model의 스코어
    - 캐글 제출 이미지
        
        ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d9788760-15e9-4a12-83e6-af02e78a18cd/Untitled.png)
        
    - 모델, 파라미터, 스코어 소개
    
    ```markdown
    * 모델 : Catboost
    * 파라미터 : random_state=42, n_estimators=1000, learning_rate=0.2, max_depth=5
    * F1 스코어 : 0.37366
    ```
    
- 링크
    
    [LIKELION_AIschool/김삼민team_ipy정리.ipynb at main · woOONi/LIKELION_AIschool](https://github.com/woOONi/LIKELION_AIschool/blob/main/Mid2_Project_김삼민/김삼민team_ipy정리.ipynb)
    
    > 자세히 알아보고 싶으시면 링크를 확인해주세요!
    > 

3-7. 프로젝트 회의록

[3팀 회의록](https://www.notion.so/b579d708c9ba4594970a1273bd7a41bd)

### 4. 프로젝트 회고 및 개선점

4-1. 피드백

발표 후 QnA 시간에 나온 질문과 피드백을 모두 작성해 주세요. 
듣는 즉시 바로 작성하면 빠뜨리지 않고 모두 적을 수 있을 거에요! 이때 개선점으로 넘어가도 좋을 반영할 부분을 발견했다면 최종 제출 전에 그 부분 위주로 정리하는 것도 좋아요. 그리고 발표 시간에 적극적으로 질문과 피드백을 주고 받으면 서로의 성장에 무척 도움이 되겠죠?

[조은 강사님 피드백]

1. 발표에 대한 피드백
- 커머스쪽 커리어, 상품 분석을 디테일하게 하고싶다는 분들이 포폴 만들기에 좋은 데이터셋이다.
- 데이터도 나뉘어져서 어떻게 조합해서 써야하는 지 배우기 좋음.
- 데이터 양이 큼 ㅠㅠ
    
    → 데이터를 샘플링해서 줄여서 모델을 돌려보는 것도 방법이다.
    
- 그리드서치 모델별 시각화한 거 좋았다.
- 재주문에 대한 분석 인상적
- 신호, 소음 구분해서 피쳐 찾는 것도 인상적.

1. 회고 활용 방법
- 회고하면서 Skill 에 대해 Focus를 맞춰서 진행하면 좋다.

4-2. 회고(개인의 문제인식~극복과정)

모든 팀원이 각자 꼼꼼히 작성해 주세요. 농도 짙은 회고는 프로젝트와 나, 팀의 성장에 밑거름이 됩니다.

| 이름 | 회고 내용 |
| --- | --- |
| 김미리 | 많이 웃고 즐기면서 프로젝트를 해서 감사했다. :)
인스타카트에 너무 친숙해져서 장바구니에 담고 구매할뻔..ㅋㅋ

혼자 했으면 파생변수랑 피쳐 만드는 것만 3일은 걸리며, 헤맸을 거 같다.
그리고... 컴퓨터 메모리의 중요성을 많이 실감한 프로젝트였다 (ㅠㅠ)
맥북에서 팬 돌아가는 소리가 이렇게 커질 수 있다니 !! 

장바구니 연관분석이나 고객 세그먼트 나눠서 군집화해보는 건 
정해진 기간내에 하지 못했지만
서브 프로젝트로 나중에 잊지않고 꼭 해봐야겠다. |
| 김상민 | 밝은 분위기에서 미드 프로젝트를 진행해서 좋았다. 즐겁고 다양한 주제를 이야기를 나누고 정보 공유를 하면서 해서 좋았습니다.
다른 노트북과 비교하면서 EDA해서 어떻게 새로운 피처를 찾아야할까 고민했지만 쉽지 않았다

고객 세그먼트, 군집화를 해보고 싶었지만 시간이 없어서 아쉬웠다.

다양한 머신러닝을 돌려서 최적의 머신러닝 모델을 알고싶었지만 너무 오래걸려서 아쉬웠습니다.

그래도 모두 마지막까지 열심히 해서 감사합니다!! |
| 김지원 | 이번 프로젝트 너무 재밌게 했다!!!!
팀원분들과 즐겁게 이야기하면서 해서 다양한 아이디어도 더 나올 수 있었던 것 같았다!! 

개인적으로 boosting 모델과 어느 한 라이브러리와 충돌이 있어 해결하는데 시간이 많이 소모되어서 아쉬웠다.

고객 세그먼트를 해보고 싶었는데 모델 충돌과 하이퍼 파라미터 그리드 서치를 하면서 시간이 많이 소모되어 하지 못한 점이 아쉬웠다. 

추후에 연관 분석을 이용하여 좀더 정확도를 높이고 싶다. |
| 양희권 | 팀원 모두 즐거운 분위기 속에서 프로젝트를 진행해서 좋았다. 데이터가 여러개로 흩어져 있어 EDA와 피쳐엔지니어링이 까다로웠다.
고객 세그먼트도 해보고 싶었고 여러가지 해보고 싶은 것이 많았지만 생각보다 시간이 부족해 아쉬웠다.
처음에는 인스타마트 데이터를 선택한 팀이 우리팀 뿐이라 걱정을 많이 했지만… 나름 잘 헤쳐나간 것 같아서 뿌듯하다!
팀원들 모두 적극적으로 열심히 해주셔서 감사하다! |
| 이호연 | 처음 데이터를 다룰 때는 막막하고 어떻게 해야하나 감이 안 왔지만 팀원들과 많은 이야기를 나누면서 방향을 잡아갈 수 있었다. EDA를 통해 어떤 인사이트를 가져올 수 있을지 찾는 방법이 쉽지 않았다. 
또한 머신러닝을 더 많이 다양하게 돌려보고 싶었는데 노트북 메모리 용량이 부족하여 그리드 서치도 못 하고 다양하게 못 해, 너무 아쉽다… 
마지막으로 팀원들 덕분에 시작도 하 기 전부터 걱정 많았던 미드2 프로젝트를 매우 즐겁고 뿌듯하게 마무리하여 기분이 매우 좋다.
팀원분들께 감사의 인사도 함께 드립니다. (- - )( _ _) |

4-3. 개선점(팀 내에서 논의 및 합의된 개선 방향)

팀에서 논의된 공통의 내용을 작성해 주세요. 개선하고자하는 이유와 방향, 기대되는 결과도 같이 작성해주시면 더욱 좋습니다.

- 고객 세그먼트, 군집화, 장바구니 연관성 분석에 대해서 시도하지 못한 점

4-4. 추후 개선 계획

프로젝트는 여전히 살아 숨쉬고 있습니다. 
개선점에 대한 회고 이후, 가능하다면 실제 액션 계획도 세워보세요. 포트폴리오에서 ‘개선 시도/경험’은 아주 긍정적인 요소로 작용한답니다. 

- 개인 서브 프로젝트로 고객 세그먼트, 군집화, 장바구니 연관성 분석 해보기

### 5. 부록

5-1. 참고자료

[‎Instacart: Grocery delivery](https://apps.apple.com/us/app/instacart/id545599256?platform=iphone)

[Instacart Company | About Us](https://www.instacart.com/company/about-us)

[온라인 장보기 서비스 ‘인스타카트’ 파헤치기 | 요즘IT](https://yozm.wishket.com/magazine/detail/1598/)

[KATI 농식품수출정보](https://www.kati.net/board/exportNewsView.do?board_seq=95765&menu_dept2=35&menu_dept3=71)

[xgb of Instacart ML 2 Notebook](https://www.kaggle.com/code/charalambos/xgb-of-instacart-ml-2-notebook)

5-2. 출처

[In-Store Navigation to Help Make Finding Items Easier](https://www.instacart.com/company/shopper-community/in-store-navigation-to-make-finding-items-easier/)

[Shopper Community Insights & News | The Instacart Blog](https://www.instacart.com/company/shopper-community/)
