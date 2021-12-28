# 지역별 식당 키워드 추출 및 감정 분석 
## 딥러닝 프레임워크 최종 프로젝트

### 프로젝트 소개
#### 문제 인식
+ 리뷰의 중요성

![리뷰의 중요성](https://user-images.githubusercontent.com/74261590/146880587-e1e6e809-f87f-4cc5-8d69-d864f62a8a94.png)

식당에 대한 정보를 판단할 때 소비자들의 리뷰의 중요성이 커지고 있음


+ 다양한 리뷰에 대한 신빙성 문제
1. 허위 리뷰

![허위리뷰](https://user-images.githubusercontent.com/74261590/146880675-d7f9cb5f-0bef-4d1a-a7ff-b642cb6622bf.jpg)
2. 악성 리뷰

![악성리뷰](https://user-images.githubusercontent.com/74261590/146880741-5cb74f3d-803f-4d38-b202-9a0900c894c2.jpg)

3. 긍정적인 리뷰 내에 부정적인 평가 존재

![긍정적인 평가내에 부정적인 평가 존재](https://user-images.githubusercontent.com/74261590/146880899-4ab216e7-f1be-4f32-b68b-50b976cc63f0.jpg)

#### 해결 방안
<table>
  <tr>
    <th>문제점</th>
    <th>해결 방안</th>
    <th>기대 효과</th>
  </tr>
 <tr>
    <td>허위 리뷰</td>
    <td>허위 리뷰 키워드 사전을 제작하여 일반 리뷰-0, 허위 리뷰-1로 라벨링을 후 지도 학습을 통해 허위 리뷰 제거</td>
    <td>허위 리뷰를 제거함으로써 리뷰에 대한 신뢰도 상승</td>
  </tr>
   <tr>
    <td>악성 리뷰</td>
    <td>악성 리뷰를 일반 리뷰-0, 비판 리뷰-1, 악성 리뷰-2로 라벨링 후 지도 학습을 통해 악성 리뷰 제거</td>
    <td>감정적이거나 악의적인 내용을 가진 리뷰 제거</td>
  </tr>
  <tr>
    <td>긍정적인 리뷰 내에 부정적인 평가 존재</td>
    <td>맛, 분위기, 서비스, 가성비, 재방문의사등의 속성 별로 리뷰를 파악하여 점수 도출</td>
    <td>소비자들이 여러 속성 별로 리뷰 파악 가능</td>
  </tr>
</table>

---------------------------------

### 프로젝트 프로세스

![프로젝트 프로세스](https://user-images.githubusercontent.com/74261590/146882619-75263567-6a35-45b4-bb68-ed2bc8acc5e5.jpg)

#### 데이터 수집
1. 식당 수집

본인이 검색하고 싶은 지역 + 맛집으로 네이버 지도에 검색

![식당 수집](https://user-images.githubusercontent.com/74261590/147182014-5b6ac26c-594c-4094-8b3f-34ae7dc0b90f.jpg)
![식당 수집 리스트](https://user-images.githubusercontent.com/74261590/147182301-2e7a8715-8db0-4f8a-b284-ff59c91379a5.jpg)
``` Python
#동적 크롤링을 위한 크롬드라이버
driver = webdriver.Chrome(executable_path=r'D:\temp\chromedriver.exe')

#사용자로부터 원하는 지역 검색
place = input('검색할 맛집 지역을 입력하세요 : ') 
place = place.replace(' ', '') 
place_url="https://m.map.naver.com/search2/search.naver?query={}맛집"

#url접속
driver.get(place_url.format(place))
```
2. 블로그 리뷰 수집

블로그 리뷰를 추출하기 위해서 url 변환 후(pcmap->m) 블로그 리뷰 수집

![블로그 리뷰 수집](https://user-images.githubusercontent.com/74261590/147184066-dacc130d-87a0-46a6-b892-468b189ff7ed.jpg)
![블로그 리뷰 수집 리스트](https://user-images.githubusercontent.com/74261590/147184111-ef835748-1106-41b2-8f0b-302a4fe634da.jpg)

``` Python
driver = webdriver.Chrome(executable_path=r'D:\temp\chromedriver.exe')

for idx,row in data2.iterrows():
    try:
        url=row['url']
        place_name=row['name']
        print("========================")
        print(place_name)
        #블로그 리뷰를 추출하기 위해 url 변환
        blog_url1 = url.replace('review/visitor#', 'home')
        blog_url2 = blog_url1.replace('pcmap','m')
        print(blog_url2)
        
        #변환한 url 접속
        driver.get(blog_url2)
        time.sleep(0.5)
        
        #식당 이름
        name_link = driver.find_element_by_xpath('//*[@id="_title"]/span[1]')
        name = name_link.text

        #음식점 링크에서 블로그 리뷰 버튼 클릭
        blog_review = '//*[@id="app-root"]/div/div/div[2]/div[1]/div/div/div[1]/div/span[3]/a'
        driver.find_element_by_xpath(blog_review).send_keys(Keys.ENTER)

        #블로그 리뷰에서 더보기 끝까지 내리기
        while True:
            try:
                time.sleep(1)
                driver.find_element_by_tag_name('body').send_keys(Keys.END)
                time.sleep(3)
                driver.find_element_by_css_selector('#app-root > div > div > div.place_detail_wrapper > div:nth-child(5) > div:nth-child(4) > div.place_section._31amG > div._2kAri > a').click()
                time.sleep(3)
                driver.find_element_by_tag_name('body').send_keys(Keys.END)
                time.sleep(1)

            except NoSuchElementException:
                print('-더보기 버튼 모두 클릭 완료-')
                break
```
3. 네이버 플레이스 방문자 리뷰 수집

추출한 url을 통해 네이버 플레이스 방문자 리뷰 수집

![네이버 플레이스 방문자 리뷰 수집](https://user-images.githubusercontent.com/74261590/147182539-c41d7c68-4db4-4e82-a1d4-154191dbe2e1.jpg)
![네이버 플레이스 방문자 리뷰 수집 리스트](https://user-images.githubusercontent.com/74261590/147182574-0fc5c30e-eebd-427b-b3e2-34705368f435.jpg)
``` Python
driver = webdriver.Chrome(executable_path=r'D:\temp\chromedriver.exe')

for idx,row in data2.iterrows():
    try:
        #식당 정보 url, 이름, 지역 가져오기
        url=row['url']
        place_name=row['name']
        region=row['region']
        print('========================')
        print(place_name + '식당')

        #url접속
        driver.get(url)
        driver.implicitly_wait(3)

        #더보기 버튼 다 누르기
        while True:
            try:
                time.sleep(1)
                driver.find_element_by_tag_name('body').send_keys(Keys.END)
                time.sleep(1)
                driver.find_element_by_css_selector('#app-root > div > div > div.place_detail_wrapper > div:nth-child(5) > div:nth-child(4) > div.place_section.cXO6M > div._2kAri > a').click()
                time.sleep(1)
                driver.find_element_by_tag_name('body').send_keys(Keys.END)
                time.sleep(1)

            except NoSuchElementException:
                print("-더보기 버튼 모두 클릭 완료-")
                break

        #파싱
        html = driver.page_source
        soup = BeautifulSoup(html, 'lxml')
        time.sleep(1)
```
#### 리뷰 전처리
1. 영어, 자음과 모음, 특수 문자 제거
2. 띄어쓰기 전처리
3. 리뷰를 문장 단위로 자르기
4. 전처리가 끝난 문장에 .을 붙임
<table>
  <tr>
    <th>전처리 전</th>
    <th>전처리 후</th>
  </tr>
 <tr>
    <td><img src="https://user-images.githubusercontent.com/74261590/147184522-7102cfe6-89b6-4a0f-960b-419d57799ecd.jpg"></td>
    <td><img src="https://user-images.githubusercontent.com/74261590/147184616-4cb08526-6a05-472d-bd02-70287a00c7f0.jpg"></td>
  </tr>
</table>

``` Python
corpus = []
def normalization(cr_review):
    for i in tqdm(range(0, len(cr_review)), desc='전처리 진행율'):
        review = re.sub('\n','', str(cr_review[i]))
        # 기본 전처리
        review = re.sub('\u200b','',review) # 폭 없는 공백 제거
        review = re.sub('\xa0','',review) # Latin1 (ISO 8859-1)비 공백 공간 제거
        review = re.sub(r'[@%\\*=()/~#&\+á?\xc3\xa1\-\|\.\;\!\-\,\_\~\$\'\"\:]', '',review) # 여러 기호 제거
        review = re.sub(r'[^가-힣-\s0-9]', '', review) # 한글과 숫자만 살리고 제거
        review = re.sub(r'\s+', ' ', review) # 다중 공백 제거
        review = re.sub(r'^\s+', '', review) # 시작 공백 제거
        # 표현 및 문법 전처리
        review = re.sub('라구요|라구여|라구용|라고염|라고여|라고용', '라고요', review)
        review = re.sub('어용|어염|어여', '어요', review)
        review = re.sub('봐용|봐염|봐여', '봐요', review)
        review = re.sub('게용|게염|게여', '게요', review)
        review = re.sub('했당', '했다', review)
        review = re.sub('았당', '았다', review)
        review = re.sub('에용|에염|에여|예염', '에요', review)
        review = re.sub('세용|세염|세여', '세요', review)
        review = re.sub('께용|께염|께여|께유', '께요', review)
        review = re.sub('해용|해염|해여|해유', '해요', review)
        review = re.sub('네용|네염|네여|네유', '네요', review)
        review = re.sub('아용|아염|아여', '아요', review)
        review = re.sub('니당', '니다', review)
        review = re.sub('괜춘', '괜찮네요', review)
        review = re.sub('이뻐', '예뻐', review)
        review = re.sub('이쁘', '예쁘', review)
        review = re.sub('이쁜', '예쁜', review)
        review = re.sub('고기집', '고깃집', review)
        review = re.sub('같아용|같아염|같아여', '같아요', review)
        review = re.sub('같네용|같네염|같네여', '같네요', review)
        review = re.sub('이구용', '이구요', review)
        review = re.sub('었 따', '었다', review)
        # 띄어쓰기
        review = spacing(review)
        # 띄어쓰기 전처리
        review = re.sub('니\s다', '니다', review)
        review = re.sub('라\s고요|라고\s요', '라고요', review)
        review = re.sub('배\s곧', '배곧', review)
        review = re.sub('또\s잇', '또잇', review)
        review = re.sub('와\s규', '와규', review)
        review = re.sub('에\s비야', '에비야', review)
        #문장 분절
        review = kss.split_sentences(review)
        corpus.append(review)
        
    return corpus
```
#### 리뷰 선별
1. 허위 리뷰 필터링

+ 데이터 라벨링

① 허위 리뷰 키워드 사전 spam_words 정의(광고 심사 지침에 있는 권고 문구를 바탕으로 제작)

② 블로그 리뷰에 spam_words에 있는 단어가 있을 경우 spam(1), 없을 경우 non_spam(0)으로 라벨링

![spam_words](https://user-images.githubusercontent.com/74261590/147190696-92f4ee9a-fae4-4df0-b481-0a25efda180c.jpg)

+ 변수 특성

데이터가 너무 적고 불균형이 심해 성능이 제대로 나오지 않아 기존 spam 데이터를 복제해서 삽입

NONSPAM(0) : SPAM(1) = 2756 : 66 -> NONSPAM(0) : SPAM(1) = 2756 : 157

![허위 리뷰 변수 특성](https://user-images.githubusercontent.com/74261590/147189668-8610bfd1-e3b7-4872-80b5-e0a2e8ada66e.jpg)

+ 모델 성능

성능 평가 지표로는 Confusion matrix(Accuracy, Recall, Precision, F1_Score)와 ROC분석을 통한 AUC 수치 활용

-Decision Tree

``` Python
dtc = DecisionTreeClassifier(random_state = 2021)

#GridSearchCV의 param_grid 설정
#불순도 함수 : gini, entropy
param_grid = {'criterion': ['gini', 'entropy'],
              'max_depth': [2, 4, 6, 8, 10, None]}

#param_grid 2X6=12
#cv=20 이므로 12X20=240번의 학습/평가가 이루어짐
dtc_cv = GridSearchCV(estimator=dtc, param_grid=param_grid, cv=20)

dtc_cv.fit(X_train, y_train)
```

<table>
  <tr>
    <th>AUC 수치</th>
    <th>Confusion Matrix</th>
  </tr>
 <tr>
    <td><img src="https://user-images.githubusercontent.com/74261590/147191098-badb8010-6e47-43b2-976c-5da753313738.png"></td>
    <td><img src="https://user-images.githubusercontent.com/74261590/147191119-3f829b49-7606-4ca8-a942-cba4a241eed0.png"></td>
  </tr>
</table>
-Random Forest

``` Pyhton
rf = RandomForestClassifier(random_state=2019)

#GridSearchCV의 param_grid 설정
#n_estimators : 최대 트리의 개수
#불순도 함수 : gini, entropy
param_grid = {'n_estimators': [5, 10, 15, 20, 25, 30, 100],
              'criterion': ['gini', 'entropy'],
              'max_depth': [2, 4, 6]}

#param_grid 7X2X3=42
#cv=10 이므로 10X42=420번의 학습/평가가 이루어짐
rf_cv = GridSearchCV(estimator=rf, param_grid=param_grid, cv=10)

rf_cv.fit(X_train, y_train)
```

<table>
  <tr>
    <th>AUC 수치</th>
    <th>Confusion Matrix</th>
  </tr>
 <tr>
    <td><img src="https://user-images.githubusercontent.com/74261590/147191198-7b82ec27-efd8-416e-aa0a-b930f8793f1c.png"></td>
    <td><img src="https://user-images.githubusercontent.com/74261590/147191229-de12d195-53cd-4e51-9569-71a2c27524e8.png"></td>
  </tr>
</table>
-Logisitic Regression

``` Python
log = LogisticRegression(random_state=2019)
log.fit(X_train, y_train)
```

<table>
  <tr>
    <th>AUC 수치</th>
    <th>Confusion Matrix</th>
  </tr>
 <tr>
    <td><img src="https://user-images.githubusercontent.com/74261590/147191283-c16ed8f0-345b-4e13-af48-42ca5465127d.png"></td>
    <td><img src="https://user-images.githubusercontent.com/74261590/147191299-20363731-2fe6-46aa-813c-c92773bcd77f.png"></td>
  </tr>
</table>
-xgboost

``` Python
xg = XGBClassifier(random_state = 2021)

#GridSearchCV의 param_grid 설정
#n_estimators : 학습기의 개수(반복 횟수)
#불순도 함수 : gini, entropy
#min_child_weight : 이것을 기준으로 추가 분기 결정
#colsample_bytree : 각 트리마다의 feature 샘플링 비율
param_grid = {'n_estimators': [5, 10, 15, 20, 25, 30, 100],
              'criterion': ['gini', 'entropy'],
              'max_depth': [2, 4, 6],
             'learning_rate':[0.1, 0.3],
             'min_child_weight':[1,3], 
              'colsample_bytree':[0.5,0.75]}

#param_grid 7X2X3X2X2X2=336
#cv=10 이므로 336X410=3360번의 학습/평가가 이루어짐
xg_cv = GridSearchCV(estimator=xg, param_grid=param_grid, cv=10)

xg_cv.fit(X_train, y_train)
```

<table>
  <tr>
    <th>AUC 수치</th>
    <th>Confusion Matrix</th>
  </tr>
 <tr>
    <td><img src="https://user-images.githubusercontent.com/74261590/147191327-0887ea31-698c-48c6-9de0-49d5a81238e6.png"></td>
    <td><img src="https://user-images.githubusercontent.com/74261590/147191344-c6a254fa-acfd-4e88-915d-7fd08a562736.png"></td>
  </tr>
</table>

가장 성능이 좋아 최종 모델로 선정

-> xgboost 모델을 사용하여 2861개의 데이터 중 67개의 허위 리뷰 제거

2. 관련 없는 내용 필터링

블로그 리뷰에는 식당에 대한 리뷰 뿐만 아니라 일상적인 내용들이 많이 있어 리뷰에 대한 내용만을 뽑아낼 필요성이 있음

FastText 모델과 Word2Vec pretrained모델과 FastText pretrained 모델을 fine tuning한 transfered 모델 사용

① 블로그 리뷰 토큰화

② 토큰화한 리뷰 데이터를 이용하여 학습

-FastText 모델

``` Python
#토큰화한 블로그 리뷰들을 이용하여 FastText 모델 생성
#size = 워드 벡터의 특징 값. 즉, 임베딩 된 벡터의 차원
#window = 컨텍스트 윈도우 크기
#min_count = 단어 최소 빈도 수 제한
#workers = 학습을 위한 프로세스 수

model = FastText(final_result, size=300, window=5, min_count=2, workers=4)
model.train(final_result, total_examples=len(final_result), epochs=500) 
```

-Word2Vec Pretrained 모델을 fine tuning한 transfered 모델1

``` Python
#미리 훈련된 한국어 모델 불러오기
PretrainedFastTextModel = Word2Vec.load('D:/채리/4학년 2학기/딥러닝프레임워크_박성호/팀플/ko.bin')

#다운로드 받은 word2vec모델이 형식이 맞지 않아 새로 저장
PretrainedFastTextModel.wv.save_word2vec_format("ko.bin.gz", binary=False)

#Fine tuning 할 새로운 Word2Vec 모델 생성
#PretrainedFastTextModel과 'vector_size'가 같은 model을 생성
TransferedModel=Word2Vec(size=PretrainedFastTextModel.vector_size, min_count=1)

TransferedModel.build_vocab([PretrainedFastTextModel.wv.vocab.keys()])

#주어진 데이터로 새로운 모델의 단어 추가
#update parameter를 True로 설정
TransferedModel.build_vocab(final_result,update=True)

#Pretrained 모델의 학습 파라미터를 기반으로 새로운 모델의 학습 파라미터 초기화
#학습파라미터를 'filepath'에 있는 값으로 모두 업데이트해줌
#lockf=0.0 : 보통은 학습 파라미터를 update하지 못하도록 lock이 걸려있음
#lockf=1 : 학습 파라미터를 update하도록 lock 해제
file_path1="D:/채리/4학년 2학기/딥러닝프레임워크_박성호/팀플/ko.bin.gz"
TransferedModel.intersect_word2vec_format(file_path1, lockf=1.0, binary=False, encoding='utf-8', unicode_errors='ignore')

#새로운 데이터 기반의 학습
TransferedModel.train(final_result, total_examples=len(final_result), epochs=500)
```

-FastText Pretrained 모델을 fine tuning한 transfered 모델2

``` Python
#모델 다운로드
urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ko.300.bin.gz', filename='cc.ko.300.bin.gz')

#미리 훈련된 한국어 모델 불러오기
PretrainedFastTextModel2=fasttext.load_facebook_model('D:/채리/4학년 2학기/딥러닝프레임워크_박성호/팀플/cc.ko.300.bin.gz')

#다운로드 받은 fasttext형식이 맞지 않아 새로 저장
PretrainedFastTextModel2.wv.save_word2vec_format("cc.ko.300.bin.gz", binary=False)

#Fine tuning 할 새로운 Word2Vec 모델 생성(FastText는 fine tuning할 방법이 없음)
#PretrainedFastTextModel2과 'vector_size'가 같은 model을 생성
TransferedModel2=Word2Vec([PretrainedFastTextModel2.wv.vocab.keys()],size=PretrainedFastTextModel2.vector_size, min_count=1)

#주어진 데이터로 새로운 모델의 단어 추가
#update parameter를 True로 설정
TransferedModel2.build_vocab(final_result,update=True)

#Pretrained 모델의 학습 파라미터를 기반으로 새로운 모델의 학습 파라미터 초기화
#학습파라미터를 'filepath'에 있는 값으로 모두 업데이트해줌
#lockf=0.0 : 보통은 학습 파라미터를 update하지 못하도록 lock이 걸려있음
#lockf=1 : 학습 파라미터를 update하도록 lock 해제
file_path2="D:/채리/4학년 2학기/딥러닝프레임워크_박성호/팀플/cc.ko.300.bin.gz"
TransferedModel2.intersect_word2vec_format(file_path2, lockf=1.0, binary=False, encoding='utf-8', unicode_errors='ignore')

#새로운 데이터 기반의 학습
TransferedModel2.train(final_result, total_examples=len(final_result), epochs=500)
```

③ 블로그 리뷰 중 식당에 대한 내용만 있는 리뷰(기준 문장)와 다른 리뷰들 사이의 유사도를 계산해서 평균을 구함

``` Python
fasttext_similarity=[]
for blog_info in temp1:
    blog_review=blog_info['blog review']
    for test_word in test_sentence:
        for word in blog_review:
            if word=='.':
                continue
            similarity=model.wv.similarity(test_word,word)
            fasttext_similarity.append(similarity)
                
fasttext_average = sum(fasttext_similarity)/len(fasttext_similarity)
```
<table>
  <tr>
    <th>모델</th>
    <th>유사도 평균</th>
  </tr>
 <tr>
    <td>FastText 모델</td>
    <td>0.01202106707845227</td>
  </tr>
   <tr>
    <td>Word2Vec pretrained 모델을 활용한 Transfered Model1</td>
    <td>0.01622760142883667</td>
  </tr>
  <tr>
    <td>FastText pretrained 모델을 활용한 Transfered Model2</td>
    <td>0.12114957328589501</td>
  </tr>
</table>

④ 리뷰에 있는 토큰과 기준 문장에 있는 모든 토큰들 사이의 유사도 계산 후 평균 계산

⑤ ④번의 값이 ③번의 유사도 평균 값보다 낮다면 그 단어 제거

⑥ 3개의 모델에서 제거한 단어들을 중복을 제거하고 모아서 원래 리뷰에서 제거

![image](https://user-images.githubusercontent.com/74261590/147196494-88c9870e-8db5-4a7b-a155-334b55754f90.png)

약 500만개의 단어 제거

3. 악성 리뷰 필터링

+ 데이터 라벨링

① 직접 라벨링 진행

② 리뷰에 감정적이거나 악의적인 내용이 있을 경우 악성 리뷰(2), 비판이나 개선 관련 내용이 있을 경우 비판 리뷰(1), 일반 리뷰(0)

![image](https://user-images.githubusercontent.com/74261590/147196920-95156090-9015-4fd7-86ec-152c8328c563.png)

+ 변수 특성

데이터가 너무 적고 불균형이 심해 성능이 제대로 나오지 않아 악성 리뷰를 약 200개 정도 추가로 수집한 후 비판 리뷰와 악성 리뷰 복제 후 삽입

일반 리뷰(0) : 비판 리뷰(1) : 악성 리뷰(2) = 13137 : 636 : 119 -> 일반 리뷰(0) : 비판 리뷰(1) : 악성 리뷰(2) = 13137 : 1272 : 636

![image](https://user-images.githubusercontent.com/74261590/147197198-e2a9212a-f622-4d5d-ac66-6a2f61b02dd2.png)

+ 모델 성능

성능 평가 지표로는 Confusion matrix(Accuracy, Recall, Precision, F1_Score)와 ROC분석을 통한 AUC 수치 활용

-Decision Tree

``` Python
dtc = DecisionTreeClassifier()

model =dtc

# Pipeline 설정
#머신 러닝 프로세스에서 파라미터를 조정하거나 스케일링 및 정규화와 같은 데이터 변환을 수행
#이때 모델 학습과 테스트를 할때 최소 2번 이상 적용해야 하는데 이러한 과정을 단순화하는 도구
vec_pipe = Pipeline([
                    ("vec", TfidfVectorizer(tokenizer=okt_tokenizer)), #Tfidf와 해당하는 모델 비교
                    ("model", model)
                    ])
    
# 하이퍼파라미터 설정
# tfidf 하이퍼파라미터 설정
vec_pipe_params = {"vec__ngram_range" : [(1,2)], # 단어 묶음 수
                    "vec__stop_words"  : [None], # 한국어 해당 안됨
                    "vec__min_df" : [3], # 학습에 포함하기 위한 최소 빈도 값
                    "vec__max_df" : [0.9]} # 학습에 포함되기 위한 최대 빈도값
    
    
# grid search
vec_gs = GridSearchCV(vec_pipe,
                        param_grid=vec_pipe_params,
                        cv=3)

# model fitting
vec_gs.fit(X_train, y_train);
    
# 예측
y_pred = vec_gs.best_estimator_.predict(X_test)
```

<table>
  <tr>
    <th>AUC 수치</th>
    <th>Confusion Matrix</th>
  </tr>
 <tr>
    <td><img src="https://user-images.githubusercontent.com/74261590/147197305-c6f12376-fbb8-4f33-ab59-3aea860b18cf.png"></td>
    <td><img src="https://user-images.githubusercontent.com/74261590/147197335-3f809641-c879-4271-a287-fb2e654f6ca2.png"></td>
  </tr>
</table>

-xgboost

``` Python
xgb= XGBClassifier()

model = xgb

# Pipeline 설정
vec_pipe = Pipeline([
                    ("vec", TfidfVectorizer(tokenizer=okt_tokenizer)), 
                    ("model", model)
                    ])
    
# 하이퍼파라미터 설정
vec_pipe_params = {"vec__ngram_range" : [(1,2)], 
                    "vec__stop_words"  : [None],
                    "vec__min_df" : [3],
                    "vec__max_df" : [0.9]}
    
    
# grid search
vec_gs2 = GridSearchCV(vec_pipe,
                        param_grid=vec_pipe_params,
                        cv=3)

# model fittingb
vec_gs2.fit(X_train, y_train);
    
# 예측
y_pred = vec_gs2.best_estimator_.predict(X_test)
```

<table>
  <tr>
    <th>AUC 수치</th>
    <th>Confusion Matrix</th>
  </tr>
 <tr>
    <td><img src="https://user-images.githubusercontent.com/74261590/147197386-205df060-f313-4841-9bc7-0027c13cce5d.png"></td>
    <td><img src="https://user-images.githubusercontent.com/74261590/147197432-2155be7e-cf3f-4d50-9edc-5924b6969dab.png"></td>
  </tr>
</table>

-LightGBM

``` Python
lgbm = LGBMClassifier()

model =lgbm

# Pipeline 설정
vec_pipe = Pipeline([
                    ("vec", TfidfVectorizer(tokenizer=okt_tokenizer)), 
                    ("model", model)
                    ])
    
# 하이퍼파라미터 설정
vec_pipe_params = {"vec__ngram_range" : [(1,2)], 
                    "vec__stop_words"  : [None],
                    "vec__min_df" : [3],
                    "vec__max_df" : [0.9]}
    
    
# grid search
vec_gs3 = GridSearchCV(vec_pipe,
                        param_grid=vec_pipe_params,
                        cv=3)

# model fitting
vec_gs3.fit(X_train, y_train);
    
# 예측
y_pred = vec_gs3.best_estimator_.predict(X_test)
```
<table>
  <tr>
    <th>AUC 수치</th>
    <th>Confusion Matrix</th>
  </tr>
 <tr>
    <td><img src="https://user-images.githubusercontent.com/74261590/147197456-6cbd0d8d-4219-493a-bc98-876dae1863be.png"></td>
    <td><img src="https://user-images.githubusercontent.com/74261590/147197471-5ba38993-6ed8-4e48-83db-b1e4da951eb6.png"></td>
  </tr>
</table>

가장 성능이 좋아 최종 모델로 선정

-> LightGBM 모델을 사용하여 13892개의 데이터 중 132개의 악성 리뷰 제거

#### 식당 별 키워드 추출 및 감성 분석 

① 감성 사전 제작

서비스, 맛, 가격, 분위기, 재방문의사 5가지 부분의 긍·부정 감성 사전을 제작

![image](https://user-images.githubusercontent.com/74261590/147197608-4707c173-0dde-4cd2-978d-df27da0681c3.png)

② 형태소 분석을 통한 토큰화

③ 구 단위로 감성 분석 진행(각 특성의 키워드와 주변 요소 함께 추출)

-키워드 + 조사 + 어절 + 어절
-키워드 + 어절 + 어절
-어절 + 키워드 + 조사

④ 키워드와 키워드가 포함된 구에서 해당 특성의 감성 사전에 포함된 긍정, 부정어가 있는지 확인해서 최종적인 긍 · 부정을 파악

``` Python
#각 특성별 키워드 추출, 문장 속에 다른 단어들과 섞여서 쓰이는 형태를 보고 감성분석을 진행해야 하니까 키워드+주변어절 추출합니다
#구 단위분석
def get_feature_keywords(feature_keywords, review):
    feature_temp = []
    for keyword in feature_keywords:
        if re.findall(keyword, review): # 리뷰에서 키워드 있는거 다 찾아서
            sub_list = ['게','고','음','며','데','만','도','면'] #연결어미
            
            for sub in sub_list:
                if sub+' ' in review: # 연결어미가 띄어쓰기랑 같이 있으면
                    review = re.sub(sub+' ', sub+',', review) # 연결어미 + 띄어쓰기를 연결어미 + , 로 대체
                    #키워드와 의미 없는 연결어미가 함께 추출되는 일이 없도록(밑에서 어절 단위로 추출함)
                
            a = re.findall(keyword +'+[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]+\s+[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]+\s+[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]+',review) # K한 한 한글
            # 키워드+조사 띄어쓰기 어절 2개가 붙어 있는거 다 찾기
            b = re.findall(keyword + '+\s+[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]+\s+[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]+',review) # K 한 한글 
            # 키워드와 띄어쓰기 어절 1개짜리 다 찾기
            c = re.findall('[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]+\s+' + keyword +'[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]+',review) # 한 K한글 예쁜 분위기가
            # 어절 + 키워드 + 어절인 부분 다 찾기
            
            # 추출한 키워드들을 feature_temp에 append
            for ngram in a:
                t = ()
                feature_temp.append(t + (ngram,keyword))
            for ngram in b:
                t = ()
                feature_temp.append(t + (ngram,keyword))
            for ngram in c:
                t = ()
                feature_temp.append(t + (ngram,keyword))
                
    return feature_temp
```
``` Python
#특성별 감성분석 인풋으로 특성별 긍정사전, 부정사전, 추출한 키워드
def get_feature_emotions(feature_good_dict,feature_bad_dict, feature_temp):
    good_feature_emotion_list = []
    bad_feature_emotion_list = []
    
    for ngrams in feature_temp: # 추출한 키워드(추출 문장, 키워드)에서
        keyword = ngrams[1] # 키워드 -> 감성사전에서 특성별 긍부정을 추출할 키워드
        ngram = ngrams[0] # 추출 문장 -> 특성별 긍부정 단어가 있는지 확인할 문장
        is_bad_feature = None
        
        good_emotion_list = feature_good_dict[keyword] # 특성별 긍정 감정어 
        bad_emotion_list = feature_bad_dict[keyword] # 특성별 부정 감정어
        for emotion in good_emotion_list: # 긍정감정어에서
            if re.findall(emotion, ngram): # 긍정어와 추출한 문장이 함께 있다면
                is_bad_feature = False  # 나쁜 특성은 아니다
        for emotion in bad_emotion_list: # 부정감정어에서
            if re.findall(emotion, ngram): # 부정어와 추출한 문장이 함께 있다면
                is_bad_feature = True    # 이 문장은 부정문이다
        for negative in negative_word_emotion: #[안, 않, 못, 없 등]의 부정어에서
            if re.findall(negative, ngram): # 부정어와 추출한 문장이 함께 있으면서
                if is_bad_feature == True: # 이 문장이 부정문이면
                    is_bad_feature = False # 이 문장은 사실 부정한 문장이 아니다
                    break
                elif is_bad_feature == False: # 이 문장이 부정한 문장이 아니면
                    is_bad_feature = True # 이 문장은 사실 부정한 문장이다
                    break
                else:
                    is_bad_feature = True # 그냥 아무 감정 없는 평범한 문장이라면 부정한 문장이다
                    break   
        if is_bad_feature:
            bad_feature_emotion_list.append(ngram) # 부정 문장에 추가
        elif is_bad_feature == False:
            good_feature_emotion_list.append(ngram) # 긍정 문장에 추가
        else:
            pass # 아무것도 아니면 패스
    return good_feature_emotion_list, bad_feature_emotion_list
```
![image](https://user-images.githubusercontent.com/74261590/147197825-f30f21c4-bce5-4c7e-93aa-93d3e5568696.png)

⑤ 식당 별로 키워드 추출 및 특성 별로 긍정리뷰/전체리뷰로 점수화

``` Python
#식당별 감성분석 결과 카운팅 및 스코어
#check_division은 restaurant_good_service_count, restaurant_good_service_count + restaurant_bad_service_count 파라미터로 받고, *100 함 -> 점수가 나옴
check_division = lambda x, y: y if y ==0 else round((x / float(y)),2) # x,y 파라미터를 받아서 y가 0이면 0, 아니면 x/y의 소수점 밑 두자리 수까지 반올림

for i in range(len(name)):
    restaurant_good_service_count = 0
    restaurant_bad_service_count = 0
    restaurant_good_atmosphere_count =0
    restaurant_bad_atmosphere_count =0
    restaurant_good_cost_count =0
    restaurant_bad_cost_count =0
    restaurant_good_visit_count = 0
    restaurant_bad_visit_count = 0
    restaurant_good_taste_count = 0
    restaurant_bad_taste_count = 0
    print(name[i]) # 식당 이름부터 출력
    reviews_list = refining(restaurant_review[i]) #식당에 대한 리뷰 전체 형태소 분석
    for review in reviews_list: #한번 돌 때마다 한 식당의 전체 리뷰에 대한 감성분석 진행
        service_temp = get_feature_keywords(service_good_feature.keys(), review)
        good_service,bad_service = get_feature_emotions(service_good_feature, service_bad_feature, service_temp)

        atmosphere_temp = get_feature_keywords(atmosphere_good_feature.keys(), review)
        good_atmosphere,bad_atmosphere = get_feature_emotions(atmosphere_good_feature, atmosphere_bad_feature, atmosphere_temp)

        cost_temp = get_feature_keywords(cost_good_feature.keys(), review)
        good_cost,bad_cost = get_feature_emotions(cost_good_feature, cost_bad_feature, cost_temp)

        visit_temp = get_feature_keywords(visit_good_feature.keys(), review)
        good_visit,bad_visit = get_feature_emotions(visit_good_feature, visit_bad_feature, visit_temp)

        taste_temp = get_feature_keywords(taste_good_feature.keys(), review)
        good_taste,bad_taste = get_feature_emotions(taste_good_feature, taste_bad_feature, taste_temp) # 맛 특성 긍부정 분석
        taste_good_emotion_temp = get_feature_keywords(taste_good_emotion, review) # 맛에 대한 긍정 감정 키워드 추출
        taste_bad_emotion_temp = get_feature_keywords(taste_bad_emotion, review) # 맛에 대한 부정 감정 키워드 추출
        good_taste2, bad_taste2 = get_taste_emotion(taste_good_emotion_temp,taste_bad_emotion_temp)
        good_taste.extend(good_taste2) # 긍정 맛 문장에 추가
        bad_taste.extend(bad_taste2) # 부정 맛 문장에 추가
        
        # 각 특성 긍부정 리스트의 크기에 따라 점수 카운트(부정은 출력 안하긴함)
        if len(good_service) > len(bad_service):
            restaurant_good_service_count += 1
        elif len(good_service) < len(bad_service):
            restaurant_bad_service_count += 1
        else:
            pass
        
        if len(good_atmosphere) > len(bad_atmosphere):
            restaurant_good_atmosphere_count += 1
        elif len(good_atmosphere) < len(bad_atmosphere):
            restaurant_bad_atmosphere_count += 1
        else:
            pass
        
        if len(good_cost) > len(bad_cost):
            restaurant_good_cost_count += 1
        elif len(good_cost) < len(bad_cost):
            restaurant_bad_cost_count += 1
        else:
            pass
            
        if len(good_visit) > len(bad_visit):
            restaurant_good_visit_count += 1
        elif len(good_visit) < len(bad_visit):
            restaurant_bad_visit_count += 1
        else:
            pass
        
        if len(good_taste) > len(bad_taste):
            restaurant_good_taste_count += 1
        elif len(good_taste) < len(bad_taste):
            restaurant_bad_taste_count += 1
        else:
            pass
        
    TT = restaurant_good_service_count + restaurant_bad_service_count + restaurant_good_taste_count + restaurant_bad_taste_count + restaurant_good_atmosphere_count + restaurant_bad_atmosphere_count + restaurant_good_cost_count + restaurant_bad_cost_count
    
    #if TT > 5: # 총 감성 카운트가 5 이상만 출력하는건데 일단 주석 처리 해 놓고 전체 출력
    print('Total review count: {}'.format(len(restaurant_review[i])))
    print('Good service: {}/{} = {}'.format(restaurant_good_service_count,restaurant_good_service_count + restaurant_bad_service_count,100*check_division(restaurant_good_service_count, restaurant_good_service_count + restaurant_bad_service_count)))
    print('Good atmosphere: {}/{} = {}'.format(restaurant_good_atmosphere_count,restaurant_good_atmosphere_count + restaurant_bad_atmosphere_count,100*check_division(restaurant_good_atmosphere_count,restaurant_good_atmosphere_count + restaurant_bad_atmosphere_count))) 
    print('Good cost: {}/{} = {}'.format(restaurant_good_cost_count,restaurant_good_cost_count + restaurant_bad_cost_count, 100*check_division(restaurant_good_cost_count,restaurant_good_cost_count + restaurant_bad_cost_count)))
    print('Good taste: {}/{} = {}'.format(restaurant_good_taste_count,restaurant_good_taste_count + restaurant_bad_taste_count,100*check_division(restaurant_good_taste_count,restaurant_good_taste_count + restaurant_bad_taste_count)))
    print('')
```
![image](https://user-images.githubusercontent.com/74261590/147197924-b6f4f17b-f8e1-4fe5-a38b-8745f1e2d696.png)


