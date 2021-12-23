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

        #식당 리뷰 개수 출력
        reviews = soup.find_all('div', attrs = {'class':'_3vfQ6'}) #리뷰 공간
        review_num = len(reviews) #특정 식당의 리뷰 총 개수
        print('리뷰 총 개수 : ' + str(review_num))

        #넘버링
        num = []
        for i in range(0, review_num):
            num.append(i+1)

        #크롤링 작업
        category = '방문자 리뷰'
        for i in range(0, review_num):
            # review 내용
            try : 
                review_content = reviews[i].find('span', attrs = {'class':'WoYOw'}).text

            except: #리뷰가 없다면
                review_content = ""

            visitor_review_dic = {
                'num':num[i],
                'name' : place_name,
                'blog or review' : category,
                'visitor review content' : review_content
            }
            visitor_review_arr.append(visitor_review_dic)
```
#### 리뷰 전처리
#### 리뷰 선별
#### 식당 별 키워드 추출 및 감성 분석 
