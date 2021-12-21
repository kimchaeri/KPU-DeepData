# 지역별 식당 키워드 추출 및 감정 분석 
## 딥러닝 프레임워크 최종 프로젝트

### 프로젝트 소개
#### 문제 인식
+ 리뷰의 중요성
+ 
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
허위 리뷰 - 허위 리뷰를 제거함으로써 리뷰에 대한 신뢰도 상승
악성 리뷰 - 감정적이거나 악의적인 내용을 가진 리뷰 제거
긍정적인 리뷰 내에 부정적인 평가 존재 - 맛, 분위기, 서비스, 가성비, 재방문의사등의 속성 별로 리뷰를 파악하여 점수 도출
