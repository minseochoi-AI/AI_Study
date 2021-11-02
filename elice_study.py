# 2021 NIPA AI기본 교육과정 학습내용 정리
# 기본 교육과정은 데이터 분석을 위한 라이브러리 학습에 중점을 둠

# 1. 모듈<<라이브러리=패키지
#  
# 객체지향 프로그래밍>>
# 코드를 완전히 하나의 틀로 만드는 것이 아니라 다양한 객체(함수.변수등)의 집합으로 만들어
# 비슷한 영역에 다시 사용 할 수 있도록 하는 것.이때 함수를 정리해놓은 모듈을 임포트하여 편리하게 사용이 가능하다.
# 여러개의 모듈을 모아놓은 집합을 패키지 또는 라이브러리라고 한다. 

# 모듈 임포트 방식 

# A.import 모듈이름(특정모듈이 가진 함수 전체를 사용할 수 있게 함)
    #모듈이름.함수이름() 형태로 사용가능 

# B.from 모듈이름 import 함수이름(모듈이 가진 특정한 함수를 사용 할 수 있게함)
    # 함수이름() 형태로 사용

# 2. numpy 모듈 배열기초
#  다차원 배열객체를 처리하기 위한 모듈 >>여러차원으로 구성된 배열정보를 처리함

# numpy 모듈 임포트 
import numpy as np

# # 배열 생성
# # np(numpy모듈의).array(함수를 불러온다) 
# # array 함수안의 인자에는 리스트 형태의 데이터가 들어가는 듯 하다.
# array1 = np.array(range(5))

# # 배열 타입 출력하기
# print(type(array1))
# # <class 'numpy.ndarray'> = numpy의  n차원 배열 이라는 의미같다.

# # 배열 모양 출력하기
# print(array1.shape)
# # 배열이 가진 행의 개수, 한 행에 들어있는 데이터의 개수를 알려줌 but 1차원 배열일시 데이터 개수만 출력함

# # 배열 차원 출력하기
# print(array1.ndim)
# # 배열이 가진 차원의 개수를 출력 

# # 배열의 크기 출력하기
# print(array1.size)
# # 배열이 담을 수 있는 데이터의 개수 = 총 인덱스를 알려줌

# # 배열에 저장된 데이터 타입 출력하기
# print(array1.dtype)
# # numpy 배열은 다른 타입의 변수를 저장 가능하다

# # 특정 인덱스에 저장된 데이터 출력
# print(array1[3])
# # 주의! 1차원을 제외한 배열은 지정한 숫자에 해당하는 n-1차원의 인덱스 정보를 출력한다.
# # 2차원 배열일 경우 3인덱스를 가진 1차원 배열 전체가 출력됨

# print(array1[:3])
# # 인덱스 값 ~ 인덱스 값-1의 모든 정보를 출력한다. 

# # 배열 생성 다른 방법
# matrix=np.array(range(1,16))
# # range를 사용 시  정수 하나만 넣으면 0부터 그 정수 -1 까지의 1차원 배열이 생성
# # but 정수,정수를 넣으면 첫번쨰 정부부터 시작해 두번쨰 입력한 정수의-1의 인덱스만큼 배열이 생성된다

# # 배열 모양 바꾸기
# matrix.shape= 3,5
# # 1행 15열로 이루어진 1차원 배열을 3행 5열 형태인 2차원 배열로 바꾼다

# # 배열속 데이터 타입 변경
# print(matrix.astype(str))

# # 인덱스 요소
# print(matrix[2,3])
# # matrix 배열에 3번쨰 행 ,4번쨰 열 정보를 출력한다.

# # 인덱스 요소 2
# print(matrix[:3,1:3])
# # [:3] = 0번쨰 행 인덱스부터 2번쨰 행 인덱스까지 
# # [1:3]= 1번쨰 열 인덱스부터 2번쨰 열 인덱스까지
# # 행 정보와 열 정보를 모두 반영해 출력함

# # 3.배열 Indexing & Slicing 실습

# # 배열생성,모양변경을 한 번에 하기
# matrix2=np.arange(1,16,1).reshape(3,5)
# # arange 함수인자(시작숫자,전체 인덱스지정 개수+1,증가값)

# print(matrix2[0,1])
# # 1행 2열 정보 출력 

# print(matrix2[2,:2])
# # 3번쨰 행, 0~1까지 열 정보 출력

# print(matrix2 > 5)
# # 배열중 5보다 큰 값에는 Trye, 작은 값에는 False 지정해 출력

# print(matrix2[matrix2 > 5])
# # 5보다 큰 값을 가진 인덱스만 출력

# 4.pandas 
# 시리즈>> 배열이 보강된 형태
# 본래 배열에서는 숫자 인덱스를 지정해 값을 다루는 것이 가능 0,1,2,3
#  series 데이터는 인덱스 이왜에 사전{dictionary}의 키 인덱스로 특정 자료를 저장하고 불러들이는 것이 가능

#pandas 모듈 임포트  
import pandas as pd

# # 시리즈 생성
# series=pd.Series([1,2,3,4], index = ['a','b','c','d'],name="Title")
# # 내가 원하는 인덱스가 int와 key 인덱스로 나뉘어저 int.key인덱스 모두 값을 찾는데 사용가능하다

# # 국가별 인구 수 시리즈 데이터를 딕셔너리를 사용하여 만들어보기
# p_dict={"korea" : 5180,
# "japan" : 12718,
# "china" : 141500,
# "usa" : 32676}

# # Series함수인자에 딕셔너리를 넣으면 딕셔너리의 key=시리즈 인덱스,value=인덱스별 데이터가 된다
# population=pd.Series(p_dict)

# # 배열에서 인덱스를 숫자로 불러오는 것, 딕셔너리 key값으로 value 값 찾기처럼 인덱스 지정으로 series 데이터 찾기가 가능
# print(population['korea'])

# # 5.pandas DataFrame
# # 여러개의 시리즈 데이터를 합쳐 하나의 인덱스(행)에 여러개의 컬럼(열) 값이 들어가는 형태
# # 크기에 따른 분류  dictionary < series < dataFrame
# # series화 단계를 안거치고 dictionary에서 바로 dataFrame에 저장 가능   

# # Population series data 사전형태 인구데이터
# population_dict = {
#     'korea': 5180,
#     'japan': 12718,
#     'china': 141500,
#     'usa': 32676
# }

# # GDP series data: 사전형태 GDP데이터
# gdp_dict = {
#     'korea': 169320000,
#     'japan': 516700000,
#     'china': 1409250000,
#     'usa': 2041280000,
# }

# # 국가이름을 key인덱스로 설정 컬럼에는 인구,  GDP데이터 삽입
# data_f=pd.DataFrame({"population":population_dict,"gdp":gdp_dict})
#  Series,dictionary 값을 DataFrame의 특정 컬럼에 넣을 때 {['컬럼이름']:딕셔너리 or 시리즈 정보를 담은 변수} 형태로 저장

# # DataFrame key 인덱스 출력 
# print(data_f.index)

# # DataFrame column값 출력
# print(data_f.columns)

# 6.masking , query
# DataFrame 안에서 조건을 만족하는 값을 추출하는 방법

# # numpy random배열 생성 (5행 2열) column에는 A,B저장
# df=pd.DataFrame(np.random.rand(5,2),columns=["A","B"])

# # dataFrame 출력
# print(df)

# # masking 연산활용 (0.5 < A, 0.3 > B) 구하기
# df_masking=df[(df['A'] < 0.5) & (df['B'] > 0.3)]
# # data_f2 데이터 프레임안의 [데이처 프레임 컬럼 "A"의 값이 0.5보다 작은 모든 데이터들
# # data_f2 데이터 프레임안의 [데이처 프레임 컬럼 "B"의 값이 0.3보다 큰 모든 데이터들
# # 어려움****

# # masking 연산결과 
# print(df_masking)

# # query함수 사용
# df_qurey=df.query("A<0.5 & B>0.3") 
# # query 함수연산이 더욱 편리하다

# # query 연산결과
# print(df_qurey)


# # GDP와 인구수 시리즈 데이터가 들어간 데이터프레임 생성
# population = pd.Series({'korea': 5180,'japan': 12718,'china': 141500,'usa': 32676})
# gdp = pd.Series({'korea': 169320000,'japan': 516700000,'china': 1409250000,'usa': 2041280000})
# df=pd.DataFrame({'population':population,'GDP':gdp})
# # :기호 실수하지 말자!

# # 1인당 GDP컬럼(gdp_per_capita) 추가
# df['gdp_per_capita']=df['GDP']/df['population']
# # 1인당 gdp = gdp를 population으로 나눈 값 

# print(df)

# # 7.DataFrame 정렬하기
# #  특정한 값을 기준으로 DataFrame을 정렬한다.
# df = pd.DataFrame({
#     'col1' : [2, 1, 9, 8, 7, 4],
#     'col2' : ['A', 'A', 'B', np.nan, 'D', 'C'],
#     'col3': [0, 1, 9, 4, 2, 3],
# })

# # 컬럼내 숫자형 데이터 정렬

# # 오름차순 정리
# sort_df1=df['col1'].sort_values
# # DataFrame의 컬럼을 지정해 sort_values함수 사용시 저정한 컬럼값만 정렬해 return됨

# sort_df2=df.sort_values('col1')
# # sort_values함수 인자에 컬럼갑 지정 할 시 DataFrame 전체가 return됨

# print(sort_df2)

# # 내림차순 정리
# sort_df3=df.sort_values('col2',ascending=False)

# # 여러개 컬럼 동시에, 다른 차순으로 정리
# sort_df4=df.sort_values(['col1','col2'],ascending=["True","False"])
# # []기호를 사용해 묶기 sort.values([컬럼1,컬럼2,*....],ascending["T","F",*.........])

# 8.집계함수
# 데이터 값을 요약하는 함수

# data = {
#     'korean' : [50, 60, 70],
#     'math' : [10, np.nan, 40]
# }
# df = pd.DataFrame(data, index = ['a','b','c'])

# # 컬럼별 데이터 개수(axis=0일 경우 열)
# data_col=df.count(axis=0)
# print(data_col)
# # 행별 데이터 개수(axis=1일 경우 행)
# data_low=df.count(axis=1)
# print(data_low)

# # 각 컬럼 별 최댓값
# data_max=df.max()
# print(data_max)

# # 각 컬럼 별 최솟값
# data_min=df.min()
# print(data_min)

# # 컬럼별 데이터 합계
# data_sum=df.sum()
# print(data_sum)

# # 컬럼속 Nan 값을 컬럼의 최솟값으로 대체
# math_min=df['math'].min()
# df['math']=df['math'].fillna(math_min)
# print(df)

# # 컬럼별 평균값 
# data_mean=df.mean()
# print(data_mean)

# 9.그룹으로 묶기 gr0upby()
# 특정 조건을 넣어 집계하고 싶을때 

# DataFrame 생성
# df = pd.DataFrame({
#     'key': ['A', 'B', 'C', 'A', 'B', 'C'],
#     'data1': [1, 2, 3, 1, 2, 3],
#     'data2': [4, 4, 6, 0, 6, 1]
# })

# # key 기준으로 묶은 값 출력
# print(df.groupby('key').sum())

# # key.data1기준으로 묶기
# print(df.groupby(['key','data1']).sum())

# # key 기준으로 묶고, data1과 data2 각각의 최솟값, 중앙값, 최댓값을 출력
# print(df.groupby('key').aggregate([min,np.median,max]))

# # key 기준으로 묶고, data1의 최솟값, data2의 합계를 출력
# print(df.groupby('key').aggregate({"data1":min,"data2":sum}))

# groupby로 묶은 뒤 data2 컬럼 평균이 0.3보다 큰 데이터만 필터링 
# def filiter1(x):
#     return x['data2'].mean()> 3

# print(df.groupby('key').filter(filiter1))


# 10 matplotlab 모듈
# 데이터 시각화용 라이브러리
# matplotlib는 아래와 같이 크게 두 가지 방법으로 사용할 수 있습니다.

# - stateless API (objected-based)

# - stateful API (state-based)

# stateless 방법은 내가 지정한 figure, 내가 지정한 ax에 그림을 그리는 방법이고,
# stateful 방법은 현재의 figure, 현재의 ax에 그림을 그리는 방법입니다.
# * figure : 그래프를 그릴 공간(종이) / ax(axes) : 그 공간 중 지금 내가 사용할 부분

# 따라서 stateless 방법은 figure와 ax를 직접 만들어야 하고, 이런 특징으로 객체지향적이라고 할 수 있습니다.
# 반면 stateful 방법은 직접 지정하지 않고, 현재의 figure와 ax를 자동으로 찾아 그곳에 plotting 하는 방식입니다.
import matplotlib.pyplot as plt

# 배열 생성 0~9
# x=np.arange(10)

# # subplot 함수 여러개의 그래프를 하나의 그림으로 그려줌
# fig,ax=plt.subplots()    
# # fig(그래프 사이즈), ax(그래프안에 그릴 부분)인데 여러개를 지정해 넣겠다(subplot)

# # ax.plot 함수로 1번째 그래프 그리기(x, y, label(그래프 선 이름), marker(값을 표시하는 점), color(점,선의 색상), linestyle(점을 잇는 선))생성
# ax.plot(
#     x, x, label="y=x",
#     linestyle='-',
#     marker='.',
#     color='blue'
# )
# # ax.plot 함수로 2번째 그래프 그리기
# ax.plot(
#     x, x**2, label='y=x^2',
#     linestyle='-.',
#     marker=',',
#     color='red'
# )

# Line Styles

# 기호	의미
#  -	실선
#  –	대시 선
#  -.   대시 점 선
#  :	점선

# Markers

# 기호	의미	기호	의미
# .	    점	      ,	   픽셀
# o	   원         s	  사각형
# v,<, ^, >	삼각형	1, 2, 3, 4	삼각선
# p	오각형	H, h	육각형

# 그래프 x,y축에 각각 x,y 텍스트 넣음
# ax.set_xlabel("x") 
# ax.set_ylabel("y")

# # 그래프 범례 legend 설정(하나의 그림에 여러개 데이터를 넣는 경우 각 데이터애 대한 설명을 붙임)
# ax.legend(
#     # 레전드 표시 위치    
#     loc="center left",
#     # 그림자
#     shadow=True,
#     # 둥근 모서리 사용여부
#     fancybox=True,
#     # 레전드의 빈공간 크기 설정
#     borderpad=2
# )

# # 그래프 저장
# print(fig.savefig("plot.png"))

# 11.bar, histogrem

# #data set 
# x = np.array(["축구", "야구", "농구", "배드민턴", "탁구"])
# y = np.array([13, 10, 17, 8, 7])
# z = np.random.randn(1000)

# # 하나의 그림에(1), 두개의 그래프를 그린다(2),size 지정 
# fig,axes=plt.subplots(1,2,figsize=(8,4))

# # 1번쨰 축에 bar(x,y 데이터 바 형태 표현) bar 형태이므로 x,y축 데이터가 각각 필요
# axes[0].bar(x,y)

# #2번쨰 축에 histogrem(z 데이터로 histogrem 그래프 생성,bins= data를 표현할 선의 개수를 지정) 
# axes[1].hist(z,bins=200) 
# # historgrem 형태(값의 빈도를 표현)이므로 x축 데이터와 표현 선 개수 필요 

# # 그래프 보기
# plt.show()

# 12. 합계 데이터 bar 표현

# # 랜덤 데이터 3개 성성
# x=np.random.rand(3)
# y=np.random.rand(3)
# z=np.random.rand(3)

# # 데이터 list 저장
# data=[x,y,z]

# # 그래프 사이즈 설정
# fig,ax=plt.subplots()

# # x축에 들어갈 3개 정보를 담기 위해 3축 생성
# x_ax=np.arange(3)

# # for 문으로 데이터 쌓아 올리기
# for i in x_ax:
#     # subplots로 지정한 크기(ax)에 bar 생성, x축은 3개 (x_ax)삽입,y축에 data[i] 값 삽입 
#     ax.bar(x_ax,data[i])
#     # bottem에 data 합계(sum)을 0,1,2 인덱스(x,y,z) 순으로 쌓아 저장
#     bottom=np.sum(data[:i],axis=0)

# plt.show()

# 대학 비교과 과정 파뿌리 데이터 가져옴 
# 13.DataFrame 도수분포표화 예시 자동차 연도별 출력, 배기량 그래프화

# 자동차 데이터 가져오기
df=pd.read_csv(r"C:\Users\auto-mpg.csv")

# 컬럼 삽입
df.columns=["mpg(연비)","cylinders(실린더수)","displacement(배기량)","horsepower(출력)","weight(차중)",
                    "acceleration(가속능력)","modelYear(출시년도)","originNumber(제조국번호)","name(모델명)"]

# # 그래프 크기 지정
# fig,ax=plt.subplots() 

# # 데이터 삽입
# ax.plot(df["modelYear(출시년도)"],df["displacement(배기량)"])

# # x,y축 이름 지정
# ax.set_xlabel("modelYear")
# ax.set_ylabel("displacement")

# # 데이터 보기
# plt.show()

# 14. 산포 그래프 scatter() 함수

# 자동차 데이터 출시년도별 출력, 가속능력 정보 산포그래프화

# value_counts로 자동차 출시년도별 개수 가져오기,sort_index로 년도 오름차순 정렬
print(df["modelYear(출시년도)"].value_counts().sort_index())

# query함수로 77년 전후기준 데이터 나누기
old_model=df[df["modelYear(출시년도)"] < 77]
new_model=df[df["modelYear(출시년도)"] >= 77]


# 그래프 크기 지정
fig,ax=plt.subplots(figsize=(30,4))

# 그래프 축 이름, 범례 지정
ax.set_xlabel("horsepower")
ax.set_ylabel("acceleration")


# 산포 그래프 그리기
ax.scatter(old_model["horsepower(출력)"],old_model["acceleration(가속능력)"],
        marker='*', color='red', s=50)
ax.scatter(new_model["horsepower(출력)"],new_model["acceleration(가속능력)"],
        marker='.', color='blue',  s=25)

# 그래프 보기
plt.show()
