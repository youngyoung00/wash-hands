from keras.models import load_model
import numpy as np
import cv2
import pygame
import time 
import datetime
from threading import Timer

model = load_model('keras_model.h5')

cap = cv2.VideoCapture(0)

size = (224, 224)

classes = ['water', 'one', 'two', 'three', 'four', 'five', 'six', 'toothbrush', 'cup', 'nothing']
    
# 횟수 카운트를 위한 변수
to_cnt = 1
cup_cnt = 1
water_cnt = 1
one_cnt = 1
two_cnt = 1
three_cnt = 1
four_cnt = 1
five_cnt = 1
six_cnt = 1
no_cnt = 1

# 현재상태를 지정하기 위한 변수
to = 1
water = 1
one = 1
two = 1
three = 1
four = 1
five = 1
six = 100
no = 1
    
WATER = cv2.imread("water.jpg")
WATER = cv2.resize(WATER, (150,150))

Water = cv2.cvtColor(WATER, cv2.COLOR_BGR2RGB)
ret, mask = cv2.threshold(Water, 1, 255, cv2.THRESH_BINARY)

ONE = cv2.imread("one.png")
ONE = cv2.resize(ONE, (150,150))

TWO = cv2.imread("two.png")
TWO = cv2.resize(TWO, (150,150))

THREE = cv2.imread("three.png")
THREE = cv2.resize(THREE, (150,150))

FOUR = cv2.imread("four.png")
FOUR = cv2.resize(FOUR, (150,150))

FIVE = cv2.imread("five.png")
FIVE = cv2.resize(FIVE, (150,150))

SIX = cv2.imread("six.png")
SIX = cv2.resize(SIX, (150,150))

TOOTHBRUSH = cv2.imread("toothbrush.jpg")
TOOTHBRUSH = cv2.resize(TOOTHBRUSH, (150,150))

CUP = cv2.imread("cup.png")
CUP = cv2.resize(CUP, (150,150))

WATER2 = cv2.imread("water2.png")
WATER2 = cv2.resize(WATER2, (150,150))

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened(): # 웹캠화면이 켜져있는 동안
    ret, img = cap.read() 
    
    h, w, _ = img.shape 
    cx = h / 2
    #img = img[:, 200:200+img.shape[0]] # 캡쳐 프레임 사이즈 조절
    img = cv2.flip(img, 1) # 좌우반전

    img_input = cv2.resize(img, size) #웹캠으로 받은 이미지 사이즈 재조정
    img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
    img_input = (img_input.astype(np.float32) / 127.0) - 1 # 이미지 정규화
    img_input = np.expand_dims(img_input, axis=0) # 차원변형

    prediction = model.predict(img_input) # 입력데이터와 비교하여 각 클래스의 값을 예측
    idx = np.argmax(prediction) # 예측 값이 가장 큰 인덱스 값

    # 웹캠 화면에 예측한 클래스 글자 출력
    cv2.putText(img, text=classes[idx], org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255, 255, 255), thickness=2)
    now = datetime.datetime.now()
    cv2.putText(img, text=str(now), org=(343, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255, 255, 255), thickness=2)
    
    roi = img[-150-10:-10, -150-10:-10]
    # Set an index of where the mask is
    roi[np.where(mask)] = 0
    
    roo = img[-150-10:-10, 10:150+10]
    
    # 모든 상황이 초기 조건일때
    if water == 1:
        if one == 1:
            if two == 1:
                if three == 1:
                    if four == 1:
                        if five == 1:
                            if six == 100:
                                if to == 1:
                                    if no == 1:
                                        roi += WATER
                                        roo[np.where(mask)] = 0
                                        roo += TOOTHBRUSH
                                        if np.argmax(prediction) == [0]: # 최대 값의 인덱스가 0이면
                                            water_cnt += 1 # water_cnt를 1개씩 더한다
                                        if np.argmax(prediction) == [7]:
                                            to_cnt += 1 # to_cnt를 1개씩 더한다

    if to == 100:
        roo[np.where(mask)] = 0
        roo += CUP
        roi += WATER2
        if np.argmax(prediction) == [7]:
            to_cnt += 1
            no_cnt += 1
        if np.argmax(prediction) == [8]:
            cup_cnt += 1
            no_cnt += 1
        if np.argmax(prediction) == [0]: # 최대 값의 인덱스가 0이면
            water_cnt += 1 # water_cnt를 1개씩 더한다
            no_cnt += 1
        if np.argmax(prediction) == [9]:
            no_cnt += 1

    if water == 100:
        roi += ONE 
        if np.argmax(prediction) == [1]:
            one_cnt += 1
        if np.argmax(prediction) == [9]:
            no_cnt += 1
            
    if one == 100:
        roi += TWO  
        if np.argmax(prediction) == [2]:
            two_cnt += 1
        if np.argmax(prediction) == [9]:
            no_cnt += 1

    if two == 100:
        roi += THREE  
        if np.argmax(prediction) == [3]:
            three_cnt += 1
        if np.argmax(prediction) == [9]:
            no_cnt += 1

    if three == 100:
        roi += FOUR   
        if np.argmax(prediction) == [4]:
            four_cnt += 1
        if np.argmax(prediction) == [9]:
            no_cnt += 1

    if four == 100:
        roi += FIVE
        if np.argmax(prediction) == [5]:
            five_cnt += 1
        if np.argmax(prediction) == [9]:
            no_cnt += 1

    if five == 100:
        roi += SIX  
        if np.argmax(prediction) == [6]:
            six_cnt += 1
        if np.argmax(prediction) == [9]:
            no_cnt += 1
            
    if not ret: 
        break
    cv2.imshow('result', img)
    
    if to == 1: # 손씻기 단계에서 nothing 카운트 횟수
        if no_cnt % 60 == 0:
            no_cnt = 1
            to = 1
            water = 1
            one = 1
            two = 1
            three = 1
            four = 1
            five = 1
            six = 100
            no = 1
    
    if to_cnt % 25 == 0:
        pygame.init() # 초기화
        pygame.mixer.music.load('깨끗하게 양치해요!.wav') # 음성파일 로드     
        pygame.mixer.music.play(0) # 음성 파일을 1번 출력
        pygame.init() # 다음을 위해 초기화
        to_cnt = 1 # 카운트를 다시 1로 초기화
        to = 100 # 현재상태 조건을 위해 값 100으로 변경

    if to == 100:
        if cup_cnt % 10 == 0:
            pygame.init() # 초기화
            pygame.mixer.music.load('물.wav') # 음성파일 로드     
            pygame.mixer.music.play(0) # 음성 파일을 1번 출력
            pygame.init() # 다음을 위해 초기화
            cup_cnt = 1 # 카운트를 다시 1로 초기화
        
        if water_cnt % 7 == 0:
            pygame.init() # 초기화
            pygame.mixer.music.load('물.wav') # 음성파일 로드     
            pygame.mixer.music.play(0) # 음성 파일을 1번 출력
            pygame.init() # 다음을 위해 초기화
            water_cnt = 1 # 카운트를 다시 1로 초기화
            
        if to_cnt % 10 == 0:
            pygame.init() # 초기화
            pygame.mixer.music.load('물.wav') # 음성파일 로드     
            pygame.mixer.music.play(0) # 음성 파일을 1번 출력
            pygame.init() # 다음을 위해 초기화
            to_cnt = 1 # 카운트를 다시 1로 초기화
            
        if no_cnt % 675 == 0: # 칫솔 단계에서 nothing 카운트 개수
            no_cnt = 1
            to = 1


    if water_cnt == 5:
        pygame.init() # 초기화
        pygame.mixer.music.load('물.wav') # 음성파일 로드     
        pygame.mixer.music.play(0) # 음성 파일을 1번 출력
        pygame.init() # 다음을 위해 초기화
        no_cnt = 1
        
    if water_cnt % 20 == 0:
        pygame.init() # 초기화
        pygame.mixer.music.load('1단계.wav') # 음성파일 로드     
        pygame.mixer.music.play(0) # 음성 파일을 1번 출력
        pygame.init() # 다음을 위해 초기화
        water_cnt = 1 # 카운트를 다시 1로 초기화
        water = 100 # 현재상태 조건을 위해 값 100으로 변경
        no_cnt = 1

    
    if one_cnt % 20 == 0:
        pygame.init()
        pygame.mixer.music.load('2단계.wav')            
        pygame.mixer.music.play(0)
        pygame.init()
        one_cnt = 1
        water = 1
        one = 100
        no_cnt = 1

    if two_cnt % 30 == 0:
        pygame.init()
        pygame.mixer.music.load('3단계.wav')            
        pygame.mixer.music.play(0)
        pygame.init()
        two_cnt = 1
        one = 1
        two = 100
        no_cnt = 1

    if three_cnt % 30 == 0:
        pygame.init()
        pygame.mixer.music.load('4단계.wav')            
        pygame.mixer.music.play(0)
        pygame.init()
        three_cnt = 1
        two = 1
        three = 100
        no_cnt = 1

    if four_cnt % 30 == 0:
        pygame.init()
        pygame.mixer.music.load('5단계.wav')            
        pygame.mixer.music.play(0)
        pygame.init()
        four_cnt = 1
        three = 1
        four = 100
        no_cnt = 1

    if five_cnt % 30 == 0:
        pygame.init()
        pygame.mixer.music.load('6단계.wav')            
        pygame.mixer.music.play(0)
        pygame.init()
        five_cnt = 1
        four = 1
        five = 100
        no_cnt = 1


    if six_cnt % 25 == 0:
        pygame.init()
        pygame.mixer.music.load('이제 깨끗해요!.wav')            
        pygame.mixer.music.play(0)
        pygame.init()
        six_cnt = 1
        five = 1
        no_cnt = 1
        
    # 0.1s동안 사용자가 'q'키를 누르기를 기다림
    if cv2.waitKey(3) == ord('q'):
        break
