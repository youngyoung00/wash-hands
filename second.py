import cv2
import tensorflow.keras
import numpy as np

## 이미지 전처리
def preprocessing(frame):
    # 사이즈 조정
    size = (224, 224)
    frame_resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    
    # 이미지 정규화
    frame_normalized = (frame_resized.astype(np.float32) / 127.0) - 1
    
    # 이미지 차원 재조정 - 예측을 위해 reshape 해줍니다.
    frame_reshaped = frame_normalized.reshape((1, 224, 224, 3))
    
    return frame_reshaped

classes = ['one', 'water', 'two', 'three', 'four', 'five', 'six']

## 학습된 모델 불러오기
model_filename = 'keras_model.h5'
model = tensorflow.keras.models.load_model(model_filename)

# 카메라 캡쳐 객체, 0=내장 카메라
capture = cv2.VideoCapture(0)

# 캡쳐 프레임 사이즈 조절
#capture.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
#capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

water_cnt = 1 # 30초간 "졸림" 상태를 확인하기 위한 변수
while cap.isOpened(): # 특정 키를 누를 때까지 무한 반복
    ret, frame = cap.read() # 한 프레임씩 읽기
    if not ret:
        break

    h, w, _ = frame.shape
    cx = h / 2
    frame = frame[:, 200:200+frame.shape[0]]
    
    # 이미지 뒤집기
    frame_fliped = cv2.flip(frame, 1)
    
    # 이미지 출력
    cv2.imshow("VideoFrame", frame_fliped)
    #cv2.putText(frame_fliped, text=classes[idx], org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255, 255, 255), thickness=2)
    
    # 1초마다 검사하며, videoframe 창으로 아무 키나 누르게 되면 종료
    if cv2.waitKey(1) == ord('q'):
        break
        
    # 데이터 전처리
    preprocessed = preprocessing(frame_fliped)

    # 예측
    prediction = model.predict(preprocessed)
    #print(prediction) # [[0.00533728 0.99466264]]
    
    if prediction[0,0] < prediction[0,1]:
        print('water')
        water_cnt += 1
        
        # 졸린 상태가 30초간 지속되면 소리 & 카카오톡 보내기
        if water_cnt % 5 == 0:
            water_cnt = 1
            print('3초간 물이 나왔어요!!!')
            #beepsound()
            #break ## 1번만 알람이 오면 프로그램을 정지 시킴 (반복을 원한다면, 주석으로 막기!)
    else:
        print('깨어있는 상태')
        water_cnt = 1
    
# 카메라 객체 반환
capture.release() 
# 화면에 나타난 윈도우들을 종료
cv2.destroyAllWindows()