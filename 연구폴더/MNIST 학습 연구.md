문서정보 : 2022.11.10. 작성, 작성자 [@SAgiKPJH](https://github.com/SAgiKPJH)

<br>

# MNIST 학습 - 숫자를 구분하는 가장 최소의 모델은 무엇인가?

- 숫자구분 하기 위한 모델을 만들어보고, 가장 효율적인 모델을 찾아본다.
- 그리고 그 이유를 분석하여 연구한다.

### 목표

- [x] : MNIST 기본형
- [ ] : MNIST Hiddenlayer 및 Epochs 조정 함수화


### 제작자
[@SAgiKPJH](https://github.com/SAgiKPJH)

### 참조

- [텐서플로 2.0 시작하기: 초보자용](https://www.tensorflow.org/tutorials/quickstart/beginner?hl=ko)
- [Tensorflow 인공지능 학습](https://github.com/SAgiKPJH/SAGI_JJU-JJUCODE-/blob/main/Project/02%20AI%20%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5%20%EB%BF%8C%EC%85%94%EB%B2%84%EB%A0%A4%20%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8/AI%20%EA%B3%B5%EB%B6%80/Tensorflow.md)

---
<br><br><br><br>

# MNIST 기본형

- MNIST 학습을 위한 기본형은 다음과 같다.
  ```python
  # TensorFlow 버전 확인
  import tensorflow as tf
  print("TensorFlow version:", tf.__version__)
  
  # 손글씨 정보 데이터 세트 (x:손글씨 이미지(28*28), y:숫자)
  # Train 6만개, Test 1만개
  mnist = tf.keras.datasets.mnist 
  (x_train, y_train), (x_test, y_test) = mnist.load_data() # x:0~255 이미지
  x_train, x_test = x_train / 255.0, x_test / 255.0 # x:0~1 이미지 변환
  
  # 신경망 구축
  model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
  ])
  # 784 > 128 (relu) > Dropout(0.2) > 10 (확률분포)
  # Dropout(0.2) : 노드 20% 무작위로 0 (과적합 방지) 
  # softmax : 0~9 각 확률들을 나타냄
  
  # 모델 설정
  # optimizer adam : 최적화 'adam' 알고리즘
  # loss 형 유형을 선택, 훈련데이터의 label 값이 정수일 때
  # metrics : 모델 평가, accuracy: 빈도계산
  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  
  # 모델 훈련, train으로 훈련, 5회
  model.fit(x_train, y_train, epochs=5)
  
  # 모델 평가, 몇 % 정확도인지 보여준다.
  model.evaluate(x_test,  y_test, verbose=2)
  ```

<br><br><br>

# MNIST Hiddenlayer 및 Epochs 조정 함수화

- 학습을 위한 사전 정의를 구상한다.
  ```python
  import tensorflow as tf
  
  mnist = tf.keras.datasets.mnist 
  (x_train, y_train), (x_test, y_test) = mnist.load_data() # x:0~255 이미지
  x_train, x_test = x_train / 255.0, x_test / 255.0 # x:0~1 이미지 변환
  ```
- MNIST 학습률 획득 함수화 작업을 한다.
  - Hiddenlayer 및 Epochs 조정할 수 있도록 함수화한다.
  ```python
  def MNIST_AI (h, t) :
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(h, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=t)
    return model.evaluate(x_test,  y_test, verbose=2)
  ```
- 실행 Test를 한다.
  ```python
  result = []
  result.append( MNIST_AI (128, 5) )
  result
  ```
  ```bash
  Epoch 1/5
  1875/1875 [==============================] - 5s 3ms/step - loss: 0.2983 - accuracy: 0.9134
  Epoch 2/5
  1875/1875 [==============================] - 5s 3ms/step - loss: 0.1451 - accuracy: 0.9572
  Epoch 3/5
  1875/1875 [==============================] - 5s 3ms/step - loss: 0.1081 - accuracy: 0.9675
  Epoch 4/5
  1875/1875 [==============================] - 5s 3ms/step - loss: 0.0882 - accuracy: 0.9728
  Epoch 5/5
  1875/1875 [==============================] - 5s 3ms/step - loss: 0.0740 - accuracy: 0.9765
  313/313 - 1s - loss: 0.0766 - accuracy: 0.9767 - 702ms/epoch - 2ms/step
  [[0.07662060856819153, 0.9767000079154968]]
  ```
  ```bash
  > result[0]
  [0.07662060856819153, 0.9767000079154968]
  ```
