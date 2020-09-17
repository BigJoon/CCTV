# Nvidia Jetson Nano: Custom Object Detection from scratch using Tensorflow, OpenCV and TensorRT
![test](https://img.shields.io/badge/language-python3-brightgreen)


## Requirements
 - Ubuntu 18.04
 - TensorFlow 1.12
 - OpenCV
 - Jetson Nano Developer Kit
 - Python

[![이미지](https://github.com/BigJoon/CCTV/blob/master/Script_Collection/whole_process.jpg)](https://github.com/BigJoon/CCTV/blob/master/Object_Detection_JetsonNano.md)

## 1. Creating Custom Dataset -Done
- pass

## 2. Pre-Processing
 - 조졌다. 이미지들이 모두 FHD급 이미지들임. 먼저 resolution을 떨구고 이미지 태깅을 맡겼어야 했는데 FHD급 그대로 xml파일이 다 만들어졌음. 혹시 이 상태에서도 이미지 resolution을 떨어뜨리면 xml에 태깅된 박스 정보들도 같은 비율로 줄이는 방법이 있나..? 있을거 같긴한데 하기 싫다. 일단 그냥 하자. =>이와 관련하여 issue 에 기록한 것이 해결책이 될 수도...? 
 - 데이터 축소시키며 박스도 줄일 것임.


## 3. Labeling - Done
 - pass


## 4. Creating TFRecords
 - 용량이 250GB 정도의 상당한 양이므로 이미지 100개씩만 뽑아서 변환을 테스트하는 것이 좋다.
 - 이 과정이 이제 핵심인데 변환 후 TFrecord를 시각화 시켜주는 툴인 tfviewer 를 사용하여 확인해본다. (이와 관련해서 웹브라우저로 볼 수 있게 링크를 추가할 예정_다만 교내에서만 접근 가능한 )
 - 학습시킬 객체 이외에 xml에 태깅된 정보들이 있을 것이다. 그러니깐 label_map에 없는 클래스가 xml파일에 있어도 되는 것인가? 학습에 영향을 주나? 결론은 영향을 주지 않는다. 하지만 학습시 읽어오게 되는 tfreocrd의 용량을 차지하므로 영향이 없진 않을 것임.
 
 
 
## 5. CNN
 - 직접 모델을 만들 수도 있지만 지금은 Pre-trained 모델을 사용할 것이다. (변환 후 inference까지 잘 된다면 pose estimation까지 포함된 기능을 수행하는 모델 만들 예정) 


## 6. Training
 - 학습이 잘 일어나고 있는지 확인 한다.
 - Tensorboard, Evaluation(2000번 마다), 중간에 기록된 모델에 대해 inference(이 때 export시켜 GPU머신에서 inference 하는 스크립트로 가는 "링크" 첨부 필요)


## 7. Export Models(with Frozen inference graph)
 - export_inference_graph.py 를 이용하여 graph를 export 시킨다.

## 8. Convert to TensorRT Engines
 - Jkjung 의 블로그를 참조하여 tensorrt로 변환한다.

 - https://github.com/jkjung-avt/tensorrt_demos#ssd
 
위 블로그를 참조한다.
 - 이후 변환된 .uff와 .engine 파일을 jetson-inference의 라이브러리 내부로 형식에 맞게 옮기면 된다.
 - **이후 inference  링크 추가 예정 **

=======================================================================================================================
=======================================================================================================================
## 첫번째 과정
 - PASCALVOC DATA 를 TFrecord로 변환한다. (tfrecord는 데이터를 시리얼화 시키는 작업이므로 딱히 용량이 줄어들지 않는다. 별도의 하드디스크에 저장하는 것을 추천한다.)
 
## 두번째 과정
 - TFrecord를 이용하여 tensorflow 1.12를 사용하여 학습 시킨다.
 - 위 과정에서 tensorflow-gpu 1.12 버전은 쿠다 9.0버전이 필요하므로 도커 사용을 권장한다.
 - https://github.com/tensorflow/models/tree/r1.13.0
 - 위 브랜치를 사용하여 학습시키면 되고 /models/research/object_detection/model_main.py 가 학습시키는 샘플 코드이다.
 
 
 <tensorflow-gpu 1.14 or tensorflow 1.15를 사용하여 학습시킬 경우 TensorRT로 변환시 지원되지 않는 BatchNormv3,Cast Layer가 들어가게 됨.
 또한 tensorflow-gpu 1.13, 1.14, 1.15로 학습 후 frozen graph로 export 할 때 1.12를 사용한다해도 같은 에러가 발생하게 된다.>
 
 => 따라서 tensorflow-gpu 1.12 버전을 사용하는 것을 추천한다. 
 
## 세번째 과정
 - tensorflow-gpu 1.12 버전에서 모델을 export 시킨다. 
 - /models/research/object_detection/export_inference_graph.py 를 사용하여 frozen graph를 export 시킨다.
 
## 네번째 과정
 - Jkjung 의 블로그를 참조하여 tensorrt로 변환한다.

 - https://github.com/jkjung-avt/tensorrt_demos#ssd
 
 윗 블로그를 참조할 것
