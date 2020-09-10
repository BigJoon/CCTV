# Run Custom Finetuned Mobilenet-SSD-V2 on Jetson Nano

## Requirements
 - TensorFlow 1.12
 - OpenCV
 - Jetson Nano Developer Kit
 - Python

[![이미지](https://github.com/BigJoon/CCTV/blob/master/Script_Collection/whole_process.jpg)](https://github.com/BigJoon/CCTV/blob/master/Object_Detection_JetsonNano.md)

## 1. Creating Custom Dataset -Done
- pass

## 2. Pre-Processing
 - 일단 이미지들을 까보면서 사이즈가 큰지 확인 그리고 상황에 따라 resolution을 감소시키기로 하자.(여기서 걱정이 우린 이미 객체들 태깅한 xml 파일도 갖고 있는데, resolutino을 떨구게 된다면 xml파일 안의 정보랑 일치하지 않게되는거 아닌가 함.)

## 3. Labeling - Done
 - pass

## 4. Creating TFRecords
 - 이 과정이 이제 핵심인데 변환 후 TFrecord를 시각화 시켜주는 툴인 tfviewer 를 사용하여 확인해본다. (이와 관련해서 웹브라우저로 볼 수 있게 링크를 추가할 예정)
 
## 5. CNN
 - 직접 모델을 만들 수도 있지만 지금은 Pre-trained 모델을 사용할 것이다. (변환 후 inference까지 잘 된다면 pose estimation까지 포함된 기능을 수행하는 모델 만들 예정) 


## 6. Training
 - 학습이 잘 일어나고 있는지 확인 한다.
 - Tensorboard, Evaluation(2000번 마다), 중간에 기록된 모델에 대해 inference(이 때 export시켜 GPU머신에서 inference 하는 스크립트로 가는 "링크" 첨부 필요)

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
