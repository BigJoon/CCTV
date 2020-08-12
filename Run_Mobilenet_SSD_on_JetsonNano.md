# Run Mobilenet-SSD-V2 on Jetson Nano

## 첫번째 과정
 - PASCALVOC DATA 를 TFrecord로 변환한다.
 
## 두번째 과정
 - TFrecord를 이용하여 tensorflow 1.12를 사용하여 학습 시킨다.
 - 위 과정에서 tensorflow-gpu 1.12 버전은 쿠다 9.0버전이 필요하므로 도커 사용을 권장한다.
 
 
 
 <tensorflow-gpu 1.14 or tensorflow 1.15를 사용하여 학습시킬 경우 TensorRT로 변환시 지원되지 않는 BatchNormv3,Cast Layer가 들어가게 됨.
 또한 tensorflow-gpu 1.13, 1.14, 1.15로 학습 후 frozen graph로 export 할 때 1.12를 사용한다해도 같은 에러가 발생하게 된다.>
 
 => 그냥 tensorflow-gpu 1.12 버전을 사용하는 것을 추천한다. 
 
## 세번째 과정
 - tensorflow-gpu 1.12 버전에서 모델을 export 시킨다.
 
## 네번째 과정
 - Jkjung 의 블로그를 참조하여 tensorrt로 변환한다.

 - https://github.com/jkjung-avt/tensorrt_demos#ssd
 
 윗 블로그를 참조할 것
