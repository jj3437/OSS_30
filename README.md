## 팀원
- 202434584 경제진
- 202434612 박시후
- 202434622 배주원(팀장)
- 202434675 최석범

# VideoAnalyzer

## 프로젝트 개요
이 프로젝트는 OpenCV와 TensorFlow를 사용하여 동영상을 분석하고, 다양한 기능을 제공합니다:
1. 객체 탐지 (사람, 개, 소, 말, 돼지 구분)
2. 사람 얼굴 모자이크 처리
3. 동영상에서 사람만 표시
4. 추가 기능: 동영상에서 움직임이 가장 많은 영역 탐지

## 실행 방법

### 설치
```bash
pip install -r requirements.txt
실행
객체 탐지:
bash
코드 복사
python main.py detect
사람 얼굴 모자이크:
bash
코드 복사
python main.py mosaic
사람만 표시:
bash
코드 복사
python main.py person
움직임 탐지:
bash
코드 복사
python main.py motion
사용된 라이브러리
TensorFlow
TensorFlow Hub
OpenCV
Numpy
주요 기능 설명
객체 탐지:

TensorFlow Hub에서 사전 학습된 EfficientDet 모델을 사용하여 사람, 개, 소, 말, 돼지 등을 탐지합니다.
결과는 output_detect.mp4에 저장됩니다.
사람 얼굴 모자이크:

OpenCV를 사용하여 얼굴 영역을 탐지하고 해당 영역을 모자이크 처리합니다.
사람만 표시:

OpenCV를 사용하여 동영상에서 사람만 추출합니다.
움직임 탐지:

OpenCV를 사용하여 동영상에서 움직임이 가장 많은 영역을 감지합니다.
참고 자료
TensorFlow Hub: EfficientDet 모델
OpenCV: 공식 문서
markdown
코드 복사
