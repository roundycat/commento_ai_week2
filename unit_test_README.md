# 🧪 `unit_test.ipynb` - README

## 📌 프로젝트 개요

`unit_test.ipynb`는 시멘틱 세그멘테이션 모델의 출력 결과를 점검하기 위한 **단위 테스트용 노트북**입니다.  
모델이 생성한 마스크 결과가 정상적으로 출력되었는지, 특정 픽셀의 클래스가 올바른지 등을 확인할 수 있습니다.

## ⚙️ 주요 기능

- 모델 출력 마스크 시각화
- 특정 위치에서의 class index 확인
- 예측 마스크의 유효성 확인 (유니크 값 확인 등)
- GT와 비교를 통한 결과 분석

## 🛠️ 환경 설정 및 종속성

### 필요한 패키지

- `torch`
- `numpy`
- `matplotlib`
- `PIL`
- `cv2`

### 설치 방법

#### pip
```bash
pip install torch torchvision matplotlib numpy
```

#### conda
```bash
conda install pytorch torchvision matplotlib numpy -c pytorch
```

## 📂 코드 구성

### 1. 기본 라이브러리 로드
```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
```

### 2. 예시 이미지와 마스크 불러오기
```python
image = Image.open("example_image.jpg")
mask = np.load("example_mask.npy")  # 클래스 마스크
```

### 3. 특정 좌표 픽셀 값 확인
```python
x, y = 100, 150
class_index = mask[y, x]
print(f"Class index at ({x},{y}):", class_index)
```

### 4. 예측 마스크 시각화
```python
plt.imshow(mask, cmap='tab20')
plt.title("Predicted Mask")
plt.axis("off")
plt.show()
```

### 5. 예측 마스크의 클래스 목록 확인
```python
unique_classes = np.unique(mask)
print("Detected classes:", unique_classes)
```

## 🖼️ 결과 예시

- 입력 이미지, 예측 마스크, GT 마스크 모두 시각적으로 확인 가능
- 특정 좌표에서 예측된 클래스 index 확인 가능

## 🔍 주의 사항

- 테스트용 노트북으로 학습 또는 배포에는 적합하지 않음
- 클래스 수가 바뀌면 컬러 매핑 및 시각화 로직도 수정 필요
- GT와 예측이 다른 경우를 시각적으로 확인할 수 있지만 정량적 비교는 포함되어 있지 않음

## ✅ 사용 목적

- 모델 디버깅 및 개발 단계에서 유효성 점검
- 예측 결과 시각화를 통한 결과 해석
- 픽셀 단위 분석 및 GT와의 시각적 비교 수행