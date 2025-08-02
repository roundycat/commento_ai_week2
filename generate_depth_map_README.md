# 📘 `generate_depth_map.ipynb` - README

## 📌 프로젝트 개요

`generate_depth_map.ipynb`는 단일 이미지를 입력받아 해당 이미지의 **깊이 맵(Depth Map)** 을 생성하는 노트북입니다.  
Microsoft의 MiDaS(Mixed Depth and Scale) 모델을 기반으로 하며, 이미지로부터 상대적 거리 정보를 추정할 수 있습니다.

## ⚙️ 주요 기능

- MiDaS 사전 학습 모델 로드 (`DPT_BEiT_L_384`)
- 단일 이미지 입력 → 전처리 → 깊이 추론 → 깊이 맵 저장 및 시각화
- GPU 또는 CPU 디바이스에서 실행 가능

## 🛠️ 환경 설정 및 종속성

### 필요한 패키지

- `torch`
- `torchvision`
- `torchaudio`
- `opencv-python`
- `matplotlib`
- `PIL`
- `urllib`

### 설치 방법

#### pip
```bash
pip install torch torchvision torchaudio
pip install opencv-python matplotlib
```

#### conda
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c conda-forge opencv matplotlib
```

## 📂 코드 구성

### 1. 라이브러리 및 모델 불러오기
```python
import torch
import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

model_type = "DPT_BEiT_L_384"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device).eval()
```

### 2. 이미지 입력 및 전처리
```python
img_path = "image.jpg"
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
input_batch = transform(img).to(device)
```

### 3. 깊이 추론 및 시각화
```python
with torch.no_grad():
    prediction = midas(input_batch)

depth_map = prediction.squeeze().cpu().numpy()

plt.imshow(depth_map, cmap='inferno')
plt.axis('off')
plt.savefig("depth_map.png")
plt.show()
```

## 🖼️ 결과 예시

- 결과는 현재 디렉토리에 `depth_map.png`로 저장됩니다.
- `matplotlib`을 통해 추정된 깊이 맵 시각화도 가능합니다.

## 🔍 주의 사항

- **상대적 깊이만 제공**되며, 절대적인 거리 정보는 포함되지 않습니다.
- 입력 이미지의 해상도가 너무 클 경우, 추론 속도가 느려질 수 있습니다.
- 모델이 자연 이미지에 최적화되어 있으므로 특수한 입력에서는 정확도가 떨어질 수 있습니다.