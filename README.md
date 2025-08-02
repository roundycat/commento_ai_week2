📘 generate_depth_map.ipynb - README
1. 📌 프로젝트 개요
generate_depth_map.ipynb는 단일 이미지로부터 깊이 맵(Depth Map) 을 생성하는 노트북입니다. Microsoft에서 개발한 MiDaS 모델을 기반으로 하며, 이미지의 픽셀 단위로 상대적인 깊이 정보를 추정할 수 있습니다.

2. ⚙️ 주요 기능
단일 이미지로부터 상대적 깊이 추정

MiDaS 사전학습 모델 활용 (DPT_BEiT_L_384 등)

추정된 깊이 맵 시각화 및 이미지 저장

3. 🛠️ 환경 설정 및 종속성
  3.1. 필요 라이브러리
    torch, torchvision, torchaudio
  
    opencv-python
  
    matplotlib
    
    urllib, os, PIL 등

  3.2. 설치 예시
  
    pip 사용 
    pip install torch torchvision torchaudio
    pip install opencv-python matplotlib
    
    conda 사용 시
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    conda install -c conda-forge opencv matplotlib

4. 📂 코드 구성 및 흐름
  4.1. 라이브러리 및 모델 로드
    필요한 패키지 로드
    torch.hub을 이용해 MiDaS 모델 다운로드

  4.2. 이미지 불러오기 및 전처리
    이미지 파일 경로 또는 URL에서 이미지 로드
    모델 입력 형식에 맞게 리사이징 및 정규화 수행

  4.3. 깊이 맵 예측
    모델을 통한 추론 수행 (no_grad)
    추정된 깊이 맵을 matplotlib를 사용해 시각화 및 저장

5. 🖼️ 사용 예시 
  img_path = "image.jpg"
  input_image = cv2.imread(img_path)

  #추론 및 저장까지 자동 수행됨
  결과는 depth_map.png로 저장됩니다.

  matplotlib에서 시각적으로 확인 가능

6. 🔍 주의 사항
  MiDaS는 상대 깊이만 예측하므로 절대 거리 측정은 불가
  고해상도 이미지 입력 시 처리 속도가 느려질 수 있음
  입력 이미지 크기와 품질에 따라 예측 결과 품질이 달라질 수 있음

