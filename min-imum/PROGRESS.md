# 🚀 OpenPi - Jetson AGX Orin 개발 진행 현황

이 문서는 Jetson AGX Orin 환경에서 OpenPi(π₀.₅) 모델을 구동하고 Dobot E6 로봇에 적용하기 위한 기술적 진행 사항을 기록합니다.

---

## 🏗️ 1. 핵심 인프라 구축 (Infrastructure)

### ✅ Torchvision CUDA 빌드 성공
- **문제**: 기존 Torchvision에 CUDA 연산 지원이 누락되어 NMS 등 핵심 Ops가 CPU에서만 동작함.
- **해결**: 
  - `torchvision` v0.20.1 소스 빌드 수행.
  - 빌드 플래그(`FORCE_CUDA=1`, `NVCC_APPEND_FLAGS`) 최적화.
  - `ninja` 빌드 시스템을 사용하여 Jetson Orin 아키텍처(aarch64)에 최적화된 바이너리 생성.
- **결과**: `torchvision.ops.nms`가 GPU에서 정상 작동함을 확인 (`verify_torchvision.py`).

### ✅ JAX/Jaxlib 하이브리드 설정
- **문제**: Jetpack 6 (CUDA 12.6, cuDNN 9.3)과 JAX 라이브러리 간의 버전 매칭 이슈로 GPU 초기화 에러 발생.
- **해결**: 
  - **PyTorch (GPU)** + **JAX (CPU)** 하이브리드 모드 채택.
  - 전용 환경 변수 `export JAX_PLATFORMS=cpu` 설정을 통해 안정적인 구동 환경 확보.
  - 모델의 핵심 연산은 PyTorch GPU 백엔드가 담당하므로 실질적 성능 손실 최소화.

---

## 🧠 2. OpenPi 모델 구동 및 디버깅 (Model & Debugging)

### ✅ Checkpoint 복원 버그 수정
- **문제**: `orbax-checkpoint` 라이브러리 업데이트로 인해 `StepMetadata` 객체를 딕셔너리처럼 인덱싱할 수 없는 오류 발생.
- **해결**: `openpi/models/model.py` 소스 코드 패치를 통해 라이브러리 버전 호환성 확보.
- **결과**: `pi05_droid` 모델 파라미터 로딩 성공.

### ✅ 추론(Inference) 테스트 완료
- `pi05_droid` 체크포인트를 사용한 엔드투엔드 추론 테스트 성공.
- 입력: 더미 이미지 및 상태 데이터.
- 출력: 15개의 타임스텝에 대한 8D 액션(7 Joint + 1 Gripper) 생성 완료.

---

## 🤖 3. Dobot E6 전이 학습 설계 (Robot Transfer)

### 📐 액션 차원 최적화 (8D → 7D)
- **분석**: Franka Panda(7축) 기준 모델을 Dobot E6(6축 + 석션)에 맞추기 위한 전략 수립.
- **결정**: 
  - **7D (6 Joint + 1 Suction)** 차원으로 모델 헤드(Action Head)를 재구성하여 Fine-tuning 진행 예정.
  - 석션 컵(On/Off)은 기존 연속형 그리퍼 제어값을 임계값(Threshold) 기반으로 변환하여 매핑.

### 🎯 전이 학습(Fine-tuning) 가이드
- **데이터셋**: Dobot E6의 실제 이동 궤적 데이터를 수집하여 LeRobot 포맷으로 변환.
- **지능 전이**: Gemma 2B 거대 모델의 시각 지능은 유지하고, 출력 레이어만 Dobot의 기하학적 구조에 맞춰 학습.

---

## 📝 4. 주요 파일 가이드

- `verify_torchvision.py`: CUDA 연산 연동 확인 스크립트.
- `test_openpi_inference.py`: OpenPi 추론 테스트 메인 스크립트.
- `openpi/models/model.py`: Orbax 호환성 패치가 적용된 핵심 모델 코드.

---
**Last Updated**: 2026-01-29
**Status**: 추론 성공 및 전이 학습 준비 완료
