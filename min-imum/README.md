# 🚀 OpenPi - Jetson AGX Orin 테스트 & 전이 학습 환경

Physical Intelligence의 **OpenPi (π₀.₅)** 모델을 Jetson AGX Orin에서 구동하고 Dobot E6 로봇에 적용하기 위한 환경입니다.

## 📋 핵심 사용 가이드 (Quick Start)

가상환경을 활성화한 후, 점검 스크립트 하나로 모든 상태를 확인할 수 있습니다.

```bash
cd ~/move-one/min-imum
source move-one/bin/activate
export JAX_PLATFORMS=cpu  # 필수: JAX 충돌 방지
python3 check_env.py      # 환경 및 GPU 연산 통합 점검
```

### 🎯 주요 실행 명령어
- **환경 점검**: `python3 check_env.py` (GPU 연산, 라이브러리 호환성 체크)
- **추론 테스트**: `python3 test_openpi_inference.py` (실제 모델 추론 동작 확인)
- **보고서 생성**: `python3 generate_activity_report.py` (활동 보고용 텍스트 생성)

---

## 🤝 서버 인수인계 사항 (Handover Notes)

다른 환경이나 서버 사용자에게 인수인계 시 다음 사항을 반드시 전달해 주세요.

1. **JAX CPU 모드 강제**: Jetson Orin의 cuDNN 버전 이슈로 JAX GPU 초기화 시 에러가 발생할 수 있습니다. `export JAX_PLATFORMS=cpu`를 반드시 선언하세요. (추론은 PyTorch GPU가 담당하므로 성능에 지장 없음)
2. **Torchvision 커스텀 빌드**: 본 환경의 `torchvision`은 CUDA Ops(NMS 등)를 위해 해당 장비에서 직접 빌드되었습니다. 삭제나 재설치 시 `build_torchvision.sh`를 통해 다시 빌드해야 합니다.
3. **Orbax 패치**: 라이브러리 버전 차이로 인해 `openpi/models/model.py`에 `StepMetadata` 관련 패치가 적용되어 있습니다. 코드를 새로 받으실 경우 해당 패치 여부를 확인하세요.
4. **Dobot E6 전이 방식**: 현재 모델은 8D(7축+그리퍼) 기반입니다. 6축인 Dobot E6에 적용하려면 `action_dim=7`로 설정하고 관절/좌표 매핑을 거치는 Fine-tuning이 필요합니다.

---

## 🖥️ 환경 정보

- **하드웨어**: Jetson AGX Orin (64GB GPU)
- **Python**: 3.10.12
- **PyTorch**: 2.5.0a0+872d972e41.nv24.08 (NVIDIA Jetson ARM64)
- **CUDA**: 12.6.68
- **패키지 관리**: uv
- **가상환경**: `move-one`

## 🚀 빠른 시작 (3단계)

### 1️⃣ 환경 설정

```bash
cd ~/move-one/min-imum
source move-one/bin/activate
python setup_openpi.py
```

이 스크립트는 다음을 수행합니다:
- NumPy 버전 호환성 해결 (NumPy 1.x 설치)
- OpenPi 필수 패키지 설치 (transformers, einops 등)
- Import 테스트

### 2️⃣ 체크포인트 다운로드

```bash
python download_checkpoint.py
```

**추천 체크포인트**: `pi05_droid` (빠른 추론 + 좋은 언어 이해)

또는 수동 다운로드:
```bash
# gsutil 설치 필요
gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_droid ./checkpoints/
```

### 3️⃣ 추론 테스트

```bash
python test_openpi_inference.py
```

## 📂 파일 구조

```
min-imum/
├── move-one/                       # Python 가상환경 (이름 변경됨)
├── openpi/                         # OpenPi 소스 코드
├── checkpoints/                    # 모델 체크포인트 (다운로드됨)
│   ├── pi05_droid/                # π₀.₅ DROID 체크포인트
│   ├── pi05_base/                 # π₀.₅ Base (Fine-tuning용)
│   └── pi0_droid/                 # π₀ DROID (이전 버전)
│
├── setup_openpi.py                # 환경 설정 스크립트 ⭐
├── download_checkpoint.py          # 체크포인트 다운로더 ⭐
├── test_openpi_inference.py       # 추론 테스트 스크립트 ⭐
│
├── test_cuda.py                   # CUDA 테스트
├── install_dependencies.py         # (구버전 - setup_openpi.py 사용 권장)
└── README.md                      # 이 파일
```

## 🤖 사용 가능한 모델

### Pi0.5 모델 (권장)

| 체크포인트 | 크기 | 설명 | 용도 |
|-----------|------|------|------|
| **pi05_droid** | ~3GB | DROID 데이터 Fine-tuned | **추론 (추천)** |
| pi05_base | ~3GB | Base 모델 | Fine-tuning용 |

### Pi0 모델 (이전 버전)

| 체크포인트 | 크기 | 설명 | 용도 |
|-----------|------|------|------|
| pi0_droid | ~2.5GB | DROID 데이터 Fine-tuned | 추론 (더 빠름) |
| pi0_base | ~2.5GB | Base 모델 | Fine-tuning용 |

**차이점**:
- **π₀.₅**: 더 나은 일반화 성능, 언어 이해 향상
- **π₀**: 더 빠른 추론 속도

## ⚡ 메모리 요구사항

| 작업 | GPU 메모리 | Jetson Orin 지원 |
|------|-----------|-----------------|
| 추론 | 8GB+ | ✅ 가능 |
| LoRA Fine-tuning | 22.5GB+ | ✅ 가능 |
| Full Fine-tuning | 70GB+ | ❌ (메모리 부족) |

## 🔧 트러블슈팅

### NumPy 버전 에러
```bash
# NumPy 1.x로 다운그레이드
uv pip install 'numpy<2'
```

### Import 에러
```bash
# 환경 재설정
python setup_openpi.py
```

### CUDA 확인
```bash
python test_cuda.py
```

### 메모리 부족
```bash
# 메모리 사용량 확인
jtop

# 불필요한 프로세스 종료
pkill -f jtop  # 예시
```

### gsutil 없음
```bash
# Google Cloud SDK 설치
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
```

## 📝 다음 단계

1. **✅ 추론 테스트 완료 후**:
   - Dobot E6 로봇 데이터 수집
   - Fine-tuning 준비

2. **Fine-tuning 예시**:
   ```bash
   cd ~/move-one/openpi
   
   # Norm stats 계산
   uv run scripts/compute_norm_stats.py --config-name dobot_e6_config
   
   # 학습 시작
   uv run scripts/train_pytorch.py dobot_e6_config --exp_name dobot_exp
   ```

3. **Policy 서버 구축**:
   ```bash
   uv run scripts/serve_policy.py policy:checkpoint \
       --policy.config=pi05_droid \
       --policy.dir=./checkpoints/pi05_droid \
       --port=8000
   ```

## 📚 참고 자료

- **OpenPi GitHub**: https://github.com/Physical-Intelligence/openpi
- **OpenPi 블로그**: https://www.physicalintelligence.company/blog/pi05
- **DROID Dataset**: https://droid-dataset.github.io/
- **NVIDIA Jetson Forum**: https://forums.developer.nvidia.com/

## 🎯 프로젝트 목표

- [x] OpenPi 환경 구축
- [ ] Pi0.5-DROID 체크포인트 다운로드
- [ ] 추론 테스트 성공
- [ ] Dobot E6 데이터로 Fine-tuning
- [ ] 실제 로봇 제어 통합
