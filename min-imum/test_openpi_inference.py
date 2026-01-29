#!/usr/bin/env python3
"""OpenPi 추론 테스트 (PyTorch 백엔드)"""

import sys
import os

# OpenPi 소스를 import 경로에 추가
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

print("=" * 70)
print("OpenPi (π₀.₅) PyTorch 추론 테스트")
print("=" * 70)

# GPU 확인
print(f"\n[시스템 확인]")
print(f"  Python: {sys.version.split()[0]}")
print(f"  PyTorch: {torch.__version__}")
print(f"  NumPy: {np.__version__}")
print(f"  CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# 체크포인트 경로 입력
print("\n" + "=" * 70)
print("사용 가능한 체크포인트:")
print("  - pi05_droid: ./checkpoints/pi05_droid")
print("  - pi05_base: ./checkpoints/pi05_base")
print("  - pi0_droid: ./checkpoints/pi0_droid")
print("=" * 70)

checkpoint_name = input("\n체크포인트 이름 입력 (기본: pi05_droid): ").strip()
if not checkpoint_name:
    checkpoint_name = "pi05_droid"

checkpoint_dir = f"./checkpoints/{checkpoint_name}"

if not os.path.exists(checkpoint_dir):
    print(f"\n✗ 체크포인트가 없습니다: {checkpoint_dir}")
    print("\n먼저 체크포인트를 다운로드하세요:")
    print("  python download_checkpoint.py")
    sys.exit(1)

print(f"\n[체크포인트]")
print(f"  경로: {checkpoint_dir}")

try:
    # OpenPi 모듈 import
    print("\n[1/4] OpenPi 모듈 로드 중...")
    from openpi.training import config as _config
    from openpi.policies import policy_config
    print("✓ OpenPi 모듈 로드 완료")
    
    # Config 로드
    print("\n[2/4] Config 로드 중...")
    config = _config.get_config(checkpoint_name)
    print(f"✓ Config 로드 완료: {checkpoint_name}")
    
    # Policy 생성
    print("\n[3/4] Policy 로드 중...")
    print("  (GPU 메모리 할당으로 시간이 걸릴 수 있습니다...)")
    
    policy = policy_config.create_trained_policy(config, checkpoint_dir)
    print("✓ Policy 로드 완료 (PyTorch 백엔드)")
    
    # 추론 테스트
    print("\n[4/4] 추론 테스트 중...")
    
    # 더미 입력 데이터 (DROID 형식)
    example = {
        "observation/exterior_image_1_left": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.zeros(8, dtype=np.float32),
        "observation/gripper_position": np.array([0.0], dtype=np.float32),
        "prompt": "pick up the red block"
    }
    
    # 추론 실행
    result = policy.infer(example)
    actions = result["actions"]
    
    print("\n" + "=" * 70)
    print("✅ 추론 성공!")
    print("=" * 70)
    print(f"\n출력:")
    print(f"  Actions shape: {actions.shape}")
    print(f"  Actions dtype: {actions.dtype}") 
    print(f"\n첫 3개 타임스텝 (8D 액션):")
    for i in range(min(3, actions.shape[0])):
        action_8d = actions[i][:8]
        print(f"  Step {i}: [{', '.join([f'{v:7.4f}' for v in action_8d])}]")
    
    # Actions 저장
    output_file = "/home/billy/openpi_inference_output.npy"
    np.save(output_file, actions)
    print(f"\n✓ Actions 저장: {output_file}")
    
    print("\n" + "=" * 70)
    print("🎉 OpenPi 모델이 정상적으로 작동합니다!")
    print("=" * 70)
    print("\n다음 단계:")
    print("  - 실제 로봇 데이터로 테스트")
    print("  - Dobot E6 데이터로 Fine-tuning")
    print("  - Policy 서버 구축")
    
except ImportError as e:
    print(f"\n✗ Import 에러: {e}")
    print("\n필요한 패키지가 설치되지 않았을 수 있습니다.")
    print("다음 명령어를 실행하세요:")
    print("  python setup_openpi.py")
    sys.exit(1)
    
except Exception as e:
    print(f"\n✗ 에러 발생: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
