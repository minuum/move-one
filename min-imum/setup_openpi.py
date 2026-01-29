#!/usr/bin/env python3
"""OpenPi 환경 설정 스크립트 (Jetson AGX Orin용)"""

import subprocess
import sys
import os

def run_cmd(cmd, description):
    """명령어 실행 헬퍼"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"실행: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"✗ 실패: {description}")
        return False
    print(f"✓ 완료: {description}")
    return True

def main():
    print("="*60)
    print("OpenPi 환경 설정 시작")
    print("="*60)
    
    # 1. NumPy 1.x로 다운그레이드 (PyTorch 호환성)
    if not run_cmd(
        "uv pip install 'numpy<2,>=1.26'",
        "[1/10] NumPy 1.x 설치 (PyTorch 호환)"
    ):
        return False
    
    # 2. JAX와 Flax 설치 (OpenPi 필수)
    if not run_cmd(
        "uv pip install jax jaxlib flax",
        "[2/10] JAX/Flax 설치 (OpenPi 백엔드)"
    ):
        return False
    
    # 3. 필수 패키지 설치
    packages = [
        "transformers==4.53.2",
        "sentencepiece",
        "pillow",
        "opencv-python-headless",  # headless 버전은 numpy 의존성 약함
        "einops",
        "beartype==0.19.0",
        "jaxtyping==0.2.36",
        "ml_collections",
        "dm-tree",
        "tqdm",
        "etils[epath]",
        "absl-py",
        "orbax-checkpoint",
        "tyro",  # OpenPi CLI 파서
        "augmax",  # OpenPi 데이터 augmentation
        "pytest",  # OpenPi 테스트 프레임워크
        "tqdm-loggable",  # OpenPi 다운로드 진행 표시
        "openpi-client",  # OpenPi 클라이언트 유틸리티
        "numpydantic",  # OpenPi 데이터 검증
        "gcsfs",  # Google Cloud Storage 지원
        "lerobot",  # OpenPi 정책 로딩 의존성
    ]
    
    for i, pkg in enumerate(packages, start=3):
        if not run_cmd(
            f"uv pip install {pkg}",
            f"[{i}/6] {pkg} 설치"
        ):
            return False
    
    # 4. NumPy 버전 재확인 및 고정
    if not run_cmd(
        "uv pip install 'numpy<2,>=1.26' --force-reinstall",
        "[최종-1] NumPy 버전 재고정"
    ):
        return False
    
    # 5. OpenPi 소스 경로 확인
    print("\n" + "="*60)
    print("[최종] OpenPi 모듈 확인")
    print("="*60)
    
    result = subprocess.run(
        "python -c 'import sys; sys.path.insert(0, \".\"); from openpi.training import config; print(\"[OK] OpenPi import success\")'",
        shell=True,
        cwd="/home/billy/move-one/min-imum"
    )
    
    if result.returncode == 0:
        print("\n" + "="*60)
        print("[SUCCESS] OpenPi environment setup complete!")
        print("="*60)
        print("\nNext steps:")
        print("  1. Download checkpoint:")
        print("     python download_checkpoint.py")
        print("\n  2. Run inference test:")
        print("     python test_openpi_inference.py")
        return True
    else:
        print("\n[ERROR] OpenPi import failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
