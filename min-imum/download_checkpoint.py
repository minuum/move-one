#!/usr/bin/env python3
"""OpenPi 체크포인트 다운로드 스크립트"""

import subprocess
import os
import sys

CHECKPOINTS = {
    "pi05_droid": {
        "url": "gs://openpi-assets/checkpoints/pi05_droid",
        "description": "Pi0.5-DROID (추천: 빠른 추론 + 좋은 언어 이해)",
        "size": "~3GB"
    },
    "pi05_base": {
        "url": "gs://openpi-assets/checkpoints/pi05_base",
        "description": "Pi0.5 Base (Fine-tuning용 기본 모델)",
        "size": "~3GB"
    },
    "pi0_droid": {
        "url": "gs://openpi-assets/checkpoints/pi0_droid",
        "description": "Pi0-DROID (이전 버전, 더 빠른 추론)",
        "size": "~2.5GB"
    },
}

def check_gsutil():
    """gsutil 설치 확인"""
    result = subprocess.run(["which", "gsutil"], capture_output=True)
    return result.returncode == 0

def download_checkpoint(checkpoint_name):
    """체크포인트 다운로드"""
    if checkpoint_name not in CHECKPOINTS:
        print(f"✗ 알 수 없는 체크포인트: {checkpoint_name}")
        return False
    
    info = CHECKPOINTS[checkpoint_name]
    checkpoint_dir = f"./checkpoints/{checkpoint_name}"
    
    print("="*60)
    print(f"체크포인트 다운로드: {checkpoint_name}")
    print("="*60)
    print(f"설명: {info['description']}")
    print(f"크기: {info['size']}")
    print(f"다운로드 위치: {checkpoint_dir}")
    print("="*60)
    
    # 디렉토리 생성
    os.makedirs("./checkpoints", exist_ok=True)
    
    if os.path.exists(checkpoint_dir):
        response = input(f"\n{checkpoint_dir}가 이미 존재합니다. 덮어쓰시겠습니까? (y/N): ")
        if response.lower() != 'y':
            print("다운로드 취소됨")
            return False
        subprocess.run(["rm", "-rf", checkpoint_dir])
    
    # gsutil로 다운로드
    print(f"\n다운로드 시작... (시간이 걸릴 수 있습니다)")
    cmd = ["gsutil", "-m", "cp", "-r", info["url"], checkpoint_dir]
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n" + "="*60)
        print(f"✅ 다운로드 완료: {checkpoint_dir}")
        print("="*60)
        return True
    else:
        print("\n✗ 다운로드 실패")
        return False

def download_checkpoint_openpi(checkpoint_name, dest_dir):
    """OpenPi 내부 다운로더 사용"""
    print(f"\n[OpenPi Downloader] {checkpoint_name} 다운로드 중...")
    
    try:
        # OpenPi 모듈 import
        import sys
        import os
        
        # 다운로드 경로 환경변수 설정
        abs_checkpoints_dir = os.path.abspath("./checkpoints")
        os.environ["OPENPI_DATA_HOME"] = abs_checkpoints_dir
        print(f"Set OPENPI_DATA_HOME = {abs_checkpoints_dir}")
        
        sys.path.insert(0, ".")
        from openpi.shared import download
        
        url = CHECKPOINTS[checkpoint_name]["url"]
        print(f"URL: {url}")
        
        # 다운로드 실행
        downloaded_path = download.maybe_download(url)
        
        print("\n" + "="*60)
        print(f"✅ 다운로드 완료: {downloaded_path}")
        print("="*60)
        
        # 심볼릭 링크 생성 (편의상)
        target_link = os.path.join(abs_checkpoints_dir, checkpoint_name)
        if not os.path.exists(target_link) and os.path.exists(downloaded_path):
             # URL 구조에 따라 깊은 경로에 있을 수 있음. 
             # 여기서는 다운로드가 완료되었다는 것만 확인
             pass
             
        return True
        
    except ImportError as e:
        print(f"\n✗ Import Error 발생: {e}")
        print("  상세 정보:")
        import traceback
        traceback.print_exc()
        print("\n  팁: 'gcsfs' 패키지가 누락되었을 수 있습니다.")
        return False
    except Exception as e:
        print(f"\n✗ 다운로드 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("OpenPi 체크포인트 다운로더")
    print("="*60)
    
    print("\n사용 가능한 체크포인트:")
    for i, (name, info) in enumerate(CHECKPOINTS.items(), 1):
        print(f"\n{i}. {name}")
        print(f"   {info['description']}")
        print(f"   크기: {info['size']}")
    
    print("\n추천: pi05_droid (빠르고 정확함)")
    
    # 사용자 선택
    choice = input("\n다운로드할 체크포인트 이름 입력 (기본: pi05_droid): ").strip()
    if not choice:
        choice = "pi05_droid"
        
    if choice not in CHECKPOINTS:
        print(f"✗ 잘못된 선택: {choice}")
        sys.exit(1)
        
    # 체크포인트 디렉토리 준비
    dest_dir = os.path.abspath(f"./checkpoints/{choice}")
    
    # gsutil 시도
    if check_gsutil():
        print("\n[Method 1] gsutil 사용")
        download_checkpoint(choice)
    else:
        print("\n[Method 2] OpenPi 내부 다운로더 사용 (gsutil 없음)")
        # 체크포인트 디렉토리 생성
        download_checkpoint_openpi(choice, "./checkpoints")
        
    sys.exit(0)

if __name__ == "__main__":
    main()
