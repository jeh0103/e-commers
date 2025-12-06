실행 방법

(1) 권장: 가상환경 생성

python3 -m venv .venv source .venv/bin/activate

(2) 필수 패키지 설치

python -m pip install --upgrade pip python -m pip install "streamlit>=1.30" pandas numpy

(3) (선택) 리스트 위험도 그라데이션용 Matplotlib 설치 (Matplotlib이 없으면 CSS 그라데이션으로 자동 대체됩니다.)

python -m pip install matplotlib

(4) 앱 실행

python -m streamlit run app_enhanced.py

💡 실행 후 브라우저에서 http://localhost:8501 로 접속하세요.
