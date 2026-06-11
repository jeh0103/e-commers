# E-commerce Customer Ops

Streamlit 프로토타입을 FastAPI 백엔드와 Jinja 기반 프론트엔드로 옮긴 CRM 운영 화면입니다.

## FastAPI 실행 방법

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

실행 후 브라우저에서 <http://127.0.0.1:8000> 로 접속하세요.

## 주요 화면

- `/` : 고객 관리 메인 화면
- `/risky` : 우선 연락 대상 목록
- `/vip` : VIP 관리 및 전환 후보 목록
- `/customer-types` : 고객유형별 고객 목록
- `/customers/{customer_id}` : 고객 상세, 문자 생성, 액션 기록

## API

- `GET /api/summary`
- `GET /api/customers`
- `GET /api/customers/{customer_id}`
- `POST /api/customers/{customer_id}/actions`
- `POST /api/customers/{customer_id}/sms-preview`
- `GET /api/risky.csv`
- `GET /api/vip-candidates.csv`

## 기존 Streamlit 프로토타입

기존 파일은 보존되어 있습니다. Streamlit 버전이 필요하면 아래처럼 실행할 수 있습니다.

```bash
python -m pip install "streamlit>=1.30" pandas numpy matplotlib
python -m streamlit run app_enhanced.py
```
