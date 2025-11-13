# honda_smart_advisor
honda-smart-advisor/
├── city-small.pdf
├── crv.pdf
├── honda_smart_advisor.py
├── .env
└── requirements.txt

AI Honda smart advisor
Các bước chạy ứng dụng
Đi đến thư mục chứa file honda_smart_advisor.py và file .env.
cd path/to/honda-smart-advisor
# 2. Tạo môi trường ảo (khuyến nghị)
python -m venv venv
source venv/bin/activate  # Trên macOS/Linux
venv\Scripts\activate     # Trên Windows
# 3. Cài đặt thư viện
pip install -r requirements.txt
# 4. Chạy ứng dụng Streamlit
venv\Scripts\activate
streamlit run honda_smart_advisor.py
http://localhost:8501
- chuc nang:
- Tư vấn xe
- So sánh xe
- Đề xuất xe