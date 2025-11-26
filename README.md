# 🍜 VN Food Recommendation RAG Engine

Hệ thống **Hybrid Search Engine** (Tìm kiếm lai) chuyên dụng cho việc gợi ý địa điểm ăn uống, kết hợp giữa **AWS Bedrock (LLM)** và **PostgreSQL (pgvector)**. Hệ thống sử dụng kỹ thuật RAG (Retrieval-Augmented Generation) để hiểu ngữ cảnh, tâm trạng người dùng và tìm kiếm dữ liệu thời gian thực.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![AWS Bedrock](https://img.shields.io/badge/AWS-Bedrock-orange.svg)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-blue)

---

## 🚀 Công Nghệ Sử Dụng (Tech Stack)

* **Backend Framework:** FastAPI (Python).
* **Database:** PostgreSQL + extension `pgvector` (Lưu trữ data nhà hàng & Vector Embedding).
* **LLM & AI Services (AWS Bedrock):**
    * **Embedding:** `amazon.titan-embed-text-v2:0` (Tạo vector 1024 chiều).
    * **Intent Parsing:** `anthropic.claude-3-haiku-20240307-v1:0` (Nhanh, rẻ, dùng để trích xuất bộ lọc).
    * **Chat/Response:** `anthropic.claude-3-5-sonnet-20241022-v2:0` (Thông minh, dùng để tổng hợp câu trả lời).
* **ORM:** SQLAlchemy.

---

## ✨ Tính Năng Nổi Bật (Key Features)

### 1. 🧠 Smart Intent Parsing (Phân tích ý định)
Sử dụng **Claude 3 Haiku** với tính năng *Tool Use* để trích xuất cấu trúc dữ liệu từ ngôn ngữ tự nhiên:
* **Chuẩn hóa địa danh:** Tự động hiểu `Q1`, `Q.Nhất` -> `Quận 1`.
* **Nhận diện Category thông minh:** Phân biệt rõ nhu cầu *Ăn* (Cơm, Phở), *Uống* (Cafe), *Nhậu* (Bar, Pub).
* **Xử lý phủ định:** Hiểu các yêu cầu như "trừ quận 4", "không ăn hải sản".

### 2. 🎭 Mood Analysis & Toxic Handling (Xử lý cảm xúc)
Hệ thống tự động phát hiện tâm trạng người dùng (đặc biệt khi người dùng tiêu cực, chửi thề, buồn chán) để thay đổi chiến lược:

| Trạng thái User | Hành động của AI |
| :--- | :--- |
| **Neutral** (Bình thường) | Tìm kiếm theo đúng yêu cầu, trả lời lịch sự, ngắn gọn. |
| **Negative** (Chửi bậy, Buồn) | 1. **Thay đổi Tone:** Chuyển sang giọng đồng cảm, "chill", xoa dịu (như bạn bè).<br>2. **Auto-Suggest:** Tự động gợi ý các món "Giải sầu" (Bia, Bar, Pub, Đồ ngọt, Lẩu). |

### 3. 🔍 Hybrid Search & Fallback Pipeline
Kết hợp sức mạnh của Keyword Search và Semantic Search:
* **Công thức:** `Score = 0.3 * (Keyword Rank) + 0.7 * (Vector Cosine Similarity)`.
* **Cơ chế Fallback (Dự phòng):** Không bao giờ trả về "Không tìm thấy" ngay lập tức.
    1.  *Strict:* Tìm chính xác mọi tiêu chí.
    2.  *Relax Price/Time:* Nếu không có, bỏ qua giá và giờ mở cửa.
    3.  *Relax District:* Nếu vẫn không có, tìm sang quận lân cận.
    4.  *Semantic Only:* Tìm dựa trên "Vibe" (ngữ nghĩa vector).

---

## 🛠️ Cài đặt & Cấu hình (Installation)

### 1. Yêu cầu hệ thống
* Python 3.9+
* PostgreSQL (đã cài extension `vector`).
* Tài khoản AWS có quyền truy cập Bedrock (Titan V2, Claude 3 Haiku, Claude 3.5 Sonnet).

### 2. Biến môi trường
Cập nhật các biến trong file code (hoặc chuyển sang file `.env`):

```python
USERNAME = "postgres"
PASSWORD = "your_password"
HOST = "localhost"
DATABASE = "food_recommendation"
AWS_REGION = "us-west-2"
```

### 3. Chạy ứng dụng

```bash
pip install boto3 fastapi uvicorn sqlalchemy psycopg2-binary pytz
python main.py
```

Server sẽ chạy tại: http://0.0.0.0:7000

## 🔌 API Documentation

### 🔍 Search Endpoint

Method: POST

**URL: /api/search**

Request Payload

**Content-Type: application/json**
```json
{

"query": "Tao đang chán đời quá, tìm chỗ nào nhậu ở Quận 1 đi",

"session_id": "session_123456",

"is_new_topic": false

}
```
- query: Câu hỏi tự nhiên của người dùng.
- session_id: ID phiên làm việc (để duy trì ngữ cảnh chat).
- is_new_topic: true nếu muốn reset lịch sử chat.

Response Example

```json
{
"answer": "Hạ hỏa nào bạn ơi, đời còn dài gái còn nhiều. Làm ly bia cho quên sự đời nhé! Dưới đây là mấy quán 'chất' ở Quận 1 cho bạn giải sầu:",
"restaurants": [
{
"id": 101,
"name": "Bia Craft Sài Gòn",
"address": "Lê Thánh Tôn, Quận 1",
"priceRange": "50000 - 150000",
"hours": "16:00 - 23:59",
"category": "Beer/Pub",
"score": "0.92"
}
],
"debug_intent": {
"district": "Quận 1",
"mood": "negative",
"target_categories": ["Quán nhậu", "Beer", "Bar"]
}
}
```

## 🗂️ Cấu trúc hệ thống (System Architecture)

1. Client gọi API với câu query.

2. RAG Service gọi Claude Haiku để phân tích Intent & Mood.

3. Search Engine thực hiện truy vấn DB:
   - Tạo Embedding từ query (Titan V2).
   - Thực thi SQL Query (Hybrid Search).
   - Nếu ít kết quả → kích hoạt Fallback Mechanism.

4. Generation:
   - Tổng hợp kết quả tìm kiếm.
   - Gửi Prompt + Context + Mood instruction sang Claude Sonnet.

5. Return: Trả về câu trả lời dạng text và JSON danh sách quán.

## 📝 Logs & Monitoring

Hệ thống tích hợp logging chi tiết để theo dõi chi phí và hiệu năng:

- [BEDROCK]: Theo dõi Token Input/Output và thời gian phản hồi của model.

- [SQL]: Theo dõi thời gian truy vấn DB và điểm số (Score) của kết quả.

- [INTENT]: Log lại các filter mà AI đã trích xuất được.
