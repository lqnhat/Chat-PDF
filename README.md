# 💬 Ask a question to the PDF

## Giới thiệu
Ứng dụng Chat PDF là một ứng dụng cho phép trò chuyện với bất kỳ file PDF nào bằng ngôn ngữ hỏi đáp tự nhiên. Bạn chỉ cần cung cấp đường dẫn đến file PDF của bạn, ứng dụng sẽ sinh ra một chatbot có khả năng trả lời các câu hỏi của bạn liên quan đến file PDF đó. Bạn có thể hỏi bất cứ điều gì, từ những câu hỏi đơn giản như "Ai là tác giả của tệp PDF này?" hay "Tệp PDF này có bao nhiêu trang?" đến những câu hỏi phức tạp hơn như "Tóm tắt nội dung chính của tệp PDF này?" hay "So sánh và đánh giá các phương pháp nghiên cứu được sử dụng trong tệp PDF này?".

Ứng dụng Chat PDF là một ứng dụng rất hữu ích cho những ai muốn khám phá và tìm hiểu thêm về các tài liệu PDF của mình. Bạn có thể sử dụng ứng dụng để học tập, nghiên cứu, làm việc, hay bất cứ mục đích nào bạn muốn. Bạn có thể trò chuyện với sách, bài báo khoa học, luận văn, hợp đồng pháp lý, hay bất cứ chủ đề nào bạn có!

Ứng dụng Chat với PDF sử dụng công nghệ của OpenAI API để xử lý ngôn ngữ tự nhiên (NLP) và sinh ra các câu trả lời. Ứng dụng sử dụng để sinh ra các câu trả lời tự động và chính xác , và GPT API để trích xuất văn bản từ PDF và tạo embedding cho các phần của tài liệu. Ứng dụng cũng sử dụng các thuật toán học máy để phân tích và hiểu nội dung của PDF, và kết hợp với các nguồn tri thức khác để cung cấp cho bạn những thông tin bổ sung và hữu ích.


## Mục tiêu
Sử dụng OpenAI API kết hợp với các kiến thức đã học về kỹ thuật prompt engineer để xây dựng ứng dụng Chatbot trên file PDF (một trong những định dạng dữ liệu phổ biến hiện nay). Bên cạnh đó, bạn còn vận dụng một số thư viện của Python liên quan đến việc đọc nội dung PDF.


## Các thư viện hỗ trợ
- [pdfplumber](https://pypi.org/project/pdfplumber/0.1.2/): thư viện python để đọc file pdfs
- [langchain](https://python.langchain.com/): thư viện để dễ dàng kết hợp các prompts. Khuyến khích sử dụng OpenAI API gốc để hiểu rõ hơn về API. Xem thêm bài viết những ý kiến trái chiều về langchain.
- OpenAI [embeddings/pinecone/huggingface](https://huggingface.co/blog/getting-started-with-embeddings) để sử dụng embeddings, thuận lợi cho việc tương tác với file pdfs dài vượt quá context length
- [Streamlit](https://app-starter-kit.streamlit.app/): thư viện giúp tạo web apps nhanh chóng bằng Python (Optional)


## Usage


1. Add your OpenAI API key into **.streamlit/secrets.toml**
```secrets.toml
OPENAI_API_KEY = "<YOUR_OPENAI_API_KEY>"
```
2. Create virtual environment and activate it.
```
$> virtualenv venv

$> source venv/bin/activate
```
3. Install all required libraries.

```
$> pip install --upgrade pip

$> pip install -r requirements.txt
```
4. Run the app.
```
$> streamlit run streamlit_app.py
```


## GitHub Codespaces

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/lqnhat/Chat-PDF)