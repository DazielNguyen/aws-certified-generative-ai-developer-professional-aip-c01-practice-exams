# **Randomized Test – AWS Certified Generative AI Developer Professional**

### **Question 01:**
Category: AIP – Testing, Validation, and Troubleshooting
A global e-commerce company is developing a product return prediction model using the Amazon SageMaker XGBoost algorithm to identify which orders are most likely to result in customer returns. Transactional and behavioral data are stored in Amazon Redshift, and the data engineering team utilizes SageMaker Data Wrangler to clean and prepare training features, including product category, customer sentiment, and delivery delays.

After deploying the model, the team observes that while the training Area Under the Curve (AUC) is exceptionally high, the validation performance drops significantly. Model explainability reports from SageMaker Clarify reveal that the model is overly sensitive to a few dominant features like product price and region, suggesting overfitting to patterns in the training set. The team decides to tune the model’s hyperparameters to reduce model complexity and improve its ability to generalize to new data.

Which hyperparameter modification would best help reduce overfitting and improve validation performance for this XGBoost model?

[x] Decrease the value of the max_depth hyperparameter to limit tree complexity and prevent overfitting on dominant patterns in the training data.

[ ] Decrease the min_child_weight parameter to allow each leaf node to split more easily, capturing additional complexity in the dataset.

[ ] Increase the max_depth hyperparameter to allow deeper trees that can better capture relationships between rare features.

[ ] Increase the colsample_bytree parameter to ensure each tree uses a greater subset of features and improves model variance.

> Giải thích. 

Trong thuật toán XGBoost (một dạng Decision Tree Boosting), tình trạng overfitting thường xảy ra khi các cây quyết định quá sâu hoặc quá phức tạp, dẫn đến việc chúng học thuộc lòng cả các nhiễu (noise) trong dữ liệu huấn luyện thay vì học quy luật chung.

#### **1. Tại sao giảm `max_depth` lại hiệu quả?**

* **`max_depth`** kiểm soát độ sâu tối đa của một cái cây. Cây càng sâu, mô hình càng có khả năng tạo ra các quy tắc phân tách rất chi tiết (và hẹp), chỉ đúng cho một vài mẫu dữ liệu huấn luyện cụ thể.
* Bằng cách **giảm** `max_depth`, bạn buộc các cây phải dừng lại ở những tầng cao hơn, nơi các quy tắc phân tách mang tính tổng quát hơn. Điều này trực tiếp làm giảm độ phức tạp và ngăn mô hình "quá nhạy cảm" với các đặc trưng chiếm ưu thế như báo cáo từ SageMaker Clarify đã chỉ ra.

#### **2. Tại sao các phương án khác không phù hợp?**

* **Giảm `min_child_weight` (Sai):** Tham số này quy định trọng số tối thiểu cần thiết trong một nút con để tiếp tục phân tách. Nếu giảm nó, mô hình sẽ dễ dàng tạo thêm nhiều nhánh nhỏ, làm tăng độ phức tạp và **làm trầm trọng thêm** tình trạng overfitting. (Để giảm overfitting, bạn nên *tăng* tham số này).
* **Tăng `max_depth` (Sai):** Như đã giải thích ở trên, việc tăng độ sâu sẽ làm mô hình phức tạp hơn và chắc chắn sẽ khiến kết quả trên tập xác thực tệ hơn do overfitting nặng hơn.
* **Tăng `colsample_bytree` (Sai):** Tham số này quy định tỷ lệ các đặc trưng được chọn ngẫu nhiên cho mỗi cây. Để giảm overfitting, người ta thường **giảm** tỷ lệ này để mô hình không quá phụ thuộc vào một vài đặc trưng mạnh (như giá và khu vực trong trường hợp này). Tăng nó lên sẽ khiến các cây có xu hướng sử dụng lại cùng các đặc trưng mạnh đó, không giúp ích cho việc tổng quát hóa.

#### **Mẹo nhỏ khi làm bài thi về XGBoost Overfitting:**

Để giảm overfitting trong XGBoost, bạn nên đi theo hướng:

* **Giảm:** `max_depth`, `eta` (learning rate), `colsample_bytree`, `subsample`.
* **Tăng:** `min_child_weight`, `gamma`, `alpha/lambda` (L1/L2 regularization).

---

### **Question 02:**

A social media platform is developing an ML model to moderate user-generated content. The platform stores large volumes of textual and document data in Amazon S3. It uses Amazon Textract to extract text from uploaded PDFs or images, and Amazon SageMaker AI to train custom language models that classify comments based on sentiment, context, and topic relevance.

The moderation team aims to enhance safety by detecting toxic or harmful language in real-time, including hate speech, harassment, and explicit threats. The solution must integrate seamlessly with the existing SageMaker AI inference pipeline, handle high throughput, and provide confidence scores for each classification, allowing content flagged as toxic to be reviewed or blocked automatically.

Which of the solutions provides a managed solution for detecting toxicity in text to support this ML pipeline?

[ ] Utilize Amazon Comprehend sentiment analysis to detect negative comments and block content automatically.

[ ] Use Amazon Translate to convert text into another language before moderation to reduce offensive content.

[ ] Use Amazon Bedrock to fine-tune a foundation model for general language understanding.

[x] Utilize Amazon Comprehend toxicity detection to identify abusive or harmful language in text.

> Giải thích: 

Đây là một bài toán về việc lựa chọn dịch vụ AI được quản lý (Managed AI Service) phù hợp nhất với mục tiêu cụ thể.

#### **1. Tại sao Amazon Comprehend Toxicity Detection là lựa chọn tốt nhất?**

* **Tính chuyên dụng:** AWS cung cấp một API cụ thể trong Amazon Comprehend được thiết kế riêng để phát hiện **Toxicity**. Nó có khả năng phân loại văn bản thành các danh mục như: *Hate speech* (ngôn từ thù ghét), *Harassment* (quấy rối), *Insult* (lăng mạ), *Threats* (đe dọa),... đúng như yêu cầu của đề bài.
* **Managed Solution (Giải pháp được quản lý):** Bạn không cần phải tự xây dựng, dán nhãn dữ liệu hay huấn luyện mô hình từ đầu. AWS đã huấn luyện mô hình này trên hàng triệu mẫu dữ liệu.
* **Confidence Scores:** API này trả về điểm tin cậy cho từng danh mục độc hại, cho phép hệ thống tự động đưa ra quyết định (ví dụ: điểm > 0.9 thì chặn ngay, điểm từ 0.5 - 0.9 thì gửi cho con người xem xét).
* **Khả năng tích hợp:** Dễ dàng gọi API này từ trong một SageMaker Inference Pipeline hoặc Lambda function.

#### **2. Tại sao các phương án khác không phù hợp?**

* **Phương án 1 (Sentiment Analysis):** Phân tích sắc thái chỉ cho biết văn bản đó là "Tích cực", "Tiêu cực" hay "Trung tính". Một câu nói có thể rất "Tiêu cực" (ví dụ: "Tôi buồn quá") nhưng không hề "Độc hại" (toxic). Ngược lại, một câu nói mỉa mai có thể có sắc thái trung tính nhưng lại mang tính quấy rối. Do đó, Sentiment Analysis không đủ để kiểm duyệt nội dung độc hại.
* **Phương án 2 (Amazon Translate):** Việc dịch ngôn ngữ không giải quyết được vấn đề phát hiện nội dung độc hại. Thậm chí, việc dịch có thể làm mất đi ngữ cảnh văn hóa và khiến việc phát hiện ngôn từ thù ghét trở nên khó khăn hơn.
* **Phương án 3 (Amazon Bedrock Fine-tuning):** Mặc dù Amazon Bedrock rất mạnh mẽ, nhưng việc tinh chỉnh (fine-tune) một mô hình nền tảng đòi hỏi nhiều nỗ lực về dữ liệu và chi phí hơn. Trong khi đó, đề bài hỏi về một "managed solution" sẵn có để phát hiện độc tính, và Comprehend Toxicity API là giải pháp "mì ăn liền" và tối ưu nhất cho mục đích này.
