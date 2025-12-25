# **Randomized Test – AWS Certified Generative AI Developer Professional**

### **Question 01:**

Category: AIP – Testing, Validation, and Troubleshooting

A global e-commerce company is developing a product return prediction model using the Amazon SageMaker XGBoost algorithm to identify which orders are most likely to result in customer returns. Transactional and behavioral data are stored in Amazon Redshift, and the data engineering team utilizes SageMaker Data Wrangler to clean and prepare training features, including product category, customer sentiment, and delivery delays.

After deploying the model, the team observes that while the training Area Under the Curve (AUC) is exceptionally high, the validation performance drops significantly. Model explainability reports from SageMaker Clarify reveal that the model is overly sensitive to a few dominant features like product price and region, suggesting overfitting to patterns in the training set. The team decides to tune the model’s hyperparameters to reduce model complexity and improve its ability to generalize to new data.

Which hyperparameter modification would best help reduce overfitting and improve validation performance for this XGBoost model?

[x] **Decrease the value of the max_depth hyperparameter to limit tree complexity and prevent overfitting on dominant patterns in the training data.**

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

Category: AIP – AI Safety, Security, and Governance

A social media platform is developing an ML model to moderate user-generated content. The platform stores large volumes of textual and document data in Amazon S3. It uses Amazon Textract to extract text from uploaded PDFs or images, and Amazon SageMaker AI to train custom language models that classify comments based on sentiment, context, and topic relevance.

The moderation team aims to enhance safety by detecting toxic or harmful language in real-time, including hate speech, harassment, and explicit threats. The solution must integrate seamlessly with the existing SageMaker AI inference pipeline, handle high throughput, and provide confidence scores for each classification, allowing content flagged as toxic to be reviewed or blocked automatically.

Which of the solutions provides a managed solution for detecting toxicity in text to support this ML pipeline?

[ ] Utilize Amazon Comprehend sentiment analysis to detect negative comments and block content automatically.

[ ] Use Amazon Translate to convert text into another language before moderation to reduce offensive content.

[ ] Use Amazon Bedrock to fine-tune a foundation model for general language understanding.

[x] **Utilize Amazon Comprehend toxicity detection to identify abusive or harmful language in text.**

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

--- 

### **Question 03:**

Category: AIP – AI Safety, Security, and Governance

An enterprise is working with an external ML lab specializing in Amazon SageMaker Feature Store and Amazon Bedrock model orchestration. The enterprise wants to keep all core training corpus assets and historical parquet feature store lineage assets inside Amazon S3 within the enterprise’s own AWS account because legal holds, privacy flags, and regional data sovereignty boundaries are enforced at the S3 layer.

The ML lab needs to perform highly iterative model development and comparison experiments directly against the enterprise-owned S3 data without copying the S3 datasets into the ML lab’s AWS account. The ML lab also needs to be able to browse the S3 training corpus from the AWS Management Console and run automated pipelines that require programmatic access.

Which of the following should be implemented?

[ ] Enroll the ML lab AWS account into the enterprise AWS Organizations hierarchy, create an organization-level Service Control Policy (SCP) granting S3 read access to the required S3 buckets, and propagate Service Control Policy inheritance so the ML lab can consume data as a member Organizational Unit (OU) with standard enterprise guardrails.

[x] **Create a delegated IAM role within the enterprise account that specifies trust exclusively toward the ML lab AWS account, and grant that role only the precise S3 level permissions required to read the feature corpus datasets and evaluate access both programmatically and via AWS console role assumption.**

[ ] Configure a VPC endpoint policy in the ML lab account that permits access to a specific S3 prefix pattern, then require the ML lab to always route all training workloads through this endpoint to enforce granular network access boundaries between both accounts.

[ ] Create an S3 Access Point in the enterprise account that points to the company’s bucket, then grant the ML lab’s AWS account IAM permissions directly on that access point’s ARN to enable both console listing and programmatic access for the ML lab.

> Giải thích: 

Để giải quyết bài toán này, chúng ta cần tìm giải pháp tối ưu nhất cho việc truy cập liên tài khoản (Cross-account access) mà vẫn đảm bảo tính bảo mật và tuân thủ các quy định về dữ liệu.

Trong môi trường AWS, khi một bên thứ ba (ML lab) cần làm việc với dữ liệu nằm trong tài khoản của khách hàng (Enterprise) mà không được phép sao chép dữ liệu, giải pháp tiêu chuẩn là sử dụng **IAM Cross-Account Role**.

#### **1. Tại sao chọn phương án này?**

* **Tính bảo mật (Security):** Doanh nghiệp (Enterprise) hoàn toàn kiểm soát quyền hạn. Họ tạo ra một Role và chỉ định rõ ràng rằng: "Tôi chỉ tin tưởng tài khoản của ML Lab này".
* **Nguyên tắc đặc quyền tối thiểu (Least Privilege):** Role này chỉ được cấp quyền `s3:Get*` và `s3:List*` trên các bucket hoặc prefix cụ thể cần thiết cho việc huấn luyện mô hình.
* **Hỗ trợ cả hai mục tiêu:**
    * **Programmatic access:** Các tập lệnh hoặc pipeline của Lab ML có thể sử dụng lệnh `sts:AssumeRole` để lấy thông tin xác thực tạm thời và truy cập dữ liệu.
    * **Console access:** Các kỹ sư của Lab ML có thể sử dụng tính năng **"Switch Role"** trên AWS Management Console để duyệt các bucket S3 của doanh nghiệp như thể họ đang ở trong tài khoản đó.


* **Tuân thủ (Compliance):** Dữ liệu không bao giờ rời khỏi bucket S3 của doanh nghiệp, đáp ứng các yêu cầu về chủ quyền dữ liệu (data sovereignty) và pháp lý (legal holds).

#### **2. Tại sao các phương án khác không phù hợp?**

* **Phương án A (VPC Endpoint Policy):** VPC Endpoint dùng để kiểm soát luồng dữ liệu giữa VPC và dịch vụ AWS (như S3) nhằm tránh đi qua internet công cộng. Nó không phải là cơ chế chính để cấp quyền truy cập liên tài khoản và không hỗ trợ việc duyệt S3 trên Console từ một tài khoản khác.
* **Phương án B (AWS Organizations & SCP):** SCP (Service Control Policy) là các "hàng rào" dùng để **giới hạn** quyền tối đa trong một tổ chức, nó không dùng để **cấp quyền** (grant access). Hơn nữa, việc đưa một đối tác bên thứ ba vào cấu trúc Organization của doanh nghiệp là không thực tế và rủi ro về mặt quản trị.
* **Phương án D (S3 Access Point):** Mặc dù Access Point giúp quản lý truy cập S3 dễ dàng hơn, nhưng nếu chỉ cấp quyền trên ARN của Access Point cho tài khoản Lab ML thì vẫn gặp khó khăn trong việc thiết lập quyền truy cập Console một cách mượt mà và an toàn so với việc sử dụng IAM Role với cơ chế AssumeRole.

---

### **Question 04:**

Category: AIP – Operational Efficiency and Optimization for Generative AI Applications

A research organization is developing an image classification solution by integrating Amazon Rekognition with Amazon SageMaker AI. The team uses Rekognition to automatically label thousands of wildlife images stored in Amazon S3, generating a high-quality annotated dataset. This labeled dataset is then used to train a supervised learning algorithm in SageMaker AI to classify different species of animals for ecological analysis.

After several training runs, the ML team observes that the model achieves 98% accuracy on the training dataset but only 76% accuracy on the validation dataset. The data scientists confirm that the dataset is clean, balanced, and correctly split. There is a suspicion that the model has become too complex and is memorizing specific image features from the training data instead of learning general patterns.

Which of the following should be implemented?

[ ] Modify the Rekognition labeling configuration to generate fewer image labels for simplicity.

[ ] Increase the model complexity by adding more convolutional layers to capture detailed patterns.

[ ] Reassign a portion of the training samples to the validation dataset. Adjust the split to 75% for training and 25% for validation to achieve a better balance.

[x] **Adjust model hyperparameters to introduce stronger regularization and retrain the model to minimize overfitting.**

> Giải thích: 

Vấn đề được mô tả trong câu hỏi (độ chính xác tập huấn luyện rất cao nhưng tập xác thực thấp) là dấu hiệu điển hình của **Overfitting (Quá khớp)**.

#### **1. Tại sao chọn Regularization (Chính quy hóa)?**

* **Bản chất của Overfitting:** Xảy ra khi mô hình quá linh hoạt hoặc phức tạp, dẫn đến việc nó học cả các "nhiễu" (noise) và các chi tiết đặc thù của tập huấn luyện thay vì học các đặc điểm chung có thể áp dụng cho dữ liệu mới.
* **Vai trò của Regularization:** Đây là kỹ thuật chính để giải quyết Overfitting. Các phương pháp chính quy hóa (như L1, L2, Dropout) sẽ áp đặt một hình phạt (penalty) lên các trọng số lớn hoặc độ phức tạp của mô hình, buộc mô hình phải học những quy luật tổng quát hơn và đơn giản hơn.

#### **2. Tại sao các phương án khác sai?**

* **Giảm nhãn của Rekognition:** Việc giảm số lượng nhãn không giải quyết được vấn đề mô hình "học vẹt" dữ liệu hiện có. Nếu dữ liệu đã sạch và chất lượng, vấn đề nằm ở cách mô hình xử lý dữ liệu đó.
* **Tăng độ phức tạp (Thêm lớp tích chập):** Đây là một lỗi sai nghiêm trọng trong trường hợp này. Việc tăng thêm lớp sẽ làm mô hình phức tạp hơn, khiến nó càng dễ dàng ghi nhớ dữ liệu huấn luyện hơn, từ đó làm tình trạng overfitting trở nên trầm trọng hơn.
* **Thay đổi tỷ lệ chia dữ liệu (75/25):** Các nhà khoa học dữ liệu đã xác nhận dữ liệu đã được chia tách đúng cách. Việc thay đổi tỷ lệ chia (ví dụ từ 80/20 sang 75/25) không giải quyết được gốc rễ của việc mô hình không có khả năng tổng quát hóa.

#### **Kết luận**

Khi gặp tình trạng **High Training Accuracy + Low Validation Accuracy**, hành động ưu tiên hàng đầu luôn là **giảm độ phức tạp của mô hình** hoặc **áp dụng Regularization**.

---

### **Question 05:**

Category: AIP – Foundation Model Integration, Data Management, and Compliance

A financial advisory firm is developing a question-answering application that uses Amazon Bedrock to access an Amazon Titan model. The application must provide detailed reasoning for complex financial queries, such as multi-step investment calculations, risk assessments, and explanations of regulatory compliance.

The engineering team notices that the model often provides correct answers but skips intermediate reasoning steps, making it difficult for auditors and clients to verify the logic. To improve transparency and accuracy, the team wants to design prompts that encourage the model to explain its reasoning step by step, ensuring that each intermediate calculation or decision is visible in the output.

Which prompt engineering technique should the team apply to improve multi-step reasoning and generate detailed explanations from the Amazon Titan model?

[ ] Implement instruction-based tuning by providing high-level task instructions.

[ ] Apply zero-shot prompting to ask the model directly for the final answer.

[ ] Utilize context window expansion by including more background information or documents in the prompt to improve answer accuracy.

[x] **Use a chain-of-thought prompting to guide the model through each step of the problem before providing a final answer.**

> Giải thích 

Trong lĩnh vực Generative AI, đặc biệt là với các mô hình nền tảng (Foundation Models) như Amazon Titan trên Bedrock, khả năng giải quyết các bài toán logic phức tạp phụ thuộc rất nhiều vào cách chúng ta đặt câu hỏi.

#### **1. Chain-of-Thought (CoT) Prompting là gì?**

* **Cơ chế:** CoT yêu cầu mô hình phân tích vấn đề thành một chuỗi các bước logic trung gian thay vì nhảy thẳng tới kết quả. Điều này thường được thực hiện bằng cách thêm các cụm từ như *"Let's think step by step"* (Hãy suy nghĩ từng bước một) hoặc cung cấp các ví dụ mẫu có sẵn các bước giải (Few-shot CoT).
* **Lợi ích:** Đối với tài chính, CoT giúp mô hình liệt kê rõ: Bước 1 tính thuế, Bước 2 tính lãi suất kép, Bước 3 so sánh với quy định,... Điều này giúp đáp ứng đúng yêu cầu của kiểm toán viên về tính minh bạch.

#### **2. Tại sao các phương án khác không phù hợp?**

* **Instruction-based tuning (Phương án 1):** Đây là quá trình huấn luyện lại mô hình (Fine-tuning) trên một tập dữ liệu chỉ dẫn. Đây là một phương pháp tốn kém, phức tạp và không phải là một "prompt engineering technique" đơn thuần mà nhóm có thể áp dụng nhanh chóng vào câu lệnh.
* **Zero-shot prompting (Phương án 2):** Đây là cách đặt câu hỏi trực tiếp mà không đưa ra ví dụ nào. Như đề bài đã nêu, cách làm này thường khiến mô hình bỏ qua các bước lập luận và chỉ đưa ra đáp án cuối cùng.
* **Context window expansion (Phương án 3):** Việc cung cấp thêm tài liệu (như kỹ thuật RAG - Retrieval-Augmented Generation) giúp mô hình có thêm thông tin chính xác để trả lời, nhưng nó không bắt buộc mô hình phải **trình bày lập luận** từng bước. Mô hình có thể có nhiều thông tin hơn nhưng vẫn chỉ đưa ra một kết quả ngắn gọn.

#### **Ví dụ về Chain-of-Thought trong tài chính:**

Thay vì hỏi: *"Lãi suất thực tế là bao nhiêu?"*
Bạn sẽ hỏi: *"Hãy tính lãi suất thực tế dựa trên lãi suất danh nghĩa 5% và lạm phát 2%. **Hãy suy nghĩ từng bước một**, bắt đầu bằng công thức, sau đó thay số và đưa ra kết luận cuối cùng."*

---

### **Question 06:**

Category: AIP – Implementation and Integration

A retail company is developing an AI-based recommendation system using Amazon Bedrock AgentCore. The system will leverage a large language model (LLM) fine-tuned in Amazon SageMaker AI to generate personalized product recommendations based on user browsing behavior.

The solution must ensure secure, real-time inference within the Bedrock environment, while also maintaining scalability and operational efficiency. Additionally, the company wants to monitor the inference latency and model performance metrics in real-time using Amazon CloudWatch.

Which solution ensures secure and scalable inference while minimizing operational overhead?

[ ] Store the model in Amazon S3, configure Bedrock AgentCore to load the model during runtime, and use CloudWatch to monitor API performance.

[ ] Host the LLM on an Amazon EC2 instance, connect it to Bedrock AgentCore via a REST API, and use CloudWatch Logs to monitor the API calls.

[x] **Import the fine-tuned model directly into Bedrock using the Custom Model Import, assign an AgentCore-managed execution role, and set up CloudWatch for real-time monitoring of model metrics.**

[ ] Deploy the fine-tuned LLM to SageMaker AI, configure an AWS Lambda function to invoke SageMaker endpoints, and monitor performance using CloudWatch metrics.

> Giải thích: 

Trong hệ sinh thái AWS hiện nay (đặc biệt là năm 2024-2025), AWS đã tối ưu hóa việc sử dụng các mô hình tùy chỉnh trong môi trường không máy chủ (serverless) thông qua Bedrock.

#### **1. Tại sao chọn Custom Model Import trong Amazon Bedrock?**

* **Tối ưu vận hành (Lowest Operational Overhead):** Khi bạn nhập mô hình vào Bedrock, AWS sẽ chịu trách nhiệm quản lý hạ tầng bên dưới (provisioning, scaling). Bạn không cần quản lý endpoint như trong SageMaker hay quản lý server như trong EC2.
* **An toàn & Tích hợp:** Bedrock cung cấp các Role IAM được quản lý để kiểm soát quyền truy cập một cách chặt chẽ. Việc suy luận diễn ra ngay trong môi trường Bedrock, giúp giảm độ trễ và tăng tính bảo mật.
* **Khả năng giám sát:** Bedrock tích hợp sẵn với CloudWatch để cung cấp các chỉ số như `InferenceLatency`, `InvocationCount`, giúp theo dõi hiệu suất theo thời gian thực mà không cần cấu hình phức tạp.

#### **2. Tại sao các phương án khác không phải là tối ưu nhất?**

* **Phương án 1 (S3 & Runtime load):** Amazon Bedrock không hoạt động theo cơ chế "load mô hình từ S3 tại thời điểm chạy" một cách thủ công như vậy. Việc này sẽ gây ra độ trễ cực lớn (cold start) và không phải là cách triển khai chuẩn.
* **Phương án 2 (EC2):** Việc tự quản lý EC2 tạo ra chi phí vận hành (operational overhead) rất lớn: bạn phải tự cài đặt driver, quản lý việc tự động mở rộng (Auto Scaling), cập nhật bản vá hệ điều hành... Điều này đi ngược lại yêu cầu "giảm thiểu chi phí vận hành".
* **Phương án 4 (SageMaker + Lambda):** Đây là một giải pháp khả thi nhưng **không tối ưu bằng Bedrock Custom Model Import**. Việc sử dụng Lambda làm trung gian để gọi SageMaker Endpoint tạo ra thêm một lớp kiến trúc cần quản lý, tăng độ trễ và chi phí. Nếu mục tiêu là sử dụng trong môi trường Bedrock, việc đưa trực tiếp mô hình vào Bedrock là con đường ngắn nhất và hiệu quả nhất.

#### **Ghi chú về tính năng:**

**Custom Model Import** cho phép bạn mang các mô hình được huấn luyện ở bất cứ đâu (miễn là định dạng được hỗ trợ như Hugging Face hoặc từ SageMaker) vào Amazon Bedrock để sử dụng như một mô hình "nhà trồng được" (First-party model), giúp tận dụng toàn bộ các tính năng của Bedrock Agent và Knowledge Bases.

---

### **Question 07:**

Category: AIP – Foundation Model Integration, Data Management, and Compliance

A financial services company is looking to implement Retrieval Augmented Generation (RAG) with a large language model (LLM) running on Amazon Bedrock. The company stores various documents, including `.csv` and `.docx` files, in an Amazon S3 bucket and also uses Amazon Textract to extract structured data from these financial documents. These documents contain sensitive financial data that must be incorporated into the model’s inference process to generate more accurate and relevant responses. The solution should be simple to manage, ensuring that security and regulatory requirements for handling sensitive financial data are met.

Which approach will address these objectives with the LEAST operational complexity?

[ ] Store document embeddings in Amazon SageMaker Data Wrangler, and connect it with Bedrock to perform RAG queries on the stored embeddings.

[x] **Develop a knowledge base in Bedrock, associate the S3 bucket as a data source, and use the Bedrock API to execute RAG queries.**

[ ] Set up a new model within Amazon SageMaker Pipelines and call it from Bedrock to perform RAG queries.

[ ] Use AWS Glue to transform and clean the data in the S3 bucket, store the processed data in Amazon Redshift, and query the data using Bedrock for RAG-based inference.

> Giải thích: 

Câu hỏi nhấn mạnh vào từ khóa quan trọng nhất: **"LEAST operational complexity"** (Độ phức tạp vận hành thấp nhất).

#### **1. Tại sao Knowledge Bases for Amazon Bedrock là lựa chọn tối ưu?**

* **Giải pháp Managed RAG:** AWS thiết kế tính năng **Knowledge Bases for Amazon Bedrock** như một giải pháp "tất cả trong một" để triển khai RAG mà không cần người dùng phải tự quản lý từng bộ phận riêng lẻ.
* **Tự động hóa quy trình:** Nó tự động hóa toàn bộ quy trình: lấy dữ liệu từ S3 -> chia nhỏ văn bản (chunking) -> tạo vector embeddings (sử dụng mô hình như Titan Embeddings) -> lưu trữ vào một cơ sở dữ liệu vector (như OpenSearch Serverless).
* **Ít quản trị nhất:** Bạn không cần viết mã để quản lý việc nhúng dữ liệu hay duy trì hạ tầng vector database phức tạp. Mọi thứ được cấu hình thông qua giao diện điều khiển hoặc API đơn giản.
* **Bảo mật:** Tích hợp sẵn với các tiêu chuẩn bảo mật của AWS, phù hợp cho dữ liệu tài chính nhạy cảm.

#### **2. Tại sao các phương án khác không phù hợp?**

* **Phương án 1 (SageMaker Data Wrangler):** Data Wrangler chủ yếu dùng để chuẩn bị dữ liệu (Data Preparation) cho Machine Learning truyền thống, không phải là công cụ chuyên dụng để lưu trữ embeddings hay phục vụ quy trình RAG cho LLM.
* **Phương án 3 (SageMaker Pipelines):** Đây là công cụ để quản lý vòng đời (workflow) huấn luyện và triển khai mô hình. Việc sử dụng nó cho RAG trong khi Bedrock đã có sẵn tính năng chuyên dụng là cực kỳ phức tạp và không cần thiết.
* **Phương án 4 (AWS Glue + Redshift):** Mặc dù Redshift hiện hỗ trợ Vector Search, nhưng quy trình thiết lập Glue để trích xuất dữ liệu, nạp vào Redshift rồi mới kết nối với Bedrock đòi hỏi rất nhiều công sức vận hành (Operational Overhead). Nó không đáp ứng tiêu chí "độ phức tạp thấp nhất".

#### **Kết luận**

Khi bạn thấy yêu cầu **"Làm RAG trên Bedrock với chi phí vận hành thấp nhất"**, hãy luôn nghĩ ngay đến **Knowledge Bases for Amazon Bedrock**.

---

### **Question 08:**

Category: AIP – Foundation Model Integration, Data Management, and Compliance

A company operates a fleet of wind turbines, solar panels, and weather stations in remote locations with unstable network connectivity. The company wants to improve its predictive maintenance capabilities by collecting and analyzing telemetry data from these devices, which include temperature, pressure, and rotational speed metrics.

An AI developer plans to apply machine learning (ML) models, including anomaly detection and predictive maintenance algorithms, to identify potential failures in advance. The telemetry data will be stored securely in Amazon S3 for further analysis and integration with AWS machine learning services. The developer plans to use Amazon SageMaker AI for model training and deployment, while leveraging Amazon Comprehend to analyze unstructured logs that could contain important context about device performance.

Which solution will meet the given requirements?

[ ] Set up a serverless application with Amazon API Gateway to collect telemetry data from the devices, then use AWS Lambda to process and deliver the data to S3.

[ ] Stream the telemetry data over Message Queuing Telemetry Transport (MQTT) to AWS IoT Core, forward it to an Amazon Kinesis Data Stream, and then configure an AWS Lambda function to process the data and send it to S3.

[ ] Route telemetry data over Message Queuing Telemetry Transport (MQTT) to AWS IoT Core, configure a rule in IoT Core to direct the data to an Amazon Data Firehose stream that delivers data to an S3.

[x] **Use AWS IoT Greengrass on each device to preprocess telemetry data locally, then batch upload the data to S3 using AWS SDK calls from the edge.**

> Giải thích: 

Từ khóa quan trọng nhất trong câu hỏi này là **"remote locations with unstable network connectivity"** (vùng sâu vùng xa với kết nối mạng không ổn định).

#### **1. Tại sao AWS IoT Greengrass là lựa chọn duy nhất đúng?**

* **Xử lý tại vùng biên (Edge Computing):** Trong điều kiện mạng không ổn định, bạn không thể dựa vào việc truyền dữ liệu liên tục (streaming) lên đám mây. AWS IoT Greengrass cho phép các thiết bị thu thập và tiền xử lý dữ liệu ngay tại chỗ mà không cần kết nối internet liên tục.
* **Hoạt động ngoại tuyến (Offline Operation):** Greengrass có thể lưu trữ dữ liệu cục bộ khi mất mạng và tự động đồng bộ hóa hoặc tải lên theo lô (batch upload) khi có kết nối trở lại. Điều này đảm bảo dữ liệu không bị mất.
* **Tiền xử lý dữ liệu:** Bạn có thể chạy các hàm Lambda hoặc các mô hình ML nhỏ ngay trên thiết bị Greengrass để lọc nhiễu hoặc phát hiện bất thường ngay lập tức mà không cần đợi gửi dữ liệu về trung tâm.

#### **2. Tại sao các phương án khác không phù hợp?**

* **Phương án 1 (API Gateway & Lambda):** API Gateway yêu cầu kết nối HTTPS ổn định và thường dùng cho ứng dụng web/mobile. Các thiết bị IoT ở vùng xa với mạng chập chờn sẽ gặp lỗi liên tục khi gọi API.
* **Phương án 2 & 3 (MQTT to IoT Core/Kinesis/Firehose):** Cả hai phương án này đều dựa trên việc truyền phát dữ liệu trực tiếp (real-time streaming). Nếu kết nối mạng bị ngắt, dữ liệu phát sinh trong thời gian đó có thể bị mất nếu thiết bị không có cơ chế lưu đệm (buffering) phức tạp tự xây dựng. Ngoài ra, việc duy trì kết nối MQTT liên tục trong môi trường mạng yếu là rất khó khăn.

#### **3. Tích hợp với ML Services**

Sau khi dữ liệu được Greengrass đưa vào S3:

* **Amazon SageMaker:** Sẽ lấy dữ liệu từ S3 để huấn luyện các mô hình dự báo hỏng hóc.
* **Amazon Comprehend:** Sẽ đọc các file log (văn bản) từ S3 để phân tích các yếu tố ngữ cảnh (ví dụ: "cánh quạt kêu to", "nhiệt độ tăng đột ngột").

#### **Gợi ý học tập:**

Khi đề bài nhắc đến **"unstable network"**, **"remote locations"**, hoặc **"low latency requirement at the edge"**, đáp án thường sẽ liên quan đến **AWS IoT Greengrass** hoặc **AWS Snowball Edge**.

---

### **Question 09:**

Category: AIP – Operational Efficiency and Optimization for Generative AI Applications

A finance company has deployed a generative AI model using Amazon Bedrock to power its customer support chatbot. The model is designed to handle various customer inquiries and deliver real-time responses. However, the company has encountered issues where the model intermittently fails to process complex queries and crashes when handling a high volume of requests. The investigation reveals that the model’s performance suffers significantly when multiple complex queries are processed simultaneously, causing memory-related errors.

The company needs to enhance its Bedrock deployment to better manage higher memory demands during peak usage, ensuring that the model can handle more complex queries without performance degradation or crashes.

Which solution will resolve this issue and improve the model’s stability?

[ ] Expand the memory capacity of the Bedrock AgentCore agent.

[x] **Increase the number of Bedrock AgentCore agent instances.**

[ ] Add more processing power to the Bedrock AgentCore agent.

[ ] Upgrade the storage capacity of the Bedrock AgentCore agent.

> Giải thích: 

Vấn đề cốt lõi ở đây là khả năng chịu tải của hệ thống khi gặp lượng truy vấn lớn (high volume) và các truy vấn phức tạp gây ngốn tài nguyên (memory-intensive).

#### **1. Tại sao tăng số lượng Instances là giải pháp?**

* **Cơ chế Horizontal Scaling (Mở rộng theo chiều ngang):** Trong kiến trúc đám mây và đặc biệt là với các dịch vụ như Bedrock Agents, khi một phiên bản (instance) duy nhất bị quá tải bộ nhớ do xử lý quá nhiều tác vụ đồng thời, cách hiệu quả nhất để phân phối tải là tăng số lượng các "worker" (các instance).
* **Xử lý song song:** Việc tăng số lượng instances cho phép hệ thống phân phối các truy vấn đến nhiều agent khác nhau. Mỗi agent sẽ có một không gian bộ nhớ độc lập để xử lý truy vấn của mình, từ đó tránh được việc một instance bị cạn kiệt bộ nhớ và treo toàn bộ hệ thống.
* **Độ ổn định:** Nếu một instance gặp sự cố, các instance khác vẫn có thể tiếp tục phục vụ khách hàng, giúp tăng tính sẵn sàng của chatbot.

#### **2. Tại sao các phương án khác không phù hợp?**

* **Phương án 1 & 3 (Mở rộng bộ nhớ/xử lý của 1 Agent):** Đây là phương pháp **Vertical Scaling** (Mở rộng theo chiều dọc). Thông thường, trong các dịch vụ Managed như Bedrock, bạn không có quyền can thiệp sâu vào việc "nâng cấp thanh RAM" hay "thêm CPU" cho một agent cụ thể. AWS quản lý các giới hạn tài nguyên này. Cách thức chuẩn để xử lý tải trong AWS là Scale Out (tăng số lượng) thay vì Scale Up (tăng sức mạnh một cái).
* **Phương án 4 (Nâng cấp dung lượng lưu trữ):** Lỗi ở đây là lỗi bộ nhớ (RAM) khi đang xử lý (runtime), không phải lỗi thiếu không gian lưu trữ (Disk/Storage). Việc tăng dung lượng lưu trữ sẽ không giúp giải quyết vấn đề mô hình bị treo do xử lý truy vấn phức tạp.

#### **Mẹo nhỏ khi đi thi:**

Khi gặp các câu hỏi liên quan đến việc hệ thống bị treo hoặc chậm do **"high volume of requests"** hoặc **"simultaneously processing"**, đáp án ưu tiên hàng đầu luôn là các giải pháp liên quan đến **Scaling** (tăng số lượng instance, sử dụng Auto Scaling, hoặc phân tán tải).

---

### **Question 10:**

Category: AIP – Testing, Validation, and Troubleshooting

A generative AI developer manages a production generative model endpoint in Amazon SageMaker AI, with Model Monitor and SageMaker Clarify enabled. The developer recently deployed a newly retrained model to replace an older version, and Model Monitor has been configured to track data quality and distribution drift. SageMaker Clarify continues to provide explainability and bias metrics for the same endpoint.

Despite the model being updated and serving new inference traffic based on a more current dataset, the monitoring job continues to show violations. The developer has confirmed that the new dataset is statistically representative of the current traffic, but the violations still persist.

Which corrective action should the developer take to resolve the persistent violations in Model Monitor?

[ ] Remove the current model endpoint and recreate it with the same configuration as the original one.

[ ] Trigger a new SageMaker model training job with the existing baseline dataset to refresh the model's performance.

[x] **Perform a baseline job on the new training data and configure Model Monitor to reference the new baseline statistics.**

[ ] Adjust the SageMaker Model Monitor threshold settings to reduce the frequency of violations.

> Giải thích: 

Vấn đề cốt lõi ở đây nằm ở cơ chế hoạt động của **Amazon SageMaker Model Monitor**.

#### **1. Cơ chế của Model Monitor là gì?**

Model Monitor hoạt động bằng cách so sánh dữ liệu thực tế thu thập được từ endpoint (Inference data) với một "tiêu chuẩn" được gọi là **Baseline**.

* **Baseline** bao gồm các số liệu thống kê (statistics) và các ràng buộc (constraints) được trích xuất từ bộ dữ liệu dùng để huấn luyện mô hình ban đầu.
* Khi có sự khác biệt đáng kể giữa dữ liệu thực tế và Baseline, hệ thống sẽ báo cáo là **Drift (Độ lệch)** hay **Violation (Vi phạm)**.

#### **2. Tại sao lỗi vi phạm vẫn xuất hiện?**

Khi nhà phát triển huấn luyện lại mô hình trên một bộ dữ liệu mới (current dataset), các đặc tính thống kê của dữ liệu đó đã thay đổi. Tuy nhiên, Model Monitor vẫn đang sử dụng **Baseline của mô hình cũ** để so sánh.

Dù dữ liệu thực tế (Inference traffic) hoàn toàn khớp với dữ liệu huấn luyện mới, nó vẫn khác với dữ liệu huấn luyện cũ. 

Do đó, Model Monitor hiểu lầm rằng dữ liệu đang bị lệch và báo cáo vi phạm.

#### **3. Tại sao các phương án khác sai?**

* **Phương án 1 (Xóa và tạo lại endpoint):** Việc này không thay đổi cấu hình của Model Monitor. Monitor vẫn sẽ tìm đến file baseline cũ để so sánh.
* **Phương án 2 (Huấn luyện lại với baseline cũ):** Điều này vô nghĩa vì bộ dữ liệu cũ đã lỗi thời và không còn phản ánh đúng thực tế lưu lượng truy cập hiện tại.
* **Phương án 4 (Điều chỉnh ngưỡng - Threshold):** Đây chỉ là cách "che giấu" triệu chứng. Nếu bạn nới lỏng ngưỡng, bạn có thể bỏ lỡ những lần dữ liệu bị lệch thật sự trong tương lai, gây rủi ro cho độ chính xác của mô hình.

#### **Kết luận**

Bất cứ khi nào bạn thay đổi dữ liệu huấn luyện (Retrain model), bạn **BẮT BUỘC** phải tạo lại một Baseline mới từ dữ liệu đó để Model Monitor có cái nhìn đúng đắn nhất về "trạng thái bình thường" mới của dữ liệu.

---

### **Question 11:**

Category: AIP – Foundation Model Integration, Data Management, and Compliance

A global e-commerce company is building a multilingual product description generator using Amazon Titan Text on Amazon Bedrock. The team plans to enhance the model’s domain understanding by fine-tuning it with proprietary product data in Amazon SageMaker AI before deploying it to production through Bedrock’s managed API.

For compliance reasons, all model artifacts must be encrypted using AWS KMS, and API calls to the Bedrock endpoint must be logged and auditable. The company also wants to continuously monitor model latency and throughput across different regions to ensure consistent performance.

Which solution provides a secure, compliant, and observable architecture for deploying and monitoring the Titan-based generative AI model?

[x] **Fine-tune the Titan model in SageMaker AI using training data stored in Amazon S3 with KMS encryption, deploy the model through Bedrock with a customer-managed KMS key, enable AWS CloudTrail for API auditing, and use Amazon CloudWatch metrics for regional performance monitoring.**

[ ] Train the Titan model entirely within SageMaker AI, export it to an Amazon EC2-based inference server with Amazon EBS encryption, and use AWS Config for compliance monitoring.

[ ] Fine-tune the Titan model in SageMaker AI using data encrypted with SSE-S3, deploy it through an API Gateway endpoint in front of Bedrock, and use Amazon Macie to monitor for data exfiltration events.

[ ] Deploy the Titan model directly on Bedrock without fine-tuning, use IAM roles to restrict access, and rely on AWS CloudTrail logs to monitor API calls.

> Giải thích

#### 1. Giải thích đáp án đúng

Phương án này đáp ứng đầy đủ các tiêu chuẩn về bảo mật, tính tuân thủ và khả năng giám sát mà hệ thống yêu cầu:

* **Mã hóa dữ liệu & Artifacts:** Việc sử dụng **AWS KMS** (Key Management Service) để mã hóa dữ liệu huấn luyện trên S3 và mã hóa mô hình (artifacts) khi triển khai trên Bedrock đảm bảo rằng dữ liệu nhạy cảm của doanh nghiệp được bảo vệ ở trạng thái nghỉ (at rest). Việc dùng **Customer-managed key** cho phép doanh nghiệp có quyền kiểm soát tối cao đối với việc xoay vòng khóa và truy cập.
* **Kiểm toán (Auditing):** **AWS CloudTrail** là dịch vụ ghi lại mọi hoạt động API trong tài khoản AWS. Nó cung cấp bằng chứng xác thực về việc ai đã gọi model, gọi lúc nào, đáp ứng yêu cầu "auditable" của đề bài.
* **Khả năng quan sát (Observability):** **Amazon CloudWatch** cung cấp các chỉ số (metrics) mặc định cho Bedrock như `InvocationLatency` (độ trễ) và `Invocations` (số lượng yêu cầu/thông lượng). Điều này giúp theo dõi hiệu suất mô hình trên nhiều vùng (regions) khác nhau một cách trực quan.
* **Quy trình kết hợp:** Việc sử dụng **Amazon SageMaker AI** để fine-tune (tinh chỉnh) mô hình rồi đẩy sang **Amazon Bedrock** để chạy inference qua API là một kiến trúc chuẩn cho các doanh nghiệp muốn tùy biến sâu nhưng vẫn muốn sự tiện lợi của serverless API.


#### 2. Tại sao các phương án còn lại sai?

* **Phương án 2:** Đề cập đến việc xuất mô hình sang máy chủ **Amazon EC2**. Điều này đi ngược lại yêu cầu "triển khai qua Bedrock’s managed API". EC2 yêu cầu bạn phải tự quản lý hạ tầng (tự cài thư viện, quản lý patch), làm tăng độ phức tạp vận hành. **AWS Config** dùng để kiểm tra cấu hình tài nguyên chứ không phải công cụ chính để ghi log API hay giám sát hiệu suất model.
* **Phương án 3:** Sử dụng **SSE-S3** (mã hóa do S3 quản lý) thường không đủ linh hoạt và an toàn bằng KMS cho các yêu cầu tuân thủ khắt khe của doanh nghiệp toàn cầu. **Amazon Macie** là dịch vụ dùng để quét và phát hiện dữ liệu nhạy cảm (như số thẻ tín dụng) bị lộ trong S3, nó không có chức năng giám sát độ trễ hay hiệu suất mô hình AI.
* **Phương án 4:** Đề xuất triển khai **trực tiếp không fine-tuning**. Điều này vi phạm kế hoạch của nhóm là "enhance the model’s domain understanding by fine-tuning". Dù bảo mật qua IAM là đúng, nhưng nó thiếu các yếu tố về mã hóa artifacts bằng KMS và giám sát hiệu suất qua CloudWatch.

#### 3. Notes: Các dịch vụ và thay đổi mới (Cập nhật 2025)

Dưới đây là một số lưu ý quan trọng về các dịch vụ Generative AI trên AWS mà bạn nên biết:

* **Amazon SageMaker AI:** Đây là tên gọi mới (đổi từ Amazon SageMaker) kể từ cuối năm 2024, nhấn mạnh vào việc đây là nền tảng toàn diện cho cả Machine Learning truyền thống và AI tạo sinh.
* **Amazon Bedrock Custom Models:** Hiện nay Bedrock đã cho phép bạn fine-tune mô hình trực tiếp trong giao diện Bedrock hoặc nhập (import) các mô hình đã được huấn luyện từ bên ngoài (như từ SageMaker) vào để sử dụng dưới dạng Serverless API.
* **Guardrails for Amazon Bedrock:** Đây là một dịch vụ "mới nổi" rất quan trọng giúp thiết lập các bộ lọc nội dung (về bạo lực, ngôn từ thù ghét hoặc lộ thông tin PII) giữa người dùng và mô hình AI.
* **Provisioned Throughput:** Khi triển khai model đã fine-tune trên Bedrock, bạn thường phải sử dụng "Provisioned Throughput" (Thông lượng dự phòng) để đảm bảo độ trễ ổn định cho các ứng dụng sản xuất (production).

---

### **Question 12:**

Category: AIP – Operational Efficiency and Optimization for Generative AI Applications

A research division is continuously retraining a Generative Forecasting Foundation model pipeline built on Amazon SageMaker AI, where base embedding creation happens in SageMaker Processing, feature encoding is performed inside the SageMaker Feature Store ingestion pipeline, and final supervised regression is trained in SageMaker Training. Last quarter, this same regression model was tuned using SageMaker AI automatic model tuning (AMT), and the results produced a known stable set of high-performing hyperparameters that were saved.

A new quarter dataset is now being ingested, and it is larger, has a different distribution, and must be used for the next model update. However, the engineering pillar requires that a new hyperparameter tuning job cannot start from zero because the compute budget is limited. There is also a requirement that if the model’s validation loss does not improve anymore, the tuning must automatically stop without manual intervention.

Which hyperparameter tuning job configuration should be used?

[ ] Run a new hyperparameter tuning job using SageMaker AMT Hyperband strategy, allowing the system to aggressively eliminate poorly performing training jobs early while exploring the new quarterly dataset with wider search ranges. Depend on Hyperband efficiency itself rather than importing prior warm start knowledge from the previously tuned model.

[x] **Start a warm start hyperparameter tuning job using the TRANSFER_LEARNING warm start type to import the previously saved tuning job results. Enable AMT Early Stopping to automatically terminate exploration as soon as validation loss stops improving when training with the new quarterly dataset.**

[ ] Configure a warm start hyperparameter tuning job with the IDENTICAL_DATA_AND_ALGORITHM warm start mode and enable AMT Early Stopping to automatically terminate exploration as soon as validation loss stops improving.

[ ] Execute a hyperparameter tuning job with a newly expanded search space using the same algorithm container. Rely on Bayesian Optimization inside SageMaker AI AMT to naturally rediscover higher-performing parameter combinations using the new quarterly dataset, which will naturally guide convergence efficiency without importing prior tuning job results.

> Giải thích: 

#### 1. Giải thích đáp án đúng

Phương án này là giải pháp tối ưu nhất để cân bằng giữa hiệu suất mô hình và giới hạn ngân sách (compute budget):

* **Warm Start với `TRANSFER_LEARNING`:** Đây là tính năng cho phép Amazon SageMaker AI sử dụng kiến thức (kết quả) từ các lần chạy Hyperparameter Tuning (HPO) trước đó. Thay vì bắt đầu dò tìm các tham số một cách ngẫu nhiên, mô hình sẽ tận dụng các tổ hợp tham số "tốt" đã biết để làm điểm tựa. Loại `TRANSFER_LEARNING` được thiết kế riêng cho trường hợp thuật toán giống nhau nhưng **dữ liệu đã thay đổi** (dataset mới của quý này).
* **AMT Early Stopping:** Đây là tính năng giải quyết trực tiếp yêu cầu "tự động dừng nếu không cải thiện". SageMaker AI sẽ giám sát các training jobs; nếu một job không cho thấy triển vọng cải thiện so với các job trước đó dựa trên objective metric (validation loss), nó sẽ bị khai tử sớm để tiết kiệm tài nguyên tính toán.
* **Tiết kiệm chi phí:** Kết hợp hai tính năng này giúp giảm tổng số giờ GPU/CPU cần thiết, đáp ứng yêu cầu khắt khe về ngân sách của bộ phận kỹ thuật.

#### 2. Tại sao các phương án còn lại sai?

* **Phương án 1 (Hyperband):** Mặc dù Hyperband rất hiệu quả trong việc loại bỏ các job kém chất lượng, nhưng phương án này lại chọn **bỏ qua (ignore)** các kết quả từ lần tuning trước. Điều này lãng phí những thông tin giá trị mà nhóm đã có, dẫn đến việc phải thăm dò lại từ đầu trên một không gian tham số rộng, không tối ưu cho ngân sách hạn hẹp.
* **Phương án 3 (IDENTICAL_DATA_AND_ALGORITHM):** Kiểu warm start này chỉ nên dùng khi bạn chạy thêm các job trên **cùng một bộ dữ liệu chính xác** (ví dụ: muốn chạy thêm 10 jobs nữa để tìm kết quả tốt hơn trên tập data cũ). Đề bài nêu rõ đây là "new quarterly dataset" với "different distribution", vì vậy dùng mode này sẽ khiến bộ điều phối (tuner) bị nhầm lẫn và không đạt hiệu quả cao.
* **Phương án 4 (Bayesian Optimization từ đầu):** Việc tin tưởng vào thuật toán Bayesian tự tìm lại các tham số tốt mà không kế thừa kết quả cũ sẽ tiêu tốn rất nhiều tài nguyên ở giai đoạn "thăm dò" (exploration). Điều này vi phạm yêu cầu "cannot start from zero because the compute budget is limited".

#### 3. Notes: Các dịch vụ và kỹ thuật trong SageMaker AI

Dưới đây là các khái niệm quan trọng bạn cần lưu ý cho kỳ thi hoặc áp dụng thực tế:

* **SageMaker AI Automatic Model Tuning (AMT):** Tự động tìm kiếm các bộ tham số (learning rate, batch size, v.v.) tốt nhất. Hiện nay nó hỗ trợ nhiều chiến lược như Bayesian, Hyperband, và Random Search.
* **Warm Start Types:**
* `TRANSFER_LEARNING`: Dùng khi dữ liệu thay đổi hoặc thuật toán có sự điều chỉnh nhỏ.
* `IDENTICAL_DATA_AND_ALGORITHM`: Dùng khi muốn tiếp tục một job HPO đã dừng hoặc mở rộng thêm số lượng thử nghiệm trên cùng dữ liệu.


* **SageMaker Feature Store:** Là kho lưu trữ trung tâm để quản lý, chia sẻ và phục vụ các đặc trưng (features) cho ML. Trong câu hỏi này, nó đóng vai trò là nơi thực hiện "feature encoding" trước khi đưa vào huấn luyện.
* **SageMaker Processing:** Một dịch vụ chạy các scripts xử lý dữ liệu (như Python, Spark) một cách độc lập, rất phù hợp cho bước tạo "base embedding" quy mô lớn trước khi train.
* **Early Stopping Logic:** SageMaker sử dụng thuật toán dựa trên sự hội tụ của đường cong học tập (learning curve) để dự đoán liệu một job có khả năng vượt qua kết quả tốt nhất hiện tại hay không. Nếu không, nó sẽ dừng job đó ngay lập tức.

---

### **Question 13:**

Category: AIP – Implementation and Integration

A global broadcasting enterprise, TD Studios, operates an intelligent content-management platform built on Amazon SageMaker and Amazon Bedrock AgentCore. The platform uses SageMaker to coordinate large-scale data preparation and model-training workflows, enabling automated feature extraction and experimentation across thousands of multimedia assets. At the same time, Bedrock AgentCore serves as the orchestration layer that connects multiple AWS AI services, allowing agents to invoke specialized models and enrich content with generative summaries and metadata for semantic search.

The enterprise recently expanded its media repository with several petabytes of newly acquired, unlabeled images, videos, podcasts, and written transcripts stored in Amazon S3. The Research and Curation division, composed primarily of content analysts with limited machine learning expertise, must now enable automatic tagging and indexing of this data to support entity-based search and cross-referencing. The enterprise wants to extend its existing Bedrock AgentCore workflow to handle this new data without performing any model training, manual labeling, or infrastructure provisioning. The solution must deliver accurate tagging across multiple modalities and achieve rapid implementation suitable for a non-specialized research team.

Which solution provides the fastest and most effective approach to automatically index and categorize the multimedia assets?

[x] **Leverage Amazon Comprehend, Amazon Transcribe, and Amazon Rekognition to categorize and tag multimedia content automatically**

[ ] Convert audio to text using Amazon Transcribe, then use SageMaker’s Neural Topic Model (NTM) and Object Detection to assign category tags across the dataset.

[ ] Configure Amazon Polly, Amazon Translate, and Amazon Lex within a SageMaker pipeline to convert audio and text into multilingual transcripts and conversational responses for cataloging media.

[ ] Use AWS Batch to periodically run containerized Python scripts that perform speech-to-text conversion, image feature extraction, and topic modeling across the media files.

> Giải thích: 

#### 1. Giải thích đáp án đúng

Đây là giải pháp tối ưu nhất cho TD Studios vì nó đáp ứng hoàn hảo tất cả các ràng buộc về kỹ thuật và nhân sự:

* **Đa phương thức (Multimodal):** Hệ thống cần xử lý hình ảnh, video, podcast (âm thanh) và văn bản. Bộ ba dịch vụ này bao phủ toàn bộ:
* **Amazon Rekognition:** Tự động phân tích hình ảnh và video để gắn thẻ (tagging) đối tượng, cảnh quay và nhận diện thực thể.
* **Amazon Transcribe:** Chuyển đổi âm thanh (podcasts, video audio) thành văn bản (Speech-to-Text).
* **Amazon Comprehend:** Phân tích các bản ghi văn bản và transcript để trích xuất thực thể (entities), từ khóa (keyphrases) và chủ đề (topics).


* **Không cần chuyên môn ML sâu:** Đây là các dịch vụ **AI được đào tạo sẵn (Pre-trained AI Services)**. Người dùng chỉ cần gọi API mà không cần biết cách xây dựng hay huấn luyện mô hình (No model training).
* **Không quản lý hạ tầng:** Các dịch vụ này hoàn toàn là serverless, phù hợp với yêu cầu "no infrastructure provisioning".
* **Tích hợp linh hoạt:** Kết quả đầu ra từ các dịch vụ này có thể dễ dàng được đẩy vào **Amazon Bedrock AgentCore** để làm giàu dữ liệu (enrichment) cho việc tìm kiếm ngữ nghĩa.

#### 2. Tại sao các phương án còn lại sai?

* **Phương án 2 (SageMaker NTM & Object Detection):** Việc sử dụng các thuật toán như Neural Topic Model (NTM) trên SageMaker đòi hỏi nhóm nghiên cứu phải thực hiện các bước chuẩn bị dữ liệu phức tạp, cấu hình máy chủ và có kiến thức nhất định về ML để tinh chỉnh mô hình. Điều này vi phạm yêu cầu "limited ML expertise".
* **Phương án 3 (Polly, Translate, Lex):** Các dịch vụ này không phục vụ mục đích lập chỉ mục (indexing).
* **Amazon Polly:** Chuyển văn bản thành giọng nói (ngược với yêu cầu).
* **Amazon Lex:** Dùng để xây dựng chatbot.
* **Amazon Translate:** Chỉ dùng để dịch ngôn ngữ, không giúp trích xuất thực thể để gắn thẻ.


* **Phương án 4 (AWS Batch & Python Scripts):** Việc viết script Python tùy chỉnh và chạy trên AWS Batch đòi hỏi kỹ năng lập trình và quản lý container/hạ tầng. Đây là cách tiếp cận "nặng" về kỹ thuật, không phải là cách "nhanh nhất và hiệu quả nhất" cho một đội ngũ chuyên viên phân tích nội dung.

#### 3. Notes: Các dịch vụ AI phục vụ phân tích nội dung

Dưới đây là các lưu ý quan trọng về các dịch vụ này trong hệ sinh thái AWS:

* **Amazon Bedrock AgentCore:** Đây là một thành phần quan trọng trong việc điều phối (orchestration). Nó cho phép các Agent tự động gọi các "Action Groups" (có thể là Lambda functions gọi đến Rekognition hoặc Transcribe) để hoàn thành các tác vụ phức tạp một cách tự động.
* **Amazon Rekognition Video:** Điểm khác biệt lớn là dịch vụ này có thể phân tích video theo thời gian thực hoặc theo lô (batch), nhận diện được cả hành động và sự thay đổi cảnh quay để đánh dấu metadata chính xác.
* **Amazon Comprehend Entity Recognition:** Dịch vụ này có khả năng tự động phân loại thực thể thành các nhóm như "Người", "Địa điểm", "Tổ chức", "Ngày tháng". Điều này cực kỳ hữu ích cho việc "entity-based search" mà doanh nghiệp yêu cầu.
* **Media2Cloud:** Một giải pháp (AWS Solution) phổ biến thường kết hợp các dịch vụ trên để giúp các công ty truyền thông nhanh chóng đưa dữ liệu từ on-premise lên S3 và tự động chạy quy trình phân tích AI này.

---

### **Question 14:**

Category: AIP – Operational Efficiency and Optimization for Generative AI Applications

A company is building an automated model-evaluation workflow for a new LLM-powered document-classification system. The GenAI developer has prepared a large custom prompt dataset in JSONL format and stored it in Amazon S3. The dataset contains more than 15,000 prompts that must be used to evaluate multiple foundation models available in Amazon Bedrock.

The developer uses Amazon SageMaker Pipelines to orchestrate preprocessing and a Bedrock evaluation job to score the responses. During execution, Bedrock returns repeated errors indicating that the evaluation request exceeds the maximum number of prompts allowed per job. The developer must comply with this service limit while ensuring the entire dataset is evaluated without manually restructuring the pipeline.

Which is the most effective way to resolve this problem?

[ ] Combine all prompts into a single JSONL file, compress it, and configure the evaluation job in Bedrock to process it as one batch.

[x] **Split the dataset into multiple smaller JSONL files of up to 1,000 prompts each, store them in S3, and run separate evaluation jobs in Bedrock orchestrated by SageMaker Pipelines.**

[ ] Enable S3 bucket versioning and set proper access permissions for SageMaker and Bedrock.

[ ] Convert the JSONL dataset into Apache Parquet format, upload it to S3, and request a quota increase so a single evaluation job can process all prompts.

> Giải thích: 

#### 1. Giải thích đáp án đúng

Giải pháp này giải quyết trực tiếp các giới hạn kỹ thuật của dịch vụ trong khi vẫn duy trì tính tự động hóa:

* **Vượt qua giới hạn Service Quota:** Hiện tại, **Amazon Bedrock Model Evaluation** có các giới hạn cứng (hard limits) về số lượng prompt tối đa trong một job đơn lẻ (thường giới hạn ở mức 1.000 prompts đối với một số loại đánh giá tự động). Với tập dữ liệu 15.000 prompts, việc chia nhỏ là bắt buộc để không vi phạm quota của AWS.
* **Tự động hóa bằng SageMaker Pipelines:** Vì developer đã sử dụng SageMaker Pipelines, họ có thể thêm một bước (step) xử lý dữ liệu đơn giản bằng **SageMaker Processing** để tự động chia nhỏ file JSONL lớn thành các file nhỏ hơn 1.000 dòng.
* **Cấu trúc Pipeline:** Sau bước chia nhỏ, pipeline có thể sử dụng vòng lặp hoặc các bước song song để khởi tạo nhiều Bedrock Evaluation Jobs. Cách tiếp cận này đảm bảo toàn bộ dữ liệu được đánh giá mà không cần sự can thiệp thủ công của con người (manual restructuring).

#### 2. Tại sao các phương án còn lại sai?

* **Phương án 1 (Gộp file & Nén):** Việc gộp 15.000 prompts vào một file duy nhất và nén lại không giúp giải quyết vấn đề giới hạn số lượng bản ghi (records) mà Bedrock có thể xử lý trong một job. Bedrock vẫn sẽ phải giải nén và đếm số lượng prompt, dẫn đến lỗi vượt giới hạn tương tự.
* **Phương án 3 (Versioning & Permissions):** Lỗi mà developer gặp phải là "request exceeds the maximum number of prompts", đây là lỗi về **Quota/Limit**, không phải lỗi về quyền truy cập (Permissions) hay phiên bản dữ liệu (Versioning). Phương án này hoàn toàn lạc đề.
* **Phương án 4 (Parquet format & Quota increase):** * Bedrock Model Evaluation yêu cầu định dạng đầu vào cụ thể (thường là JSONL), việc chuyển sang Parquet có thể khiến job không đọc được dữ liệu.
* Yêu cầu tăng quota (Quota increase) không phải lúc nào cũng được chấp nhận ngay lập tức hoặc có thể tăng lên mức 15.000 (gấp 15 lần giới hạn mặc định) trong một số trường hợp giới hạn cứng của hệ thống.

#### 3. Notes: Lưu ý về Model Evaluation trên Amazon Bedrock

Dưới đây là các điểm quan trọng khi triển khai đánh giá mô hình Generative AI trên AWS:

* **Các loại hình Evaluation:** Bedrock hỗ trợ hai loại đánh giá chính:
1. **Automatic Evaluation:** Sử dụng các thuật toán chuẩn (như ROUGE, BLEU) hoặc các LLM khác để chấm điểm dựa trên độ chính xác, tính độc hại, v.v.
2. **Human Evaluation:** Gửi kết quả cho con người (đội ngũ nội bộ hoặc Amazon Mechanical Turk) để đánh giá các yếu tố cảm tính như phong cách, độ thân thiện.


* **JSONL Format:** Cấu trúc file JSONL cho Bedrock cực kỳ quan trọng. Mỗi dòng phải là một đối tượng JSON hợp lệ chứa `prompt` và (tùy chọn) `referenceResponse` để đối chiếu.
* **SageMaker Pipelines Integration:** Để tích hợp Bedrock vào SageMaker Pipelines, bạn thường sử dụng một **LambdaStep** hoặc **PythonSdkStep** để gọi API `CreateEvaluationJob` của Bedrock.
* **Monitoring:** Kết quả của các job đánh giá thường được xuất ra Amazon S3 dưới dạng các tệp JSON chi tiết, cho phép bạn phân tích sâu về hiệu suất của từng model Foundation (như Claude 3.5 Sonnet vs Titan Text).

---

### **Question 15:**

Category: AIP – Foundation Model Integration, Data Management, and Compliance

A Generative AI engineering team at TD GenAILabs is developing a large-scale natural language processing system designed to process multilingual customer feedback from global data sources. The workflow uses Amazon SageMaker Data Wrangler to automate text cleaning, normalization, and tokenization, ensuring that all transformations are applied consistently before model training. The workflow also integrates Amazon Titan Text Embeddings to generate high-quality vector representations of text, enabling deeper semantic understanding and improving similarity analysis in later stages of the pipeline.

The team plans to train a Word2Vec-style model on SageMaker AI to generate embeddings that will support semantic similarity analysis and topic clustering. The dataset contains over one million sentences with inconsistent casing, mixed encodings, and minor typographical errors such as: “An Apple a DAY Keeps the doctor Away.” The preprocessing process must therefore ensure consistent sanitization, reproducibility, and embedding quality across multiple training cycles to maintain model accuracy and stability.

Which of the following operations should be implemented in the preprocessing phase to correctly sanitize and prepare the dataset for embedding and downstream predictions? (Select THREE.)

[ ] Apply part-of-speech tagging to identify grammatical elements and retain only the verbs and nouns.

[x] **Normalize the text by converting every word in the sentence to lowercase.**

[ ] Replace every word with its corresponding synonym using a lexical database before tokenization.

[x] **Segment the sentence into individual word units through tokenization.**

[ ] Convert all tokens into fixed-length character n-grams before Word2Vec training to capture subword features.

[x] **Exclude common non-informative words from the dataset using an English stop-word dictionary.**

> Giải thích: 

#### 1. Giải thích các đáp án đúng

Để xây dựng một mô hình Word2Vec (hoặc các mô hình nhúng tương tự) hiệu quả, quá trình tiền xử lý phải tập trung vào việc giảm nhiễu và làm sạch dữ liệu:

* **Normalize to lowercase (Chuyển về chữ thường):** Đây là bước quan trọng để giải quyết vấn đề "inconsistent casing" (như ví dụ: "Apple", "DAY"). Trong Word2Vec, "Apple" và "apple" sẽ được coi là hai vector khác nhau nếu không đưa về cùng một định dạng. Chuyển về chữ thường giúp gộp các từ này lại, tăng mật độ dữ liệu cho mỗi từ và cải thiện chất lượng nhúng.
* **Tokenization (Phân tách từ):** Đây là bước nền tảng. Word2Vec hoạt động dựa trên mối quan hệ giữa các từ đơn lẻ trong một cửa sổ ngữ cảnh (context window). Việc chia câu thành các "token" (đơn vị từ) là bắt buộc để thuật toán có thể học được các vector biểu diễn.
* **Exclude stop-words (Loại bỏ từ dừng):** Các từ như "an", "a", "the", "is" xuất hiện với tần suất rất cao nhưng mang lại rất ít giá trị về mặt ngữ nghĩa (semantic understanding). Loại bỏ chúng giúp mô hình tập trung tài nguyên tính toán vào các từ mang ý nghĩa thực sự, từ đó cải thiện độ chính xác của phân tích cụm (clustering) và tương đồng ngữ nghĩa.

#### 2. Tại sao các phương án còn lại sai?

* **Apply POS tagging (Chỉ giữ lại động từ và danh từ):** Việc chỉ giữ lại động từ và danh từ sẽ làm mất đi ngữ cảnh quan trọng. Word2Vec cần các từ xung quanh (bao gồm cả tính từ, trạng từ) để hiểu ngữ nghĩa của một từ. Việc lọc quá gắt gao này sẽ phá hủy cấu trúc của câu.
* **Replace with synonyms (Thay thế bằng từ đồng nghĩa):** Điều này làm thay đổi bản chất của dữ liệu gốc. Word2Vec được thiết kế để tự học quan hệ đồng nghĩa thông qua vị trí của chúng trong không gian vector. Nếu bạn thay thế chúng trước, bạn đang làm mất đi khả năng học hỏi các sắc thái ngôn ngữ của mô hình.
* **Character n-grams:** Mặc dù kỹ thuật này (như trong FastText) rất tốt để xử lý các từ hiếm hoặc lỗi chính tả, nhưng đề bài yêu cầu cụ thể là huấn luyện một mô hình **Word2Vec-style**. Word2Vec truyền thống hoạt động dựa trên các token từ hoàn chỉnh, không phải character n-grams.


#### 3. Notes: Các dịch vụ và kỹ thuật trong SageMaker AI

* **Amazon SageMaker Data Wrangler:** Đây là một công cụ cực kỳ mạnh mẽ cho phép bạn làm sạch và chuẩn bị dữ liệu (Data Prep) bằng giao diện trực quan hoặc code. Nó có sẵn các "Transform" như *Lowercase*, *Strip whitespace*, và *Tokenize* mà không cần viết nhiều code.
* **Amazon Titan Text Embeddings:** Đây là một mô hình nhúng (Embedding model) được cung cấp dưới dạng dịch vụ trên Bedrock. Điểm khác biệt là nó đã được huấn luyện sẵn (pre-trained) trên quy mô khổng lồ. Tuy nhiên, trong đề bài này, đội ngũ chọn tự huấn luyện Word2Vec trên SageMaker AI để kiểm soát hoàn toàn bộ dữ liệu đặc thù.
* **Reproducibility (Tính tái lặp):** Khi sử dụng SageMaker Data Wrangler, mọi bước biến đổi dữ liệu được lưu lại thành một quy trình (flow). Điều này đảm bảo rằng khi có tập dữ liệu mới của quý tiếp theo, bạn chỉ cần chạy lại flow đó để có kết quả tiền xử lý giống hệt, giúp duy trì tính ổn định cho mô hình.
* **Handling Mixed Encodings:** Trong thực tế, trước khi thực hiện 3 bước trên, bạn thường cần một bước chuẩn hóa Unicode để đảm bảo các ký tự đặc biệt hoặc mã hóa khác nhau (UTF-8, Latin-1) được đưa về cùng một chuẩn.

---

### **Question 16:**

Category: AIP – AI Safety, Security, and Governance

A large retail company is looking to leverage Amazon SageMaker AI notebooks and Amazon Comprehend to process and analyze customer feedback data. The company uses Amazon S3 buckets to store large datasets of customer reviews, and the data needs to be accessed securely for machine learning analysis.

Given the sensitivity of the data, the company mandates that all resources must remain within a secure Amazon Virtual Private Cloud (VPC). Additionally, all communication must occur over the AWS network.

Which solution will satisfy the given requirements?

[ ] Deploy the SageMaker AI notebook in a private subnet with a route to an internet gateway and ensure that all data requests to S3 are routed through an external proxy.

[ ] Deploy the SageMaker AI notebook in a private subnet within a VPC and use a VPC Peering connection with another VPC where S3 is accessible.

[ ] Deploy the SageMaker AI notebook in a private subnet, use a NAT gateway to provide outbound internet access for S3, and restrict access to only specific S3 buckets.

[x] **Deploy the SageMaker AI notebook in a private subnet within a VPC and ensure that the VPC has private endpoints for both SageMaker AI and S3.**

> Giải thích: 

#### 1. Giải thích đáp án đúng

Giải pháp này đáp ứng hoàn hảo hai yêu cầu cốt lõi của đề bài: **Tài nguyên nằm trong VPC bảo mật** và **Mọi giao tiếp phải diễn ra trong mạng nội bộ AWS (AWS network)**.

* **VPC Private Subnet:** Việc đặt SageMaker Notebook trong subnet riêng tư đảm bảo nó không có địa chỉ IP công cộng và không thể bị truy cập trực tiếp từ Internet.
* **VPC Endpoints (AWS PrivateLink):** Đây là yếu tố then chốt.
* **Interface VPC Endpoint** cho SageMaker AI và **Gateway VPC Endpoint** cho S3 cho phép traffic di chuyển trực tiếp từ VPC đến các dịch vụ AWS thông qua hạ tầng mạng riêng của AWS.
* Dữ liệu **không bao giờ đi qua Internet công cộng**. Điều này thỏa mãn yêu cầu về bảo mật dữ liệu nhạy cảm và tuân thủ quy định của công ty.


* **Tính bảo mật:** Bằng cách sử dụng Endpoint, bạn có thể áp dụng thêm **Endpoint Policies** để giới hạn quyền truy cập (ví dụ: chỉ cho phép VPC này truy cập vào một Bucket S3 cụ thể), tăng cường khả năng quản trị.

#### 2. Tại sao các phương án còn lại sai?

* **Phương án 1 (Internet Gateway & External Proxy):** Việc sử dụng Internet Gateway và External Proxy có nghĩa là dữ liệu sẽ đi ra khỏi mạng nội bộ AWS để đến proxy trước khi quay lại S3. Điều này vi phạm yêu cầu "all communication must occur over the AWS network" và tạo ra rủi ro bảo mật tiềm ẩn.
* **Phương án 2 (VPC Peering):** VPC Peering dùng để kết nối hai VPC với nhau. Tuy nhiên, S3 là một dịch vụ công cộng của AWS (public service), không nằm trong một VPC cụ thể của khách hàng. Do đó, VPC Peering không phải là giải pháp để truy cập S3 một cách riêng tư.
* **Phương án 3 (NAT Gateway):** NAT Gateway cho phép các tài nguyên trong private subnet truy cập Internet. Mặc dù nó giúp kết nối tới S3, nhưng traffic vẫn đi qua các điểm truy cập công cộng (public endpoints) của S3. Điều này không tối ưu bằng VPC Endpoint và không đáp ứng triệt để yêu cầu "giao tiếp trong mạng AWS" một cách khép kín nhất.

#### 3. Notes: Các dịch vụ và lưu ý về Bảo mật AI (Cập nhật 2025)

Dưới đây là những điểm quan trọng về bảo mật hạ tầng cho các dự án AI/ML trên AWS:

* **SageMaker AI (Tên mới):** Như đã đề cập ở các câu trước, AWS đã hợp nhất các dịch vụ dưới thương hiệu SageMaker AI để bao hàm cả GenAI và ML truyền thống.
* **Interface Endpoint vs. Gateway Endpoint:**
* **S3 & DynamoDB:** Sử dụng **Gateway Endpoints** (miễn phí, dựa trên bảng định tuyến - route table).
* **Hầu hết các dịch vụ khác (SageMaker, Comprehend, Bedrock):** Sử dụng **Interface Endpoints** (có phí, dựa trên PrivateLink/ENI).


* **Amazon Comprehend với VPC:** Tương tự như SageMaker, nếu bạn muốn gọi các tác vụ phân tích văn bản của Comprehend từ trong VPC mà không qua Internet, bạn cũng cần tạo một **Interface VPC Endpoint** cho Comprehend.
* **IAM Roles & S3 Bucket Policies:** Ngoài việc thiết lập hạ tầng mạng (VPC), bạn luôn cần áp dụng nguyên tắc **Least Privilege** (Quyền hạn tối thiểu) thông qua IAM Role gắn vào SageMaker Notebook để đảm bảo chỉ những người dùng/dịch vụ hợp lệ mới có thể đọc dữ liệu nhạy cảm.

---

### **Question 17:**

Category: AIP – Foundation Model Integration, Data Management, and Compliance

A global market research firm stores thousands of recorded analyst briefings in Amazon S3. The data science division uses Amazon SageMaker AI JumpStart to experiment with pretrained NLP models for text classification and Amazon Polly to synthesize sample voice outputs for model validation. The next project phase requires grouping all recorded discussions by the conversation themes, such as equity strategy, macroeconomic policy, and currency trends. The AI developer must choose an approach that automatically categorizes these recordings by topic, while minimizing custom development effort.

Which solution will meet this requirement most efficiently?

[ ] Build a custom data labeling workflow in SageMaker AI Ground Truth, then execute Amazon Transcribe jobs to convert audio to text and use SageMaker AI semantic segmentation to group the labeled outputs.

[ ] Run Amazon Transcribe jobs followed by an Amazon Comprehend custom classifier to assign predefined topic labels to each transcript.

[ ] Generate transcripts through Amazon Transcribe jobs, then sequentially train both the SageMaker AI semantic segmentation algorithm and Neural Topic Model (NTM) algorithm to manually group textual content into themes.

[x] **Configure Amazon Transcribe jobs to process all recordings into text, followed by an Amazon Comprehend topic detection job that identifies and clusters conversation topics.**

> Giải thích

#### 1. Giải thích đáp án đúng

Đây là giải pháp hiệu quả nhất (most efficient) vì nó sử dụng các dịch vụ AI được quản lý hoàn toàn (fully managed) để giải quyết bài toán mà không cần huấn luyện mô hình tùy chỉnh:

* **Chuyển đổi âm thanh sang văn bản:** **Amazon Transcribe** là lựa chọn tiêu chuẩn để chuyển đổi hàng ngàn bản ghi âm (analyst briefings) từ S3 thành văn bản (text) một cách tự động và quy mô lớn.
* **Phân nhóm theo chủ đề (Topic Modeling):** Yêu cầu của đề bài là "grouping discussions by conversation themes". **Amazon Comprehend Topic Detection** được thiết kế chính xác cho việc này. Nó sử dụng thuật toán học không giám sát (unsupervised learning) để tự động kiểm tra các tập tài liệu và xác định các chủ đề chung (như equity strategy, macro policy) mà **không cần người dùng phải gán nhãn dữ liệu trước** hoặc định nghĩa nhãn.
* **Tối thiểu hóa nỗ lực phát triển:** Cả hai dịch vụ này đều hoạt động thông qua API hoặc bảng điều khiển AWS, giúp giảm thiểu tối đa việc viết code tùy chỉnh, đáp ứng yêu cầu "minimizing custom development effort".

#### 2. Tại sao các phương án còn lại sai?

* **Phương án 1 (SageMaker AI Ground Truth):** Giải pháp này yêu cầu "custom data labeling" (gán nhãn thủ công). Điều này cực kỳ tốn thời gian và công sức, đi ngược lại mục tiêu "giảm thiểu nỗ lực phát triển". Ngoài ra, *Semantic Segmentation* thường dùng cho xử lý hình ảnh (phân đoạn pixel), không phải là thuật toán chính cho phân loại văn bản.
* **Phương án 2 (Comprehend Custom Classifier):** Để dùng *Custom Classifier*, bạn phải có một tập dữ liệu đã được gắn nhãn sẵn để huấn luyện mô hình nhận diện các chủ đề cụ thể. Đề bài không nói rằng công ty đã có sẵn các nhãn này, nên việc xây dựng bộ phân loại tùy chỉnh sẽ tốn nhiều công sức hơn là dùng tính năng Topic Detection có sẵn.
* **Phương án 3 (Training NTM manually):** Việc tự huấn luyện thuật toán *Neural Topic Model (NTM)* trên SageMaker AI yêu cầu kiến thức chuyên sâu về Machine Learning, cấu hình hạ tầng, chuẩn bị dữ liệu và điều chỉnh tham số. Đây không phải là cách tiếp cận "hiệu quả nhất" khi đã có dịch vụ AI chuyên dụng như Comprehend thực hiện việc này chỉ với vài cú click.

#### 3. Notes: Các dịch vụ AI phục vụ phân tích ngôn ngữ (NLP)

Dưới đây là các lưu ý quan trọng để phân biệt các công cụ xử lý ngôn ngữ trên AWS:

* **Amazon Comprehend Topic Detection:** Khác với phân loại (classification), Topic Detection là **unsupervised**. Nó tự tìm ra các nhóm từ thường xuyên xuất hiện cùng nhau và nhóm các tài liệu lại. Kết quả trả về là một danh sách các chủ đề và danh sách các tài liệu thuộc về mỗi chủ đề đó.
* **Amazon SageMaker AI JumpStart:** Đây là một kho chứa (hub) các mô hình nền tảng (Foundation Models) và các thuật toán được xây dựng sẵn. Dù nó rất mạnh mẽ để thực hiện các thí nghiệm nâng cao, nhưng đối với các tác vụ phổ biến như trích xuất chủ đề, các dịch vụ AI cấp cao như Comprehend thường nhanh và rẻ hơn.
* **Amazon Polly:** Trong câu hỏi này, Polly chỉ là "nhiễu" (distractor). Polly dùng để chuyển văn bản thành giọng nói (Text-to-Speech), phục vụ cho việc xác thực mô hình chứ không đóng góp vào quy trình phân tích và phân loại nội dung audio đầu vào.
* **Amazon Transcribe Call Analytics:** Một tính năng mở rộng của Transcribe có thể tự động xác định các đặc tính của cuộc hội thoại như tâm trạng (sentiment), các thực thể được nhắc đến, và thậm chí là tóm tắt nội dung, rất hữu ích cho các bản ghi âm hội thoại chuyên sâu.

---
### **Question 18:**

Category: AIP – Foundation Model Integration, Data Management, and Compliance

A data science team is developing a computer vision model using Amazon SageMaker AI to classify product images that customers upload. In addition to image features, the team plans to integrate Amazon Comprehend to analyze customer feedback text associated with each image. During data preparation, the team noticed that one of the numerical features representing image brightness seemed to affect the model’s convergence rate during training.

The lead ML engineer wants to explore how the brightness values are distributed in the dataset before deciding to apply normalization. The engineer uses SageMaker Data Wrangler for feature engineering and data preprocessing.

Which actions should the engineer take to best understand the range and distribution of the brightness feature values before transformation?

[ ] The engineer should export the dataset to Amazon S3 and use AWS Glue DataBrew to create a box plot visualization of the brightness feature.

[ ] The engineer should use Comprehend to perform sentiment analysis on the brightness values to determine if normalization is needed.

[ ] The engineer should use SageMaker Clarify to detect data bias in the brightness feature before performing any normalization.

[x] **The engineer should use the SageMaker Data Wrangler histogram visualization to inspect the range of values for the brightness feature and identify any outliers.**

> Giải thích: 

#### 1. Giải thích đáp án đúng

Đây là cách tiếp cận trực tiếp và hiệu quả nhất khi bạn đã sử dụng **SageMaker Data Wrangler** cho quy trình làm việc (workflow) của mình:

* **Histogram (Biểu đồ tần suất):** Là công cụ tiêu chuẩn để quan sát phân phối (distribution) của một đặc tính số học (numerical feature). Nó giúp kỹ sư nhìn rõ giá trị độ sáng tập trung ở đâu, dải giá trị (range) rộng hay hẹp, và liệu có các giá trị ngoại lai (outliers) quá cao hoặc quá thấp hay không.
* **Tích hợp sẵn (Built-in):** SageMaker Data Wrangler cung cấp các công cụ trực quan hóa tích hợp sẵn. Kỹ sư không cần phải rời khỏi môi trường làm việc hoặc xuất dữ liệu sang dịch vụ khác, giúp tiết kiệm thời gian và duy trì tính nhất quán của quy trình xử lý dữ liệu (data lineage).
* **Hỗ trợ quyết định:** Dựa trên hình dạng của histogram (ví dụ: nếu nó bị lệch - skewed), kỹ sư có thể quyết định áp dụng các kỹ thuật chuẩn hóa (normalization) hoặc chuẩn hóa theo phân phối chuẩn (standardization) một cách chính xác.

#### 2. Tại sao các phương án còn lại sai?

* **Phương án 1 (AWS Glue DataBrew):** Mặc dù AWS Glue DataBrew cũng có khả năng trực quan hóa mạnh mẽ, nhưng việc phải xuất dữ liệu ngược lại Amazon S3 rồi mới mở DataBrew tạo ra các bước thừa không cần thiết (overhead) khi bạn đang làm việc trực tiếp trong SageMaker Data Wrangler.
* **Phương án 2 (Amazon Comprehend):** Đây là một phương án gây nhiễu hoàn toàn sai về mặt kỹ thuật. Amazon Comprehend là dịch vụ phân tích ngôn ngữ tự nhiên (NLP) dùng cho văn bản. Nó không thể thực hiện "sentiment analysis" trên các giá trị số (brightness values) và càng không thể giúp xác định nhu cầu chuẩn hóa dữ liệu hình ảnh.
* **Phương án 3 (SageMaker Clarify):** SageMaker Clarify chủ yếu được dùng để phát hiện độ lệch (bias) liên quan đến các thuộc tính nhạy cảm (như giới tính, sắc tộc) hoặc phân tích sự đóng góp của các đặc tính (feature attribution). Dù nó có thể hiển thị phân phối, nhưng mục đích chính của nó là kiểm tra tính công bằng và giải thích mô hình, không phải là công cụ cơ bản nhất để kiểm tra dải giá trị đơn thuần cho việc chuẩn hóa như Histogram.

#### 3. Notes: Các tính năng phân tích dữ liệu trong SageMaker AI

Dưới đây là một số dịch vụ và tính năng quan trọng phục vụ việc tiền xử lý và hiểu dữ liệu:

* **SageMaker Data Wrangler:** Hiện tại đã được hợp nhất sâu vào Amazon SageMaker Studio, cho phép bạn thực hiện hơn 300 phép biến đổi dữ liệu mà không cần code. Ngoài Histogram, nó còn hỗ trợ *Scatter plots*, *Box plots*, và *Target leakage analysis*.
* **Normalization vs. Standardization:**
* **Normalization (Min-Max Scaling):** Đưa dữ liệu về dải [0, 1]. Phù hợp khi bạn biết rõ dải giá trị và không có nhiều nhiễu.
* **Standardization (Z-score Scaling):** Đưa dữ liệu về dạng có trung bình bằng 0 và độ lệch chuẩn bằng 1. Thường giúp mô hình hội tụ nhanh hơn nếu dữ liệu có phân phối hình chuông (Gaussian).


* **Amazon Comprehend (Tích hợp):** Trong bài toán này, Comprehend được dùng để xử lý phần văn bản (customer feedback). Kết quả từ Comprehend (ví dụ: điểm cảm xúc - sentiment score) có thể được kết hợp với các đặc tính hình ảnh (image features) trong SageMaker để tạo ra một mô hình đa phương thức (multimodal model).
* **Data Insights Report:** Một tính năng trong Data Wrangler tự động tạo báo cáo về chất lượng dữ liệu, cảnh báo về các cột có giá trị bị thiếu hoặc các đặc tính có biến động (variance) quá thấp.

---
### **Question 19:**

Category: AIP – AI Safety, Security, and Governance

An organization is developing a machine learning-based fraud detection system to process real-time transactions. The system leverages multiple APIs for generating embeddings from transaction descriptions, which are then used to assess the likelihood of fraud. These APIs are integrated with Amazon Comprehend for natural language processing tasks and Amazon SageMaker AI to handle advanced machine learning tasks for analyzing transaction patterns and building custom fraud detection models.

The organization’s security policy requires rotating API tokens every 3 months to reduce exposure risks. The solution must automate token rotation, maintain strong security, and integrate smoothly with existing AWS services. It must also ensure secure token storage, automatic updates within applications, and continuous operation without downtime.

Which solution will best address these requirements?

[ ] Use AWS Secrets Manager to store the tokens, monitor API usage with AWS CloudTrail, and rely on Amazon EventBridge to trigger token rotation.

[ ] Use AWS Systems Manager Parameter Store to store the tokens and rely on an AWS Lambda function for automatic rotation.

[ ] Use AWS Key Management Service (AWS KMS) with customer-managed keys to store the tokens and rely on Amazon EventBridge to trigger rotation events.

[x] **Use AWS Secrets Manager to store the tokens and rely on an AWS Lambda function to perform the rotation process.**

> Giải thích: 

#### 1. Giải thích đáp án đúng

Đây là giải pháp tiêu chuẩn của AWS (Best Practice) để quản lý vòng đời của các thông tin nhạy cảm như API tokens:

* **AWS Secrets Manager:** Được thiết kế chuyên biệt để lưu trữ, quản lý và xoay vòng (rotate) các bí mật (secrets). Dịch vụ này hỗ trợ lưu trữ các thông tin dưới dạng cặp key-value và có tính năng **phiên bản hóa (versioning)**, giúp đảm bảo hệ thống không bị gián đoạn (no downtime) trong khi quá trình xoay vòng đang diễn ra.
* **AWS Lambda:** Secrets Manager tích hợp trực tiếp với Lambda để thực hiện logic xoay vòng. Khi đến thời hạn (ví dụ: 3 tháng), Secrets Manager sẽ kích hoạt hàm Lambda. Hàm này sẽ thực hiện các bước: tạo token mới từ nhà cung cấp API, cập nhật token mới vào Secrets Manager và kiểm tra tính hợp lệ.
* **Bảo mật và Tự động hóa:** Giải pháp này loại bỏ việc lưu trữ token "cứng" (hard-coded) trong mã nguồn hoặc tệp cấu hình. Ứng dụng sẽ gọi API Secrets Manager để lấy token mới nhất mỗi khi cần, đảm bảo tính liên tục và an toàn tuyệt đối theo đúng yêu cầu của tổ chức.

#### 2. Tại sao các phương án còn lại sai?

* **Phương án 1 (EventBridge & CloudTrail):** Mặc dù EventBridge có thể lên lịch kích hoạt, nhưng việc sử dụng CloudTrail để giám sát mức độ sử dụng không phải là cơ chế để thực hiện xoay vòng. Phương án này thiếu một thành phần thực thi logic xoay vòng (như Lambda).
* **Phương án 2 (Systems Manager Parameter Store):** SSM Parameter Store (loại SecureString) có thể lưu trữ token, nhưng nó **không có tính năng tự động xoay vòng tích hợp** mạnh mẽ như Secrets Manager. Bạn sẽ phải tự xây dựng toàn bộ cơ chế quản lý phiên bản và kích hoạt, điều này làm tăng nỗ lực phát triển và rủi ro vận hành.
* **Phương án 3 (AWS KMS):** KMS được dùng để tạo và quản lý **khóa mã hóa** (encryption keys), không phải là dịch vụ dùng để **lưu trữ** các API tokens của bên thứ ba. Đây là sự nhầm lẫn phổ biến về chức năng của các dịch vụ bảo mật.

#### 3. Notes: Các dịch vụ và lưu ý quan trọng (Cập nhật 2025)

Dưới đây là một số điểm cần lưu ý khi thiết kế hệ thống bảo mật cho AI/ML:

* **Secrets Manager vs. Parameter Store:** * Sử dụng **Secrets Manager** khi bạn cần tính năng tự động xoay vòng và quản lý các bí mật phức tạp (có phí).
* Sử dụng **Parameter Store** cho các cấu hình thông thường hoặc bí mật đơn giản không yêu cầu xoay vòng tự động (miễn phí cho các tham số tiêu chuẩn).


* **VPC Endpoints cho Secrets Manager:** Để đảm bảo tính bảo mật cao nhất (như yêu cầu của đề bài), bạn nên sử dụng **Interface VPC Endpoint (AWS PrivateLink)** để các ứng dụng trong VPC (như SageMaker Notebooks) truy cập Secrets Manager mà không cần đi qua Internet công cộng.
* **IAM Policy:** Luôn áp dụng nguyên tắc **Least Privilege** (Quyền hạn tối thiểu). Chỉ cho phép hàm Lambda thực hiện xoay vòng có quyền truy cập vào secret cụ thể và chỉ cho phép ứng dụng Fraud Detection có quyền `GetSecretValue`.
* **Integration with SageMaker AI:** Trong các pipeline của SageMaker, bạn có thể sử dụng thư viện `boto3` để truy xuất API tokens từ Secrets Manager ngay trong quá trình huấn luyện hoặc suy luận (inference), đảm bảo thông tin luôn được cập nhật mới nhất.

---
### **Question 20:**

Category: AIP – AI Safety, Security, and Governance

A financial services company is modernizing its machine learning (ML) platform using Amazon SageMaker AI and Amazon Rekognition to automate fraud detection and document verification workflows. The AI development team uses Rekognition to extract and analyze visual data from scanned identification documents, and then processes these insights in dedicated SageMaker notebook instances to train models that classify fraudulent patterns.

The administrator must ensure that each AI developer can access only the assigned SageMaker notebook instance while maintaining shared access to the Rekognition APIs and centralized training data stored in Amazon S3.

Which solution will meet this requirement?

[ ] Enable port forwarding on the SageMaker notebook instances to block external network access for unauthorized users.

[ ] Configure SageMaker lifecycle configurations to prevent developers from accessing unassigned notebook instances.

**[x] Create an IAM policy for each AI developer’s IAM user that grants SageMaker permissions only for the ARN of their assigned notebook instance.**

[ ] Create a shared SageMaker notebook instance and restrict developer access by configuring JupyterLab role-based permissions.

> Giải thích: 

#### 1. Giải thích đáp án đúng

Đây là giải pháp tuân thủ **Nguyên tắc quyền hạn tối thiểu (Principle of Least Privilege)** của AWS để quản lý truy cập:

* **IAM Policy dựa trên ARN:** Mỗi tài nguyên trong AWS (bao gồm cả SageMaker Notebook Instance) đều có một mã định danh duy nhất gọi là **ARN (Amazon Resource Name)**. Bằng cách viết một chính sách IAM chỉ cho phép hành động `sagemaker:CreatePresignedNotebookInstanceUrl` hoặc `sagemaker:StartNotebookInstance` trên một ARN cụ thể, quản trị viên đảm bảo rằng Developer A không thể mở hoặc can thiệp vào Notebook của Developer B.
* **Quyền truy cập chung (Shared Access):** Các quyền truy cập vào **Amazon Rekognition APIs** và **Amazon S3** (dữ liệu huấn luyện tập trung) có thể được định nghĩa trong một chính sách IAM khác hoặc gán trực tiếp vào IAM Role mà tất cả các developer đều sử dụng. Điều này cho phép họ làm việc độc lập trên môi trường tính toán của mình nhưng vẫn dùng chung tài nguyên dữ liệu và dịch vụ AI của công ty.
* **Tính quy mô:** Giải pháp này tận dụng hạ tầng quản lý định danh (IAM) có sẵn của AWS, giúp dễ dàng kiểm soát, kiểm toán và thay đổi quyền hạn mà không cần can thiệp vào cấu hình bên trong máy chủ.

#### 2. Tại sao các phương án còn lại sai?

* **Phương án 1 (Port forwarding):** Port forwarding là một kỹ thuật mạng, không phải là cơ chế quản lý danh tính và quyền truy cập (IAM) chính thống của AWS. Việc chặn truy cập mạng không thay đổi được việc một người dùng có quyền gọi API của AWS để điều khiển tài nguyên hay không.
* **Phương án 2 (Lifecycle configurations):** Lifecycle configurations là các tập lệnh chạy khi khởi tạo hoặc bắt đầu Notebook (ví dụ: cài đặt thư viện). Chúng không được thiết kế để kiểm soát quyền truy cập của người dùng (Who can access what). Việc dùng script để chặn người dùng truy cập là cách làm không an toàn và dễ bị bỏ qua.
* **Phương án 4 (Shared SageMaker instance):** Dùng chung một Notebook Instance gây ra rủi ro lớn về bảo mật và xung đột tài nguyên. JupyterLab không hỗ trợ mạnh mẽ việc phân quyền Role-based cho nhiều người dùng trên cùng một thực thể máy chủ đơn lẻ theo cách mà IAM thực hiện ở cấp độ dịch vụ AWS.

#### 3. Notes: Các dịch vụ và lưu ý về Quản trị AI

Dưới đây là một số lưu ý quan trọng để tối ưu hóa quản trị nền tảng ML:

* **SageMaker AI (Tên gọi mới):** AWS đã cập nhật tên gọi từ SageMaker sang SageMaker AI để phản ánh sự mở rộng sang các mô hình nền tảng và Generative AI.
* **Amazon Rekognition:** Trong ngữ cảnh này, dịch vụ được dùng để trích xuất văn bản từ hình ảnh (OCR) hoặc so sánh khuôn mặt trên giấy tờ định danh. Đây là một dịch vụ "Pre-trained", không cần người dùng quản lý máy chủ.
* **Tag-based Access Control:** Thay vì tạo chính sách cho từng ARN (có thể gây mệt mỏi nếu có hàng trăm developer), bạn có thể sử dụng **Tags**. Ví dụ: Chỉ cho phép Developer truy cập vào Notebook có Tag `Owner: [Tên_Developer]`. Đây là cách quản lý hiện đại và linh hoạt hơn.
* **Sagemaker Studio (Update 2025):** Hiện nay AWS khuyến khích sử dụng SageMaker Studio vì nó hỗ trợ **Space** – các không gian làm việc riêng tư cho từng người dùng hoặc nhóm, giúp việc quản lý quyền truy cập trở nên trực quan và dễ dàng hơn so với việc quản lý từng Notebook Instance riêng lẻ.

---
### **Question 21:**

Category: AIP – Foundation Model Integration, Data Management, and Compliance

A global online publishing company is developing a generative AI-based content recommendation system to suggest relevant tags for articles. The raw data for the articles, stored in JSON format, is housed in an Amazon S3 bucket. The data scientist is using the Amazon SageMaker Neural Topic Model (NTM) algorithm to build a model that can generate the most appropriate tags for each article. To enhance the model’s performance, Amazon Titan is being utilized to improve natural language understanding and generate contextual insights from the article content, leveraging Titan’s powerful pre-trained models for tag suggestions.

During the evaluation phase, the model’s recommendations include common stopwords such as “and,” “of,” and “in,” along with some rare words that appear only in a few specific articles. After working with the content team to review the model’s output, the data scientist determines that while the rare words are unusual, the words could still be valuable for generating more relevant tags. The challenge is to find a way to exclude the stopwords from the model’s tag predictions while still retaining the rare but meaningful words.

How can the model be refined to exclude stopwords and retain rare words in tags?

[ ] Employ the CountVectorizer function from scikit-learn to preprocess the articles by removing stopwords while retaining rare words, then upload the processed data back to the S3 bucket.

[ ] Index the article content using Amazon OpenSearch, configuring it to filter out stopwords and retain rare terms during indexing, then store the indexed data in the S3 bucket.

[ ] Use Amazon Comprehend’s entity detection capabilities to detect the keywords while removing stopwords from the articles, and then update the data in the S3 bucket with the cleaned version.

**[x] Apply SageMaker Processing to run a custom script that removes stopwords and filters out irrelevant terms, while preserving rare but meaningful words, and then store the cleaned data in an S3 bucket.**

> Giải thích:

#### 1. Giải thích đáp án đúng

Đây là giải pháp linh hoạt và mạnh mẽ nhất để xử lý dữ liệu phức tạp trong hệ sinh thái AI/ML của AWS:

* **Tính tùy biến cao (Custom Logic):** Yêu cầu của đề bài rất đặc thù: loại bỏ từ dừng (stopwords) nhưng **phải giữ lại** các từ hiếm (rare words). Các công cụ tự động đôi khi sẽ loại bỏ cả hai (vì từ hiếm thường bị coi là nhiễu). Với **SageMaker Processing**, bạn có thể viết một script Python (sử dụng thư viện như `nltk` hoặc `spaCy`) để thực hiện chính xác logic này: lọc theo danh sách stopwords cụ thể và giữ lại các từ có tần suất thấp nhưng mang ý nghĩa chuyên môn.
* **Xử lý quy mô lớn:** SageMaker Processing tự động hóa việc cung cấp hạ tầng (infrastructure provisioning), chạy script trên các tập dữ liệu lớn từ S3, sau đó lưu kết quả trở lại S3. Điều này đảm bảo tính ổn định cho môi trường sản xuất.
* **Tích hợp vào Pipeline:** Bước xử lý này có thể dễ dàng trở thành một phần trong **SageMaker Pipeline**, giúp quy trình từ làm sạch dữ liệu đến huấn luyện mô hình NTM và tích hợp Amazon Titan trở nên hoàn toàn tự động và có khả năng tái lặp.

#### 2. Tại sao các phương án còn lại sai?

* **Phương án 1 (scikit-learn CountVectorizer):** Mặc dù `CountVectorizer` có thể loại bỏ stopwords, nhưng nó thường được dùng trong môi trường local hoặc notebook đơn lẻ. Việc xử lý thủ công rồi upload lại S3 không mang tính tự động hóa và khó quản lý khi dữ liệu lớn (thousands of articles). Nó thiếu khả năng mở rộng (scalability) so với SageMaker Processing.
* **Phương án 2 (Amazon OpenSearch):** OpenSearch là dịch vụ tìm kiếm và phân tích, không phải là công cụ chuyên dụng để tiền xử lý dữ liệu cho việc huấn luyện mô hình ML. Việc index dữ liệu rồi xuất ngược lại S3 là một quy trình vòng vèo, tốn kém tài nguyên và không cần thiết cho mục đích này.
* **Phương án 3 (Amazon Comprehend Entity Detection):** Comprehend Entity Detection dùng để nhận diện các thực thể như "Tên người", "Địa điểm", "Tổ chức". Nó không được thiết kế để làm sạch văn bản tổng quát theo yêu cầu giữ lại "từ hiếm" (rare words) mà không phải là thực thể. Việc ép dùng Comprehend sẽ không giải quyết được triệt để bài toán về các từ chuyên ngành hiếm gặp.

#### 3. Notes: Các lưu ý về NTM và Tiền xử lý dữ liệu

* **Neural Topic Model (NTM):** Đây là một thuật toán "Unsupervised Learning" (Học không giám sát). Nó rất nhạy cảm với stopwords vì các từ này xuất hiện quá nhiều, khiến mô hình lầm tưởng chúng là "chủ đề" chính. Việc làm sạch dữ liệu trước khi đưa vào NTM là bước sống còn.
* **Amazon Titan Integration:** Việc sử dụng Titan để lấy "contextual insights" cho thấy xu hướng kết hợp giữa **Topic Modeling truyền thống** và **LLMs (Large Language Models)**. Titan giúp hiểu ngữ cảnh sâu hơn, trong khi NTM giúp phân loại cấu trúc theo các nhóm chủ đề cố định.
* **TF-IDF (Term Frequency-Inverse Document Frequency):** Một lưu ý kỹ thuật nhỏ là khi giữ lại "rare words", bạn có thể sử dụng chỉ số TF-IDF trong script của SageMaker Processing. Những từ hiếm nhưng xuất hiện tập trung trong một số bài viết sẽ có điểm TF-IDF cao, giúp mô hình NTM nhận diện chúng là các tag đặc trưng.
* **SageMaker Processing Containers:** Bạn có thể sử dụng các container có sẵn của AWS (như Scikit-learn container) hoặc mang theo container riêng (Bring Your Own Container - BYOC) nếu script xử lý của bạn yêu cầu các thư viện ngôn ngữ đặc biệt.

---
### **Question 22:**

Category: AIP – Operational Efficiency and Optimization for Generative AI Applications

A national utility company manages hourly power consumption data for 150 types of smart meters deployed across multiple regions. Over the past 30 years, the company has collected a large volume of historical usage data enriched with weather metrics, regional energy prices, and maintenance logs.

The data science team currently uses Amazon SageMaker AI Data Wrangler to perform feature engineering and data preprocessing, and SageMaker AI Canvas (a no-code ML service) to generate baseline forecasts for monthly energy demand. However, the team now plans to build a custom machine learning (ML) model that can forecast future meter-usage patterns across all meter types.

Which option satisfies the requirement with the LEAST operational overhead?

[ ] Use SageMaker AI Autopilot to automatically create and tune a single predictive model for all meter types using the combined dataset.

**[x] Build a single global forecasting model using the SageMaker AI DeepAR forecasting algorithm trained on all meter types to capture shared consumption patterns.**

[ ] Train multiple forecasting models using the SageMaker AI Prophet algorithm, one for each meter type, to capture seasonal consumption patterns for individual devices.

[ ] Create a single global forecasting model using the SageMaker AI XGBoost algorithm trained on all meter types to predict future consumption patterns.

> Giải thích: 

#### 1. Giải thích đáp án đúng

Để giải quyết bài toán dự báo cho nhiều chuỗi thời gian (150 loại công tơ mét) với "nỗ lực vận hành ít nhất" (least operational overhead), **DeepAR** là sự lựa chọn tối ưu vì:

* **Global Model (Mô hình toàn cầu):** Thay vì phải huấn luyện và quản lý 150 mô hình riêng lẻ cho từng loại thiết bị, DeepAR cho phép huấn luyện một mô hình duy nhất trên toàn bộ tập dữ liệu. Nó có khả năng học các mô hình tiêu thụ chung (shared patterns) giữa các loại công tơ mét khác nhau.
* **Deep Learning (RNN/LSTM):** DeepAR sử dụng mạng thần kinh tái phát, cực kỳ mạnh mẽ trong việc xử lý dữ liệu chuỗi thời gian có độ phức tạp cao, đặc biệt khi có các yếu tố bổ trợ (như dữ liệu thời tiết, giá năng lượng) mà đề bài đã nêu.
* **Xử lý dữ liệu đa dạng:** DeepAR hoạt động rất tốt khi các chuỗi thời gian có thang đo khác nhau (ví dụ: một hộ gia đình dùng ít điện vs một nhà máy dùng nhiều điện) nhờ cơ chế tự động điều chỉnh tỷ lệ (scaling) tích hợp sẵn.
* **Tính dự báo xác suất:** DeepAR không chỉ đưa ra một con số duy nhất mà còn cung cấp các khoảng tin cậy (confidence intervals), giúp công ty tiện ích quản lý rủi ro tốt hơn trong việc điều phối năng lượng.

#### 2. Tại sao các phương án còn lại sai?

* **Phương án 1 (SageMaker AI Autopilot):** Autopilot là công cụ tuyệt vời để tìm ra mô hình tốt nhất cho các bài toán Tabular (phân loại/hồi quy), nhưng nó không được tối ưu hóa đặc thù cho bài toán **Time-series forecasting** quy mô lớn với nhiều chuỗi thời gian đan xen như DeepAR. Việc ép Autopilot xử lý dữ liệu này sẽ tốn nhiều công sức chuẩn bị dữ liệu hơn.
* **Phương án 3 (Prophet - Multiple Models):** Yêu cầu "huấn luyện một mô hình cho mỗi loại công tơ" tạo ra **operational overhead khổng lồ**. Bạn sẽ phải quản lý, lưu trữ và theo dõi 150 mô hình riêng biệt. Prophet cũng là mô hình đơn biến (univariate), khó tận dụng được các mối tương quan chéo giữa các thiết bị khác nhau.
* **Phương án 4 (XGBoost):** XGBoost là một thuật toán mạnh mẽ cho dữ liệu bảng, nhưng để dùng cho chuỗi thời gian, bạn phải tự thực hiện "feature engineering" rất phức tạp (tạo các cột lag, rolling window, v.v.). Điều này vi phạm tiêu chí "least operational overhead" so với một thuật toán chuyên dụng cho time-series như DeepAR.

#### 3. Notes: Các dịch vụ và lưu ý về Dự báo (Forecasting)

* **SageMaker AI Canvas:** Là công cụ No-code rất tốt cho Business Analysts để tạo dự báo nhanh (baseline). Tuy nhiên, khi chuyển sang "Custom ML Model", chúng ta cần các thuật toán chuyên sâu hơn có thể can thiệp vào mã nguồn và tham số.
* **DeepAR vs. Amazon Forecast:** * **DeepAR:** Là thuật toán nằm trong SageMaker, phù hợp cho các Data Scientists muốn kiểm soát sâu quy trình huấn luyện và tích hợp vào Pipeline.
* **Amazon Forecast:** Là dịch vụ AI cấp cao hơn (managed service), tự động chọn thuật toán (bao gồm cả DeepAR, Prophet, CNN-QR) và xử lý dữ liệu. Nếu đề bài không yêu cầu "Custom ML model on SageMaker", Amazon Forecast thường là đáp án cho "ít nỗ lực nhất".
* **Dữ liệu bổ trợ (Dynamic Variables):** Trong dự báo năng lượng, thời tiết (nhiệt độ) là biến quan trọng nhất. DeepAR cho phép đưa các biến này vào dưới dạng "dynamic features" để tăng độ chính xác đáng kể.
* **Cold Start Problem:** DeepAR có khả năng dự báo cho các thiết bị mới lắp đặt (chưa có nhiều dữ liệu lịch sử) bằng cách học từ các thiết bị tương tự đã có trong tập huấn luyện.

---
### **Question 23:**

Category: AIP – Operational Efficiency and Optimization for Generative AI Applications

A global logistics company is developing an AI-powered fleet safety monitoring platform to improve driver safety and minimize accident-related costs. The platform analyzes dashboard camera feeds from delivery trucks to detect behaviors that indicate driver distraction or fatigue. The system is being tested on long-haul trucks that operate primarily on overnight routes, where drivers encounter challenging nighttime conditions such as low visibility, glare from oncoming headlights, and inconsistent illumination from streetlights during extended driving periods. Amazon Rekognition Custom Labels is used to train a computer vision model capable of identifying visual patterns such as yawning, looking away, or using a mobile device, while Amazon SageMaker AI is used to handle the end-to-end machine learning workflow, including model training, tuning, and evaluation.

The company’s AI team collected 1,000 labeled driver training images in a controlled facility to represent different attention states such as “alert,” “sleepy,” and “distracted.” During training in SageMaker AI, the team observed that the training loss decreases quickly with each epoch, but the validation accuracy remains low. When tested in real driving conditions, the model often misclassifies drivers under varying lighting and road environments. The AI team must find a way to improve the model’s ability to generalize to new, unseen data without collecting additional samples or modifying the model’s architecture.

Which combination of actions should be implemented to address this issue? (Select TWO.)

**[x] Perform data augmentation on the training dataset using randomized transformations and preprocessing operations.**

[ ] Increase the total number of training epochs within the current model configuration.

[ ] Adjust the optimizer settings to use a higher learning rate across all training iterations.

[ ] Execute gradient checking after initializing model parameters and performing initial training runs.

**[x] Add L2 regularization to the neural network during the optimization phase of the training cycle.**

> Giải thích: 

#### 1. Giải thích các đáp án đúng

Tình trạng "training loss giảm nhanh nhưng validation accuracy thấp" là dấu hiệu điển hình của **Overfitting** (Quá khớp). Mô hình đang "học thuộc lòng" 1.000 ảnh trong phòng thí nghiệm thay vì học các đặc trưng tổng quát. Để khắc phục mà không cần thu thập thêm dữ liệu hay thay đổi kiến trúc, chúng ta sử dụng:

* **Data Augmentation (Tăng cường dữ liệu):** Đây là kỹ thuật tạo ra các biến thể mới từ dữ liệu sẵn có. Bằng cách áp dụng các phép biến đổi như: thay đổi độ sáng (để mô phỏng ánh sáng ban đêm/đèn đường), thêm nhiễu (glare), xoay hoặc lật ảnh, mô hình sẽ được tiếp xúc với nhiều tình huống "giả lập" thực tế hơn. Điều này giúp mô hình nhận diện được hành vi "buồn ngủ" ngay cả khi điều kiện ánh sáng thay đổi, từ đó cải thiện khả năng tổng quát hóa (**generalization**).
* **L2 Regularization (Điều chuẩn L2):** Kỹ thuật này thêm một "hình phạt" (penalty) vào hàm mất mát dựa trên độ lớn của các trọng số (weights). Nó ngăn cản các trọng số trở nên quá lớn, giúp mô hình trở nên "mượt" hơn và ít nhạy cảm hơn với các chi tiết nhiễu trong tập huấn luyện. Đây là một trong những cách hiệu quả nhất để kiểm soát overfitting trong mạng thần kinh mà không cần thay đổi cấu trúc model.

#### 2. Tại sao các phương án còn lại sai?

* **Increase the total number of training epochs (Tăng số lượng epoch):** Nếu mô hình đã bị overfitting, việc tăng thêm số vòng lặp huấn luyện sẽ chỉ khiến mô hình "học thuộc lòng" tập training kỹ hơn nữa, làm khoảng cách giữa training loss và validation loss càng lớn hơn.
* **Higher learning rate (Tăng tốc độ học):** Tăng learning rate có thể khiến quá trình tối ưu hóa bị mất ổn định, nhảy qua điểm tối ưu và thường không giúp ích gì cho việc cải thiện khả năng tổng quát hóa khi đã bị overfitting.
* **Gradient checking (Kiểm tra gradient):** Đây là kỹ thuật dùng để gỡ lỗi (debug) quá trình tính toán đạo hàm trong khi lập trình thuật toán lan truyền ngược (backpropagation). Nó không có tác dụng cải thiện độ chính xác hay giải quyết vấn đề overfitting của mô hình.

#### 3. Notes: Các lưu ý về Computer Vision và SageMaker AI

* **Amazon Rekognition Custom Labels:** Dịch vụ này cho phép bạn huấn luyện mô hình nhận diện vật thể/cảnh quan đặc thù (như hành vi tài xế) chỉ với một lượng dữ liệu nhỏ. Nó tự động áp dụng nhiều kỹ thuật tối ưu, nhưng khi gặp dữ liệu quá đặc thù trong phòng thí nghiệm, overfitting vẫn có thể xảy ra.
* **SageMaker Training & Augmentation:** Trong thực tế, bạn có thể thực hiện Data Augmentation ngay trong **SageMaker Training Job** bằng cách sử dụng thư viện như `Albumentations` hoặc `torchvision` trong file script huấn luyện.
* **Nighttime Vision Challenges:** Đối với các bài toán lái xe ban đêm, việc Augmentation tập trung vào **Color Jittering** (thay đổi độ sáng/độ tương phản) và **Gaussian Noise** là cực kỳ quan trọng để mô phỏng điều kiện ánh sáng yếu và nhiễu của camera dashboard.
* **Early Stopping:** Một kỹ thuật khác thường được dùng kèm với L2 Regularization trong SageMaker AI là **Early Stopping** (dừng sớm). Nếu validation loss bắt đầu tăng trong khi training loss vẫn giảm, SageMaker sẽ tự động dừng quá trình huấn luyện để tránh overfitting.

---
### **Question 24:**

Category: AIP – Implementation and Integration

A data scientist is tasked with developing a fraud detection model for an e-commerce platform. The dataset consists of transaction records where fraudulent transactions are much less frequent than legitimate ones, resulting in a class imbalance.

To enrich the dataset, voice-based transaction logs or customer support calls may be transcribed using Amazon Transcribe and analyzed with Amazon Comprehend, but the core model will be built using structured data.

The data scientist has limited experience with advanced model development techniques but aims to minimize the operational overhead while ensuring the model is fair and unbiased. To accelerate development, the data scientist is planning to use Amazon SageMaker AI.

Which approach will fulfill the given requirements?

[ ] Use SageMaker Studio to preprocess the data and apply SMOTE, then use SageMaker Reinforcement Learning to build a fraud detection model and check for bias with SageMaker Clarify.

**[x] Use SageMaker Studio to preprocess and balance the data using the synthetic minority oversampling technique (SMOTE), then develop a fraud detection model with SageMaker JumpStart. Afterward, use SageMaker Clarify to check for bias and finalize the model for deployment.**

[ ] Use SageMaker Studio for data processing and model development, integrating the synthetic minority oversampling technique (SMOTE) into the workflow. Once the model is trained, use Amazon Augmented AI (Amazon A2I) for bias detection before deployment.

[ ] Use SageMaker Studio to preprocess the data and apply the synthetic minority oversampling technique (SMOTE) to balance the dataset. Build the model using SageMaker Pipelines and use SageMaker Clarify for bias detection before deployment.

> Giải thích:

#### 1. Giải thích đáp án đúng

Giải pháp này xử lý tối ưu các thách thức về dữ liệu mất cân bằng và yêu cầu ít kinh nghiệm ML:

* **Xử lý mất cân bằng (Class Imbalance):** Sử dụng kỹ thuật **SMOTE** (Synthetic Minority Over-sampling Technique) để tạo thêm các mẫu giả lập cho lớp thiểu số (giao dịch gian lận), giúp mô hình không bị thiên kiến về lớp đa số.
* **Hỗ trợ người dùng ít kinh nghiệm:** **SageMaker JumpStart** cung cấp các giải pháp và mô hình huấn luyện sẵn (pre-trained) được tối ưu hóa cho các bài toán cụ thể như Fraud Detection. Điều này giúp "giảm thiểu chi phí vận hành" (minimize operational overhead) đúng như đề bài yêu cầu.
* **Đảm bảo tính công bằng:** **SageMaker Clarify** là công cụ chuyên dụng của AWS để phát hiện sai lệch (bias) trong tập dữ liệu và giải thích các dự đoán của mô hình, đảm bảo hệ thống "fair and unbiased".

#### 2. Tại sao các phương án còn lại sai?

* **Phương án 1 (Reinforcement Learning):** Học tăng cường (Reinforcement Learning) thường dùng cho điều khiển robot hoặc game, không phải là phương pháp tiêu chuẩn và ít nỗ lực nhất cho bài toán phát hiện gian lận dựa trên dữ liệu bảng (tabular data).
* **Phương án 3 (Amazon A2I):** **Amazon Augmented AI (A2I)** được dùng để đưa con người vào vòng lặp kiểm tra (human-in-the-loop) đối với các dự đoán có độ tin cậy thấp, không phải là công cụ chính để phát hiện sai lệch (bias detection) trong quá trình phát triển mô hình.
* **Phương án 4 (SageMaker Pipelines):** Mặc dù Pipelines rất tốt để tự động hóa, nhưng việc tự xây dựng toàn bộ model trong Pipeline từ đầu đòi hỏi nhiều kinh nghiệm phát triển hơn so với việc dùng giải pháp có sẵn từ **JumpStart**.

#### 3. Notes (Giải thích dịch vụ & Lưu ý)

* **SMOTE (Synthetic Minority Over-sampling Technique):** Một thuật toán tạo ra các mẫu mới dựa trên các mẫu lớp thiểu số hiện có thay vì chỉ sao chép chúng. Nó giúp mô hình học được ranh giới quyết định tốt hơn.
* **SageMaker JumpStart:** Giống như một "cửa hàng" các giải pháp ML có sẵn. Bạn có thể triển khai mô hình chỉ với vài cú click, phù hợp với người có "limited experience".
* **SageMaker Clarify:** Cung cấp các chỉ số như *Difference in Positive Proportions (DPP)* để đo lường sự bất công giữa các nhóm dữ liệu (ví dụ: theo khu vực địa lý hoặc loại thẻ).
* **Amazon Transcribe & Comprehend (Nhiễu):** Trong câu hỏi này, việc nhắc đến Transcribe và Comprehend chỉ là thông tin bổ sung về cách "làm giàu dữ liệu", không quyết định lựa chọn kiến trúc lõi của model.

---
### **Question 25:**

Category: AIP – Operational Efficiency and Optimization for Generative AI Applications

An e-commerce company leverages both Amazon SageMaker AI to train a machine learning (ML) model that predicts product-interest scores for users, and Amazon Personalize to generate real-time personalized product recommendations based on that model’s output.

The AI developer wants to visualize recommendation results across four distinct dimensions: the user’s interest score (first dimension) plotted against the product’s prior conversion rate (second dimension), then visualize a third dimension (product category) via color, and a fourth dimension (number of impressions for that product) via the size of each data point. The goal is to spot segments where high-interest, high-conversion products still receive low impressions.

Which approach best satisfies the given requirements?

[ ] Use the SageMaker Canvas Box Plot visualization to compare distributions and use a fill pattern for the third dimension.

[ ] Use the SageMaker Canvas Bar Chart visualization to group products by category and simultaneously apply bar color and height to represent interest score and conversion rate.

**[x] Apply the SageMaker Canvas scatter plot visualization and map the third dimension (product category) to scatter point color and the fourth dimension (number of impressions) to scatter point size.**

[ ] Visualize the data using the SageMaker Data Wrangler scatter plot visualization and color data points by the third feature to represent all four dimensions.

> Giải thích: 

#### 1. Giải thích đáp án đúng

Để trực quan hóa dữ liệu có **4 chiều (4 dimensions)** trên cùng một biểu đồ, biểu đồ phân tán (Scatter Plot) là công cụ mạnh mẽ nhất:

* **Trục X và Y (Chiều 1 & 2):** Việc vẽ *interest score* đối lập với *conversion rate* (cả hai đều là dữ liệu số) trên hai trục tọa độ giúp xác định mối tương quan giữa mức độ quan tâm của người dùng và hiệu quả bán hàng thực tế.
* **Màu sắc (Chiều 3 - Color):** Gán *product category* (dữ liệu phân loại) vào màu sắc của các điểm dữ liệu giúp người dùng dễ dàng phân biệt các nhóm sản phẩm khác nhau trong cùng một không gian.
* **Kích thước điểm (Chiều 4 - Size):** Đây là yếu tố then chốt để thể hiện chiều thứ tư (*number of impressions*). Khi gán số lượng hiển thị vào kích thước điểm (tạo thành biểu đồ bong bóng - Bubble Chart), nhà phân tích có thể ngay lập tức nhận ra các sản phẩm tiềm năng nhưng đang nhận được ít sự chú ý (điểm nằm ở góc trên bên phải nhưng có kích thước rất nhỏ).
* **Công cụ:** **SageMaker AI Canvas** cung cấp giao diện no-code cho phép cấu hình linh hoạt các thuộc tính Color và Size này chỉ bằng vài thao tác kéo thả.

#### 2. Tại sao các phương án còn lại sai?

* **Phương án 1 (Box Plot):** Biểu đồ hộp dùng để xem phân phối dữ liệu (trung vị, tứ phân vị) của một biến số theo các nhóm. Nó không thể hiện được mối quan hệ giữa hai biến số độc lập (X vs Y) và không hỗ trợ thay đổi kích thước điểm để biểu diễn chiều thứ tư.
* **Phương án 2 (Bar Chart):** Biểu đồ cột chủ yếu dùng để so sánh số lượng giữa các danh mục. Việc dùng cả chiều cao và màu sắc cho hai biến số khác nhau trên cùng một cột thường gây rối mắt và không thể hiện rõ ràng các điểm dữ liệu đơn lẻ (từng sản phẩm) như Scatter Plot.
* **Phương án 4 (Data Wrangler Scatter Plot):** Mặc dù Data Wrangler có biểu đồ phân tán, nhưng mô tả trong phương án này chỉ nhắc đến việc dùng màu sắc để đại diện cho chiều thứ ba, hoàn toàn bỏ qua chiều thứ tư (*Size*). Do đó, nó không đáp ứng đủ yêu cầu "trực quan hóa cả 4 chiều" của đề bài.

#### 3. Notes (Giải thích dịch vụ & Lưu ý)

* **SageMaker AI Canvas:** Là dịch vụ Machine Learning không cần lập trình (no-code), cho phép người dùng kinh doanh hoặc các nhà phân tích dữ liệu xây dựng mô hình dự báo và trực quan hóa dữ liệu một cách nhanh chóng.
* **Amazon Personalize:** Dịch vụ này sử dụng các thuật toán tương tự như của Amazon.com để tạo ra các đề xuất sản phẩm, xếp hạng tùy chỉnh và phân khúc khách hàng theo thời gian thực.
* **Mô hình đa chiều (Multivariate Analysis):** Trong phân tích dữ liệu, việc kết hợp X, Y, Color và Size là kỹ thuật phổ biến nhất để quan sát 4 thuộc tính cùng lúc mà không cần dùng đến các không gian 3D phức tạp.
* **Spotting Segments (Phân khúc mục tiêu):** Mục tiêu của công ty là tìm "high-interest, high-conversion, low impressions". Trên biểu đồ Scatter Plot này, đó chính là những **điểm nhỏ nằm ở phía trên bên phải**. Đây là những sản phẩm "mỏ vàng" cần được tăng cường hiển thị để tối đa hóa doanh thu.

---
### **Question 26:**

Category: AIP – Implementation and Integration

A financial analytics firm uses Amazon SageMaker AI and Amazon Comprehend to detect potential fraud patterns across millions of real-time financial transactions. Comprehend extracts critical text features such as transaction context, merchant type, and sentiment from unstructured financial notes, while SageMaker AI consumes this enriched dataset to train and deploy a fraud detection model that classifies transactions as legitimate or suspicious.

A newly optimized model version has recently been retrained in SageMaker AI using improved training features and hyperparameters. The data science team must evaluate the new model’s prediction accuracy and latency in production without impacting the throughput of the currently deployed model. Additionally, the deployment process must introduce minimal operational overhead and must not require any changes to the way the inference endpoint is invoked by clients.

Which of the following options will satisfy the given requirements?


**[x] Modify the existing SageMaker AI endpoint configuration by adding the new model as a ProductionVariant through the ProductionVariant API, and set a small InitialVariantWeight compared to the existing model’s ProductionVariant VariantWeight to control the percentage of traffic routed to it.**

[ ] Deploy both models in separate SageMaker endpoints and use Amazon CloudWatch metrics to compare their results in post-processing.

[ ] Register the new model version in SageMaker Model Registry and configure an event trigger in AWS Lambda to automatically swap the endpoint to the new model after initial validation.

[ ] Configure an Amazon API Gateway endpoint that splits traffic between the current and the new SageMaker endpoints for A/B testing.

> Giải thích: 

#### 1. Giải thích đáp án đúng

Đây là phương pháp **Canary Deployment** hoặc **A/B Testing** tích hợp sẵn trong SageMaker, đáp ứng hoàn hảo các ràng buộc của đề bài:

* **Không ảnh hưởng đến lưu lượng hiện tại:** Bằng cách thiết lập `InitialVariantWeight` nhỏ (ví dụ: 0.1 cho model mới và 0.9 cho model cũ), bạn chỉ điều hướng một phần nhỏ traffic thực tế sang model mới để kiểm tra độ chính xác và độ trễ (latency).
* **Không thay đổi cách gọi Endpoint:** Vì cả hai model (cũ và mới) đều nằm dưới cùng **một Endpoint duy nhất**, khách hàng (client) vẫn gọi đến một URL duy nhất mà không cần biết có bao nhiêu phiên bản model đang chạy phía sau.
* **Tối thiểu hóa chi phí vận hành (Minimal Overhead):** Bạn không cần thiết lập thêm hạ tầng như API Gateway hay Lambda. Việc quản lý chia tải được thực hiện hoàn toàn bởi cơ chế quản lý Endpoint của SageMaker.
* **Giám sát hiệu quả:** Bạn có thể so sánh trực tiếp các chỉ số `Invocations`, `ModelLatency`, và `OverheadLatency` của từng biến thể (variant) ngay trên Amazon CloudWatch.

#### 2. Tại sao các phương án còn lại sai?

* **Phương án 2 (Separate Endpoints):** Việc triển khai hai endpoint riêng biệt yêu cầu client phải thay đổi code để gọi đến cả hai nơi, hoặc bạn phải tự xây dựng một bộ điều phối traffic ở phía trước. Điều này vi phạm yêu cầu "no changes to the way the inference endpoint is invoked".
* **Phương án 3 (Auto-swap via Lambda):** Việc tráo đổi (swap) toàn bộ endpoint là kỹ thuật **Blue/Green Deployment**. Nó không cho phép bạn đánh giá model mới "trong sản xuất mà không ảnh hưởng đến throughput" một cách an toàn (vì nếu model mới lỗi, 100% traffic sẽ bị ảnh hưởng ngay lập tức). Nó cũng không hỗ trợ việc so sánh song song latency giữa hai bản.
* **Phương án 4 (API Gateway):** Mặc dù API Gateway có thể chia traffic, nhưng nó làm tăng thêm một lớp hạ tầng cần quản lý (operational overhead) và yêu cầu client phải đổi URL từ SageMaker Endpoint sang API Gateway URL.

#### 3. Notes (Giải thích dịch vụ & Lưu ý)

* **Production Variant:** Trong SageMaker, một Endpoint có thể chứa nhiều biến thể. Mỗi biến thể có thể sử dụng các loại máy chủ (instance type) và số lượng máy chủ khác nhau.
* **Variant Weight:** Là trọng số tương đối để chia tải. Traffic được chia theo tỷ lệ: . Nếu bạn muốn 10% traffic vào model mới, bạn đặt Weight mới là 1 và Weight cũ là 9.
* **Inference Pipeline:** Trong bài toán này, vì có sự kết hợp giữa **Amazon Comprehend** và **SageMaker**, thực tế bạn có thể sử dụng *Inference Pipelines* để gộp bước trích xuất đặc trưng văn bản và bước dự đoán gian lận vào một quy trình duy nhất.
* **Shadow Testing (Lưu ý nâng cao):** Nếu đề bài yêu cầu "không được sai sót trên traffic thực", AWS có tính năng **Shadow Variant**. Model mới sẽ nhận được một bản sao của traffic (copy), kết quả của nó được ghi log lại nhưng không trả về cho client. Tuy nhiên, trong câu hỏi này, cách dùng `ProductionVariant` với trọng số nhỏ là phương án tiêu chuẩn nhất.

---
### **Question 27:**

Category: AIP – AI Safety, Security, and Governance

A global e-commerce company is developing a product recommendation engine using Amazon SageMaker AI to personalize shopping experiences for millions of users. The data science team trains a deep learning model that predicts product affinity based on customer purchase history, browsing behavior, and geographic location. The model is deployed to a SageMaker real-time inference endpoint and integrated into the company’s recommendation API.

After the model goes live, the data ethics team discovers that specific product categories are being shown disproportionately to customers from certain regions. The organization must verify whether this imbalance is caused by underlying bias in the training dataset or by biased decision patterns in the deployed model predictions. The company must also generate automated reports that explain which input features most strongly influence the model’s recommendations to support transparency and compliance in personalization outcomes.

Which solution should the company implement to identify, measure, and explain potential bias across both the dataset and the model outputs?

[ ] Use SageMaker Data Wrangler to manually rebalance the dataset by filtering and transforming data before retraining the recommendation model.

**[x] Use SageMaker Clarify to detect and measure data bias, evaluate model fairness, and generate feature attribution explainability reports for compliance and transparency**

[ ] Use Amazon Personalize to automatically adjust recommendation weights in real-time to reduce category bias without performing explicit fairness evaluation.

[ ] Use SageMaker Model Monitor to track endpoint metrics such as latency, drift, and accuracy without analyzing model bias or feature influence.

> Giải thích: 

#### 1. Giải thích đáp án đúng

**SageMaker Clarify** là dịch vụ chuyên biệt của AWS được thiết kế để giải quyết chính xác các vấn đề về đạo đức AI (AI Ethics) và tính minh bạch mà công ty đang gặp phải:

* **Phát hiện Bias trong dữ liệu (Pre-training Bias):** Clarify có thể phân tích tập dữ liệu huấn luyện để xác định xem có sự mất cân bằng nào không (ví dụ: một vùng địa lý có ít dữ liệu mua sắm hơn vùng khác).
* **Đánh giá tính công bằng của mô hình (Post-training Bias):** Sau khi mô hình được huấn luyện hoặc triển khai, Clarify đo lường xem các dự đoán có thiên vị cho một nhóm đối tượng cụ thể nào không bằng các chỉ số như *Disparate Impact* hay *Statistical Parity*.
* **Giải thích tính năng (Feature Attribution):** Sử dụng thuật toán **SHAP (SHapley Additive exPlanations)**, Clarify cung cấp các báo cáo giải thích mức độ đóng góp của từng đầu vào (như vùng địa lý, lịch sử mua hàng) vào kết quả đầu ra. Điều này giúp đáp ứng yêu cầu "giải thích những tính năng nào ảnh hưởng mạnh nhất" để phục vụ kiểm toán và tính minh bạch.
* **Báo cáo tự động:** Clarify có khả năng xuất các báo cáo chi tiết, trực quan giúp các bên liên quan (như đội ngũ đạo đức dữ liệu) dễ dàng hiểu và ra quyết định.

#### 2. Tại sao các phương án còn lại sai?

* **Phương án 1 (SageMaker Data Wrangler):** Mặc dù Data Wrangler có thể dùng để làm sạch và cân bằng lại dữ liệu, nhưng nó là một bước **thực thi (action)**, không phải là một bước **phân tích và đo lường (measurement/explanation)** toàn diện. Nó không thể tự động giải thích tầm quan trọng của tính năng (feature attribution) hay đo lường bias trong các dự đoán thực tế của mô hình.
* **Phương án 3 (Amazon Personalize):** Đây là một dịch vụ đề xuất được quản lý hoàn toàn. Việc điều chỉnh trọng số (weighting) có thể giúp giảm thiên kiến một cách tạm thời, nhưng nó không cung cấp các báo cáo phân tích bias hay giải thích tính năng chi tiết để phục vụ yêu cầu tuân thủ và minh bạch như đề bài mong muốn.
* **Phương án 4 (SageMaker Model Monitor):** Dịch vụ này tập trung vào việc giám sát sức khỏe của endpoint (độ trễ, lỗi) và sự trôi dạt dữ liệu (data drift). Mặc dù nó có thể tích hợp với Clarify để theo dõi bias theo thời gian, nhưng bản thân Model Monitor (nếu không cấu hình phần Clarify) sẽ không giải thích được tầm ảnh hưởng của tính năng (feature influence).

#### 3. Notes (Giải thích dịch vụ & Lưu ý)

* **SHAP Values:** Đây là "chìa khóa" trong SageMaker Clarify để giải thích mô hình. Nó phân bổ giá trị dự đoán cho từng đặc tính đầu vào, giúp bạn trả lời câu hỏi: "Tại sao khách hàng này lại được đề xuất sản phẩm X?".
* **Bias Metrics:** Bạn nên làm quen với một số chỉ số như:
* **CI (Class Imbalance):** Đo lường sự khác biệt về số lượng mẫu giữa các nhóm trong tập dữ liệu.
* **DPL (Difference in Positive Proportions):** Đo lường sự khác biệt về tỷ lệ dự đoán tích cực (ví dụ: được đề xuất mua hàng) giữa hai nhóm (ví dụ: Thành phố A và Thành phố B).


* **Transparency & Compliance (Tính minh bạch và Tuân thủ):** Trong các ngành nhạy cảm như tài chính hay thương mại điện tử toàn cầu, việc có các báo cáo từ Clarify là bắt buộc để chứng minh rằng thuật toán không phân biệt đối xử dựa trên các thuộc tính nhạy cảm như vùng miền hay sắc tộc.

---
### **Question 28:**

Category: AIP – Implementation and Integration

A data science team is running an anti-financial crime workload using Amazon SageMaker Training and SageMaker Feature Store. The team stores transactional features inside the Feature Store offline store, and those features are periodically retrieved and consumed by SageMaker model training jobs inside a scheduled retraining pipeline. The binary classifier is used for real-time fraud detection involving regulated payment flows, but the team is consistently observing a very high count of false negatives, even though the overall model accuracy is greater than 96%. The core issue is the severe class imbalance condition, where fraud transactions represent less than one-half of one percent of the entire dataset, and historical fraud samples remain extremely limited.

The goal is to increase the model’s ability to correctly detect fraudulent cases. The team wants to directly correct this imbalance problem before running the next retraining cycle.

Which solution will increase the fraudulent case detection performance?

[ ] Enable automatic model tuning in SageMaker using Bayesian Optimization to find the best hyperparameters by running multiple training jobs. Increase the number of hyperparameter tuning jobs to explore a broader range of hyperparameter values and potentially improve model performance.

**[x] Integrate a preprocessing step that applies the Synthetic Minority Oversampling Technique (SMOTE) on the minority fraudulent transaction class only before the training run begins.**

[ ] Perform random oversampling on the non-fraudulent transactions to equalize batch sizes during training.

[ ] Enable early stopping in SageMaker to automatically halt training when the model's accuracy on the validation set no longer improves.

> Giải thích: 

#### 1. Giải thích đáp án đúng

Vấn đề cốt lõi mà đội ngũ đang gặp phải là **Class Imbalance** (Mất cân bằng lớp) cực kỳ nghiêm trọng (tỷ lệ gian lận < 0,5%). Điều này dẫn đến hiện tượng **Accuracy Paradox**: Mô hình đạt độ chính xác cao (96%) nhưng thực tế chỉ là do nó dự đoán tất cả đều là "không gian lận", dẫn đến tỷ lệ **False Negatives** (bỏ lọt tội phạm) rất cao.

* **SMOTE (Synthetic Minority Oversampling Technique):** Đây là giải pháp tiêu chuẩn để xử lý mất cân bằng lớp. Thay vì chỉ sao chép các mẫu gian lận hiện có, SMOTE tạo ra các mẫu "giả" (synthetic) mới bằng cách nội suy giữa các điểm dữ liệu của lớp thiểu số (minority class).
* **Tăng khả năng phát hiện:** Bằng cách tăng số lượng mẫu gian lận trong tập huấn luyện, mô hình sẽ có đủ dữ liệu để học được các đặc trưng và ranh giới quyết định của hành vi gian lận, từ đó cải thiện đáng kể chỉ số **Recall** (khả năng phát hiện đúng các ca gian lận thực tế).
* **Quy trình:** Bước này nên được tích hợp vào pipeline tiền xử lý (như dùng SageMaker Processing hoặc Data Wrangler) trước khi dữ liệu được đưa vào SageMaker Training.

#### 2. Tại sao các phương án còn lại sai?

* **Phương án 1 (Hyperparameter Tuning):** Tinh chỉnh tham số có thể cải thiện hiệu năng tổng thể, nhưng nếu tập dữ liệu vẫn bị mất cân bằng trầm trọng, thuật toán tối ưu hóa sẽ vẫn ưu tiên tối thiểu hóa lỗi trên lớp đa số để đạt Accuracy cao. Nó không giải quyết được nguồn gốc của vấn đề là thiếu dữ liệu lớp thiểu số.
* **Phương án 3 (Random oversampling on non-fraudulent):** Đây là một lỗi logic. Việc tăng thêm mẫu cho lớp **non-fraudulent** (lớp đa số vốn đã quá nhiều) sẽ chỉ làm tình trạng mất cân bằng trở nên tồi tệ hơn. Nếu muốn cân bằng, người ta phải *Undersampling* lớp đa số hoặc *Oversampling* lớp thiểu số.
* **Phương án 4 (Early Stopping):** Tính năng này giúp tránh Overfitting bằng cách dừng huấn luyện khi Accuracy trên tập validation không tăng nữa. Tuy nhiên, như đã phân tích, Accuracy trong trường hợp này là một thước đo sai lệch. Dừng sớm dựa trên Accuracy không giúp mô hình học được cách phát hiện gian lận tốt hơn.

#### 3. Notes (Giải thích dịch vụ & Lưu ý)

* **SageMaker Feature Store (Offline Store):** Dữ liệu giao dịch lịch sử thường được lưu ở Offline Store (dưới dạng S3/Athena). Khi lấy dữ liệu này ra để train, đây là thời điểm lý tưởng để áp dụng SMOTE.
* **Số liệu về mất cân bằng:** Trong các hệ thống tài chính thực tế, tỷ lệ gian lận thường dao động từ **0,1% đến 0,5%**. Nếu không xử lý dữ liệu, mô hình "ngây thơ" nhất (luôn đoán là Legitimate) sẽ đạt Accuracy 99,5% nhưng giá trị sử dụng bằng 0.
* **Các chỉ số thay thế Accuracy:** Trong các bài toán Fraud Detection, bạn nên sử dụng **F1-Score**, **Precision-Recall Curve**, hoặc **AUC-ROC** thay vì Accuracy để đánh giá mô hình.
* **Cost-sensitive Learning:** Ngoài SMOTE, một kỹ thuật khác có thể dùng trong SageMaker là điều chỉnh **Weight** (trọng số) của hàm mất mát, khiến mô hình bị "phạt" nặng hơn nếu dự đoán sai một ca gian lận so với dự đoán sai một ca hợp pháp.

---
### **Question 29:**

Category: AIP – Operational Efficiency and Optimization for Generative AI Applications

An AI development team at a publishing company is tasked with automating the summarization of large volumes of text, such as news articles, legal documents, and academic papers. The team needs to fine-tune a pre-trained large language model (LLM) for this task, but prefers a low-code/no-code (LCNC) solution. To ensure the summaries are accessible to a global audience, Amazon Translate is utilized to adapt the generated summaries for different regions and languages. The team now needs a solution to fine-tune the LLM with minimal manual intervention while integrating seamlessly with the existing workflow.

To meet these needs, Amazon SageMaker AI is being considered for fine-tuning the LLM, as it provides a low-code environment for experimentation. The team is focused on automating model training and fine-tuning to improve efficiency and reduce manual intervention, ensuring scalability and ease of use.

Which solution will best meet the team’s requirements?

[ ] Leverage SageMaker Script Mode to fine-tune an LLM on Amazon EC2 instances, enabling custom training scripts to optimize model performance with flexibility.

[ ] Use SageMaker Training Jobs to fine-tune an LLM deployed through a custom API endpoint, automating the model training process with scalable resources.

[ ] Utilize SageMaker Studio for fine-tuning an LLM deployed on Amazon EC2 instances, simplifying the training process with an interactive and intuitive environment.

**[x] Configure SageMaker Autopilot to fine-tune an LLM deployed via SageMaker JumpStart, streamlining model customization with automatic setup and minimal user intervention.**
> Giải thích: 

#### 1. Giải thích đáp án đúng

Đây là giải pháp hoàn hảo nhất đáp ứng các yêu cầu về **Low-code/No-code (LCNC)** và **tối thiểu hóa can thiệp thủ công**:

* **SageMaker JumpStart:** Cung cấp quyền truy cập vào các mô hình nền tảng (Foundation Models) đã được huấn luyện sẵn cho các tác vụ như tóm tắt văn bản. Nó cho phép triển khai và tinh chỉnh (fine-tuning) các mô hình này một cách nhanh chóng.
* **SageMaker Autopilot cho LLMs:** Đây là tính năng mới (Cập nhật 2024-2025) cho phép tự động hóa quy trình fine-tuning cho các mô hình ngôn ngữ lớn. Người dùng chỉ cần cung cấp tập dữ liệu (ví dụ: cặp văn bản gốc và bản tóm tắt), Autopilot sẽ tự động chọn các siêu tham số (hyperparameters) tối ưu và quản lý hạ tầng tính toán.
* **Minimal Manual Intervention:** Sự kết hợp này loại bỏ nhu cầu viết mã lập trình phức tạp để quản lý vòng đời huấn luyện, phù hợp với tiêu chí LCNC của đội ngũ phát triển.
* **Tích hợp:** Quy trình này dễ dàng kết hợp với **Amazon Translate** trong một workflow tự động để tạo ra các bản tóm tắt đa ngôn ngữ cho khán giả toàn cầu.

#### 2. Tại sao các phương án còn lại sai?

* **Phương án 1 (SageMaker Script Mode):** Đây là phương pháp dành cho các chuyên gia ML muốn kiểm soát hoàn toàn mã nguồn huấn luyện. Nó yêu cầu kỹ năng lập trình cao (High-code), hoàn toàn trái ngược với yêu cầu **low-code/no-code** của đề bài.
* **Phương án 2 (Custom API endpoint & Training Jobs):** Việc tự xây dựng Training Jobs cho một custom API endpoint đòi hỏi nhiều bước cấu hình thủ công về hạ tầng, container và mã nguồn điều phối, không tối ưu về mặt "minimal manual intervention".
* **Phương án 3 (SageMaker Studio & EC2):** Mặc dù SageMaker Studio cung cấp giao diện trực quan, nhưng việc triển khai thủ công trên EC2 và tự quản lý quá trình huấn luyện vẫn đòi hỏi nhiều thao tác kỹ thuật và kiến thức về quản lý máy chủ hơn so với giải pháp tự động hóa hoàn toàn của Autopilot.

#### 3. Notes: Xu hướng Low-code AI trên AWS (2025)

Dưới đây là những lưu ý quan trọng về các dịch vụ AI "ít dùng code" mà bạn nên ghi nhớ:

* **SageMaker JumpStart:** Hiện nay hỗ trợ hàng trăm mô hình từ các "ông lớn" như Meta (Llama), Mistral, và cả Amazon (Titan). Bạn có thể fine-tune trực tiếp từ giao diện bảng điều khiển (Console).
* **Amazon Bedrock vs. JumpStart:**
* **Bedrock:** Là dịch vụ Serverless hoàn toàn (No-code), tốt nhất cho việc gọi API và fine-tuning nhanh các mô hình đóng (Closed source).
* **JumpStart:** Cho phép bạn kiểm soát sâu hơn vào máy chủ (Instance) và phù hợp khi bạn muốn dùng các mô hình mã nguồn mở (Open source) trong môi trường SageMaker AI.


* **LLM Fine-tuning (Instruction Tuning):** Quá trình mà Autopilot thực hiện thường là *Instruction Fine-tuning*, giúp mô hình hiểu và thực hiện các yêu cầu cụ thể (như "Hãy tóm tắt văn bản này theo phong cách báo chí").
* **Amazon Translate:** Khi kết hợp với LLM, bạn thường sử dụng Translate để dịch bản tóm tắt cuối cùng thay vì dịch toàn bộ tài liệu gốc, giúp tiết kiệm chi phí và giữ được ngữ cảnh tóm tắt đồng nhất.

---
### **Question 30:**

Category: AIP – Implementation and Integration

A global e-commerce company is building an AI-powered customer service assistant to handle order inquiries, refunds, and personalized product recommendations. The assistant must understand customer questions, maintain conversation context, and update order records securely.

The company plans to use Amazon Bedrock for generative AI to manage reasoning and dialogue, and Amazon SageMaker AI to analyze historical customer interactions and optimize personalized recommendations. Customer session data and order history must be stored securely for compliance and accurate responses.

Which solution best satisfies the company’s requirements?

[ ] Use Amazon Lex V2 to build a conversational chatbot for customer interactions and store conversation transcripts in Amazon S3 for historical analysis.

**[x] Use Amazon Bedrock AgentCore to develop an AI agent capable of reasoning, planning, and executing workflows for order management. Integrate the agent with Amazon DynamoDB to store and retrieve customer session data, order history, and interaction context for each user conversation.**

[ ] Use Amazon Kendra to search for answers to product manuals and FAQs. Combined with AWS Lambda to manage refund requests and data updates.

[ ] Use Amazon Titan Text G1 for conversation handling and maintain customer session states in a Python dictionary within the application memory for short-term interactions.

> Giải thích: 

#### 1. Giải thích đáp án đúng

Đây là kiến trúc hiện đại nhất (State-of-the-art) trên AWS để xây dựng các trợ lý AI có khả năng thực thi tác vụ (Action-oriented AI):

* **Amazon Bedrock AgentCore (Agents for Amazon Bedrock):** Đây là "bộ não" của hệ thống. Các Agent này sử dụng mô hình ngôn ngữ lớn (LLM) để hiểu ý định của người dùng, tự lập kế hoạch (Reasoning/Planning) và quyết định xem cần gọi API nào để hoàn thành yêu cầu (như kiểm tra đơn hàng hay hoàn tiền).
* **Action Groups & Lambda:** Agent kết nối với các hệ thống thực tế thông qua "Action Groups". Khi người dùng muốn cập nhật đơn hàng, Agent sẽ kích hoạt AWS Lambda để thực hiện thay đổi đó một cách bảo mật.
* **Amazon DynamoDB:** Đây là lựa chọn tối ưu cho **Session Management** (Quản lý phiên làm việc). Nó cung cấp tốc độ truy xuất cực nhanh (mili giây) và khả năng mở rộng vô hạn. Lưu trữ context tại đây giúp Agent "nhớ" được những gì khách hàng đã nói ở các lượt hội thoại trước, đảm bảo tính liên quán và cá nhân hóa.
* **Bảo mật và Tuân thủ:** DynamoDB và Bedrock đều hỗ trợ mã hóa bằng AWS KMS và tích hợp IAM, đáp ứng yêu cầu lưu trữ dữ liệu đơn hàng và lịch sử tương tác một cách an toàn.

#### 2. Tại sao các phương án còn lại sai?

* **Phương án 1 (Lex V2 + S3):** Amazon Lex V2 là dịch vụ xây dựng chatbot dựa trên quy tắc (rule-based) hoặc ý định (intent) truyền thống. Mặc dù nó có thể tích hợp AI, nhưng nó không mạnh về khả năng "reasoning" (suy luận) và tự lập kế hoạch như Bedrock Agents. S3 lưu trữ transcript là để phân tích sau này, không phải là giải pháp tốt để quản lý **active session state** (trạng thái phiên đang hoạt động) trong thời gian thực.
* **Phương án 3 (Kendra + Lambda):** Amazon Kendra là dịch vụ tìm kiếm thông minh (RAG). Nó rất tốt để tìm câu trả lời trong tài liệu hướng dẫn, nhưng nó không phải là một công cụ quản lý hội thoại (conversation manager) hay tác vụ suy luận. Kiến trúc này thiếu đi thành phần cốt lõi để duy trì ngữ cảnh hội thoại (context management).
* **Phương án 4 (Titan Text G1 + Python dictionary):** Sử dụng **Python dictionary** để lưu trữ trạng thái phiên là một sai lầm nghiêm trọng về kiến trúc. Dữ liệu trong bộ nhớ (application memory) sẽ bị mất khi ứng dụng khởi động lại hoặc khi mở rộng (scaling). Nó không bền vững, không bảo mật và không thể chia sẻ giữa nhiều máy chủ khác nhau trong hệ thống e-commerce toàn cầu.

#### 3. Notes: Các dịch vụ và lưu ý quan trọng (Update 2025)

Dưới đây là những khái niệm bạn nên nắm vững khi làm việc với Generative AI trên AWS:

* **Agents for Amazon Bedrock:** Cho phép bạn xây dựng các ứng dụng có thể thực thi các tác vụ phức tạp bằng cách kết nối mô hình nền tảng (Foundation Models) với các nguồn dữ liệu của công ty và các API hệ thống.
* **Knowledge Bases for Amazon Bedrock:** Đây là giải pháp RAG (Retrieval-Augmented Generation) được quản lý hoàn toàn. Bạn có thể đẩy tài liệu FAQ hoặc hướng dẫn sản phẩm vào đây để Agent có thể trả lời dựa trên thông tin chính xác của công ty.
* **Session Persistence:** Trong AI Agent, việc lưu giữ lịch sử hội thoại (Conversation History) vào DynamoDB thường được cấu hình thông qua tham số `sessionId`. Điều này giúp duy trì trí nhớ ngắn hạn và dài hạn cho chatbot.
* **Amazon SageMaker AI Integration:** Kết quả phân tích hành vi từ SageMaker có thể được đẩy vào DynamoDB như các "User Features". Khi Agent hoạt động, nó sẽ đọc các features này từ DynamoDB để đưa ra các lời khuyên mua sắm (Product Recommendations) cực kỳ chính xác.

---
### **Question 31:**

Category: AIP – Foundation Model Integration, Data Management, and Compliance

A global e-commerce company uses Amazon Bedrock with the Amazon Titan foundation model (FM) to power a multilingual customer support chatbot. In addition to Titan FM, the company also uses Amazon Comprehend to perform entity recognition and sentiment analysis on incoming customer messages. However, the support chatbot often provides generic answers because it lacks access to the company’s proprietary order management and product documentation data stored in Amazon S3 and an internal database.

The AI developer wants to enhance the chatbot’s responses with retrieval-augmented generation (RAG) so that Titan FM can generate more accurate and contextually relevant replies based on real-time, private company data, without retraining the model.

Which of the following options will satisfy this requirement?

[ ] Fine-tune the Amazon Titan foundation model with the company’s support data using Amazon SageMaker AI.

[ ] Increase the temperature parameter for the Amazon Titan model to improve contextual understanding.

**[x] Set up a Bedrock knowledge base and integrate it with the company’s private data sources.**

[ ] Use the Comprehend custom classification to provide the model with retrieval capabilities.

> Giải thích: 

#### 1. Giải thích đáp án đúng

Đây là giải pháp trực tiếp và hiệu quả nhất để triển khai **RAG (Retrieval-Augmented Generation)** trên AWS:

* **Knowledge Bases for Amazon Bedrock:** Đây là một dịch vụ được quản lý hoàn toàn (fully managed) giúp kết nối các mô hình nền tảng (như Amazon Titan) với các nguồn dữ liệu riêng tư của công ty (S3, Databases).
* **Quy trình tự động:** Khi khách hàng đặt câu hỏi, Bedrock sẽ tự động thực hiện các bước:
1. **Chuyển đổi (Vectorization):** Chuyển câu hỏi của khách hàng thành vector.
2. **Truy xuất (Retrieval):** Tìm kiếm các đoạn thông tin liên quan nhất từ dữ liệu nội bộ trong S3.
3. **Tăng cường (Augmentation):** Gửi các đoạn thông tin này cùng với câu hỏi gốc đến mô hình Titan.
4. **Phát sinh (Generation):** Mô hình Titan tạo ra câu trả lời dựa trên ngữ cảnh thực tế của công ty (thông tin đơn hàng, tài liệu sản phẩm).


* **Không cần huấn luyện lại:** Đúng như yêu cầu của đề bài, RAG giúp mô hình có kiến thức mới mà **không cần thực hiện fine-tuning** tốn kém và phức tạp.


#### 2. Tại sao các phương án còn lại sai?

* **Phương án 1 (Fine-tune):** Fine-tuning là quá trình huấn luyện lại mô hình để thay đổi hành vi hoặc phong cách ngôn ngữ. Nó không phải là cách tốt nhất để cập nhật các dữ liệu thay đổi liên tục như trạng thái đơn hàng (order management). Đề bài cũng yêu cầu cụ thể là "without retraining the model".
* **Phương án 2 (Increase Temperature):** Tham số `temperature` điều chỉnh tính sáng tạo/ngẫu nhiên của mô hình. Tăng temperature chỉ khiến mô hình đưa ra các câu trả lời bay bổng hơn, thậm chí dễ gây ra hiện tượng "ảo tưởng" (hallucination), chứ không giúp mô hình tiếp cận được dữ liệu riêng tư trong S3.
* **Phương án 4 (Comprehend Custom Classification):** Amazon Comprehend dùng để phân loại văn bản (ví dụ: xác định đây là khiếu nại hay khen ngợi). Nó không có khả năng truy xuất dữ liệu từ S3 và cung cấp ngữ cảnh cho LLM để tạo ra câu trả lời RAG.

#### 3. Notes (Giải thích dịch vụ & Lưu ý)

* **RAG (Retrieval-Augmented Generation):** Là kỹ thuật cung cấp cho AI một "cuốn sách tra cứu" (dữ liệu nội bộ) để nó trả lời chính xác hơn, thay vì chỉ dựa vào kiến thức đã học trong quá khứ.
* **Vector Database:** Để Knowledge Base hoạt động, Bedrock cần một nơi lưu trữ dữ liệu đã được vector hóa. Các lựa chọn phổ biến là **Amazon OpenSearch Serverless**, **Pinecone**, hoặc **Amazon Aurora**.
* **Amazon Titan:** Là dòng mô hình nền tảng do AWS phát triển, bao gồm các mô hình cho văn bản (Text), nhúng (Embeddings) và hình ảnh (Image). Trong RAG, bạn thường dùng *Titan Text Embeddings* để mã hóa dữ liệu và *Titan Text* để tạo câu trả lời.
* **Tính bảo mật:** Dữ liệu được truy xuất thông qua Bedrock Knowledge Base luôn nằm trong phạm vi kiểm soát của AWS và không được dùng để huấn luyện lại các mô hình công cộng, đảm bảo an toàn cho dữ liệu doanh nghiệp.

---
### **Question 32:**

Category: AIP – Implementation and Integration

A global publishing company manages a large repository of multimedia content, including PDFs, images, audio files, and video recordings from news articles, interviews, and webinars. The company seeks to automate the extraction of insights such as key topics and relevant entities to help journalists and editors quickly access information. The company aims to build a generative AI-powered research assistant capable of answering natural language queries like, “Which article mentioned climate adaptation in Southeast Asia and included aerial images?”

The company plans to use Amazon Bedrock Data Automation (BDA) for processing unstructured media and extracting structured insights, and Amazon SageMaker AI to host the generative AI model for context‑aware response generation.

Which approach will best meet the requirements?

[ ] Utilize Bedrock Data Automation (BDA) to process media files, analyze the structured content using Amazon Comprehend for entity and sentiment extraction, and then forward the results to a foundation model hosted on Amazon EC2 for generating answers.

**[x] Leverage Bedrock Data Automation (BDA) to process documents, images, audio, and video, index the results in Bedrock Knowledge Bases for semantic search, and then input the retrieved context into a foundation model via SageMaker AI for response generation.**

[ ] Use Bedrock Data Automation (BDA) to process the media files, store the raw content in Amazon S3, and deploy a custom AWS Lambda function to create your own vector database outside of Bedrock Knowledge Bases before passing it to SageMaker AI.

[ ] Employ Bedrock Data Automation (BDA) to process all media types and directly supply the structured output into a foundation model via SageMaker AI, bypassing the use of a knowledge base.

> Giải thích: 

#### 1. Giải thích đáp án đúng

Đây là giải pháp tối ưu nhất vì nó kết hợp khả năng xử lý đa phương thức (multimodal) mạnh mẽ với kiến trúc RAG (Retrieval-Augmented Generation) hiện đại:

* **Amazon Bedrock Data Automation (BDA):** Đây là một dịch vụ mới (mạnh hơn các dịch vụ AI riêng lẻ trước đây) chuyên dùng để tự động hóa việc trích xuất dữ liệu từ các tệp không cấu trúc đa dạng (PDF, hình ảnh, âm thanh, video). BDA có khả năng hiểu ngữ cảnh của video và hình ảnh để tạo ra các mô tả và siêu dữ liệu (metadata) có cấu trúc.
* **Bedrock Knowledge Bases:** Đóng vai trò là "bộ nhớ" và công cụ tìm kiếm. Sau khi BDA trích xuất thông tin, dữ liệu này cần được lập chỉ mục (index) vào một cơ sở dữ liệu vector. Việc sử dụng Knowledge Bases giúp thực hiện **Semantic Search** (tìm kiếm theo ý nghĩa), cho phép trả lời các câu hỏi phức tạp như "bài báo nào có hình ảnh từ trên không" bằng cách khớp các mô tả ảnh do BDA tạo ra.
* **SageMaker AI Integration:** Sử dụng SageMaker AI để host mô hình ngôn ngữ lớn (LLM) giúp tùy chỉnh quy trình tạo câu trả lời (generation). Mô hình sẽ nhận ngữ cảnh đã được tìm thấy từ Knowledge Bases để trả lời người dùng một cách chính xác và tự nhiên.
* **Giải quyết bài toán đa phương thức:** Đây là cách duy nhất trong các phương án xử lý hiệu quả cả video và hình ảnh để phục vụ việc trả lời các câu hỏi cụ thể về nội dung hình ảnh.

#### 2. Tại sao các phương án còn lại sai?

* **Phương án 1:** Sử dụng **Amazon EC2** để host foundation model làm tăng đáng kể chi phí vận hành (operational overhead) so với SageMaker AI. Ngoài ra, việc dùng Comprehend sau BDA là thừa thãi vì BDA đã có khả năng trích xuất thực thể và hiểu nội dung rất sâu.
* **Phương án 3:** Yêu cầu tự tạo cơ sở dữ liệu vector bằng **AWS Lambda** bên ngoài Knowledge Bases. Điều này tạo ra sự phức tạp không cần thiết và tốn nhiều công sức bảo trì ("manual intervention"), đi ngược lại với xu hướng sử dụng các dịch vụ quản lý hoàn toàn (managed services) của AWS.
* **Phương án 4:** Bỏ qua **Knowledge Base**. Nếu không có Knowledge Base để tìm kiếm và lọc dữ liệu, mô hình AI trên SageMaker sẽ không thể "truy cập nhanh" vào hàng triệu tài liệu. Bạn không thể gửi toàn bộ kho nội dung vào một prompt duy nhất vì giới hạn cửa sổ ngữ cảnh (context window).

#### 3. Notes: Dịch vụ công nghệ mới (Update 2025)

* **Amazon Bedrock Data Automation (BDA):** Đây là một bước tiến lớn so với việc sử dụng kết hợp Rekognition + Transcribe + Textract. BDA sử dụng các mô hình nền tảng để trích xuất dữ liệu, giúp nó hiểu được các logic phức tạp trong tài liệu hoặc các hành động trong video mà các dịch vụ thế hệ cũ khó làm được.
* **Multimodal RAG (RAG đa phương thức):** Đây là xu hướng mới nhất trong năm 2025. Thay vì chỉ tìm kiếm văn bản, hệ thống giờ đây có thể tìm kiếm dựa trên nội dung hình ảnh (aerial images) và âm thanh nhờ vào việc chuyển đổi tất cả các loại media này sang cùng một không gian vector thông qua BDA.
* **SageMaker AI (tên mới):** AWS đã đồng bộ thương hiệu SageMaker AI để bao gồm cả các công cụ Generative AI, giúp tích hợp mượt mà hơn với Amazon Bedrock.
* **Semantic Search vs Keyword Search:** Trong câu hỏi về "climate adaptation", tìm kiếm ngữ nghĩa (semantic) sẽ tìm được cả những bài viết dùng từ "global warming" hoặc "environmental changes", điều mà tìm kiếm từ khóa truyền thống có thể bỏ lỡ.

---
### **Question 33:**

Category: AIP – AI Safety, Security, and Governance

A financial services organization is developing a document classification model to detect fraudulent claims from scanned forms. The team uses Amazon Textract to extract text and structured data from thousands of claim documents, then builds a training dataset in Amazon SageMaker AI. During model evaluation, the data scientists notice that the model performs very well on “legitimate claim” documents but frequently misclassifies “fraudulent claim” samples.

To understand the cause, the team uses SageMaker Clarify and observes that the pretraining bias analysis reveals a significant skew in the dataset. The team realizes this uneven class distribution leads to biased predictions and poor generalization to minority classes.

What issue is most likely causing the model’s poor performance on fraudulent claim detection?

[ ] The issue is due to overfitting, where the model has memorized training examples and performs poorly on unseen data, regardless of label distribution.

[ ] The issue is due to a high learning rate, which causes the optimization process to skip over the optimal weights during training.

[ ] The issue is due to insufficient text extraction, where Textract failed to extract all key fields, leading to missing features for the model.

**[x] The issue is due to class imbalance (CI) in the training dataset, where the minority class has too few samples, causing the model to learn biased decision boundaries.**

> Giải thích: 

#### 1. Giải thích đáp án đúng

Vấn đề này là một kịch bản kinh điển trong Machine Learning cho ngành tài chính, nơi các hành vi gian lận (fraud) thường chiếm tỷ lệ cực kỳ nhỏ so với các giao dịch hợp lệ:

* **Class Imbalance (CI - Mất cân bằng lớp):** Đây là chỉ số chính mà **SageMaker Clarify** sử dụng để đo lường sự chênh lệch số lượng mẫu giữa các nhóm (facets). Khi một lớp (ví dụ: "legitimate claims") chiếm đa số tuyệt đối, mô hình sẽ có xu hướng tối ưu hóa hàm mất mát (loss function) bằng cách dự đoán mọi thứ thuộc về lớp đa số để đạt độ chính xác (accuracy) cao nhất.
* **Biased Decision Boundaries:** Do có quá ít dữ liệu về "fraudulent claims", mô hình không học được các đặc trưng nhận dạng tinh vi của lớp này. Ranh giới quyết định (decision boundary) sẽ bị đẩy lệch về phía lớp thiểu số, dẫn đến việc bỏ sót (false negatives) rất nhiều trường hợp gian lận thực tế.
* **Dấu hiệu nhận biết:** Hiệu suất rất tốt trên lớp đa số nhưng cực kỳ kém trên lớp thiểu số là "triệu chứng" đặc trưng của mất cân bằng dữ liệu, chứ không phải do hạ tầng hay tốc độ học.

#### 2. Tại sao các phương án còn lại sai?

* **Overfitting (Quá khớp):** Nếu bị overfitting, mô hình thường đạt kết quả cực tốt trên **toàn bộ** tập huấn luyện nhưng lại kém trên **tập kiểm thử (test set)**. Ở đây, vấn đề xảy ra ngay cả khi đánh giá dựa trên sự phân bổ nhãn, và Clarify đã chỉ ra sự sai lệch (skew) trong tập dữ liệu tiền huấn luyện, điều này hướng trực tiếp tới cấu trúc dữ liệu hơn là việc học thuộc lòng.
* **High Learning Rate (Tốc độ học cao):** Tốc độ học quá cao thường dẫn đến việc mô hình không thể hội tụ (loss dao động mạnh hoặc bị nổ gradient). Nó sẽ làm giảm hiệu suất tổng thể của mô hình trên tất cả các lớp, thay vì chỉ gây ra lỗi trên một lớp cụ thể như lớp thiểu số.
* **Insufficient Text Extraction (Trích xuất văn bản thiếu):** Mặc dù việc thiếu trường dữ liệu có thể ảnh hưởng đến độ chính xác, nhưng nếu lỗi này xảy ra ngẫu nhiên trên cả hai loại tài liệu, nó sẽ không tạo ra một sự sai lệch đặc thù chỉ tập trung vào lớp gian lận như kết quả phân tích bias từ SageMaker Clarify đã chỉ ra.

#### 3. Notes: Giải thích dịch vụ & Lưu ý quan trọng

* **SageMaker Clarify:** Là công cụ cung cấp các chỉ số đo lường bias. Trong đó, **Class Imbalance (CI)** là chỉ số cơ bản nhất:

* **Amazon Textract:** Không chỉ trích xuất chữ (OCR), Textract còn nhận diện được bảng (tables) và biểu mẫu (forms). Trong bài toán bảo hiểm, việc trích xuất các ô tích hoặc chữ ký là cực kỳ quan trọng để phát hiện gian lận.
* **Cách khắc phục:** Để giải quyết vấn đề này, đội ngũ có thể áp dụng các kỹ thuật đã học ở các câu trước như **SMOTE** (tăng cường lớp thiểu số), **Undersampling** (giảm bớt lớp đa số), hoặc sử dụng **Cost-sensitive learning** (tăng trọng số lỗi cho lớp gian lận).
* **Confusion Matrix:** Để đánh giá chính xác các mô hình mất cân bằng, nhà khoa học dữ liệu nên nhìn vào **Recall** và **F1-Score** cho lớp "Fraudulent" thay vì nhìn vào tổng Accuracy.

---
### **Question 34:**

Category: AIP – AI Safety, Security, and Governance

A large healthcare organization is developing an AI-powered system to predict patient health outcomes using both historical and real-time medical data. The system will use Amazon SageMaker AI to train machine learning models on historical data stored in an on-premises Microsoft SQL Server database, and then utilize Amazon Comprehend Medical to extract relevant insights from unstructured clinical notes. To comply with strict regulatory guidelines, sensitive data such as patient records must never leave the on-premises data center, while non-sensitive data can be securely transferred to Amazon S3 for periodic model retraining. All data transfers to the cloud must be done through a secure Internet Protocol security (IPsec) connection.

The organization needs a solution to securely upload only the non-sensitive data from the MySQL database to S3 each day for retraining the model, without violating data localization regulations.

Which solution will satisfy the given requirements?

[ ] Utilize Amazon Data Firehose to stream non-sensitive transaction data into S3. Ensure that the data transfer happens over an IPsec-protected connection, and leverage AWS Lambda to filter out sensitive data before uploading.

**[x] Set up an AWS Glue job to connect to the Microsoft SQL Server database, extract only the non-sensitive data, and transfer it to S3 over an AWS Site-to-Site VPN connection for model retraining.**

[ ] Employ SageMaker AI Data Wrangler to directly connect to the Microsoft SQL Server database, filter sensitive data, and upload the sanitized data to S3 over an AWS Direct Connect connection with IPsec encryption for secure model retraining.

[ ] Configure AWS Database Migration Service (AWS DMS) to replicate non-sensitive data from the Microsoft SQL Server database into S3, and transfer it over an IPsec connection.

> Giải thích: 

#### 1. Giải thích đáp án đúng

Giải pháp này đáp ứng đầy đủ các yêu cầu khắt khe về bảo mật, kết nối và quy định về dữ liệu y tế:

* **Trích xuất có chọn lọc:** **AWS Glue** là dịch vụ tích hợp dữ liệu (ETL) mạnh mẽ. Nó có thể kết nối với Microsoft SQL Server (on-premises) thông qua JDBC, thực hiện các script để chỉ trích xuất những cột/hàng chứa dữ liệu phi nhạy cảm (non-sensitive), đảm bảo dữ liệu nhạy cảm không bao giờ rời khỏi trung tâm dữ liệu.
* **Kết nối bảo mật (IPsec):** **AWS Site-to-Site VPN** sử dụng giao thức **IPsec** để tạo một đường ống mã hóa an toàn giữa mạng on-premises của tổ chức y tế và AWS VPC. Điều này thỏa mãn yêu cầu về phương thức truyền tải dữ liệu.
* **Tự động hóa định kỳ:** Glue jobs có thể được lập lịch (scheduled) để chạy hàng ngày, tự động hóa quy trình cập nhật dữ liệu cho việc huấn luyện lại mô hình trên S3.
* **Xử lý dữ liệu không cấu trúc:** Sau khi dữ liệu được chuyển vào S3, **Amazon Comprehend Medical** có thể được tích hợp để xử lý các ghi chú lâm sàng, trích xuất thực thể y tế (như tên thuốc, mã bệnh) một cách bảo mật và tuân thủ HIPAA.

#### 2. Tại sao các phương án còn lại sai?

* **Phương án 1 (Amazon Data Firehose):** Firehose chủ yếu dùng để stream dữ liệu thời gian thực (real-time). Việc sử dụng Lambda để lọc dữ liệu *sau khi* nó đã vào cloud (trước khi ghi vào S3) có thể vi phạm quy định "sensitive data must never leave the on-premises data center" nếu không được cấu hình cực kỳ cẩn thận. Ngoài ra, việc thiết lập Firehose kết nối trực tiếp với SQL Server on-premises phức tạp hơn nhiều so với Glue.
* **Phương án 3 (SageMaker Data Wrangler & Direct Connect):** * **AWS Direct Connect** là kết nối vật lý riêng biệt, khác với yêu cầu về kết nối **IPsec** (thường đi qua Internet công cộng nhưng được mã hóa).
* Data Wrangler chủ yếu dùng để khám phá và chuẩn bị dữ liệu trong giao diện trực quan, không phải là công cụ tối ưu nhất cho các tác vụ ETL định kỳ, quy mô lớn từ database on-premises so với AWS Glue.


* **Phương án 4 (AWS DMS):** AWS Database Migration Service (DMS) thường được dùng để di chuyển toàn bộ database hoặc sao chép dữ liệu liên tục (CDC). Mặc dù nó có thể lọc dữ liệu, nhưng mục đích chính của nó là di chuyển dữ liệu (migration), trong khi AWS Glue mạnh hơn về khả năng biến đổi và tích hợp dữ liệu phức tạp cho các pipeline AI/ML.

#### 3. Notes: Các dịch vụ và lưu ý quan trọng (Update 2025)

* **Amazon Comprehend Medical:** Đây là dịch vụ NLP chuyên sâu cho ngành y tế, có khả năng tự động nhận diện và gỡ bỏ (redaction) các thông tin định danh cá nhân (PHI) để đảm bảo tuân thủ HIPAA.
* **AWS Glue Interactive Sessions:** Tính năng mới giúp các Data Scientist có thể viết và kiểm tra script Glue ngay từ notebook, giúp quy trình từ on-premises lên S3 trở nên linh hoạt hơn.
* **Data Localization (Nội địa hóa dữ liệu):** Trong ngành healthcare, việc dữ liệu nhạy cảm không được rời khỏi on-premises là yêu cầu pháp lý phổ biến (như GDPR hoặc các luật y tế địa phương). Việc lọc dữ liệu tại nguồn (tại SQL Server) bằng Glue là một kiến trúc chuẩn để đảm bảo tuân thủ.
* **Hybrid Cloud AI:** Đây là mô hình kết hợp giữa hạ tầng tại chỗ (để lưu trữ dữ liệu nhạy cảm) và điện toán đám mây (để tận dụng sức mạnh tính toán của SageMaker), giúp tối ưu hóa cả bảo mật và hiệu năng.

---
### **Question 35:**

Category: AIP – Operational Efficiency and Optimization for Generative AI Applications

A global insurance technology company built a claims automation system using Amazon SageMaker AI endpoint and Amazon Rekognition. Rekognition analyzes uploaded vehicle images to detect the severity and type of damage, while SageMaker uses this information, along with other structured features such as vehicle age, mileage, and estimated repair cost, to predict claim approval probability and estimated settlement time.

During initial training, the model assigned greater weight to the damage severity score and repair cost estimate as the most influential features for decision-making. However, recent observations suggest that the deployed model on the SageMaker AI endpoint might now be emphasizing vehicle age more heavily than damage severity, resulting in inconsistent predictions and potential policy compliance issues.

The data science team must implement a monitoring solution that detects when the feature weight or attribution importance of input variables shifts in production and automatically triggers alerts for investigation.

Which of the following should be implemented?

[ ] Deploy SageMaker Clarify to perform bias and explainability analysis on the training dataset. Use Amazon CloudWatch to alert if Clarify reports significant changes in feature attribution or fairness metrics.

[ ] Implement a baseline for model quality using the ModelQualityMonitor class. The baseline will evaluate key performance metrics such as accuracy and recall, with periodic checks to identify any significant shifts in model performance. Set up an Amazon CloudWatch if the model’s quality metrics diverge from the baseline.

[ ] Enable SageMaker DataCapture to log inference inputs and outputs. Build a custom pipeline to analyze feature distributions and model responses over time. Use Amazon CloudWatch to alert when significant shifts in input patterns or predictions are detected.

**[x] Use ModelExplainabilityMonitor class with a SHAP-based baseline to detect feature attribution drift in production. Regularly compare how the model assigns importance to input features against the baseline, and configure Amazon CloudWatch to alert stakeholders when attribution values drift beyond acceptable thresholds.**

> Giải thích: 

#### 1. Giải thích đáp án đúng

Vấn đề mà công ty bảo hiểm đang gặp phải là **Feature Attribution Drift** (Sự trôi dạt thuộc tính tính năng). Đây là tình trạng mà tầm quan trọng của các yếu tố đầu vào (ví dụ: tuổi xe vs. mức độ hư hại) bị thay đổi trong môi trường thực tế so với lúc huấn luyện.

* **ModelExplainabilityMonitor:** Đây là một thành phần của **SageMaker Model Monitor** được thiết kế riêng để theo dõi tính minh bạch và giải thích của mô hình trong sản xuất. Nó giúp trả lời câu hỏi: "Tại sao mô hình lại đưa ra quyết định này ngay lúc này?".
* **SHAP-based Baseline:** SageMaker Clarify sử dụng giá trị **SHAP (SHapley Additive exPlanations)** để gán điểm số quan trọng cho từng tính năng. Bằng cách thiết lập một "baseline" (mức cơ sở) từ tập dữ liệu huấn luyện, hệ thống có thể so sánh xem trong thực tế, mô hình có đang "ưu tiên" quá mức cho tuổi xe thay vì mức độ hư hại hay không.
* **Feature Attribution Drift Detection:** Nếu sự đóng góp của tính năng *Vehicle Age* tăng lên vượt ngưỡng cho phép so với baseline, `ModelExplainabilityMonitor` sẽ phát hiện ra sự sai lệch này.
* **Amazon CloudWatch Alerts:** Tích hợp với CloudWatch cho phép tự động gửi cảnh báo (Email/SMS) cho đội ngũ Data Science ngay khi phát hiện trôi dạt, giúp xử lý kịp thời trước khi gây ra sai sót hàng loạt về bồi thường.

#### 2. Tại sao các phương án còn lại sai?

* **Phương án 1 (Clarify on training dataset):** SageMaker Clarify thực hiện phân tích trên tập huấn luyện là để phát hiện bias *trước khi* triển khai. Tuy nhiên, đề bài yêu cầu một giải pháp giám sát **trong sản xuất (in production)** để phát hiện sự thay đổi theo thời gian.
* **Phương án 2 (ModelQualityMonitor):** Lớp này tập trung vào các chỉ số hiệu suất kỹ thuật như `Accuracy`, `Precision`, và `Recall`. Mặc dù nó cho biết mô hình có đang dự đoán sai hay không, nhưng nó **không giải thích được tại sao** (không chỉ ra được sự thay đổi trọng số của các tính năng đầu vào).
* **Phương án 3 (DataCapture & Custom Pipeline):** Mặc dù `DataCapture` là bước cần thiết để lấy dữ liệu, nhưng việc "tự xây dựng pipeline tùy chỉnh" để phân tích sự thay đổi trọng số tính năng là cực kỳ phức tạp và tốn kém nguồn lực (operational overhead), trong khi AWS đã cung cấp sẵn `ModelExplainabilityMonitor` để làm việc này một cách tự động.

#### 3. Notes: Giải thích dịch vụ & Lưu ý (Update 2025)

* **SageMaker Model Monitor:** Bao gồm 4 loại giám sát chính:
1. **Data Quality:** Phát hiện sự thay đổi trong phân phối dữ liệu đầu vào (Data Drift).
2. **Model Quality:** Theo dõi các chỉ số hiệu suất (Accuracy, v.v.) bằng cách so sánh dự đoán với nhãn thực tế thu thập được sau đó.
3. **Model Bias Drift:** Giám sát xem các chỉ số công bằng (fairness) có thay đổi theo thời gian không.
4. **Model Explainability Drift:** (Đáp án câu này) Giám sát sự thay đổi trong tầm quan trọng của các tính năng (Feature Attribution).


* **SHAP (SHapley Additive exPlanations):** Là phương pháp dựa trên lý thuyết trò chơi để giải thích kết quả của bất kỳ mô hình ML nào bằng cách phân bổ "phần thưởng" (giá trị dự đoán) cho các "người chơi" (tính năng đầu vào).
* **Inference Pipeline:** Trong hệ thống này, dữ liệu hình ảnh từ **Rekognition** chảy qua một pipeline để trích xuất điểm số hư hại, sau đó mới kết hợp với dữ liệu bảng để đưa vào **SageMaker AI**. Việc giám sát cần thực hiện ở bước cuối cùng (Predictive Model).

---
### **Question 36:**

Category: AIP – Foundation Model Integration, Data Management, and Compliance

A financial technology company manages multiple machine learning models that predict credit default risk for real-time loan applications. These models are versioned and stored in the Amazon SageMaker Model Registry, which tracks approved models for production deployment. The engineering team deploys models to a SageMaker real-time inference endpoint running on accelerated instance types, backed by Reserved Instances to optimize long-term cost efficiency.

A new model version has been approved after demonstrating improved accuracy during evaluation. However, during a previous deployment, the team observed latency spikes and request failures immediately after switching models in production. To minimize downtime and mitigate the risk of performance degradation while still validating the new model in live traffic, the team must select an appropriate deployment strategy that offers safe rollout and automatic rollback capabilities.

Which deployment configuration best meets these requirements?

[ ] Deploy both models using a multi-model endpoint configuration. Dynamically select the model version at runtime based on an API request parameter.

[ ] Use SageMaker batch transform to validate the new model offline. Promote directly to full production using a single update event.

[ ] Use a shadow testing deployment to send duplicate inference requests to the new model. Log results for later comparison, without affecting live predictions.

[ ] Configure a blue/green deployment with canary traffic shifting and a traffic size of 10%. Gradually route requests to the new model while maintaining the existing version as a fallback.

> Giải thích: 



---
### **Question 37:**

Category: AIP – Operational Efficiency and Optimization for Generative AI Applications

A publishing company uses Amazon Comprehend to analyze the sentiment and key phrases in customer feedback about its books. The company also uses a text-to-text foundation model (FM) on Amazon Bedrock to summarize and retrieve insights from thousands of customer reviews.

The company has accumulated a large volume of diverse customer feedback collected from various regions. These reviews contain casual language, local expressions, and abbreviations that differ from standard writing styles. The data science team observed that the model sometimes misinterprets these phrases, resulting in summaries that fail to fully reflect the tone and intent of the reviewers. This inconsistency affects the company’s ability to make accurate business decisions based on the generated insights.

Which solution provides the most efficient and cost-effective approach to improve the model’s understanding of customer feedback?

[ ] Use Amazon SageMaker Data Wrangler to preprocess customer feedback data, remove slang and abbreviations, and standardize the language before sending it to the Bedrock model for summarization.

[ ] Customize the current foundation model by applying fine-tuning using labeled datasets of customer feedback that reflect informal wording, abbreviations, and expressions.

[ ] Implement Custom Entity Recognition (CER) to extract slang terms and abbreviations from customer feedback and use these extracted entities as metadata inputs to Bedrock during text generation.

[ ] Launch a new large-scale training job in Amazon SageMaker AI using the model-parallelism library to build a domain-specific language model trained entirely on historical customer reviews.

> Giải thích: 



---
### **Question 38:**

Category: AIP – Operational Efficiency and Optimization for Generative AI Applications

An e-commerce platform is developing a generative AI (GenAI) application designed to improve the shopping experience. The application will handle customer requests such as personalized product recommendations, real-time language translation, and automated product descriptions. Each task will be managed by a distinct foundation model (FM). The platform is powered by AWS services such as Amazon API Gateway, AWS Lambda, and Amazon Bedrock to orchestrate these tasks, while Amazon Personalize is utilized to scale personalized recommendations across multiple regions.

The application must meet the following requirements:

+ Direct inference requests to specific foundation models (FMs) based on task type (e.g., translation, product recommendations) and customer configuration.

+ Alter the routing behavior at runtime without redeploying or restarting the system.

+ Optimize for low response times, system reliability, and cross-region operation across multiple providers.

+ Implement failover mechanisms to switch to a standby model or AWS Region in case of primary model or region failure.

Which approach delivers the needed functionality with the least operational effort?

[ ] Configure API Gateway to route requests to a Lambda function for each task type. Store model configurations in Amazon S3. Use AWS Elastic Load Balancing (ELB) to distribute traffic across regions and set up Amazon Route 53 for automatic regional failover. Update routing logic by modifying the S3 configuration file and redeploying the Lambda function to apply the changes.

[ ] Set up a Flask-based model router in Amazon ECS, with routing data stored in Amazon Aurora. Route inference requests via API Gateway to the Flask application, which selects and invokes the corresponding model using the Bedrock SDK. Set up Amazon CloudWatch alarms to monitor errors and initiate updates to the model routing table.

[ ] Create a Lambda function with AppConfig Agent Lambda extension to dynamically fetch model routing rules from AWS AppConfig. Use AWS Step Functions to manage task-specific workflows, incorporating a failover strategy with circuit breaker functionality. Ensure model invocations use Bedrock regional endpoints, retrying in a secondary region in case of failure.

[ ] Deploy a Kubernetes router on Amazon EKS, using ConfigMaps for routing rules. Forward requests from API Gateway to an Ingress controller that sends them to the router, which invokes the appropriate FM through the Bedrock SDK. Manage failover by updating ConfigMaps and restarting Pods when a Region or model fails.

> Giải thích: 



---
### **Question 39:**

Category: AIP – Foundation Model Integration, Data Management, and Compliance

A startup is training an autonomous vehicle detection model using Amazon SageMaker AI with labeled image data stored in an Amazon S3 bucket. The dataset is annotated using SageMaker Ground Truth, which contains thousands of high-resolution images.

During model training, the team observes slow startup times and low GPU utilization because the training job downloads data sequentially from S3. The company wants to keep the S3 bucket as the primary data repository but improve data access performance and training throughput for SageMaker AI without duplicating data or significantly altering the training script.

Which solution should be implemented to optimize SageMaker AI training performance while maintaining the existing S3-based workflow?

[ ] Create an Amazon FSx for Lustre file system and connect it to the existing S3 bucket. Update the SageMaker AI training job to access the dataset directly from the FSx mount.

[ ] Enable S3 Transfer Acceleration to reduce latency when downloading data during each training job.

[ ] Copy the dataset from S3 to local Amazon EBS volumes before each SageMaker AI training run to eliminate data transfer delays.

[ ] Use Amazon EFS to store the training data and mount it directly to the SageMaker AI training container for faster sequential reads.

> Giải thích: 



---
### **Question 40:**

Category: AIP – Operational Efficiency and Optimization for Generative AI Applications

A global robotics manufacturing enterprise is building a next-generation, Generative AI-powered automated quality control vision pipeline. The engineering team uses Amazon SageMaker AI to optimize embedding generation and model parameter tuning for transformer architectures specifically designed for high-speed visual defect scoring, and uses Amazon Rekognition Custom Labels to train specialized defect classifiers from thousands of labeled factory images capturing micro-scratches, misalignment, surface distortion, and abnormal texture patterns. Each production lane has an industrial PC configured with AWS IoT Greengrass and a long-running AWS Lambda function that uploads every captured image to Amazon S3. A Python-based Lambda function invokes a custom model that runs on a SageMaker endpoint, and inference results are returned to a local web service that triggers mechanical diverters to prevent defective items from reaching final shipment.

This workflow worked during the pilot with one machine, but after scaling across hundreds of units, inference latency increased beyond the acceptable SLA (Service Level Agreement) for real-time processing. Deep analysis confirms that the internet outbound throughput limit at the production facility is now saturated due to continuously streaming raw, uncompressed images to the cloud for every inference call.

Which solution is the most cost-effective fix for this performance issue while maintaining inference accuracy?

[ ] Provision a high-capacity 10 Gbps AWS Direct Connect link to the closest AWS Region for uploading the generated images and expand the SageMaker endpoint capacity by using larger instances and additional endpoint instances.

[ ] Configure IoT Greengrass to invoke the existing Lambda inference function only after batching multiple image frames together into a single S3 upload event, and increase the Lambda memory size to accelerate decompression and preprocessing before calling the SageMaker endpoint.

[ ] Enable S3 Transfer Acceleration to improve cross-region upload performance from the production site and configure automatic scaling on the SageMaker endpoint to handle higher parallel request volume generated by all industrial PCs.

[ ] Host the inference Lambda code and the ML model on the IoT Greengrass core running on each industrial PC and perform the defect detection workflow locally, then return only the reduced inference output to the local web service.

> Giải thích: 



---
### **Question 41:**

Category: AIP – Operational Efficiency and Optimization for Generative AI Applications

A global retail company is transforming its customer support system with a conversational AI chatbot built using Amazon Lex for natural language understanding and automated conversation flows. Amazon Transcribe is integrated for speech-to-text conversion, allowing customers to voice inquiries and interact naturally with the system. This setup improves the customer experience by enabling spoken communication, but the company now aims to automate the chatbot’s responses by leveraging its vast knowledge base of product documentation, FAQs, and support articles.

The solution must quickly and accurately retrieve relevant information from the company’s documentation library. It should seamlessly integrate with existing AWS tools to provide real-time, dynamic responses to customer queries, reducing the need for manual updates or custom model training.

What approach will provide the desired result with the least amount of development work?

[ ] Store the company documentation in an Amazon Bedrock Knowledge Base, then use Amazon Comprehend to analyze the customer queries and extract relevant insights from the documentation to provide accurate responses.

[ ] Use a BERT-based model in Amazon SageMaker AI, store documentation in Amazon S3, and rely on Lex to handle queries and invoke the SageMaker endpoint for responses.

[ ] Train a Bidirectional Attention Flow (BiDAF) model using customer questions and company documentation, deploy it via Amazon SageMaker AI, and integrate it with the chatbot through the SageMaker Runtime InvokeEndpoint API to provide responses.

[ ] Utilize Amazon Kendra for indexing company documents and integrate it with the chatbot through the Kendra Query API for dynamic response generation.

> Giải thích: 


---
### **Question 42:**

Category: AIP – AI Safety, Security, and Governance

An AI developer is creating a convolutional neural network (CNN) model using Amazon SageMaker AI. To preprocess the large amount of unstructured data, the developer uses Amazon Textract to extract text from scanned documents and Amazon Rekognition to analyze images for object detection. To protect sensitive data, the developer is responsible for blocking all external network access during model training to prevent any possibility of data being compromised or leaked by malicious code that may be unintentionally present in the training container.

Which of the following options is the MOST secure protection for the training job?

[ ] Encrypt the dataset using AWS Key Management Service (KMS) before processing it in Rekognition and Textract.

[ ] Configure SageMaker AI to use a private Amazon VPC endpoint for accessing Textract and Rekognition during training.

[ ] Enable SageMaker AI Model Monitor to prevent data leakage during model training.

[ ] Activate network isolation during the training job.

> Giải thích: 


---
### **Question 43:**

Category: AIP – Operational Efficiency and Optimization for Generative AI Applications

A Generative AI engineering team is developing a supervised image recognition model to accurately identify pandas from a diverse dataset of wildlife photographs. Amazon SageMaker Ground Truth is used to label 1,000 panda images and split the data into training and testing sets, reserving 100 images as a constant test set. Amazon Rekognition Custom Labels is then configured to train a custom image classification model capable of recognizing distinctive panda features such as fur patterns, posture, and environmental background.

During the evaluation phase, the engineering team observes that the image classifier misclassifies several test images. A closer analysis reveals that in more than 75% of the misclassified images, the pandas appear upside down, indicating the model’s sensitivity to image orientation. The team must enhance the model’s ability to correctly identify pandas regardless of orientation, without collecting an entirely new dataset or extensively modifying the existing training pipeline.

Which approach will most effectively enhance the model’s accuracy in addressing this specific misclassification issue?

[ ] Apply transfer learning techniques to reuse a proven vision model’s base layers while retraining task-specific layers for panda identification.

[ ] Expand the existing training dataset by introducing data augmentation techniques such as image rotation, flipping, and scaling.

[ ] Raise the number of training epochs to prolong optimization and reinforce feature representation within the existing dataset.

[ ] Implement normalization preprocessing steps to make all images share a common scale and brightness distribution.

> Giải thích: 



---
### **Question 44:**

Category: AIP – Implementation and Integration

A multinational company is gathering a diverse set of multimedia data in various languages, including Spanish, from customer interactions, video recordings, and written documents. The company operates in multiple countries and needs to extract valuable insights by converting audio and video content into text, translating it into English, and summarizing it efficiently. The company already uses Amazon Kendra for indexing and searching large volumes of documents and Amazon Textract to extract text from scanned documents. These services help the company manage and process content in different languages, but an efficient solution is still needed to translate and summarize the content for a global audience.

The company is now looking for a solution that can process audio and video data, translate it into English, and summarize it quickly using a large language model (LLM). The solution should minimize deployment time, ensuring that it can scale efficiently to meet the company’s global needs.

Which option will best fulfill these requirements in the shortest time possible?

[ ] Use AWS Glue to clean and prepare the data, then use Amazon Translate to translate the data into English, and summarize the content using Amazon Lex to create a conversational summary.

[ ] Leverage Amazon Translate to translate the text into English, apply a pre-trained model in Amazon SageMaker AI for analysis, and summarize the content using the Claude Anthropic model in Amazon Bedrock.

[ ] Utilize Amazon Transcribe for audio and video-to-text conversion, Amazon Translate for translating the content into English, and Amazon Bedrock with the Jamba model for summarizing the text.

[ ] Train a custom model in Amazon SageMaker AI to process the data into English, then deploy an LLM in SageMaker AI for summarizing the content.

> Giải thích: 




---
### **Question 45:**

Category: AIP – Foundation Model Integration, Data Management, and Compliance

A team of AI engineers is developing a churn prediction model using Amazon SageMaker AI. The team imports customer interaction and billing data into SageMaker Data Wrangler for data preparation and feature engineering. During exploratory data analysis, the team discovered that only about 8% of customers in the dataset have churned, causing a significant class imbalance.

The team plans to launch a SageMaker training job using an XGBoost model to predict churn. However, initial training results show that the model consistently predicts the majority “non-churn” class, leading to poor recall for churned customers. The data scientist must address the imbalance to improve model performance and ensure the training dataset adequately represents both classes.

Which approach should the team take to resolve this issue before starting the SageMaker training job?

[ ] Apply a Random Undersampling in SageMaker Data Wrangler to remove samples from the majority non-churn class before training the model.

[ ] Enable SageMaker Model Monitor to detect data drift and class imbalance after deployment, and use its reports to adjust the model’s prediction thresholds manually.

[ ] Use SageMaker Clarify to analyze the class imbalance and generate bias metrics. Document the imbalance findings before retraining the model.

[ ] Perform the Synthetic Minority Oversampling Technique (SMOTE) in SageMaker Data Wrangler to rebalance the churn dataset before running the SageMaker training job.

> Giải thích: 


---
### **Question 46:**

Category: AIP – AI Safety, Security, and Governance

A financial services enterprise is standardizing on Amazon SageMaker AI for feature engineering and model lifecycle, while Amazon Fraud Detector scores real-time card transactions using model endpoints hosted on SageMaker AI. Data scientists routinely train and tune machine learning (ML) models on highly confidential payment data inside a dedicated VPC.

The security team must prevent data exfiltration paths from notebooks, training jobs, and hosted endpoints, especially through public internet routes or unauthorized client locations, without breaking internal access for analysts on the corporate network.

Which solution will meet this requirement while minimizing data egress from SageMaker AI? (Select THREE.)

[ ] Attach a NAT gateway to the SageMaker AI subnets to allow outbound internet access for package installations during training and inference.
[ ] Use SageMaker AI Lifecycle Configurations to install custom packages and configure notebook environments.
[ ] Route all SageMaker AI API/Runtime calls through VPC interface endpoints (AWS PrivateLink).
[ ] Run training jobs and models with network isolation enabled (EnableNetworkIsolation=true).
[ ] Enforce IAM policies that restrict notebook presigned URLs to approved corporate IP ranges using aws:SourceIp and related conditions.
[ ] Implement Amazon CloudWatch Logs ingestion for all SageMaker notebook and training job activities and set up AWS Security Hub alerts when unusual outbound network egress occurs.

> Giải thích: 



---
### **Question 47:**

Category: AIP – AI Safety, Security, and Governance

A developer is working on an advanced machine learning project using Amazon SageMaker AI. The project involves training a deep learning model using large datasets stored in Amazon S3. Additionally, the developer needs to store model artifacts, logs, and evaluation results back to a different S3 bucket upon completion of the training job. To ensure security and proper access control, the developer must grant the SageMaker notebook instance appropriate permissions to read from and write to the specific S3 buckets.

Which approach should be used to securely enable this access?


[ ] Define a bucket policy on the S3 bucket that allows the SageMaker AI notebook instance by its ARN to perform `s3:GetObject`, `s3:PutObject`, and `s3:ListBucket` actions.

[ ] Use AWS IAM identity federation to provide temporary access to the S3 bucket by configuring the SageMaker notebook instance to assume a federated role for accessing the data.

[ ] Create an S3 access point for the SageMaker notebook instance, granting it access to the necessary data, and configure the access point to allow only the required actions (`s3:GetObject`, `s3:PutObject`, and `s3:ListBucket`).

[ ] Allow the SageMaker notebook instance to perform `s3:GetObject`, `s3:PutObject`, and `s3:ListBucket` operations by attaching a policy to its associated IAM role that grants access to the designated S3 buckets.

> Giải thích: 


---
### **Question 48:**

Category: AIP – AI Safety, Security, and Governance

An enterprise is currently leveraging Amazon SageMaker Feature Store for cross-model reusable feature lineage and SageMaker Clarify for model bias monitoring pre-deployment. A new model is now planned to be deployed using a SageMaker AI endpoint. The enterprise security team mandates that all inference traffic must remain inside private subnets, so the SageMaker AI endpoint must use a VPC configuration with no public internet route. Expected request payload sizes are consistently 6 MB to 11 MB, and model inference execution time during operational peak can take 18–22 minutes per request. The finance engineering team also requires that costs must remain minimal while still supporting request/response style inference.

Which solution is the most suitable approach to satisfy this requirement?

[ ] Deploy a SageMaker asynchronous endpoint inside private subnets and include VPC configuration parameters during endpoint creation.

[ ] Use SageMaker multi-model endpoint architecture inside private subnets with VPC configuration applied as part of endpoint deployment procedures.

[ ] Configure a SageMaker Batch Transform job to run within private subnets and attach the appropriate VPC configuration parameters during endpoint creation.

[ ] Use SageMaker Neo compiled model packaging and deploy the compiled artifact to a SageMaker real-time inference endpoint inside private subnets using VPC configuration.

> Giải thích: 



---
### **Question 49:**

Category: AIP – AI Safety, Security, and Governance

A financial services company is developing a machine learning (ML) model in Amazon SageMaker AI to detect real-time fraudulent transactions. The company plans to integrate this model with Amazon Kinesis Data Streams for continuous transaction data ingestion and Amazon DynamoDB for storing transaction metadata.

The model must be trained on an extensive historical dataset stored in Amazon S3. The company must ensure that all data is encrypted in transit and at rest, and that access to sensitive data and model results is strictly controlled. The company also wants to track model performance over time to ensure that the model remains effective and that predictions align with business rules.

Which solution will secure the ML workflow, control access, and allow for continuous monitoring of model performance?

[ ] Utilize SageMaker AI to train the model with data from S3, store the model and metadata in DynamoDB, and configure Amazon VPC endpoints for secure communication between the SageMaker AI instance, DynamoDB, and Kinesis Data Streams. Use AWS CloudTrail to monitor access and activity.

[ ] Use SageMaker AI to train the model with data from S3, configure Amazon Data Firehose to ingest the transaction data into SageMaker AI, and use AWS Glue to catalog the transaction data. Use Amazon CloudWatch for model performance monitoring and IAM policies to restrict access.

[ ] Use SageMaker AI to train the model with data from S3, enable S3 server-side encryption (SSE-S3) for data at rest, store model results in S3 with IAM roles to control access, and use Amazon Macie for monitoring sensitive data exposure.

[ ] Use SageMaker AI to train the model with data from S3, enable S3 server-side encryption (SSE-KMS) for data at rest, configure IAM roles to control access to the model and data, and use Amazon CloudWatch to monitor model performance metrics and logging.

> Giải thích: 


---
### **Question 50:**

Category: AIP – Foundation Model Integration, Data Management, and Compliance

A sporting goods manufacturer uses Amazon SageMaker AI Canvas for automated time-series feature engineering and SageMaker AI Studio notebooks for custom modeling. To complement a legacy forecasting stack built on traditional exponential smoothing (ETS) models, the analytics team stores weekly sales data by SKU, color, and size variants in Amazon S3 and enriches it with promotional and holiday indicators.

With the introduction of a brand-new product variant that has no direct sales history, the forecasting initiative requires an approach capable of learning shared patterns across existing SKUs and producing reliable demand predictions for the new variant despite sparse or zero historical data.

Which approach best satisfies these requirements?

[ ] Use SageMaker AI to train the built-in DeepAR algorithm across all related SKUs and then generate a forecast for the new variant.

[ ] Use SageMaker AI to train a Linear Learner regression model using historical sales data as features and forecast values as labels for all SKUs.

[ ] Use SageMaker AI to train a Random Cut Forest (RCF) model to detect anomalies in historical sales data and project future demand levels for the new variant.

[ ] Use SageMaker AI to train a K-means clustering model to group similar SKUs and infer demand patterns for the new variant based on the nearest cluster.

> Giải thích


---
### **Question 51:**

Category: AIP – Foundation Model Integration, Data Management, and Compliance
A global manufacturing company operates factories in remote regions where reliable internet access is often unavailable. The company wants to deploy a machine learning (ML) solution to detect the dimensions of packages.

The company has collected thousands of hours of video footage from the production lines, which will be used for training an ML model. To facilitate this, the company plans to leverage Amazon SageMaker AI for model training and Amazon Rekognition to assist with initial video labeling and annotation for training data preparation. Given the remote locations of the factories, the company needs a deployment strategy that ensures the ML model can make real-time decisions regarding package routing without relying on constant cloud connectivity.

Which of the following solutions would best meet the company’s needs?

Use SageMaker’s built-in Object Detection algorithm to train the model. Deploy the trained model to an AWS IoT Greengrass core with AWS Lambda handling the decision logic at the factory.
Deploy a Convolutional Neural Network (CNN) in SageMaker AI using Amazon Kinesis Video Streams to analyze the video footage in real time. Use Amazon EventBridge to trigger downstream actions for routing packages based on the detected dimensions.
Train the model using SageMaker AI and deploy it to Amazon Elastic Kubernetes Service (Amazon EKS) clusters running in each factory. Use Amazon SQS to queue routing decisions and send them to the cloud for processing.
Use Rekognition Custom Labels to train the model and deploy it using Amazon EC2 instances at each factory. Use Amazon EventBridge to monitor inference results and trigger routing actions.

---
### **Question 52:**

Category: AIP – AI Safety, Security, and Governance
An ML engineering team is building a secure, multi-service pipeline that uses Amazon SageMaker AI for model training and Amazon Comprehend for entity extraction. The pipeline is triggered by Amazon EventBridge and orchestrated using AWS Step Functions.

To comply with internal security policies, the team provisions a VPC interface endpoint for the SageMaker AI Service API within a single public subnet of the Amazon VPC. The goal is to ensure that only specific Amazon EC2 instances and IAM users can invoke SageMaker API operations through these instances.

Which combination of actions should the team take to secure the traffic to the SageMaker Service API? (Select TWO.)

Attach a custom VPC endpoint policy that explicitly grants access to selected IAM identities.
Enable private DNS for the VPC endpoint to ensure that traffic remains within the VPC.
Deploy an additional VPC endpoint for SageMaker AI Runtime to isolate inference traffic.
Configure the security group linked to the endpoint network interface to allow traffic only from approved instances.
Enable VPC Flow Logs to monitor traffic patterns. Use AWS Lambda to automatically block unauthorized access to the SageMaker API endpoint.

---
### **Question 53:**

Category: AIP – Implementation and Integration
A cloud development team is using Amazon Q Developer, a generative AI-powered conversational assistant, to accelerate work on AWS applications. The team wants Q Developer to deliver more dynamic and contextually relevant responses by accessing live data from databases and external APIs that provide up-to-date information.

To ensure that Q Developer receives real-time and dynamic contextual data during conversations, the team has decided to use the Model Context Protocol (MCP). A solution is needed that can easily connect to these data sources and enable low-latency access, providing the most relevant information to the assistant during interactions.

Which solution will meet this requirement?

Set up an API gateway to connect the databases and the external APIs.
Configure the MCP server to connect to the databases and external APIs.
Use Amazon Quick Suite to integrate with the databases and the external APIs.
Utilize Amazon Q Developer CLI with MCP to connect to the databases and the external APIs.

---
### **Question 54:**

Category: AIP – AI Safety, Security, and Governance
A data science team is leveraging Amazon SageMaker AI to build and deploy machine learning models for predictive analytics. The team also uses Amazon Comprehend to perform natural language processing (NLP) on text data as part of their analysis pipeline. The SageMaker notebook instances are deployed within an isolated Amazon VPC to ensure that the development work is secure. VPC interface endpoints were set up to establish private connectivity between the VPC and the SageMaker API. The team later discovers that unauthorized users from outside the VPC can still access the notebook instances through the internet, raising security concerns.

How can the team limit access to the SageMaker notebook instances, ensuring only authorized VPC users can connect?


Configure an IAM policy that allows sagemaker:CreatePresignedNotebookInstanceUrl and sagemaker:DescribeNotebookInstance actions exclusively from VPC interface endpoints. Ensure this policy is applied to the appropriate IAM users, groups, and roles.
Apply VPC Endpoint Policies to control which IAM users or services can access SageMaker AI through the VPC interface endpoint, providing more granular access control for interactions with SageMaker AI.
Update the security group for the notebook instances to restrict incoming traffic to only the CIDR blocks associated with the VPC. Apply this security group across all interfaces linked to the SageMaker notebook instances.
Set up VPC Traffic Mirroring to capture traffic to and from the notebook instances and identify unauthorized access attempts, enabling enhanced monitoring.


---
### **Question 55:**

Category: AIP – Operational Efficiency and Optimization for Generative AI Applications
An AI developer is building a fraud detection model using Amazon SageMaker Autopilot to generate and evaluate candidate models automatically. To further refine model performance, the developer integrates automatic model tuning (AMT) in SageMaker AI to optimize several hyperparameters related to model learning rate, batch size, and regularization strength.

During multiple tuning jobs, the developer notices that some training jobs run unnecessarily long when the model’s validation accuracy stops improving early in the process. The tuning configuration must be adjusted appropriately to optimize resource usage and reduce overall tuning time, allowing SageMaker AI to decide automatically when to stop poorly performing training jobs.

Which configuration step should be taken to address this requirement?

Modify the objective metric in the tuning job definition to use a stricter validation threshold, ensuring underperforming models are ignored automatically.

Enable early stopping by setting the TrainingJobEarlyStoppingType parameter to the AUTO value in the tuning job configuration.
Configure the tuning strategy to use Bayesian optimization, ensuring that all training jobs complete fully before evaluating results.

Increase the MaxRuntimeInSeconds parameter in the tuning job configuration to allow more time for underperforming training jobs to complete.


---
### **Question 56:**

Category: AIP – Operational Efficiency and Optimization for Generative AI Applications
A global hotel chain operates across multiple regions, collecting vast amounts of customer interaction and booking data. The platform uses Amazon Lex to power an AI-driven chatbot that handles booking requests, customer inquiries, and preferences. Amazon Comprehend is integrated to process and analyze customer feedback and booking reviews to enhance the accuracy of the chatbot’s responses. The data, collected from various sources such as web activity logs, booking histories, and customer reviews, is continuously streamed into AWS.

The company seeks to automate the process of identifying high-demand rooms in real-time and providing up-to-date visual insights into booking trends for hotel managers. The solution must stream the booking data to Amazon S3 in near real-time, apply machine learning models to detect demand fluctuations and forecast trends, and provide an automated visualization dashboard that continuously displays the most current and accurate insights as new data flows in, reflecting the latest room demand and booking patterns.

Which approach delivers the desired setup with the least development time?

Use Amazon Data Firehose to stream the booking data into Amazon S3, process the data with AWS Glue, and detect high-demand outliers using the Random Cut Forest (RCF) model in Amazon SageMaker AI. Visualize the results in Amazon QuickSight.
Push booking data to S3 with Amazon Kinesis Data Streams, use a Random Cut Forest (RCF) model in Amazon SageMaker AI to detect demand anomalies, and visualize the results in Amazon QuickSight.
Stream booking data to S3 using Amazon Kinesis Data Streams, process the data with Amazon Athena, and apply AWS Glue for data enrichment. Use Amazon QuickSight to visualize demand trends and anomalies.
Utilize Amazon Data Firehose for direct streaming of booking data to S3 and employ Amazon QuickSight ML Insights for anomaly detection, followed by visualizing the insights in QuickSight.


---
### **Question 57:**

Category: AIP – Testing, Validation, and Troubleshooting
A data analytics company is developing a personalized news recommendation platform that delivers tailored article suggestions to readers. The AI development team uses Amazon Comprehend to perform sentiment analysis and entity recognition on articles and reader feedback data. The extracted insights are then stored in Amazon S3 and used by an Amazon Personalize solution to generate ranked recommendations for each user session.

The company has trained multiple recommendation models to improve model accuracy and wants to evaluate the effectiveness through A/B testing in a beta production environment. The system must route live inference traffic between multiple model variants, monitor real-time engagement metrics, and seamlessly direct 100% of traffic to the best-performing model.

Which solution will meet these requirements in the most operationally efficient way?

Use AWS CodeDeploy with blue/green deployment strategies and an Application Load Balancer (ALB) to alternate traffic between model versions during A/B testing. Gradually route 100% of traffic to the model with the highest engagement metrics.
Deploy the models on Amazon EC2 instances behind an Application Load Balancer (ALB) to perform A/B testing, then manually adjust the ALB weights when a model shows higher engagement.
Create a separate Amazon SageMaker AI endpoint for each model and configure Amazon API Gateway to distribute traffic for A/B testing based on weighted routing rules.
Use Amazon SageMaker AI multi-variant endpoints to deploy all model versions behind a single endpoint. Configure traffic weights for A/B testing and update routing to send all inference requests to the best-performing model once identified.

---
### **Question 58:**

Category: AIP – Foundation Model Integration, Data Management, and Compliance
A financial analytics company is building a customer-facing chatbot that uses a large language model (LLM) to answer domain-specific questions about investment products and compliance policies. The company has thousands of PDF documents stored in an Amazon S3 bucket that contain detailed policy explanations and financial regulations.

The AI engineering team wants the chatbot to retrieve precise answers from these documents instead of relying solely on the LLM’s pre-trained knowledge. The goal is to implement a Retrieval-Augmented Generation (RAG) approach to ground the model’s responses with proprietary data. The team is evaluating whether to use Amazon Kendra or Amazon Bedrock to manage this retrieval pipeline efficiently while maintaining scalability and minimal infrastructure overhead.

Which solution provides the most effective and AWS-managed way to integrate proprietary document retrieval with an LLM for this RAG-based chatbot?

Deploy Kendra as an independent search engine to index the documents in the S3 bucket. Configure the LLM to query Kendra’s search results directly for every user request.
Configure a knowledge base in Bedrock. Add the S3 bucket as the connected data source, and utilize the Bedrock API to perform RAG queries that dynamically combine document retrieval with LLM generation.
Fine-tune the LLM in Bedrock on the text extracted from the S3 bucket so that the model permanently learns the organization’s policies and eliminates the need for document retrieval.
Use Kendra to extract document embeddings then store it manually in an Amazon DynamoDB table. Query the embeddings from a custom inference endpoint for every RAG request.


---
### **Question 59:**

Category: AIP – Implementation and Integration
A company manages a large-scale analytics platform that stores high-volume transaction data in a PostgreSQL database. The dataset includes hundreds of millions of records with customer details, purchase amounts, product categories, and regions. The analytics team wants to use Amazon SageMaker AI to predict which customers are likely to churn in the next 90 days and improve retention strategies.

The company needs a solution that automates data extraction and preparation from the PostgreSQL database for scheduled cleaning, normalization, and handling of missing values. After refining the data, the team considers using Amazon SageMaker Feature Store to store and manage customer features for consistent inputs in machine learning workflows. Analysts should then be able to build and deploy a no-code churn-prediction model and easily generate predictions on new customer data without support from data scientists.

Which is the best option to achieve this requirement?

Use AWS Glue DataBrew to extract the data from PostgreSQL, clean and normalize the dataset, and write the prepared data back into Amazon S3. Import the cleaned dataset into Amazon SageMaker Canvas to build a no-code churn-prediction model and generate predictions for business analysts.
Use AWS Glue DataBrew to prepare and clean the data from the PostgreSQL database. Use Amazon SageMaker Studio to build, train, and deploy a custom churn-prediction model using a notebook-based workflow for data scientists.
Use AWS Glue DataBrew to extract the data and build the churn-prediction model directly within DataBrew. Generate predictions that are written into Amazon Redshift.
Use AWS Database Migration Service (AWS DMS) to replicate data from the PostgreSQL database into Amazon S3 continuously. Use AWS Glue DataBrew to clean and normalize the replicated dataset before importing it into Amazon SageMaker Canvas to build and deploy the churn-prediction model.


---
### **Question 60:**

Category: AIP – Implementation and Integration
A publishing company is developing an internal generative AI platform to automate the creation of research article drafts and editorial summaries. The company uses Amazon Comprehend for text preprocessing and data cleaning, and Amazon Bedrock Agents to automate content retrieval, citation validation, and orchestration of large language model (LLM) tasks across the editorial workflow. An AI developer has recently built and fine-tuned a large language model (LLM) outside of Amazon SageMaker AI and stored the model artifacts in an Amazon S3 bucket for internal use.

The AI developer wants to make the model available to the data specialist team through SageMaker Canvas, allowing the data specialists to experiment with the model’s text-generation capabilities through a no-code interface. The AI developer and the data specialist team belong to the same SageMaker AI domain, and the AI developer must ensure the model is properly registered and accessible within SageMaker AI.

Which combination of steps must be taken for the AI developer to enable SageMaker Canvas access to the model? (Select TWO.)

The AI developer must register the model in the SageMaker Model Registry to enable the data specialist team's access via SageMaker Canvas.
The AI developer must convert the model into a TensorFlow or PyTorch format for SageMaker Canvas compatibility.
The data specialist team must create a shared workspace within SageMaker Canvas that allows both the AI developer and data specialists to access the model.
The AI developer is required to set up a SageMaker endpoint for the model.
The data specialist team must be granted the necessary permissions to access the S3 bucket where the model artifacts are stored.


---
### **Question 61:**

Category: AIP – Foundation Model Integration, Data Management, and Compliance
A large-scale e-commerce platform is developing an AI-powered email filtering system that automatically identifies and flags malicious, spam, and phishing emails before reaching end users. The AI development team will implement this solution using Amazon SageMaker AI for model training and deployment, taking advantage of SageMaker’s managed environment for building and fine-tuning machine learning models. The system will rely on Amazon Comprehend to extract key entities and sentiment from incoming emails, helping the model identify potential threats. The training dataset consists of several thousand human-labeled emails, where each email is paired with a label of “spam” or “not spam”.

The AI development team aims to improve the model’s performance by applying transfer learning using a Bidirectional Encoder Representations from Transformers (BERT) model that was pretrained on a large corpus of text data. The goal is to fine-tune the pretrained BERT model on the labeled email dataset, which will allow the model to learn how to better classify email content. The pretrained BERT model weights need to be correctly loaded and used as the initialization point for fine-tuning, ensuring the model can effectively detect spam without requiring a full retraining process.

Which approach will correctly initialize the BERT model to achieve this requirement?

Initialize the model with pretrained weights, convert the output layer into a multi-task classifier that predicts multiple text classes beyond spam detection, and train this classifier using the labeled dataset.
Apply pretrained model parameters across all layers, then discard the existing final layer. Introduce a custom classifier and train it using the labeled data for spam detection.
Load the pretrained model weights for every layer and place an external classifier on top of the primary model output vector. Train the newly added classifier with the labeled dataset.
Use the pretrained model weights for all transformer layers and attach a second classifier layer in parallel with the existing output layer. Train only this additional classifier using the labeled dataset.




---
### **Question 62:**


Category: AIP – Operational Efficiency and Optimization for Generative AI Applications
A global e-commerce platform utilizes Amazon SageMaker AI to implement a real-time recommendation system that provides personalized product suggestions. The model is developed using SageMaker Studio. Once trained, the model is hosted on a SageMaker endpoint, enabling it to deliver inferences in real time. Additionally, the company employs Amazon Comprehend to analyze customer feedback and sentiment derived from reviews and product ratings. This analysis helps the company better understand customer preferences and enhance its recommendations.

As the business prepares for major sales events, the operations team observes that increased customer activity results in significant delays when retrieving product recommendations, ultimately leading to a poor user experience. To address this issue, there is a need to adjust the target tracking scaling policy on the SageMaker endpoint. The goal is to ensure effective scaling during high-traffic periods, preventing latency from adversely affecting the customer experience.

Which solution will best optimize the scaling of the SageMaker inference endpoint?

Use AWS Lambda to periodically restart the SageMaker endpoint during peak traffic to refresh instance performance.
Configure a scheduled scaling policy to increase the capacity of the SageMaker inference endpoint before the sales events begin.
Implement a step scaling policy for the SageMaker inference endpoint that scales based on resource utilization metrics such as CPU and memory usage.
Increase the instance size of the SageMaker endpoint to a larger instance type to accommodate higher traffic during sales events.



---
### **Question 63:**


Category: AIP – Operational Efficiency and Optimization for Generative AI Applications
A data science team is developing an automated pipeline that integrates Amazon SageMaker AI and Amazon Comprehend to analyze large volumes of customer feedback data. The team first uses a PySpark script to perform text preprocessing, tokenization, and feature engineering on millions of feedback records stored in Amazon S3. The preprocessed data is then passed to Comprehend for sentiment and entity extraction, and subsequently used to train multiple sentiment classification models in SageMaker AI.

The AI developer must determine how varying the PySpark feature transformation parameters and sample sizes affects overall model accuracy and inference performance.

Which solution will meet this requirement most effectively?

Use SageMaker Experiments tracker to log PySpark parameters and model metrics while executing the script as a SageMaker processing job.
Use SageMaker Autopilot to automatically choose the best PySpark preprocessing configuration and feature-engineering parameters for model optimization.
Use SageMaker Debugger hook to capture feature engineering metrics and execution logs during PySpark script execution within a SageMaker training job.
Use SageMaker Model Monitor to detect differences in PySpark data transformation parameters before each training iteration.




---
### **Question 64:**

Category: AIP – Operational Efficiency and Optimization for Generative AI Applications
An AI engineer trains a Convolutional Neural Network (CNN)–based computer vision model to generate detailed object representations for a custom dataset. The dataset consists of millions of high-resolution labeled images stored in Amazon S3. The engineer uses Amazon Rekognition for feature extraction on raw images before feeding it into an Amazon SageMaker training job for model fine-tuning.

During training, the engineer observed that model training took significantly longer than expected due to slow data reads from S3. The training job uses an Amazon EC2 On-Demand Instance and currently accesses data using File mode.

The engineer wants to improve I/O performance during training without modifying the model architecture, data preprocessing scripts, or the training infrastructure.

Which action should the engineer take to optimize training performance most efficiently?

Increase the instance size of the EC2 On-Demand Instance to a larger GPU type for higher throughput.
Set the SageMaker training job to Pipe mode instead of File mode to enhance training throughput.
Change the SageMaker training job’s data input configuration to FastFile mode to stream data directly from S3 without other changes.
Use Rekognition Custom Labels to optimize image access and retrain the model.






---
### **Question 65:**


Category: AIP – Foundation Model Integration, Data Management, and Compliance
A global e-commerce company has deployed a customer service chatbot powered by Amazon Bedrock for real-time query processing. The chatbot uses Amazon SageMaker AI for model training and inference and utilizes a cross-region inference profile to optimize throughput. The company must ensure that input prompts and output results are transmitted securely across regions, with data remaining within the designated geography (the US) to meet compliance requirements. Additionally, the system must stay scalable and responsive during peak traffic bursts, leveraging AWS services to handle traffic fluctuations and ensure consistent performance during high-demand periods.

Which approach should satisfy the requirement?

Configure Bedrock with a cross-region inference profile tied to the US geography and use Amazon Comprehend to analyze and categorize customer queries before routing the inference request, ensuring that only relevant queries are processed for improved throughput.
Implement a cross-region inference profile tied to the US geography and route requests to destination Regions within the US. Utilize SageMaker AI to handle model inference at peak times by enabling Provisioned Throughput, ensuring consistent performance during traffic bursts.
Use Bedrock with a global cross-region inference profile, allowing the system to route requests to the best-performing AWS Regions worldwide, increasing throughput and enhancing model performance. Use Amazon Kendra for query-based search to enhance user experience and ensure relevant responses.
Use Bedrock with a cross-region inference profile tied to the US geography, ensuring that requests from US Regions are routed to the optimal destination Regions within the US, while ensuring that input prompts and output results remain encrypted during transmission across Regions. Additionally, use AWS Step Functions to manage traffic bursts and ensure that service quotas are not exceeded.





---
### **Question 66:**


Category: AIP – Implementation and Integration
A financial technology company manages a large-scale payment platform that processes millions of transactions daily. The company uses Amazon Comprehend to analyze unstructured text data from customer reviews and support tickets to identify sentiment trends and potential user disputes. Amazon SageMaker AI is also used to generate periodic behavior-based risk scores for customer accounts.

Recently, the company has noticed an increasing number of fraudulent transactions, especially from newly registered accounts attempting high-value payments. The existing batch-based model in SageMaker cannot flag these activities quickly enough to prevent losses.

The data science team is looking for a real-time fraud detection solution that can automatically assess and reject fraudulent transactions at the moment of occurrence. The solution must require minimal operational effort.

Which option satisfies these requirements?

Use Amazon Lookout for Vision to detect anomalies in uploaded transaction receipt images and classify them as fraudulent or legitimate.
Use the Amazon Fraud Detector prediction API to automatically approve or deny transactions that are identified as fraudulent.
Use SageMaker AI to train a new supervised model for fraud detection and deploy it on Amazon EC2 using custom inference code.
Use Comprehend to extract entities from transaction metadata and forward them to SageMaker AI to retrain a fraud detection model.




---
### **Question 67:**


Category: AIP – Foundation Model Integration, Data Management, and Compliance
A company operates a customer support chatbot that uses Amazon Bedrock to send user queries to an Amazon Nova Pro large language model (LLM) for generating conversational responses. The chatbot is integrated with Amazon Kendra to retrieve relevant knowledge base articles, which are appended to the prompt before the request is sent to the model.

Users have reported that when similar questions are asked multiple times, different responses are sometimes returned, even though the retrieved Kendra results remain unchanged. The generative AI developer must configure the system to produce responses that are more consistent, deterministic, and less random, without modifying the knowledge retrieval process.

What approach solves these requirements?


Modify the inference parameters by lowering both the temperature value and the top_k sampling threshold.

Modify the inference parameters by increasing both the temperature value and the top_k sampling threshold.

Modify the inference parameters by lowering the temperature value and increasing the top_p sampling threshold.

Modify the inference parameters by lowering both the temperature value and the top_p sampling threshold.


---
### **Question 68:**


Category: AIP – Testing, Validation, and Troubleshooting
A travel company is developing a virtual assistant using Amazon Lex to help customers find vacation packages based on themes such as “relaxation,” “adventure,” and “culture.” The chatbot uses an AWS Lambda function to query an Amazon DynamoDB table that stores package details by category.

During testing, a Generative AI Developer observes that the chatbot sometimes fails to recognize user inputs such as “thrill-seeking,” “sightseeing,” or “chill,” even though these terms correspond to existing categories. The company is exploring Amazon Titan models to enhance future natural language understanding and embedding capabilities but requires an immediate solution that does not involve modifying the Lambda function or database.

Which action should the Generative AI Developer take to improve the chatbot’s ability to recognize these user inputs?

Define the unrecognized words as synonyms linked to current enumeration values in the custom slot type.
Update the slot type definition to include the unrecognized words as part of its enumeration list.
Add runtime hints for the slot values to guide Lex in resolving similar user inputs.
Train a new Lex intent with the unrecognized words as sample utterances.




---
### **Question 69:**


Category: AIP – Operational Efficiency and Optimization for Generative AI Applications
A multinational e-commerce enterprise, TD Conversation, is developing a Generative AI-powered speech understanding system to transcribe and classify short customer voice messages submitted through its global support platform. Each message, lasting up to 2 minutes, may contain 150 unique product names with uncommon spellings or localized pronunciations. The AI team has built a labeled dataset of 5,000 voicemail transcripts using Amazon SageMaker Ground Truth, enriched with metadata like accent, noise level, and speaker ID. Additionally, Amazon Comprehend is integrated to extract domain-specific entities like product codes and issue categories for downstream summarization.

During the model prototyping phase, developers must frequently test and refine the acoustic and vocabulary parameters of the automatic speech recognition (ASR) workflow to improve recognition accuracy for brand-specific terms. The solution must enable rapid customization iterations and maximize transcription accuracy without requiring full retraining of the ASR model from scratch.

Which solution will improve transcription accuracy for product names while supporting frequent ASR model updates?

Implement a voice bot using Amazon Lex where each product name is configured as a slot entry. Leverage Lex’s synonym feature to capture alternate pronunciations and spelling variations, refining the custom slot list throughout the testing phase.
Use Amazon Kendra to index the transcribed audio data and automatically retrieve context for similar product names. Use the search feedback mechanism to adjust ASR interpretations for improved term recognition.
Configure an ASR customization workflow in Amazon Transcribe by creating a custom vocabulary that defines every product name and pronunciation variant. After observing misrecognized words during development, manually update and redeploy the vocabulary for improved performance.
Leverage Amazon Bedrock to fine-tune a foundation model that analyzes transcribed text and generates improved acoustic embeddings for the ASR pipeline. Integrate these embeddings into subsequent transcription tasks to enhance context understanding.



---
### **Question 70:**

Category: AIP – Implementation and Integration
A data science team at a retail company wants to predict customer churn based on various historical transactional data. The dataset contains 10,000 records, each with 1,500 attributes, such as demographic information, purchasing patterns, and customer service interactions. The team is looking for an automated way to build a model using Amazon SageMaker AI, which can predict whether a customer will churn while also identifying the most relevant features contributing to the prediction. To deepen insights, the team uses Amazon Comprehend to analyze sentiment and key phrases in customer interactions.

Which of the following solutions will best fulfill these requirements while minimizing manual effort?

Leverage SageMaker Autopilot to automatically train a classification model for forecasting customer churn. Then, utilize insights from SageMaker Clarify to determine which features most significantly influence the predictions.
Use the k-means algorithm in SageMaker AI to cluster customers based on purchasing patterns. After clustering, use the resulting clusters to predict churn based on customer behavior.
Use SageMaker Data Wrangler to automatically train a churn prediction model and rely on its quick model visualization feature to generate accurate importance scores for deployment decisions.
Use SageMaker Ground Truth to label customer churn data, then build a custom TensorFlow model to predict churn and analyze feature weights post-training.




---
### **Question 71:**


Category: AIP – Foundation Model Integration, Data Management, and Compliance
A leading technology company has developed a powerful Retrieval Augmented Generation (RAG) application that enhances user interactions by providing relevant responses through advanced search techniques. The application uses a vector database to store embeddings of documents, understanding the meaning behind user queries rather than relying on keyword matches. As part of the cloud migration, the text repository has been moved to Amazon S3, containing terabytes of unstructured documents such as technical manuals and customer support logs. To support the data processing pipeline, Amazon SageMaker AI is used for building and training custom machine learning models that analyze and classify documents, and Amazon Comprehend is utilized for natural language processing tasks such as sentiment analysis and entity recognition.

The company needs a solution that integrates smoothly with the S3 bucket, scales effectively, and ensures high performance when retrieving relevant, context-aware results from the vast repository. The goal is to provide users with accurate and contextually relevant results based on the deeper meaning of queries, offering a more dynamic and engaging user experience.

Which AWS solution will optimize the company’s RAG application and enable semantic search?

Ingest documents from S3 into Amazon Kendra using the Kendra S3 connector, then perform semantic search queries with Kendra's built-in search engine.
Use AWS Lambda to process the files and generate embeddings. Store the embeddings in Amazon DynamoDB. Use Amazon QuickSight to perform the semantic searches.
Generate embeddings for documents using a custom script in SageMaker notebooks, store the embeddings in SageMaker Feature Store, and run semantic searches with SQL queries.
Leverage Amazon Textract to extract text from the documents in the S3 bucket. Store the extracted data in Amazon Redshift for analytics and perform semantic searches using Amazon OpenSearch Service.



---
### **Question 72:**


Category: AIP – Foundation Model Integration, Data Management, and Compliance
A multinational digital payments provider is designing a real-time fraud detection platform using Amazon SageMaker AI for training and hosting machine learning models, and Amazon Comprehend for analyzing unstructured transaction descriptions.

The AI development team has consolidated years of historical transaction data from multiple regions into a single data lake. Each record in this dataset contains the following fields:

client_identifier (string)

account_category (integer)

payment_value (float)

account_duration (integer)

operation_status (string) with possible values “legitimate” or “suspicious”

Before deploying the model to production, the AI developer must prepare the data to ensure compatibility with SageMaker AI built-in algorithms and maintain a valid label structure for classification.

Which preprocessing step should the AI developer perform before training the model in SageMaker AI?


Exclude the client_identifier field and encode operation_status into numeric labels, then proceed with launching the model training phase in SageMaker AI.

Retain all fields and use Comprehend to transform operation_status into sentiment-based numeric scores before starting the training job.

Exclude both client_identifier and operation_status fields to reduce data correlation, and initiate model training using the remaining attributes.
Convert all fields into string format to maintain data consistency, and then start the model training phase in SageMaker AI.



---
### **Question 73:**


Category: AIP – Implementation and Integration
A financial services organization has built a custom machine learning (ML) model designed for real-time fraud detection. The model, which is hosted on the company’s on-premises infrastructure, is less than 5 GB in size and processes up to 50 concurrent requests simultaneously. The company is exploring AWS services to migrate its model to the cloud while minimizing infrastructure management. The development team is familiar with Amazon Rekognition for image-based analysis and Amazon Textract for document analysis, and is now looking for the best service to deploy the fraud detection model.

Which of the following will meet the given requirements with the least operational overhead?

Deploy the fraud detection model on a highly available Amazon EC2 instance in an auto-scaling group. Configure an application load balancer to route the incoming requests to the EC2 instance.
Create a model configuration within Amazon SageMaker AI, then deploy the custom fraud detection model on an asynchronous SageMaker endpoint.
Create a model configuration within Amazon SageMaker AI, then deploy the custom fraud detection model on a serverless SageMaker endpoint.
Deploy the custom fraud detection model in Amazon SageMaker Neo to optimize the model, then host the optimized model on a SageMaker real‑time endpoint.



---
### **Question 74:**

Category: AIP – Operational Efficiency and Optimization for Generative AI Applications
A digital media company operates a web platform where customers upload high-resolution product and marketing images that require automated text generation for accessibility and search optimization. Each uploaded image can reach up to 60 MB in size, and the platform experiences unpredictable spikes in traffic during global promotional campaigns.

All uploaded images are securely stored in an Amazon S3 bucket, which acts as the central repository for incoming media files. The company uses Amazon Rekognition to analyze images, detecting objects, scenes, and other visual features that form the foundation of metadata. The extracted information is passed to Amazon Bedrock, which uses a foundation model to generate detailed natural language captions describing the image content accurately and contextually.

The ML workflow automatically starts processing when new images are uploaded to the S3 bucket. The GenAI developer must design a solution that scales automatically to handle fluctuating workloads, maintains high availability during traffic spikes, and operates with minimal infrastructure management overhead.

Which solution fulfills these requirements with minimal infrastructure management and operational effort?

Build a containerized processing pipeline using Amazon ECS on Fargate that runs on a schedule to handle uploaded images and insert processed data into Amazon Aurora.
Launch an Amazon EC2 Auto Scaling group to host an inference application that monitors the primary S3 bucket for new image uploads and stores processed results in a separate S3 bucket.
Deploy an Amazon SageMaker Asynchronous Inference endpoint with a scaling policy that automatically adjusts capacity and processes inference requests for each image in the S3 bucket.
Use Amazon SQS to queue image-processing tasks and trigger AWS Lambda functions that run Rekognition and Bedrock processing for each uploaded image.




---
### **Question 75:**


Category: AIP – Implementation and Integration
A GenAI developer working for a global technology enterprise is designing a real-time conversational AI assistant that must support thousands of concurrent interactions across multiple business units. Incoming text inputs are enriched with entity and sentiment analysis through Amazon Comprehend, while specialized classification tasks are handled by hosted models on Amazon SageMaker AI. The assistant’s generative responses are produced by an AWS Lambda function that invokes an Amazon Bedrock foundation model with response streaming enabled to deliver token-by-token output with minimal latency.

The development team is standardizing on an event-driven, serverless architecture that uses Amazon API Gateway WebSocket APIs to support persistent bidirectional communication. The system must maintain conversational state across multi-step interactions, manage active connections and retries, and perform disconnect cleanup to prevent stale records or orphaned sessions. The organization requires a fully managed solution that reduces complexity and operational overhead while ensuring reliability, scalability, and secure communication across all client sessions.

Which set of actions will deliver the required functionality while maintaining the lowest operational overhead? (Select THREE.)


Grant the Lambda function an IAM role that includes both bedrock:InvokeModelWithResponseStream for Bedrock streaming and execute-api:ManageConnections for WebSocket message operations tied to the API Gateway API ID.
Set up an API Gateway WebSocket API to trigger a Lambda function that manages connection events and stores session state in Amazon ElastiCache for Redis for use during subsequent message processing.

Maintain session context and active connection IDs in an Amazon DynamoDB table, leveraging $connect/$disconnect triggers and TTL for lifecycle management.
Use Amazon DynamoDB to accumulate streaming tokens for each session and rely on a Lambda function subscribed to DynamoDB Streams to transmit tokens to WebSocket endpoints.
Configure AWS Step Functions to control the interaction workflow by calling Bedrock, updating session state, and coordinating Lambda tasks responsible for forwarding messages to WebSocket connections.
Deploy an API Gateway WebSocket API with defined routes and integrate it with a Lambda function that oversees connection events and pushes streamed messages to clients via the Management API.