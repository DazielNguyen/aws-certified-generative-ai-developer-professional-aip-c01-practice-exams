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

Category: AIP – AI Safety, Security, and Governance

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

--- 

### **Question 03:**

Category: AIP – AI Safety, Security, and Governance

An enterprise is working with an external ML lab specializing in Amazon SageMaker Feature Store and Amazon Bedrock model orchestration. The enterprise wants to keep all core training corpus assets and historical parquet feature store lineage assets inside Amazon S3 within the enterprise’s own AWS account because legal holds, privacy flags, and regional data sovereignty boundaries are enforced at the S3 layer.

The ML lab needs to perform highly iterative model development and comparison experiments directly against the enterprise-owned S3 data without copying the S3 datasets into the ML lab’s AWS account. The ML lab also needs to be able to browse the S3 training corpus from the AWS Management Console and run automated pipelines that require programmatic access.

Which of the following should be implemented?

[ ] Enroll the ML lab AWS account into the enterprise AWS Organizations hierarchy, create an organization-level Service Control Policy (SCP) granting S3 read access to the required S3 buckets, and propagate Service Control Policy inheritance so the ML lab can consume data as a member Organizational Unit (OU) with standard enterprise guardrails.

[x] Create a delegated IAM role within the enterprise account that specifies trust exclusively toward the ML lab AWS account, and grant that role only the precise S3 level permissions required to read the feature corpus datasets and evaluate access both programmatically and via AWS console role assumption.

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

[x] Adjust model hyperparameters to introduce stronger regularization and retrain the model to minimize overfitting.

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

[x] Use a chain-of-thought prompting to guide the model through each step of the problem before providing a final answer.

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

[x] Import the fine-tuned model directly into Bedrock using the Custom Model Import, assign an AgentCore-managed execution role, and set up CloudWatch for real-time monitoring of model metrics.

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

[x] Develop a knowledge base in Bedrock, associate the S3 bucket as a data source, and use the Bedrock API to execute RAG queries.

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

[x] Use AWS IoT Greengrass on each device to preprocess telemetry data locally, then batch upload the data to S3 using AWS SDK calls from the edge.

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

[x] Increase the number of Bedrock AgentCore agent instances.

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

[x] Perform a baseline job on the new training data and configure Model Monitor to reference the new baseline statistics.

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

[x] Fine-tune the Titan model in SageMaker AI using training data stored in Amazon S3 with KMS encryption, deploy the model through Bedrock with a customer-managed KMS key, enable AWS CloudTrail for API auditing, and use Amazon CloudWatch metrics for regional performance monitoring.

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

[x] Start a warm start hyperparameter tuning job using the TRANSFER_LEARNING warm start type to import the previously saved tuning job results. Enable AMT Early Stopping to automatically terminate exploration as soon as validation loss stops improving when training with the new quarterly dataset.

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

[x] Leverage Amazon Comprehend, Amazon Transcribe, and Amazon Rekognition to categorize and tag multimedia content automatically

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

[x] Split the dataset into multiple smaller JSONL files of up to 1,000 prompts each, store them in S3, and run separate evaluation jobs in Bedrock orchestrated by SageMaker Pipelines.

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

[x] Normalize the text by converting every word in the sentence to lowercase.

[ ] Replace every word with its corresponding synonym using a lexical database before tokenization.

[x] Segment the sentence into individual word units through tokenization.

[ ] Convert all tokens into fixed-length character n-grams before Word2Vec training to capture subword features.

[x] Exclude common non-informative words from the dataset using an English stop-word dictionary.

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
