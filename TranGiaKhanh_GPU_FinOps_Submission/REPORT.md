# BÁO CÁO KẾT QUẢ LAB: GPU FINOPS & COST OPTIMIZATION
**Sinh viên thực hiện:** Trần Gia Khánh  
**Mã số sinh viên (MSSV):** 2A202600293

## 1. Giới thiệu
### Mục tiêu của bài lab
Bài lab nhằm mục đích cung cấp kiến thức thực tiễn và kỹ năng vận hành tối ưu chi phí GPU (FinOps) trong môi trường điện toán đám mây. Mục tiêu cụ thể bao gồm:
*   Giám sát tài nguyên GPU theo thời gian thực (Utilization, Memory, Power).
*   Xây dựng hệ thống theo dõi chi phí (Cost Tracking) và phân bổ chi phí theo Workload.
*   Quản lý và tận dụng thị trường Spot Instance để giảm thiểu chi phí.
*   Triển khai cơ chế tự động giãn nở (Autoscaling) dựa trên hiệu suất sử dụng.
*   Phân tích lãng phí (Waste Analysis) và áp dụng các kỹ thuật tối ưu như Mixed Precision (AMP).

### Tổng quan về GPU FinOps
GPU FinOps là một kỷ luật văn hóa và thực hành quản lý tài chính đám mây, giúp các đội ngũ kỹ thuật (Engineering), tài chính (Finance) và kinh doanh (Business) cùng hợp tác để tối ưu hóa giá trị kinh doanh của tài nguyên GPU. Trong bối cảnh chi phí GPU (như A100, H100) cực kỳ đắt đỏ, FinOps giúp chuyển đổi từ việc "chi tiêu không kiểm soát" sang "chi tiêu hiệu quả" thông qua ba giai đoạn: **Inform** (Thông báo), **Optimize** (Tối ưu) và **Operate** (Vận hành).

---

## 2. Phân tích kết quả thực hiện

### Part 1-7: Phân tích từ Mock Cluster (Giả lập)
Trong phần này, chúng ta đã kết nối với một cụm GPU giả lập thông qua Gateway để thực hiện các kịch bản vận hành.

*   **Cluster Monitoring Insights:** Hệ thống giám sát 4-5 Node với các loại GPU đa dạng (T4, A100, V100). Kết quả cho thấy mức sử dụng trung bình (Avg Utilization) dao động từ **43.4% đến 72.2%**. Nhiệt độ và công suất tiêu thụ (Power Draw) được theo dõi chặt chẽ để đảm bảo sức khỏe phần cứng.
*   **Cost Tracking Observations:** Mỗi Workload (ResNet, BERT, LLM) khi chạy đều được gắn nhãn giá tiền theo giờ. Tổng chi phí tiêu thụ được ghi nhận khoảng **$2.5589**, chiếm **2.6%** ngân sách (Budget).
*   **Spot Instance Savings Analysis:** Bằng cách sử dụng các GPU Spot (T4, A100), chúng ta đã đạt được mức tiết kiệm lên tới **70%** so với giá On-demand. Hệ thống cũng mô phỏng thành công việc xử lý sự kiện thu hồi (Preemption), giúp ứng dụng tự động lưu trạng thái hoặc chuyển vùng.
*   **Autoscaling Behavior:** Khi Utilization vượt ngưỡng **70%**, cơ chế Autoscaler đã kích hoạt lệnh `SCALE_UP`, tăng số lượng Node từ 4 lên 5 để đáp ứng tải trọng, sau đó ổn định lại ở mức 57.7%.
*   **Waste Analysis & Recommendations:** Hệ thống phát hiện mức lãng phí trung bình là **11.7%** (do GPU nhàn rỗi). Khuyến nghị quan trọng nhất là chuyển đổi sang Spot Instance để tiết kiệm thêm ~65% chi phí.

### Part 8: Phân tích Real GPU Training (Trên Kaggle/Colab)
Thực hiện huấn luyện mô hình ResNet-18 thực tế trên GPU Tesla T4 để so sánh hiệu năng.

*   **FP32 vs Mixed Precision (AMP):**
    *   **FP32 (Baseline):** Thời gian huấn luyện 123.2 giây, tiêu thụ 0.82 GB Memory. Chi phí ước tính $0.01198.
    *   **Mixed Precision (AMP):** Thời gian huấn luyện chỉ còn **75.5 giây (nhanh hơn 1.63 lần)**, tiêu thụ 0.60 GB Memory. Chi phí giảm xuống còn **$0.00734**.
*   **Cost Savings Achieved:** Sử dụng AMP giúp tiết kiệm **38.7%** chi phí trực tiếp trên cùng một Workload mà không làm giảm đáng kể độ chính xác của mô hình.
*   **GPU Utilization Patterns:** Trong quá trình huấn luyện, GPU Utilization đạt mức trung bình ~92.7% cho FP32, cho thấy tài nguyên được tận dụng tối đa, giảm thiểu thời gian GPU nhàn rỗi.

### Part 8.5: Phân tích nâng cao (Advanced Analysis)
*   **Multi-GPU Scaling Efficiency:** Phân tích cho thấy khi tăng số lượng GPU, tốc độ xử lý tăng nhưng hiệu năng biên giảm dần (Scaling Efficiency giảm từ 100% xuống 80% khi dùng 8 GPU). Cấu hình tối ưu nhất về chi phí thường là cụm 2-4 GPU.
*   **Project Cost Forecasting:** Dự báo cho một dự án AI dài hơi (Data Prep, Training, Tuning) cho thấy tổng chi phí dự kiến kèm 20% dự phòng (contingency) giúp doanh nghiệp lập kế hoạch tài chính chính xác hơn.
*   **Optimization Strategy Prioritization:** Các chiến lược được xếp hạng theo thang điểm Tiết kiệm/Công sức. Kết quả cho thấy **Mixed Precision** và **Spot Instances** là hai chiến lược "Low Effort - High Impact" cần ưu tiên hàng đầu.

---

## 3. Kết luận và học hỏi

### Những kỹ năng FinOps đã học
*   Kỹ năng đọc và phân tích Metric GPU thông qua `pynvml` và API.
*   Kỹ năng xây dựng Dashboard tích hợp để theo dõi chi phí theo thời gian thực.
*   Khả năng thiết kế kịch bản xử lý tự động khi chi phí vượt ngưỡng hoặc tài nguyên bị lãng phí.

### Các chiến lược cost optimization hiệu quả
1.  **Sử dụng AMP (Mixed Precision):** Giảm thời gian chiếm dụng GPU, từ đó giảm chi phí trực tiếp.
2.  **Tận dụng Spot Instance:** Chiến lược hiệu quả nhất để giảm chi phí hóa đơn lên đến 70-90%.
3.  **Right-sizing:** Chọn đúng loại GPU cho đúng tác vụ (ví dụ: dùng T4 cho inference thay vì A100).
4.  **Autoscaling:** Tắt bớt GPU khi không có tải để tránh chi phí "Idle".

### Ứng dụng thực tế trong projects
Các kiến thức này có thể áp dụng ngay vào các dự án Deep Learning lớn tại doanh nghiệp để kiểm soát ngân sách đám mây (AWS/GCP/Azure). Đặc biệt là việc thiết lập hệ thống cảnh báo (Alerting) và báo cáo lãng phí định kỳ sẽ giúp đội ngũ kỹ sư có ý thức hơn về mặt tài chính khi thực hiện các thí nghiệm AI.

---
*Báo cáo được tổng hợp dựa trên dữ liệu từ file `gpu_finops_lab_submission.ipynb`.*
