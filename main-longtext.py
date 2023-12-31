from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

device = torch.device("cpu")
import textwrap

model = T5ForConditionalGeneration.from_pretrained("NlpHUST/t5-small-vi-summarization")
tokenizer = T5Tokenizer.from_pretrained("NlpHUST/t5-small-vi-summarization")
model.to(device)

src = "Sau quá trình làm tính năng làm khô cho điều hòa, các thành viên đã đạt được kết quả như sau: An đã nghiên cứu " \
      "và áp dụng công nghệ hiện có để thiết kế tính năng làm khô cho điều hòa. Thực hiện các bài test và điều chỉnh " \
      "để đảm bảo tính ổn định và hiệu quả của tính năng. Ví dụ: Thiết kế chế độ làm khô thông qua việc điều chỉnh " \
      "nhiệt độ và tốc độ quạt. Chi đã thực hiện việc tích hợp tính năng làm khô vào giao diện người dùng của điều " \
      "hòa. Kiểm tra và đảm bảo tính tương thích của tính năng trên các môi trường và thiết bị khác nhau. Ví dụ: Thêm " \
      "các tùy chọn và giao diện điều khiển liên quan đến tính năng làm khô trong ứng dụng điều hòa. Long đã xây dựng " \
      "cơ sở dữ liệu và lưu trữ các thông tin liên quan đến tính năng làm khô. Thiết kế và triển khai các chức năng " \
      "quản lý và cài đặt cho tính năng này. Ví dụ: Lưu trữ thông tin về chế độ làm khô, ghi nhận lịch sử sử dụng " \
      "tính năng. Linh thực hiện kiểm thử và tối ưu tính năng làm khô để đảm bảo hoạt động ổn định và hiệu quả. Ghi " \
      "nhận và khắc phục các lỗi hoặc khó khăn gặp phải trong quá trình sử dụng tính năng. Ví dụ: Kiểm tra tính năng " \
      "làm khô trên các môi trường và tình huống sử dụng thực tế. Tiếp theo, chúng ta sẽ giao công việc thiết kế tính " \
      "năng làm mát bằng hơi nước cho điều hòa cho đơn vị quản lý A và 2 người với các thông tin sau: Đơn vị quản lý " \
      "A phụ trách công việc: Thiết kế tính năng làm mát bằng hơi nước cho điều hòa. Mô tả: Đơn vị quản lý A sẽ thiết " \
      "kế và triển khai tính năng làm mát bằng hơi nước cho điều hòa. Họ sẽ nghiên cứu và áp dụng các công nghệ hiện " \
      "đại để tạo ra tính năng làm mát hiệu quả và tiết kiệm năng lượng. Ví dụ: Tích hợp bộ phun sương vào điều hòa " \
      "để tạo ra hơi nước làm mát. Thời gian thực hiện: Từ ngày 5/6 đến ngày 15/6. Huế phụ trách  công việc: Xác định " \
      "yêu cầu và tính năng cần thiết. Mô tả: Huế sẽ nghiên cứu và xác định yêu cầu cần thiết cho tính năng làm mát " \
      "bằng hơi nước. Họ sẽ tìm hiểu về công nghệ phun sương và xác định các tính năng và chức năng phù hợp với điều " \
      "hòa. Ví dụ: Xác định công suất phun sương, kiểm soát độ ẩm. Thời gian thực hiện: Từ ngày 1/6 đến ngày " \
      "5/6.Hương phụ trách công việc: Thiết kế giao diện người dùng. Mô tả: Hương sẽ thiết kế giao diện người dùng " \
      "cho tính năng làm mát bằng hơi nước. Họ sẽ tạo ra các giao diện thân thiện và dễ sử dụng để người dùng có thể " \
      "điều chỉnh và kiểm soát tính năng này. Ví dụ: Tạo giao diện điều khiển đơn giản, hiển thị thông tin về chế độ " \
      "làm mát bằng hơi nước. Thời gian thực hiện: Từ ngày 1-6 đến ngày 10/6.Các thành viên được yêu cầu hoàn thành " \
      "công việc đúng thời hạn để đảm bảo tiến độ dự án và chất lượng tính năng làm mát cho điều hòa."
tokenized_text = tokenizer.encode(src, return_tensors="pt").to(device)
model.eval()
summary_ids = model.generate(
    tokenized_text,
    max_length=1024,
    num_beams=10,
    repetition_penalty=5.0,
    length_penalty=1.0,
    early_stopping=True
)
output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print(textwrap.fill(output, width=100))
