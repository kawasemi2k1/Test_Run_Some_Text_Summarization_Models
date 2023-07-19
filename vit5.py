from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

device = torch.device("cpu")
import textwrap

model = T5ForConditionalGeneration.from_pretrained("NlpHUST/t5-small-vi-summarization")
tokenizer = T5Tokenizer.from_pretrained("NlpHUST/t5-small-vi-summarization")
model.to(device)

src = "Tối 12/11, phát biểu tại Hội nghị cấp cao Diễn đàn hợp tác kinh tế châu Á - Thái Bình Dương (APEC) với sự tham " \
      "dự của Tổng thống Nga Vladimir Putin, Chủ tịch Trung Quốc Tập Cận Bình và Tổng thống Mỹ Joe Biden, " \
      "Chủ tịch nước Nguyễn Xuân Phúc đã đề nghị APEC, ở thời khắc đặc biệt này, các thành viên cần vượt qua khác " \
      "biệt để chung tư duy cùng hành động vì lợi ích của chính mình và của cả cộng đồng mà trước hết là phục hồi bền " \
      "vững sau đại dịch và tăng trưởng bao trùm.Các hội nghị của APEC năm 2021 được tiến hành hoàn toàn trực tuyến " \
      "vì chủ nhà New Zealand vẫn đang đóng cửa biên giới để phòng chống COVID-19.Phát biểu khai mạc Hội nghị đa " \
      "phương cuối cùng của thế giới trong năm nay, Thủ tướng Jacinda Ardern cho rằng mỗi nền kinh tế trong APEC đều " \
      "trải qua đại dịch COVID-19 khác nhau và ứng phó với đại dịch theo những cách riêng nhưng tất cả đều đối mặt " \
      "với các vấn đề cơ bản giống nhau, bao gồm thúc đẩy tiêm chủng, duy trì sản xuất, kinh doanh và việc làm cho " \
      "người dân, bảo đảm đi lại an toàn giữa các quốc gia, đi cùng với phục hồi kinh tế mạnh mẽ và bao trùm.Cùng bày " \
      "tỏ quyết tâm hơn bao giờ hết của APEC vượt qua đại dịch, đẩy nhanh quá trình phục hồi kinh tế, đi cùng với ứng " \
      "phó với biến đổi khí hậu và thúc đẩy tăng trưởng bao trùm cho tất cả người dân, Chủ tịch nước Nguyễn Xuân Phúc " \
      "nhấn mạnh cuộc chiến cam go đầy đau thương, mất mát của thế giới với đại dịch COVID-19 hai năm qua buộc các " \
      "nền kinh tế phải suy ngẫm, nhìn nhận lại về nhiều vấn đề khu vực và toàn cầu, nhất là sự dễ tổn thương, " \
      "thiếu sẵn sàng trước dịch bệnh và biến đổi khí hậu cũng như những bất cập và hạn chế của hệ thống quản trị " \
      "toàn cầu trong xử lý khủng hoảng cùng với sự bất bình đẳng trong và giữa các nền kinh tế.Chủ tịch nước Nguyễn " \
      "Xuân Phúc nhấn mạnh lịch sử cho thấy, mỗi lần vượt qua khủng hoảng, APEC lại càng chứng tỏ sức sống mạnh mẽ và " \
      "vai trò gắn kết của mình. Trong khó khăn hôm nay, hơn bao giờ hết, APEC- nơi đóng góp hơn 60% GDP và gần một " \
      "nửa thương mại toàn cầu cần tiếp tục phát huy vai trò là động lực tăng trưởng kinh tế toàn cầu, đi cùng với " \
      "khẳng định vai trò là trung tâm khởi xướng các ý tưởng sáng tạo và xu thế phát triển mới. Đồng thời, " \
      "APEC cần chủ động mở rộng liên kết kinh tế trong phục hồi, tăng trưởng bền vững, dẫn dắt sự định hình của kinh " \
      "tế thế giới sau đại dịch và góp phần củng cố quản trị kinh tế toàn cầu hiệu quả, công bằng, minh bạch.Trong " \
      "tuyên bố chung của Hội nghị và Kế hoạch hành động thực hiện Tầm nhìn APEC 2040 được Chủ tịch nước Nguyễn Xuân " \
      "Phúc và các nhà lãnh đạo thông qua khẳng định quyết tâm sử dụng tất cả các công cụ kinh tế vĩ mô hiện có để " \
      "giải quyết các hậu quả bất lợi của đại dịch COVID-19, duy trì sự phục hồi kinh tế, đồng thời duy trì tính bền " \
      "vững tài khóa dài hạn.Các nhà lãnh đạo cam kết thúc đẩy sản xuất và cung cấp vaccine phòng COVID-19 thông qua " \
      "chuyển giao công nghệ và xóa bỏ các hạn chế xuất khẩu đối với thiết bị y tế. Đi cùng với tăng cường hợp tác " \
      "trong việc xét nghiệm COVID-19 và hộ chiếu vaccine khi mở cửa trở lại biên giới và khi đi lại của người dân " \
      "giữa các nền kinh tế tăng lên. Các nhà lãnh đạo cũng cam kết sẽ ngừng tăng trợ cấp cho khai thác và sử dụng " \
      "nhiên liệu hóa thạch, tạo cơ sở cho việc thảo luận về các vấn đề liên quan đến biến đổi khí hậu trong các cuộc " \
      "họp APEC sau này.Chủ tịch nước Nguyễn Xuân Phúc và các nhà lãnh đạo cũng nhất trí Thái Lan sẽ giữ trọng trách " \
      "Chủ tịch APEC năm 2022. Thủ tướng New Zealand Jacindar Ardern đã chuyển giao trọng trách này cho Thủ tướng " \
      "Prayut Chan-ocha mà biểu tượng là chiếc mái chèo của người Maori, để Thái Lan đưa con thuyền APEC cùng hợp tác " \
      "hài hòa, cùng thay đổi ở hiện tại để hướng tới tương lai chung.Chủ tịch nước Nguyễn Xuân Phúc khẳng định Việt " \
      "Nam sẽ hợp tác cùng Thái Lan và các thành viên tổ chức thành công Năm APEC 2022 với 3 ưu tiên, là Rộng mở cho " \
      "tất cả các cơ hội, Kết nối trong tất cả các phương diện và Cân bằng trong mọi khía cạnh."
tokenized_text = tokenizer.encode(src, return_tensors="pt").to(device)
model.eval()
summary_ids = model.generate(
    tokenized_text,
    max_length=256,
    num_beams=5,
    repetition_penalty=2.5,
    length_penalty=1.0,
    early_stopping=True
)
output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print(textwrap.fill(output, width=100))
