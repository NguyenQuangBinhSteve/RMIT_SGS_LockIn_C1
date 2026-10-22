# Quyết định 02 – Chốt 3 câu hỏi và điều chỉnh (khoảng 11:10 VN)

## Q1 – Cách nộp: đồng ý
Nộp bằng output của notebook đúng quy ước tên: `kaggle competitions submit rmit-hackathon-2026 -k qucanhtrnnguyn/rmit-sgs-lock-in-c1 -v <version> -f submission.csv -m "<ghi chú>"`. Cách này thỏa cả "Submit notebooks" lẫn "Submit a CSV". Lệnh chỉ chạy sau khi qua cổng, và do người dùng/đồng đội chạy, agent không chạy.

## Q2 – Kernel E: đồng ý đổi tên
- Kernel E mới: title `RMIT_SGS_LockInExp_C1`, id `qucanhtrnnguyn/rmit-sgs-lockinexp-c1`, code_file `RMIT_SGS_LockInExp_C1.ipynb`, private, **không bao giờ dùng để nộp**.
- Nếu smoke E đã push dưới tên cũ `lockin-c1-exp`: chỉ dùng kết quả để đo thời gian và kiểm tra mmBERT có chạy được hay không. Từ E1 trở đi chỉ push dưới tên mới. Sau khi lấy xong số liệu, đề nghị **người dùng** xóa kernel cũ trên Kaggle (agent không tự xóa).
- Kernel probe, nếu cần, cũng phải theo quy ước: `RMIT_SGS_LockInProbe_C1` (`qucanhtrnnguyn/rmit-sgs-lockinprobe-c1`). Chỉ chạy khi smoke mới vẫn ERROR mà log trống, với CPU hoặc GPU và `-t 300`.
- Bài nộp chỉ xây trên MỘT model. Kernel E chỉ để so sánh.

## Q3 – Chọn bài cuối: đồng ý, chỉnh một điểm
- Mặc định để Kaggle tự chọn.
- Khi có ≥ 2 bài hợp lệ thì chọn tay 2 bài: (1) bài có OOF cao nhất, (2) bài có public cao nhất. Nếu 2 bài trùng nhau thì bài thứ hai là **bài có OOF cao thứ hai**, không lấy public cao thứ hai. Lý do: OOF dùng group-fold trên 14000 dòng ổn định hơn public, vốn chỉ là một tập con của 6000 dòng, và luật nhấn mạnh "robust generalisation".
- Nếu Kaggle chỉ cho chọn 1 bài thì chọn bài có OOF cao nhất, trừ khi public của nó thấp hơn bài tốt nhất quá 0.01.

## Điều chỉnh kỹ thuật
- **Gradient checkpointing chỉ bật cho large.** base và mmBERT-base vừa VRAM T4 16GB ở batch 8 × 512 mà không cần GC, chạy nhanh hơn khoảng 25–30%, và tránh được rủi ro mmBERT + GC chưa thử.
- Nếu mmBERT lỗi với fp16 (loss NaN hoặc lỗi kernel attention), thử lại một lần với `--no-fp16`, kèm budget 75. Còn lỗi nữa thì bỏ mmBERT, slot E để trống.
- Lịch trong `plans/01` giữ nguyên.
