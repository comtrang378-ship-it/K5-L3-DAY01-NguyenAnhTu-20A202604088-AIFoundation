# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202604088
- **Họ và tên:** Nguyễn Anh Tú
- **SEED** (in ở ô đầu notebook): 1525

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | Có |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | CÓ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | CÓ |

## 2. Kết quả chính

                                 precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            1.000   0.723      0.839        0.824   
2. BoW + MLP (deep learning)         1.000   0.881      0.937        0.974   
3a. Encoder (MiniLM) + kNN           0.857   0.950      0.901        0.974   
3b. Encoder (MiniLM) + LogReg        1.000   0.931      0.964        0.947   
3c. Encoder (MiniLM) + MLP head      0.979   0.921      0.949        0.974   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)           24.641       5666.694       0.063   
2. BoW + MLP (deep learning)        19.497      15969.702       0.326   
3a. Encoder (MiniLM) + kNN           5.155         63.379      27.155   
3b. Encoder (MiniLM) + LogReg        7.432       1293.581      31.519   
3c. Encoder (MiniLM) + MLP head      8.976        894.642      12.798   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.127           0.0  
2. BoW + MLP (deep learning)          9.279           0.0  
3a. Encoder (MiniLM) + kNN           46.421           0.0  
3b. Encoder (MiniLM) + LogReg        43.187           0.0  
3c. Encoder (MiniLM) + MLP head      34.932           0.0  
• Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.
• SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.
• Inference đo TỪ VĂN BẢN THÔ → NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.
• train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab.

## 3. Dự đoán và thực tế

1. dl_relu: bỏ ReLU thì F1 ra sao

Đoán: F1 tương đương hoặc gần Logistic Regression (LR tự code có F1 test 0.839).
Ra: F1 = 0.937, bằng đúng MLP có ReLU và cao hơn LR khoảng 0.10.
Vì sao lệch: Lập luận “hai lớp Linear gộp thành một phép tuyến tính” của bạn đúng. Nhưng mạng không ReLU vẫn được train bằng Adam, mini-batch, 30 epoch trong PyTorch. LR ở Phần 1 là gradient descent thuần, full-batch, 300 epoch, nên chưa hội tụ tốt (recall train chỉ 0.837). Khoảng cách F1 đến từ cách tối ưu, không phải từ phi tuyến. Với bài BoW nhỏ này, ReLU gần như không đóng góp gì.

2. emb_search: top kết quả gần nhất với câu truy vấn “complimentary holiday…”

Đoán: các tin gần nhất sẽ là ham, vì ham chiếm đa số và truy vấn không chứa từ spam quen thuộc như free, win, prize.
Ra: cả 6 tin gần nhất đều là spam (cosine 0.531–0.563), toàn tin kiểu “won… holiday… CALL NOW”.
Vì sao lệch: Tìm kiếm cosine so sánh nghĩa của câu, không so tỉ lệ lớp. Tỉ lệ ham/spam trong dữ liệu không ảnh hưởng thứ hạng. “Selected to receive a complimentary holiday, reply now to claim” có nghĩa gần như một tin trúng thưởng. Embedding hiểu “complimentary” ≈ “free/prize” nên không cần trùng từ.

3. dl_curve: MLP so với Logistic Regression

Đoán: val loss giảm rồi chững hoặc tăng. MLP có F1 “tương đương hoặc nhỉnh hơn” LR.
Ra: phần val loss đúng (thấp nhất ở epoch 10). Nhưng MLP đạt F1 0.937 so với 0.839, recall tăng từ 0.723 lên 0.881. Đó là một khoảng cách lớn, không phải “nhỉnh”.
Vì sao lệch: Cùng nguyên nhân với mục 1. Kết quả dl_relu cho thấy mạng tuyến tính cũng đạt 0.937, nên lợi thế đến từ việc LR tự code còn underfit (precision 1.0 nhưng bỏ sót nhiều spam), không phải từ khả năng học tổ hợp từ.

## 4. Phân tích 50 case khó

Model	F1 test thường (SUB)	F1 50 case khó	Mức tụt
1. BoW + LogReg	0.824	0.350	−0.47
2. BoW + MLP	0.974	0.350	−0.62
3a. Emb + kNN	0.974	0.746	−0.23
3b. Emb + LogReg	0.947	0.618	−0.33
3c. Emb + MLP head	0.974	0.604	−0.37

BoW + MLP tụt nhiều nhất (−0.62), và hai model BoW cùng dừng ở F1 0.350, accuracy 0.527.

Con số 53% accuracy dễ gây hiểu nhầm. BoW đúng 100% ở nhóm “Tưởng spam nhưng không phải” và “Nói về spam”, nhưng chỉ vì nó gần như đoán ham cho mọi thứ. Với precision ≈ 1.0 và F1 0.35, recall ước tính chỉ khoảng 0.2 (bắt được khoảng 6/29 tin spam). Ở các nhóm spam, BoW đúng chỉ 11% (thao túng/lừa đảo), 10% (spam giọng tự nhiên) và 0–14% (né từ khoá).

Embedding tụt ít hơn nhưng vẫn yếu ở spam giọng tự nhiên (10–50%). Cả 6 case mà tất cả model đều sai (#12, 13, 15, 16, 19, 31) thì 5 case thuộc nhóm này.

Case 1 — #21 (thao túng/lừa đảo): “Hi Mum, I dropped my phone in the toilet, this is my new number. Can you send £200 to my friend…” (BoW đoán ham, 3c cũng sai)

BoW học từ dữ liệu SMS spam cũ, nên dấu hiệu spam là call, txt, free, stop, www. Tin này không có từ nào như vậy. Ngược lại, các từ như “my”, “me”, “i” lại kéo mạnh về ham.
Mức độ lừa đảo nằm ở ý đồ (giả danh người thân kèm yêu cầu chuyển tiền), không nằm ở từ vựng. Tập train gần như không có kiểu tin này, nên mô hình không có trọng số nào để bắt.
Embedding khá hơn ở nhóm này (3b: 33%) nhưng 3c vẫn sai, vì nó chỉ gom các tin “gần nghĩa” với tin train, mà tin train không chứa mẫu lừa đảo này.

Case 2 — #31 (né từ khoá), tất cả model đều sai: “U have been s.e.l.e.c.t.e.d for a c.a.s.h r.e.w.a.r.d, c.a.l.l 09061 701 461”

Hàm tokenize dùng regex [a-z0-9£$€]+ nên chuỗi s.e.l.e.c.t.e.d bị tách thành từng chữ cái đơn lẻ. Từ “call”, “cash” biến mất khỏi vocab, nên BoW chỉ còn thấy các chữ lẻ và dãy số.
Encoder MiniLM cũng chia nhỏ chuỗi này thành các mảnh vô nghĩa, nên vector không còn gần cụm “prize/cash” nữa. Tức là embedding chỉ hiểu nghĩa khi chữ còn nguyên vẹn.
Cách sửa nằm ở bước chuẩn hoá đầu vào: gộp các chữ cách nhau bằng dấu chấm hoặc khoảng trắng trước khi tokenize, thay vì chờ model “tự hiểu”.

Một điểm tương phản đáng nêu: nhóm “Nói về spam” cho thấy embedding có điểm yếu riêng. Ví dụ #47 (“lol I just got a text saying I won £1000… people still fall for that?”) bị kNN đoán spam, vì embedding mã hoá chủ đề của câu (trúng thưởng) nhưng không phân biệt được thái độ của người viết (chê spam chứ không phải gửi spam).

Kết luận: điểm cao trên tập test thường chỉ chứng minh model nhớ được phân phối của tập train. Với bộ 50 case này (gồm cả spam không dùng từ khoá), mỗi cách tiếp cận lộ một điểm mù khác nhau.

## 5. Thử thách cá nhân

Đề #2: ML · Giảm vocab còn 100 từ và tăng lên 5000 từ (build_vocab(..., max_size)). Train lại, báo F1 test. Giải thích xu hướng.

Cách làm: Dùng `build_vocab(texts_tr, max_size=N)` của notebook với N ∈ {100, 2000, 5000} (2000 là mốc đối chiếu). Giữ nguyên `vectorize` (đặc trưng 0/1), `train_logreg` (lr=0.5, 300 epoch) và đánh giá trên `texts_te` chia theo SEED (70/15/15). Chỉ thay đổi `max_size`, các thành phần khác giữ cố định.

Kết quả: F1-Score (Spam) trên Test với Vocab max_size = 100  : 0.8192, F1-Score (Spam) trên Test với Vocab max_size = 5000 : 0.9209

### Giải thích xu hướng
- **N = 100:** vocab chỉ giữ các từ xuất hiện trong nhiều tin nhất. Trong các từ khoá trên, ⟨liệt kê từ còn / từ mất⟩. So với N=2000, ⟨recall / precision⟩ thay đổi ⟨…⟩, nên F1 ⟨giảm bao nhiêu⟩. Nguyên nhân là ⟨thiếu từ đặc trưng spam → underfit / giải thích khác nếu số liệu không ủng hộ⟩.
- **N = 2000 → 5000:** tập train có 3.619 tin, nên các từ thêm vào chủ yếu hiếm ⟨số từ có df ≤ 2 trong hạng 2000–5000, nếu đã đếm⟩. F1 test thay đổi ⟨+/− …⟩, còn khoảng cách F1 train − F1 test đổi từ ⟨…⟩ sang ⟨…⟩. ⟨Kết luận: lợi ích chững lại / có dấu hiệu overfit / vẫn tăng nhẹ, chọn đúng một theo số liệu⟩.

## 6. Phương án cho sản phẩm thật

Phương án Embedding (Sentence-Transformers MiniLM) kết hợp với mô hình tuyến tính (Logistic Regression / Linear Head) chạy offline/on-premise trực tiếp trên cụm server của hệ thống (hoặc phương án 1. Bag-of-Words + Logistic Regression nếu tối ưu phần cứng thuần túy). Lý do: Tránh dùng các mô hình LLM lớn gọi qua API mạng bên ngoài (như OpenAI, Anthropic) vì độ trễ mạng (network latency) và thời gian sinh token của LLM sẽ làm vượt ngưỡng p95 rất dễ dàng, đồng thời chi phí đội lên cực lớn.

Thông lượng (Throughput): 10 triệu tin/ngày tương đương trung bình $\approx 116$ tin/giây (và giờ cao điểm có thể lên tới vài trăm tin/giây).Độ trễ (Latency p95): Một mô hình nhúng nhẹ như MiniLM chạy qua ONNX Runtime / TensorRT hoặc mô hình BoW kết hợp Logistic Regression trên CPU/GPU thông thường chỉ mất trung bình từ 2 ms đến 15 ms/tin, hoàn toàn đáp ứng tốt cam kết SLA p95 < 50 ms.Chi phí: Chạy cục bộ giúp chi phí hạ tầng quy về $\approx 0\$ API cost$, chỉ tính tiền khấu hao server, rẻ hơn hàng trăm lần so với việc gọi API LLM cho 10 triệu request mỗi ngày.

Xong khi làm vậy ta cần đánh đổi độ chính xác ngữ nghĩa sâu: Chấp nhận không thông minh và linh hoạt bằng các LLM cỡ lớn (như GPT-4o hay Claude) trong việc bóc tách các câu lừa đảo cực kỳ tinh vi, dùng từ lóng phức tạp hoặc ẩn dụ cao cấp. Cùng với đó chi phí đầu tư hạ tầng ban đầu (Engineering Effort): Cần đội ngũ kỹ thuật xây dựng đường ống xử lý (Pipeline), đóng gói mô hình (Model Serving) bằng C++/ONNX hoặc Triton Inference Server để tối ưu hóa tốc độ thay vì chỉ gọi API đơn thuần.

## 7. AI log và điều đã học

Phần 3 & 4, (Embedding & LLM)	Viết hàm gọi API cho mô hình LLM dạng zero-shot prompt classification và hàm parse_label trích xuất nhãn từ kết quả trả về.	Hàm llm_classify_batch kèm theo prompt mẫu và đoạn regex bóc tách từ khóa spam/ham.	Thêm bước xử lý ngoại lệ (try-except) khi phản hồi trả về từ mô hình bị trống hoặc định dạng lỗi, đồng thời thêm cơ chế lưu cache kết quả để tránh gọi lại API trùng lặp.

Hôm nay, mình đã hiểu sâu sắc rằng độ phức tạp của mô hình không đồng nghĩa với hiệu quả thực tiễn trong bài toán sản phẩm quy mô lớn. Trước đây, mình thường mặc định các mô hình Deep Learning hoặc LLM lớn luôn là lựa chọn tối ưu, nhưng qua bài lab này, sự kết hợp giữa mô hình truyền thống (Logistic Regression) với đặc trưng gọn nhẹ (BoW/Embedding) lại mang đến lợi thế áp đảo về tốc độ (đáp ứng tiêu chuẩn p95 < 50ms) và chi phí vận hành bằng 0. Đồng thời, mình cũng nhận ra tầm quan trọng của việc kiểm soát phân phối dữ liệu và tối ưu hóa đường ống xử lý (pipeline) thay vì chỉ tập trung tinh chỉnh độ chính xác của mô hình.
