# 🚀 START HERE - CRC-Select Implementation

## Bạn đang ở đâu?

Đã có **infrastructure hoàn chỉnh** cho CRC-Select project. Sẵn sàng chạy experiments!

## Đọc gì trước?

**Đọc theo thứ tự này:**

1. **README.md** (5 phút) - Hiểu project là gì, mục tiêu gì
2. **QUICK_START.md** (5 phút) - Cách chạy từng bước
3. **IMPLEMENTATION_STATUS.md** (3 phút) - Biết đã làm gì, còn gì
4. File này - bạn đang đọc đúng rồi! 

## Làm gì tiếp theo?

### Option A: Chạy Experiments Ngay (khuyến nghị)

```bash
# 1. Cài đặt
cd /home/admin1/Desktop/selectivenet
pip install -r requirements.txt

# 2. Chạy Phase 1: Baseline (chạy background)
screen -S baseline
python run_baseline.py --dataset cifar_10 --model_name baseline --seed 42
# Nhấn Ctrl+A, sau đó D để detach

# 3. Trong lúc đợi, đọc code và thiết kế Phase 4
```

**Kết quả:** Sau 6-12 giờ sẽ có baseline results

### Option B: Đọc Code Trước

```bash
# Xem các modules chính
cat crc_utils.py         # CRC core logic
cat data_utils.py        # Data handling
cat eval_utils.py        # Evaluation
cat run_baseline.py      # Example experiment
```

## Cấu trúc Project

```
📁 /home/admin1/Desktop/selectivenet/
├── 📄 Core modules (DONE)
│   ├── crc_utils.py           - CRC calibration & metrics
│   ├── data_utils.py          - Data splitting & OOD loading
│   ├── eval_utils.py          - Plots & tables
│   └── crc_select_trainer.py - Training components (partial)
│
├── 📄 Experiment scripts
│   ├── ✅ run_baseline.py         - Phase 1 (READY)
│   ├── ✅ run_post_hoc_crc.py     - Phase 3 (READY)
│   ├── ❌ run_crc_select.py       - Phase 4 (TODO - CRITICAL!)
│   └── ❌ run_all_experiments.py  - Phase 5 (TODO)
│
├── 📄 Documentation (DONE)
│   ├── README.md              - Main doc
│   ├── QUICK_START.md         - Step-by-step guide
│   ├── IMPLEMENTATION_STATUS.md - Progress tracking
│   ├── IMPLEMENTATION_SUMMARY.md - Overall summary
│   └── THIS FILE               - You are here!
│
└── 📁 Original SelectiveNet code (unchanged)
    ├── models/
    ├── train.py
    └── selectivnet_utils.py
```

## Điều Quan Trọng Cần Biết

### ✅ Đã Hoàn Thành (70%)

- ✅ CRC utilities (calibration, risk control, metrics)
- ✅ Data utilities (splitting, OOD loading)
- ✅ Evaluation utilities (plots, tables)
- ✅ Baseline experiment script (Phase 1)
- ✅ Post-hoc CRC script (Phase 3)
- ✅ Complete documentation

### ❌ Chưa Làm (30% - CRITICAL)

- ❌ **Phase 4: `run_crc_select.py`** - CRC-Select core training
  - Cần implement alternating optimization loop
  - Ước tính: 6-8 giờ làm việc
  - ĐÂY LÀ PHẦN QUAN TRỌNG NHẤT!

- ❌ **Phase 5: Full comparison**
  - Sau khi Phase 4 xong
  - Chạy tất cả methods, multiple seeds
  - Generate paper figures

## 3 Bước Tiếp Theo

### Bước 1: Chạy Baseline (TODAY)

```bash
cd /home/admin1/Desktop/selectivenet

# Chạy trong screen (để có thể detach)
screen -S exp1
python run_baseline.py --dataset cifar_10 --model_name baseline --seed 42

# Detach: Ctrl+A, D
# Reattach: screen -r exp1
```

**Thời gian:** 6-12 giờ (chạy background OK)

### Bước 2: Implement CRC-Select (THIS WEEK)

**File cần tạo:** `run_crc_select.py`

**Template:**
```python
# Tham khảo run_baseline.py
# Thay training loop = custom loop với:
# - CRCSelectCallback
# - Alternating calibration
# - Risk penalty với mu update

# Pseudocode:
for epoch in range(epochs):
    if epoch % 5 == 0:
        q = calibrate_crc(model, x_cal, y_cal, alpha)
    
    train_one_epoch(model, x_train, y_train, q, mu)
    
    cal_risk = evaluate(model, x_cal, y_cal)
    mu = update_mu(mu, cal_risk, alpha)
```

**Ước tính:** 6-8 giờ

### Bước 3: Full Comparison (NEXT WEEK)

- Chạy cả 3 methods: SelectiveNet, Post-hoc CRC, CRC-Select
- Multiple seeds (5 seeds)
- Generate all figures
- Write paper draft

## Các Lệnh Hữu Ích

```bash
# Check GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Monitor training
tail -f checkpoints/*_history.pkl

# Check results
cat results/baseline_results.json | python -m json.tool

# View plots
xdg-open results/baseline_rc_curve.png  # Linux
# open results/baseline_rc_curve.png    # Mac

# List checkpoints
ls -lh checkpoints/

# Check disk space
df -h
```

## Troubleshooting

### "Module not found"

```bash
pip install -r requirements.txt
```

### "SVHN dataset not found"

```bash
mkdir -p datasets
cd datasets
wget http://ufldl.stanford.edu/housenumbers/test_32x32.mat
cd ..
```

### "Out of memory"

Edit model code, reduce batch_size from 128 to 64:
```python
# In models/cifar10_vgg_selectivenet.py line ~237
batch_size = 64
```

### "Training too slow"

- Check if using GPU (see command above)
- Or reduce epochs for testing: modify script to use `epochs=50`

## Expected Timeline

| Week | Tasks | Deliverable |
|------|-------|-------------|
| Week 1 | Run baseline + post-hoc CRC | Baseline results |
| Week 2 | Implement Phase 4 (CRC-Select) | Working CRC-Select |
| Week 3 | Full comparison + ablations | All results + figures |
| Week 4 | Write paper + polish | Paper draft |

## Mục Tiêu Cuối Cùng

**Nộp paper:** ICML/NeurIPS/ICLR 2026

**Claim:**
- CRC-Select achieves 10-15% higher coverage than post-hoc CRC
- 50% reduction in OOD dangerous acceptance
- Formal risk control maintained

**Figures needed:**
1. RC curves (main result)
2. OOD comparison
3. Violation rate
4. Training dynamics
5. Ablations

## Questions?

1. Về lý thuyết → Đọc **README.md** phần "Theoretical Foundations"
2. Về cách chạy → Đọc **QUICK_START.md**
3. Về tiến độ → Đọc **IMPLEMENTATION_STATUS.md**
4. Về tổng quan → Đọc **IMPLEMENTATION_SUMMARY.md**

## Contact / Notes

Project location: `/home/admin1/Desktop/selectivenet/`

Based on: https://github.com/geifmany/selectivenet

Research goal: A* conference paper (ICML/NeurIPS/ICLR)

---

**🎯 ACTION NOW:** Chạy `python run_baseline.py` để bắt đầu!

```bash
cd /home/admin1/Desktop/selectivenet
python run_baseline.py --dataset cifar_10 --model_name baseline --seed 42
```

Good luck! 🚀

