# 🚀 HƯỚNG DẪN CHẠY EXPERIMENTS - HOÀN CHỈNH

## ✅ Code Đã Sẵn Sàng!

Tất cả code đã được triển khai xong. Bạn chỉ cần chạy!

---

## 📋 Các Scripts Có Sẵn

| Script | Mục đích | Thời gian chạy |
|--------|----------|----------------|
| `run_baseline.py` | Phase 1: SelectiveNet baseline | 6-12h |
| `run_post_hoc_crc.py` | Phase 3: Post-hoc CRC | 5 phút |
| `run_crc_select.py` | Phase 4: CRC-Select (CORE) | 8-12h |
| `run_all_experiments.py` | Chạy TẤT CẢ với nhiều seeds | 3-5 ngày |
| `analyze_results.py` | Phân tích và visualization | 1 phút |
| `quick_test.py` | Test nhanh (debug) | 5-10 phút |

---

## 🎯 Chạy Từng Bước (Khuyến Nghị)

### Bước 1: Test Nhanh (Optional - để kiểm tra)

```bash
cd /home/admin1/Desktop/selectivenet

# Test CRC-Select với 10 epochs
python quick_test.py --method crcselect --epochs 10
```

**Mục đích:** Đảm bảo code chạy được, không crash. Kết quả không có ý nghĩa!

---

### Bước 2: Chạy Baseline (Phase 1)

```bash
# Trong screen/tmux để chạy background
screen -S baseline

cd /home/admin1/Desktop/selectivenet
python run_baseline.py \
    --dataset cifar_10 \
    --model_name baseline \
    --alpha 0.5 \
    --ood svhn \
    --seed 42

# Detach: Ctrl+A, D
# Reattach: screen -r baseline
```

**Output:**
- `checkpoints/baseline_cov0.8.h5` - model checkpoint
- `results/baseline_results.json` - metrics
- `results/baseline_rc_curve.png` - RC curve

**Thời gian:** 6-12 giờ (tùy GPU/CPU)

---

### Bước 3: Chạy Post-hoc CRC (Phase 3)

**Sau khi Bước 2 xong!**

```bash
cd /home/admin1/Desktop/selectivenet
python run_post_hoc_crc.py \
    --dataset cifar_10 \
    --baseline baseline \
    --coverage 0.8 \
    --model_name posthoc_crc \
    --seed 42
```

**Output:**
- `results/posthoc_crc_results.json`
- `results/posthoc_crc_rc_curve.png`

**Thời gian:** ~5 phút (không train, chỉ calibrate)

---

### Bước 4: Chạy CRC-Select (Phase 4 - CORE!)

**Đây là contribution chính của paper!**

```bash
# Trong screen
screen -S crcselect

cd /home/admin1/Desktop/selectivenet
python run_crc_select.py \
    --dataset cifar_10 \
    --model_name crc_select \
    --alpha 0.05 \
    --coverage 0.8 \
    --recalibrate_every 5 \
    --epochs 300 \
    --seed 42

# Detach: Ctrl+A, D
```

**Output:**
- `checkpoints/crc_select.h5` - trained model
- `checkpoints/crc_select_crc_history.pkl` - CRC training history (q, mu, etc.)
- `results/crc_select_results.json` - final metrics

**Thời gian:** 8-12 giờ

**Quan trọng:** Sau khi chạy xong, check:
- `results/crc_select_results.json` → coverage có cao hơn post-hoc không?
- `checkpoints/crc_select_crc_history.pkl` → q có giảm theo epochs không?

---

### Bước 5: Phân Tích Kết Quả

```bash
cd /home/admin1/Desktop/selectivenet
python analyze_results.py --exp_name comparison
```

**Output:**
- `results/analysis/methods_comparison.png` - so sánh các methods
- `results/analysis/crc_select_crc_history.png` - training dynamics
- Bảng tổng hợp in ra console

---

## 🔥 Chạy Toàn Bộ (Multiple Seeds - Cho Paper)

**Khi đã test OK ở trên, chạy full comparison:**

```bash
screen -S fullexp

cd /home/admin1/Desktop/selectivenet
python run_all_experiments.py \
    --dataset cifar_10 \
    --exp_name final_comparison \
    --alpha 0.05 \
    --coverage 0.8 \
    --seeds 42 43 44 45 46 \
    --ood svhn

# Detach: Ctrl+A, D
```

**Output:**
- `results/final_comparison_aggregated.json` - tổng hợp kết quả
- `results/final_comparison_figures/` - tất cả figures cho paper
  - `rc_curve_comparison.png`
  - `ood_comparison.png`
  - `main_results.txt`

**Thời gian:** 3-5 ngày (5 seeds × 3 methods × 8-12h)

**Lưu ý:** Có thể chạy parallel nếu có nhiều GPU:
```bash
# Terminal 1
python run_all_experiments.py --seeds 42 43 --exp_name exp_seed42_43

# Terminal 2
python run_all_experiments.py --seeds 44 45 --exp_name exp_seed44_45

# Terminal 3
python run_all_experiments.py --seeds 46 --exp_name exp_seed46

# Sau đó merge results
```

---

## 📊 Kết Quả Mong Đợi

Sau khi chạy xong tất cả, bạn sẽ có:

### Metrics

| Method | Coverage @ α=0.05 | Risk | DAR (OOD) | Violation Rate |
|--------|------------------|------|-----------|----------------|
| SelectiveNet | ~80% | ~0.07 ❌ | ~30% ⚠️ | ~60% |
| Post-hoc CRC | ~60% | ~0.048 ✅ | ~20% | <5% |
| **CRC-Select** | **~70%** ✨ | **~0.049** ✅ | **~12%** ✨ | **<5%** |

### Figures (Paper-Ready)

1. **RC Curve** - CRC-Select dominates post-hoc
2. **OOD Comparison** - CRC-Select lowest DAR
3. **Violation Analysis** - CRC-Select maintains control
4. **Training Dynamics** - q decreases (selector learns!)

---

## 🐛 Debugging

### Nếu gặp lỗi "Module not found"

```bash
cd /home/admin1/Desktop/selectivenet
pip install -r requirements.txt
```

### Nếu Out of Memory

Edit `models/cifar10_vgg_selectivenet.py`:
```python
# Line ~237: giảm batch_size
batch_size = 64  # thay vì 128
```

### Nếu training không converge

Trong `run_crc_select.py`, thử:
- Tăng `--recalibrate_every` lên 10
- Giảm `--mu_init` xuống 0.1
- Giảm `--mu_lr` xuống 0.001

### Check training progress

```bash
# Xem log
tail -f nohup.out

# Xem checkpoints
ls -lh checkpoints/

# Load và xem CRC history
python -c "import pickle; h=pickle.load(open('checkpoints/crc_select_crc_history.pkl','rb')); print('q values:', h['q'])"
```

---

## 📁 Cấu Trúc Kết Quả

Sau khi chạy xong:

```
selectivenet/
├── checkpoints/
│   ├── baseline_cov0.8.h5
│   ├── crc_select.h5
│   ├── crc_select_crc_history.pkl  ← CRC training history
│   └── ...
│
├── results/
│   ├── baseline_results.json
│   ├── posthoc_crc_results.json
│   ├── crc_select_results.json
│   ├── final_comparison_aggregated.json  ← Main results
│   │
│   ├── final_comparison_figures/  ← Paper figures
│   │   ├── rc_curve_comparison.png
│   │   ├── ood_comparison.png
│   │   └── main_results.txt
│   │
│   └── analysis/  ← Additional analysis
│       ├── methods_comparison.png
│       ├── violation_analysis.png
│       └── ...
```

---

## ⚡ Quick Commands Reference

```bash
# 1. Test nhanh
python quick_test.py --method crcselect --epochs 10

# 2. Baseline
python run_baseline.py --dataset cifar_10 --model_name baseline --seed 42

# 3. Post-hoc CRC
python run_post_hoc_crc.py --baseline baseline --model_name posthoc --seed 42

# 4. CRC-Select
python run_crc_select.py --model_name crcselect --alpha 0.05 --seed 42

# 5. Full comparison
python run_all_experiments.py --seeds 42 43 44 --exp_name final

# 6. Analyze
python analyze_results.py --exp_name final

# Check running jobs
screen -ls

# Reattach to job
screen -r baseline  # or crcselect

# Kill job
screen -X -S baseline quit
```

---

## 📝 Checklist Trước Khi Submit Paper

- [ ] Chạy xong 5 seeds cho cả 3 methods
- [ ] CRC-Select coverage > post-hoc CRC ít nhất 5%
- [ ] Violation rate < 10% across seeds
- [ ] DAR: CRC-Select < SelectiveNet ít nhất 30%
- [ ] Figures generated và trông professional
- [ ] Results reproducible (chạy lại cùng seed = cùng kết quả)
- [ ] Code released trên GitHub
- [ ] README có reproduction instructions

---

## 🎓 Next Steps After Experiments

1. **Viết paper** (Intro, Method, Experiments, Related Work)
2. **Supplementary materials** (thêm ablations, thêm datasets)
3. **Code release** (GitHub repo với instructions)
4. **Submit!** (ICML/NeurIPS/ICLR)

---

## 💡 Tips

1. **Chạy trong screen/tmux** để không bị mất khi disconnect
2. **Monitor GPU usage:** `watch -n 1 nvidia-smi`
3. **Save checkpoints often** - đã tự động trong code
4. **Log output:** `python run_baseline.py 2>&1 | tee logs/baseline.log`
5. **Parallel runs** nếu có nhiều GPU

---

## 🆘 Cần Giúp?

1. Check `QUICK_START.md` cho hướng dẫn cơ bản
2. Check `IMPLEMENTATION_STATUS.md` cho technical details
3. Check `README.md` cho theory
4. Xem code comments - có giải thích chi tiết

---

**Sẵn sàng chạy! Bắt đầu với Bước 1 hoặc chạy quick test!** 🚀

```bash
cd /home/admin1/Desktop/selectivenet
python quick_test.py --method crcselect --epochs 10
```

