# CRC-Select: Risk-Controlled Selective Prediction via Joint Training

Implementation of CRC-Select, a method that integrates **Conformal Risk Control (CRC)** into **Selective Prediction** training to achieve provably risk-controlled rejection with higher coverage than post-hoc methods.

## 🎯 Key Idea

Traditional selective prediction (SelectiveNet) learns to reject uncertain predictions, but doesn't provide formal risk guarantees. Post-hoc CRC calibration can control risk but may be overly conservative. 

**CRC-Select** jointly trains the selector with CRC objectives, learning a selection strategy that:
- ✅ Achieves formal risk control at level α (e.g., ≤5% error rate)
- ✅ Maximizes coverage (accepts more samples than post-hoc methods)
- ✅ Handles OOD shift better (rejects dangerous OOD samples)

### Workflow Comparison

```
POST-HOC CRC (2-stage):
Train SelectiveNet → Apply CRC → Done
(Selector doesn't know about CRC → conservative)

CRC-SELECT (End-to-End):
Train ← Recalibrate CRC ← Train ← Recalibrate CRC ← ...
(Selector learns "what CRC likes" → higher coverage)
```

## 📊 Expected Results

| Method | Coverage @ α=0.05 | DAR (OOD) | Risk Violation |
|--------|-------------------|-----------|----------------|
| SelectiveNet | ~75% | **High** ↗ | May violate |
| Post-hoc CRC | ~60% | Medium | ✓ Controlled |
| **CRC-Select** | **~70%** | **Low** ↘ | ✓ Controlled |

*DAR = Dangerous Acceptance Rate (lower is better)*

## 🏗️ Project Structure

```
CRC-Selection/
├── README.md                           # This file
├── IMPLEMENTATION_SUMMARY.md           # Technical summary
├── HOÀN_THÀNH_100%.md                  # Completion report
└── TRIỂN_KHAI_HOÀN_TẤT.md             # Vietnamese summary

selectivenet/                           # Main codebase
├── Core modules (NEW - 4 files, ~1200 lines)
│   ├── crc_utils.py                   # CRC calibration & metrics ✅
│   ├── data_utils.py                  # Data splitting & OOD ✅
│   ├── eval_utils.py                  # Evaluation & visualization ✅
│   └── crc_select_trainer.py          # Training components ✅
│
├── Experiment scripts (NEW - 6 files, ~1600 lines)
│   ├── run_baseline.py                # Phase 1: SelectiveNet ✅
│   ├── run_post_hoc_crc.py            # Phase 3: Post-hoc CRC ✅
│   ├── run_crc_select.py              # Phase 4: CRC-Select ✅ ⭐
│   ├── run_all_experiments.py         # Full comparison ✅
│   ├── analyze_results.py             # Analysis & plots ✅
│   └── quick_test.py                  # Quick debugging ✅
│
├── Documentation (NEW - 7 files, ~2700 lines)
│   ├── START_HERE.md                  # 👈 Read first!
│   ├── RUN_EXPERIMENTS.md             # Complete guide
│   ├── QUICK_START.md                 # Quick guide
│   ├── IMPLEMENTATION_STATUS.md       # Progress tracking
│   ├── IMPLEMENTATION_SUMMARY.md      # Technical details
│   └── requirements.txt               # Dependencies
│
├── Original SelectiveNet (unchanged)
│   ├── models/                        # VGG architectures
│   │   ├── cifar10_vgg_selectivenet.py
│   │   ├── svhn_vgg_selectivenet.py
│   │   └── catdog_vgg_selectivenet.py
│   ├── train.py                       # Original training
│   └── selectivnet_utils.py           # Original utilities
│
└── Output folders (created by scripts)
    ├── checkpoints/                   # Model weights
    ├── results/                       # JSON results
    └── results/figures/               # Paper plots

Total NEW: 17 files, ~4300 lines of code + documentation
Status: ✅ 100% COMPLETE - READY TO RUN
```

## 🚀 Getting Started - Multiple Pathways

### 🤔 Which Path Should I Take?

```
START HERE
    │
    ├─ First time using this code?
    │  └─ YES → 🏃 Quick Test (10 min)
    │
    ├─ Want to validate the approach?
    │  └─ YES → 🎯 Standard Flow (1-2 days)
    │
    ├─ Preparing for paper submission?
    │  └─ YES → 🔬 Full Research (3-5 days)
    │
    ├─ Already ran experiments?
    │  └─ YES → 📊 Analysis Only (5 min)
    │
    └─ Need specific component only?
       └─ YES → 🛠️ Individual Components
```

### Path Overview

| Path | Time | Goal | Recommended For |
|------|------|------|-----------------|
| **🏃 Quick Test** | 10 min | Test code works | First-time users, debugging |
| **🎯 Standard Flow** | 1-2 days | Core comparison (1 seed) | Quick validation, demos |
| **🔬 Full Research** | 3-5 days | Full comparison (5 seeds) | Paper submission, publication |
| **📊 Analysis Only** | 5 min | Analyze existing results | Review, visualization |
| **🛠️ Individual** | Varies | Run specific parts | Custom experiments |

---

## 🏃 PATH 1: Quick Test (10 Minutes)

**Goal:** Verify code works without waiting for full training

### Step 1: Install

```bash
cd /home/admin1/Desktop/selectivenet
pip install -r requirements.txt
```

### Step 2: Quick Test

```bash
# Test CRC-Select with just 10 epochs (not real results!)
python quick_test.py --method crcselect --epochs 10
```

**Expected:** Script runs without errors (results meaningless)

**Next:** If successful, proceed to Path 2 or Path 3

---

## 🎯 PATH 2: Standard Flow (1-2 Days)

**Goal:** Run core experiments to validate the approach

### Step 1: Setup

```bash
cd /home/admin1/Desktop/selectivenet
pip install -r requirements.txt

# Optional: Download SVHN for OOD evaluation
mkdir -p datasets && cd datasets
wget http://ufldl.stanford.edu/housenumbers/test_32x32.mat
cd ..
```

### Step 2: Baseline (6-12 hours)

```bash
# Run in background with screen
screen -S baseline
python run_baseline.py \
    --dataset cifar_10 \
    --model_name baseline \
    --seed 42
# Press Ctrl+A then D to detach
# Reattach: screen -r baseline
```

**Output:** `results/baseline_results.json`, `checkpoints/baseline_cov*.h5`

### Step 3: Post-hoc CRC (5 minutes)

```bash
# After baseline completes
python run_post_hoc_crc.py \
    --baseline baseline \
    --model_name posthoc \
    --seed 42
```

**Output:** `results/posthoc_results.json`

### Step 4: CRC-Select - CORE (8-12 hours)

```bash
screen -S crcselect
python run_crc_select.py \
    --model_name crcselect \
    --alpha 0.05 \
    --seed 42
# Ctrl+A, D to detach
```

**Output:** `results/crcselect_results.json`, `checkpoints/crcselect_crc_history.pkl`

### Step 5: Compare Results

```bash
python analyze_results.py --exp_name comparison
```

**Expected Results:**
- CRC-Select coverage ~10-15% higher than post-hoc
- Risk controlled for both (≤ α)
- CRC-Select has lower OOD DAR

---

## 🔬 PATH 3: Full Research Workflow (3-5 Days)

**Goal:** Complete comparison with multiple seeds for paper

### Step 1: Setup (same as Path 2)

### Step 2: Full Comparison - All Methods, Multiple Seeds

```bash
screen -S fullexp
python run_all_experiments.py \
    --dataset cifar_10 \
    --exp_name final_comparison \
    --alpha 0.05 \
    --seeds 42 43 44 45 46 \
    --ood svhn
# Ctrl+A, D to detach
```

**This runs:**
- SelectiveNet baseline (5 seeds)
- Post-hoc CRC (5 seeds)
- CRC-Select (5 seeds)

**Total time:** 3-5 days (can parallelize with multiple GPUs)

### Step 3: Generate All Figures

```bash
# After experiments complete
python analyze_results.py --exp_name final_comparison
```

**Output:** `results/final_comparison_figures/`
- `rc_curve_comparison.png` - Main result
- `ood_comparison.png` - OOD safety
- `violation_analysis.png` - Risk control stability
- `main_results.txt` - Table for paper

### Step 4: Ablation Studies (Optional)

```bash
# Different recalibration frequencies
python run_crc_select.py --recalibrate_every 1 --model_name crc_T1
python run_crc_select.py --recalibrate_every 10 --model_name crc_T10
python run_crc_select.py --recalibrate_every 20 --model_name crc_T20

# Different calibration set sizes
python run_crc_select.py --cal_ratio 0.1 --model_name crc_cal10
python run_crc_select.py --cal_ratio 0.3 --model_name crc_cal30

# Compare results
python analyze_results.py --exp_name ablations
```

---

## 📊 PATH 4: Analysis Only (5 Minutes)

**Goal:** Analyze existing results (if experiments already run)

```bash
cd /home/admin1/Desktop/selectivenet

# Analyze specific experiment
python analyze_results.py --exp_name final_comparison

# Check CRC training dynamics
ls checkpoints/*_crc_history.pkl

# View results
cat results/final_comparison_aggregated.json | python -m json.tool
```

---

## 🛠️ PATH 5: Individual Components

Run specific parts only:

### Option A: Just Baseline

```bash
python run_baseline.py --dataset cifar_10 --model_name mybaseline --seed 42
```

### Option B: Just Post-hoc CRC

```bash
# Requires existing baseline
python run_post_hoc_crc.py --baseline mybaseline --model_name myposthoc --seed 42
```

### Option C: Just CRC-Select

```bash
python run_crc_select.py --model_name mycrcselect --alpha 0.05 --seed 42
```

---

## 🔄 Parallel Execution (Multiple GPUs)

If you have multiple GPUs, run seeds in parallel:

```bash
# Terminal 1 (GPU 0)
CUDA_VISIBLE_DEVICES=0 python run_all_experiments.py --seeds 42 43 --exp_name exp_gpu0

# Terminal 2 (GPU 1)
CUDA_VISIBLE_DEVICES=1 python run_all_experiments.py --seeds 44 45 --exp_name exp_gpu1

# Terminal 3 (GPU 2)
CUDA_VISIBLE_DEVICES=2 python run_all_experiments.py --seeds 46 --exp_name exp_gpu2

# Merge results later
# (Results are independent, just combine JSON files)
```

---

## 📋 Detailed Step-by-Step Guides

For detailed instructions on each step:

### 📖 Phase 1: Baseline SelectiveNet

**What:** Train SelectiveNet at multiple coverage levels  
**Time:** 6-12 hours  
**Prerequisite:** None

```bash
python run_baseline.py \
    --dataset cifar_10 \
    --model_name baseline \
    --alpha 0.5 \
    --seed 42
```

**Arguments:**
- `--dataset`: Dataset to use (`cifar_10`, `svhn`, `catsdogs`)
- `--model_name`: Name for saving checkpoints
- `--alpha`: SelectiveNet loss weight (0-1, default 0.5)
- `--seed`: Random seed for reproducibility

**Output:**
- `checkpoints/baseline_cov*.h5` - Model checkpoints (6 coverage levels)
- `results/baseline_results.json` - Metrics (coverage, risk, accuracy, DAR)
- `results/baseline_rc_curve.png` - Risk-Coverage curve

**Check Success:**
```bash
ls checkpoints/baseline_cov*.h5  # Should see 6 files
cat results/baseline_results.json | python -m json.tool
```

---

### 📖 Phase 2: OOD Evaluation

**Included in Phase 1** - OOD metrics automatically computed if SVHN dataset available

To ensure OOD evaluation:
```bash
mkdir -p datasets && cd datasets
wget http://ufldl.stanford.edu/housenumbers/test_32x32.mat
cd ..
```

---

### 📖 Phase 3: Post-hoc CRC

**What:** Apply CRC calibration to trained SelectiveNet  
**Time:** 5 minutes (no training)  
**Prerequisite:** Phase 1 must be complete

```bash
python run_post_hoc_crc.py \
    --dataset cifar_10 \
    --baseline baseline \
    --coverage 0.8 \
    --model_name posthoc_crc \
    --seed 42
```

**Arguments:**
- `--baseline`: Name of baseline model from Phase 1
- `--coverage`: Which coverage model to use (0.8 recommended)
- `--model_name`: Name for this experiment

**Output:**
- `results/posthoc_crc_results.json` - Coverage at different α values
- `results/posthoc_crc_rc_curve.png` - RC curve

**Check Success:**
```bash
cat results/posthoc_crc_results.json | python -m json.tool
# Look for: "alpha_0.05" -> "coverage" and "risk"
```

---

### 📖 Phase 4: CRC-Select (CORE CONTRIBUTION)

**What:** Train with alternating CRC optimization  
**Time:** 8-12 hours  
**Prerequisite:** None (trains from scratch)

```bash
python run_crc_select.py \
    --dataset cifar_10 \
    --model_name crcselect \
    --alpha 0.05 \
    --coverage 0.8 \
    --recalibrate_every 5 \
    --epochs 300 \
    --seed 42
```

**Key Arguments:**
- `--alpha`: Target risk level (e.g., 0.05 = 5%)
- `--coverage`: Target coverage rate
- `--recalibrate_every`: Recalibrate CRC every T epochs (default 5)
- `--mu_init`: Initial penalty weight (default 1.0)
- `--mu_lr`: Dual learning rate (default 0.01)

**Output:**
- `checkpoints/crcselect.h5` - Trained model
- `checkpoints/crcselect_crc_history.pkl` - **CRC training history (q, μ, risk, coverage)**
- `results/crcselect_results.json` - Final metrics

**Check Success:**
```bash
# Check if training completed
ls checkpoints/crcselect*.h5

# Inspect CRC history (q should decrease over time)
python -c "import pickle; h=pickle.load(open('checkpoints/crcselect_crc_history.pkl','rb')); print('q values:', h['q'][:5])"

# Check final results
cat results/crcselect_results.json | python -m json.tool
```

**Debug if q not decreasing:**
- Try larger `--recalibrate_every` (10 or 20)
- Try smaller `--mu_init` (0.1 or 0.5)
- Check if selector is learning (coverage should be near target)

---

### 📖 Phase 5: Full Comparison

**What:** Run all methods with multiple seeds  
**Time:** 3-5 days (or parallelize)  
**Prerequisite:** None (runs everything)

```bash
python run_all_experiments.py \
    --dataset cifar_10 \
    --exp_name final_comparison \
    --alpha 0.05 \
    --coverage 0.8 \
    --seeds 42 43 44 45 46
```

**Arguments:**
- `--exp_name`: Name for organizing results
- `--seeds`: List of seeds (space-separated)
- `--skip_baseline`: Skip if baseline already run
- `--skip_posthoc`: Skip if post-hoc already run
- `--skip_crcselect`: Skip if CRC-Select already run

**Output:**
- `results/final_comparison_aggregated.json` - Summary across seeds
- `results/final_comparison_figures/` - All paper figures
  - `rc_curve_comparison.png`
  - `ood_comparison.png`
  - `violation_analysis.png`
  - `main_results.txt`

**Check Success:**
```bash
# View summary
cat results/final_comparison_aggregated.json | python -m json.tool

# View figures
xdg-open results/final_comparison_figures/rc_curve_comparison.png

# Should show:
# - CRC-Select coverage > Post-hoc CRC (by ~10%)
# - Both have risk ≤ α
# - CRC-Select has lower OOD DAR
```

## 📈 Evaluation Metrics

### Primary Metrics

1. **Coverage @ Risk α**: Maximum coverage achieving risk ≤ α
   - Higher is better
   - Main comparison metric

2. **Risk @ Coverage c**: Actual risk achieved at coverage c
   - Lower is better
   - Check if ≤ α (violation?)

3. **Dangerous Acceptance Rate (DAR)**: Fraction of OOD samples accepted
   - Lower is better
   - Measures OOD safety

### Secondary Metrics

4. **Violation Rate**: How often risk > α across seeds
5. **Coverage Gap**: CRC-Select vs Post-hoc coverage difference
6. **Calibration Stability**: Variance in q across recalibration steps

## 🔬 Experimental Design

### Datasets

| Dataset | Type | # Train | # Test | # Classes |
|---------|------|---------|--------|-----------|
| CIFAR-10 | ID | 50k | 10k | 10 |
| SVHN | OOD | - | 26k | - |
| CIFAR-100 | OOD | - | 10k | - |

### Data Splits

- **Train:** 80% of original train (for model updates)
- **Calibration:** 20% of original train (for CRC, held-out)
- **Test:** Original test set (for final evaluation)

### Hyperparameters

**SelectiveNet:**
- Coverage targets: [0.95, 0.9, 0.85, 0.8, 0.75, 0.7]
- Alpha (SelectiveNet loss weight): 0.5
- Epochs: 300
- Batch size: 128

**CRC:**
- Target risk α: [0.01, 0.05, 0.10]
- Loss function: Cross-entropy (monotone)
- Calibration set size: 20% of train

**CRC-Select:**
- Recalibrate q every: 5 epochs
- Initial μ: 1.0
- Dual learning rate: 0.01
- Coverage regularizer weight: 32

### Ablations (Phase 5)

1. **Recalibration frequency:** T ∈ {1, 5, 10, 20} epochs
2. **Calibration set size:** {10%, 20%, 30%}
3. **Risk penalty:** with/without μ dual update
4. **Loss function:** Cross-entropy vs 0/1 surrogate

## 📝 Code Implementation Details

### CRC Calibration (stop-gradient)

```python
# In crc_utils.py
def crc_calibrate(risk_scores, selection_scores, alpha, threshold=0.5):
    # 1. Get accepted samples
    accepted_mask = selection_scores >= threshold
    accepted_risks = risk_scores[accepted_mask]
    
    # 2. Compute CRC threshold q (quantile-based)
    q = np.quantile(accepted_risks, 1.0 - alpha)
    
    return q  # Stop-gradient: no backprop through q
```

### Alternating Training Loop

```python
# In crc_select_trainer.py
for epoch in range(epochs):
    if epoch % recalibrate_every == 0:
        # Calibration step (no gradient)
        q = compute_crc_threshold(model, x_cal, y_cal, alpha)
        
    # Training step (q fixed)
    for batch in train_loader:
        loss = l_pred + lambda_cov * l_cov + mu * l_risk(q)
        loss.backward()  # q is constant here
        optimizer.step()
        
    # Dual update for mu
    if cal_risk > alpha:
        mu += lr * (cal_risk - alpha)
```

### Risk Penalty Term

```python
def l_risk(predictions, selection_scores, q):
    """Penalize accepting high-risk samples."""
    risk_scores = compute_risk_scores(predictions, labels)
    weighted_risk = selection_scores * risk_scores
    return mean(weighted_risk)
```

## 🎓 Theoretical Foundations

### CRC Guarantee

Under exchangeability, CRC controls:

$$
\mathbb{E}[\text{Risk}(f, g) \mid \text{Accepted}] \leq \alpha
$$

where Risk is a monotone loss (e.g., 0/1, cross-entropy, FN-weighted).

### CRC-Select Intuition

- **Post-hoc:** Selector doesn't know CRC will be applied → may be conservative
- **Joint training:** Selector learns "what CRC likes" → less conservative, higher coverage
- **Stop-gradient:** Allows alternating optimization without instability

### Why Alternating Works

This is **bilevel optimization:**
- **Outer:** max coverage
- **Inner:** CRC ensures risk ≤ α

Standard approach: alternate between levels (also used in fairness constraints, rate-constrained learning).

## 📚 References

1. **SelectiveNet:** Geifman & El-Yaniv. "SelectiveNet: A Deep Neural Network with an Integrated Reject Option." ICML 2019.

2. **Conformal Risk Control:** Angelopoulos et al. "Conformal Risk Control." ICLR 2024.

3. **Related work:**
   - Learning to Reject (Bartlett & Wegkamp, 2008)
   - Selective Classification (El-Yaniv & Wiener, 2010)
   - Conformal Prediction (Vovk et al., 2005)

## 🛠️ Implementation Status

- [x] Phase 0: Setup & code structure
- [x] Phase 1: SelectiveNet baseline - `run_baseline.py` ✅
- [x] Phase 2: OOD evaluation harness ✅
- [x] Phase 3: Post-hoc CRC baseline - `run_post_hoc_crc.py` ✅
- [x] **Phase 4: CRC-Select core - `run_crc_select.py` ✅**
- [x] Phase 5: Full comparison - `run_all_experiments.py` ✅
- [x] Analysis & visualization - `analyze_results.py` ✅
- [ ] Phase 6: Medical imaging extension (FN-weighted loss)

**Status:** ✅ **READY TO RUN EXPERIMENTS!**

All core implementation complete. See `START_HERE.md` and `RUN_EXPERIMENTS.md` for detailed instructions.

## 🐛 Troubleshooting & FAQ

### Common Issues

**Q: "Module not found" error**
```bash
cd /home/admin1/Desktop/selectivenet
pip install -r requirements.txt
```

**Q: Out of memory (GPU)**
- Edit model files, reduce `batch_size` from 128 to 64
- Or use CPU (slower but works)

**Q: Training very slow**
```bash
# Check if using GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**Q: Baseline model not found (Phase 3)**
- Must run Phase 1 first
- Check model name matches: `--baseline baseline` should match `--model_name baseline` from Phase 1

**Q: CRC-Select not improving coverage**
- Check `q` values in `checkpoints/*_crc_history.pkl` - should decrease
- If not decreasing:
  - Increase `--recalibrate_every` to 10 or 20
  - Decrease `--mu_init` to 0.1
  - Check selector is learning (coverage near target)

**Q: Results not reproducible**
- Ensure same seed: `--seed 42`
- Check GPU determinism (may have small variance)

**Q: How to monitor training?**
```bash
# View logs
tail -f nohup.out

# Check checkpoints
ls -lh checkpoints/

# Reattach to screen session
screen -r baseline  # or crcselect
```

### Performance Tips

**Use screen/tmux for long jobs:**
```bash
screen -S jobname
python run_baseline.py ...
# Ctrl+A, D to detach
# screen -r jobname to reattach
```

**Parallel execution with multiple GPUs:**
```bash
CUDA_VISIBLE_DEVICES=0 python run_crc_select.py --seed 42 &
CUDA_VISIBLE_DEVICES=1 python run_crc_select.py --seed 43 &
```

**Save logs:**
```bash
python run_baseline.py 2>&1 | tee logs/baseline.log
```

---

## 🤝 Contributing

This is a research project. Main development in `/home/admin1/Desktop/selectivenet/`.

**File Structure:**
- Core logic: `crc_utils.py`, `data_utils.py`, `eval_utils.py`
- Experiment runners: `run_*.py`
- Documentation: `*.md` files

## 📄 License

Based on SelectiveNet (original repo: https://github.com/geifmany/selectivenet)

## 📧 Contact

For questions about CRC-Select implementation, open an issue.

---

## 🎯 Quick Command Reference

```bash
# Navigate to project
cd /home/admin1/Desktop/selectivenet

# Quick test (10 min)
python quick_test.py --method crcselect --epochs 10

# Phase 1: Baseline (6-12h)
python run_baseline.py --dataset cifar_10 --model_name baseline --seed 42

# Phase 3: Post-hoc CRC (5 min)
python run_post_hoc_crc.py --baseline baseline --model_name posthoc --seed 42

# Phase 4: CRC-Select (8-12h) ⭐
python run_crc_select.py --model_name crcselect --alpha 0.05 --seed 42

# Phase 5: Full comparison (3-5 days)
python run_all_experiments.py --seeds 42 43 44 --exp_name final

# Analyze results
python analyze_results.py --exp_name final
```

---

## 🎓 Next Steps - Personalized Guide

### If You're A...

**🆕 First-time User:**
```bash
cd /home/admin1/Desktop/selectivenet
cat START_HERE.md              # Read first
python quick_test.py           # Test (10 min)
cat RUN_EXPERIMENTS.md         # Learn how to run
```

**🎯 Researcher (Quick Validation):**
```bash
# Standard Flow (1-2 days, 1 seed)
python run_baseline.py --seed 42        # 6-12h
python run_post_hoc_crc.py --seed 42    # 5 min
python run_crc_select.py --seed 42      # 8-12h
python analyze_results.py               # 1 min
```

**📝 PhD Student (Paper Submission):**
```bash
# Full Research (3-5 days, 5 seeds)
python run_all_experiments.py --seeds 42 43 44 45 46
# Wait 3-5 days (or parallelize)
python analyze_results.py --exp_name final_comparison
# Get paper-ready figures in results/figures/
```

**🔧 Developer (Custom Experiments):**
```bash
# Individual components
python run_baseline.py --dataset svhn    # Different dataset
python run_crc_select.py --alpha 0.01    # Different risk level
python run_crc_select.py --recalibrate_every 10  # Ablation
```

**📊 Reviewer (Analysis Only):**
```bash
# If results already exist
cd /home/admin1/Desktop/selectivenet
python analyze_results.py --exp_name <experiment_name>
ls results/*.json                        # See all results
ls results/figures/                      # See all plots
```

---

## 📚 Documentation Index

| File | Purpose | Read When |
|------|---------|-----------|
| **`START_HERE.md`** | Entry point | First time 👈 |
| **`RUN_EXPERIMENTS.md`** | Complete guide | Before running |
| `QUICK_START.md` | Quick reference | Need fast help |
| `IMPLEMENTATION_STATUS.md` | Technical details | Deep dive |
| `IMPLEMENTATION_SUMMARY.md` | Overall summary | Overview |
| This README | Project overview | Context |

All files located in `/home/admin1/Desktop/selectivenet/`

---

## ✅ Pre-Flight Checklist

Before running experiments:

- [ ] **Installed dependencies:** `pip install -r requirements.txt`
- [ ] **GPU available (optional):** `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`
- [ ] **Disk space:** Need ~10GB for checkpoints + results
- [ ] **Time allocated:** Quick test (10 min), Standard (1-2 days), Full (3-5 days)
- [ ] **SVHN downloaded (optional):** For OOD evaluation
- [ ] **Read documentation:** At least `START_HERE.md`

---

## 🚀 Start Now!

**Choose ONE command to start:**

```bash
# Option 1: Quick test (safest start)
cd /home/admin1/Desktop/selectivenet && python quick_test.py

# Option 2: Read guide first
cd /home/admin1/Desktop/selectivenet && cat START_HERE.md

# Option 3: Jump to experiments
cd /home/admin1/Desktop/selectivenet && python run_baseline.py --seed 42
```

**Questions?** See Troubleshooting section above or read `RUN_EXPERIMENTS.md`

Good luck! 🎉

