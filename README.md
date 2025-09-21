# VisualMimic

This is the official code base for the paper **VisualMimic: Visual Humanoid Loco-Manipulation via Motion Tracking and Generation**.

## 🔥 News
- [2025/09/22] Sim2Sim pipeline and checkpoints released!

## 🛠️ Installation

The code is tested on Ubuntu 20.04.

```bash
conda create -n visualmimic python=3.8
conda activate visualmimic
pip install -r requirements.txt
```

## 🚀 Usage

### Sim2Sim

```bash
cd sim2sim
python sim2sim.py --task kick_ball # kick_box, push_box, lift_box
```

## 📜 Release Progress

- [x] Sim2Sim pipeline
- [x] Checkpoints for all tasks
- [ ] Low-level tracker training code
- [ ] Low-level tracker checkpoint
- [ ] High-level policy training code
- [ ] Sim2Real pipeline

## 🤝 Contact

If you have any questions, please contact yinshaofeng04@gmail.com.