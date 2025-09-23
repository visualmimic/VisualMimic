# VisualMimic
[[project page]](https://visualmimic.github.io/) [[arXiv]](TBD)

 **VisualMimic: Visual Humanoid Loco-Manipulation via Motion Tracking and Generation**, arXiv 2025.
 Shaofeng Yin*, Yanjie Ze*, Hong-Xing Yu, C. Karen Liu†, Jiajun Wu† (*Equal contribution, †Equal advising)
*Stanford University*

<table>
<tr>
<td align="center">
  <img src="asset/videos/sim2sim/kick_ball_sim2sim.gif" width="300"/>
  <br/>
  <b>Kick Ball</b>
</td>
<td align="center">
  <img src="asset/videos/sim2sim/kick_box_sim2sim.gif" width="300"/>
  <br/>
  <b>Kick Box</b>
</td>
</tr>
<tr>
<td align="center">
  <img src="asset/videos/sim2sim/lift_box_sim2sim.gif" width="300"/>
  <br/>
  <b>Lift Box</b>
</td>
<td align="center">
  <img src="asset/videos/sim2sim/push_box_sim2sim.gif" width="300"/>
  <br/>
  <b>Push Box</b>
</td>
</tr>
</table>


## News
- [2025/09/24] Sim2Sim pipeline and checkpoints released.

## Installation

The code is tested on Ubuntu 20.04.

```bash
conda create -n visualmimic python=3.8
conda activate visualmimic
pip install -r requirements.txt
```

## Usage

### Sim2Sim

```bash
cd sim2sim
python sim2sim.py --task kick_ball # kick_box, push_box, lift_box
```


## Release Progress

- [x] Sim2Sim pipeline
- [x] Checkpoints for real world tasks
- [ ] Low-level tracker training code
- [ ] Low-level tracker checkpoint
- [ ] High-level policy training code
- [ ] Sim2Real pipeline

## Contact

If you have any questions, please contact yinshaofeng04@gmail.com.
