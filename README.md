# augset

| Original Video |
|----------------|
| ![org](assets/gifs/025-bg-02-018.gif) |

<table>
 <tr>
  <td align="center"><img src="assets/gifs/025-bg-02-018-1.gif" width="100%" height="auto" /></td>
  <td align="center"><img src="assets/gifs/025-bg-02-018-2.gif" width="100%" height="auto" /></td>
  <td align="center"><img src="assets/gifs/025-bg-02-018-3.gif" width="100%" height="auto" /></td>
  <td align="center"><img src="assets/gifs/025-bg-02-018-4.gif" width="100%" height="auto" /></td>
  <td align="center"><img src="assets/gifs/025-bg-02-018-5.gif" width="100%" height="auto" /></td>
 </tr>
 <tr>
  <td align="center"><img src="assets/gifs/025-bg-02-018-6.gif" width="100%" height="auto" /></td>
  <td align="center"><img src="assets/gifs/025-bg-02-018-7.gif" width="100%" height="auto" /></td>
  <td align="center"><img src="assets/gifs/025-bg-02-018-8.gif" width="100%" height="auto" /></td>
  <td align="center"><img src="assets/gifs/025-bg-02-018-9.gif" width="100%" height="auto" /></td>
  <td align="center"><img src="assets/gifs/025-bg-02-018-10.gif" width="100%" height="auto" /></td>
 </tr>
</table>

## Installation
You should set the environment variable manually as follows if you want to build a local GPU environment for Grounded-SAM:
```bash
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export CUDA_HOME=/path/to/cuda-11.3/
```

#### Segment Anything
```bash
python -m pip install -e segment_anything
```

#### GroundingDino
```bash
python -m pip install -e GroundingDINO
```

#### Blending Utils
```bash
python -m pip install pyblur
```

[Update pyblur](https://github.com/lospooky/pyblur/issues/5#issue-309942237)

## Pre-Trained Weights
```
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```
