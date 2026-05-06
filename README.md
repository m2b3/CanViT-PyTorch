# CanViT (Canvas Vision Transformer) -- PyTorch

[![PyPI Downloads](https://static.pepy.tech/badge/canvit-pytorch)](https://pepy.tech/projects/canvit-pytorch)

<p align="center">
  <img src="assets/canvas_attention_across_scales.png" alt="Canvas attention across scales — two example trajectories showing glimpses, canvas crops, and full canvas PCA/change maps over multiple timesteps." width="100%">
</p>

_[CanViT: Toward Active-Vision Foundation Models](https://arxiv.org/abs/2603.22570) (arXiv:2603.22570)_

**Yohaï-Eliel Berreby, Sabrina Du, Audrey Durand, B. Suresh Krishna**

Reference PyTorch implementation of CanViT, the Canvas Vision Transformer.

### News

- **2026-04-06**: First finetuned IN1k checkpoint: [`canvitb16-add-vpe-finetune-g128px-s512px-in1k-2026-04-06`](https://huggingface.co/canvit/canvitb16-add-vpe-finetune-g128px-s512px-in1k-2026-04-06), with new `CanViTForImageClassification` API.
  - 🎉 CanViT sets a new SOTA on **active-vision IN1k classification**, with **84.5% top-1 accuracy**, up from [AdaptiveNN](https://github.com/LeapLabTHU/AdaptiveNN)'s previous best of 82.2%.
- **2026-03-23**: Preprint v1 ([arXiv:2603.22570](https://arxiv.org/abs/2603.22570)).
  - 🎉 CanViT sets a new SOTA on **active ADE20K segmentation**, with **45.9% ADE20K mIoU**, obtained using linear probing from frozen weights.
- **2026-02-18**: Initial code and [first pretrained checkpoint](https://huggingface.co/canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02) release.

---

CanViT is a scalable recurrent architecture for fine-grained vision, and the first **Active-Vision Foundation Model (AVFM)**: a foundation model for active vision that is both task-agnostic and policy-agnostic.

CanViT processes scenes through sequences of localized glimpses, integrating observations over time into a persistent scene-wide latent workspace — the **canvas** — via **Canvas Attention**, an efficient asymmetric cross-attention mechanism which is based on Scene-Relative Rotary Position Embeddings and eliminates canvas-side QKVO projections.

CanViT-B is pretrained on 1 billion glimpses taken from 13.2 million ImageNet-21k scenes, via **policy-agnostic passive-to-active dense distillation** from a frozen high-resolution DINOv3 ViT-B teacher, without human annotations.

CanViT's scene-wide output features at each timestep are linearly decodable into dense predictions without post-hoc upscaling; a frozen-weights CanViT-B evaluated with linear probing outperforms all prior dense active vision models by a wide margin on ADE20K scene parsing, at a fraction of the cost, while offering significantly greater flexibility.

CanViT generalizes natively across policies, sequence length, glimpse size and canvas size, enabling high-resolution and long-horizon continual pretraining alongside task-specific policy learning.

CanViT enables low-latency high-resolution dense vision, running at hundreds of sequential frames per second on commodity hardware.

## Checkpoints

We release checkpoints on HuggingFace under the [`canvit`](https://huggingface.co/canvit) namespace.

| Checkpoint | Description |
|------------|-------------|
| [`canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02`](https://huggingface.co/canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02) | Pretrained on IN21k via dense distillation from DINOv3 |
| [`canvitb16-add-vpe-finetune-g128px-s512px-in1k-2026-04-06`](https://huggingface.co/canvit/canvitb16-add-vpe-finetune-g128px-s512px-in1k-2026-04-06) | Finetuned for ImageNet-1k classification (trained on TPU v6e via [torch_xla](https://github.com/pytorch/xla)) |

## Quickstart

We recommend [`uv`](https://docs.astral.sh/uv/) for dependency management.

```bash
uv add "canvit-pytorch @ git+https://github.com/m2b3/CanViT-PyTorch.git"
```

A [`canvit-pytorch`](https://pypi.org/project/canvit-pytorch/) package is also available on PyPI but is updated less often — we recommend the git version in most cases.

```python
from canvit_pytorch import CanViTForPretrainingHFHub, Viewpoint, sample_at_viewpoint
from canvit_pytorch.preprocess import preprocess
from PIL import Image
import torch

# CanViT is integrated with the HuggingFace Hub.
model = CanViTForPretrainingHFHub.from_pretrained(
    "canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02"
).eval()

# Replace with the image of your choice
image = Image.open("test_data/Cat03.jpg").convert("RGB")
image = preprocess(512)(image)
image = image.unsqueeze(0)  # [1, 3, 512, 512]

# CanViT is a recurrent model.
state = model.init_state(batch_size=1, canvas_grid_size=32)

# Let's process a first glimpse: centered, zoomed-out.
# You can use any viewpoint you like, as long as it is within bounds.
# CanViT was trained on viewpoints covering 0.25% to 100%
# of a scene's surface area.
with torch.inference_mode():
    vp = Viewpoint.full_scene(batch_size=1, device=image.device)
    glimpse = sample_at_viewpoint(spatial=image, viewpoint=vp, glimpse_size_px=128)
    out = model(glimpse=glimpse, state=state, viewpoint=vp)

# Let's inspect the structure of what we get back.
# The canvas contains the model's working understanding of
# the scene at any given time, and is linearly decodable 
# into dense predictions upon token-wise LayerNorm.
# See `demos/basic.py` for how to visualize the canvas.
canvas_spatial = model.get_spatial(out.state.canvas)  # [1, 1024, 1024]
canvas_spatial = canvas_spatial.unflatten(1, (32, 32))  # [1, 32, 32, 1024] — spatial feature map
out.state.recurrent_cls  # [1, 1, 768] — global CLS token
out.local_patches        # [1, 64, 768] — glimpse patch features

# Now let's do a second glimpse: zoom into the top-left quadrant
# You can do this repeatedly: CanViT is recurrent with a large but constant-size canvas.
with torch.inference_mode():
    vp2 = Viewpoint(centers=torch.tensor([[-.5, -.5]]), scales=torch.tensor([.5]))
    glimpse2 = sample_at_viewpoint(spatial=image, viewpoint=vp2, glimpse_size_px=128)
    out2 = model(glimpse=glimpse2, state=out.state, viewpoint=vp2)
    
# You can use CanViT with frozen weights, fine-tune it, learn a policy on top...
# Or pretrain your own; it's fast.
# Start building!
```

### ImageNet-1k Classification

`CanViTForImageClassification` provides a unified interface for classification. Two construction paths, same forward pass:

**From a finetuned checkpoint** (CanViT + head trained on IN1k):

```python
from canvit_pytorch import CanViTForImageClassification, Viewpoint, sample_at_viewpoint
from canvit_pytorch.preprocess import preprocess
from PIL import Image
import torch

clf = CanViTForImageClassification.from_pretrained(
    "canvit/canvitb16-add-vpe-finetune-g128px-s512px-in1k-2026-04-06"
).eval()
```

**From the frozen pretrained CanViT checkpoint + a [DINOv3 linear probe](https://huggingface.co/canvit/dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe)**:

```python
clf = CanViTForImageClassification.from_pretrained_with_probe(
    pretrained_repo="canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02",
    probe_repo="canvit/dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe",
).eval()
```

**Both have the same forward pass:**

```python
image = preprocess(512)(Image.open("test_data/Cat03.jpg").convert("RGB")).unsqueeze(0)
state = clf.init_state(batch_size=1, canvas_grid_size=32)

with torch.inference_mode():
    vp = Viewpoint.full_scene(batch_size=1, device=image.device)
    glimpse = sample_at_viewpoint(spatial=image, viewpoint=vp, glimpse_size_px=128)
    logits, state = clf(glimpse=glimpse, state=state, viewpoint=vp)

print(logits.argmax(dim=-1))  # ImageNet-1k class index
```

### ADE20K Semantic Segmentation

`CanViTForSemanticSegmentation` bundles a CanViT and a `SegmentationProbe` head into one model. `forward` returns per-pixel logits at canvas-grid resolution; `predict` adds bilinear upsampling.

```python
from canvit_pytorch import CanViTForSemanticSegmentation

# Frozen CanViT + the flagship ADE20K probe (45.9% mIoU, 512px / 64x64 canvas):
seg = CanViTForSemanticSegmentation.from_pretrained_with_probe(
    pretrained_repo="canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02",
    probe_repo="canvit/probe-ade20k-40k-s512-c64-in21k",
).eval()

state = seg.init_state(batch_size=1, canvas_grid_size=64)
logits, state = seg(glimpse=glimpse, state=state, viewpoint=vp)               # [B, n_cls, 64, 64]
upsampled, state = seg.predict(glimpse=glimpse, state=state, viewpoint=vp,
                               target_size=(1024, 1024))                       # [B, n_cls, 1024, 1024]
```

The standalone `SegmentationProbe` head is also exported from `canvit_pytorch` for use on any spatial feature map. Published probes: [canvit ADE20K segmentation probes collection](https://huggingface.co/collections/canvit/canvit-ade20k-segmentation-probes).

## Demos

```bash
git clone https://github.com/m2b3/CanViT-PyTorch.git
cd CanViT-PyTorch

# Classification with sequential glimpses
uv run --extra demo python demos/classify.py                # finetuned checkpoint
uv run --extra demo python demos/classify.py --mode frozen  # frozen CanViT + fused probe

# Canvas PCA visualization with two viewing strategies
uv run --extra demo python demos/basic.py
```

## Supported platforms

- **CPU**
- **CUDA** (tested on RTX 4090, H100 SXM 80GB)
- **TPU** via [torch_xla](https://github.com/pytorch/xla) 2.9.0 (tested on TPU v6e)

We aim to maintain compatibility with [`torch.export`](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export.html) and [ONNX Runtime](https://onnxruntime.ai/). Please [file an issue](https://github.com/m2b3/CanViT-PyTorch/issues) if you encounter problems.

## See also

- [CanViT-pretrain](https://github.com/m2b3/CanViT-pretrain) — pretraining harness (passive-to-active dense distillation from DINOv3)
- [CanViT-MLX](https://github.com/yberreby/CanViT-MLX) — MLX implementation for Apple Silicon (experimental)
- [CanViT-NNX](https://github.com/yberreby/CanViT-NNX) — JAX/Flax NNX implementation (experimental)

## Troubleshooting

If you encounter errors loading pretrained checkpoints, ensure you are using the latest version of the package:

```bash
uv lock --upgrade-package canvit-pytorch && uv sync
```

## Citation

If you use this work, please cite our preprint:

```bibtex
@article{berreby2026canvit,
  title={CanViT: Toward Active-Vision Foundation Models},
  author={Berreby, Yoha{\"i}-Eliel and Du, Sabrina and Durand, Audrey and Krishna, B. Suresh},
  year={2026},
  eprint={2603.22570},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2603.22570}
}
```

## Contact 

Open an issue in this repository or email me@yberreby.com.

## License

MIT. See [LICENSE](LICENSE) for details.
