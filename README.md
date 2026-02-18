# CanViT (Canvas Vision Transformer) -- PyTorch

**Yohaï-Eliel Berreby, Sabrina Du, Audrey Durand, Suresh Krishna**

Reference PyTorch implementation of CanViT (Canvas Vision Transformer).

_This is an early release. For details, a preprint version of our manuscript "CanViT: Toward Active Vision Foundation Models" will be available in the coming weeks._

---

CanViT is a scalable recurrent architecture for fine-grained vision, and the first **Active Vision Foundation Model (AVFM)**: a foundation model for active vision that is both task-agnostic and policy-agnostic.

CanViT processes scenes through sequences of localized glimpses, integrating observations over time into a persistent scene-wide latent workspace — the **canvas** — via **Canvas Attention**, an efficient asymmetric cross-attention mechanism which is based on Scene-Relative Rotary Position Embeddings and eliminates canvas-side QKVO projections.

CanViT-B is pretrained on 1 billion glimpses taken from 13.5 million ImageNet-21k scenes, via **policy-agnostic passive-to-active dense distillation** from a frozen DINOv3 ViT-B teacher, without human annotations.

## Quickstart

```bash
uv add canvit-pytorch
```

```python
from canvit_pytorch import CanViTForPretrainingHFHub, Viewpoint, sample_at_viewpoint
from canvit_pytorch.preprocess import preprocess
from PIL import Image
import torch

model = CanViTForPretrainingHFHub.from_pretrained(
    "canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02"
).eval()

image = preprocess(512)(Image.open("test_data/Cat03.jpg").convert("RGB"))
image = image.unsqueeze(0)  # [1, 3, 512, 512]

state = model.init_state(batch_size=1, canvas_grid_size=32)
vp = Viewpoint.full_scene(batch_size=1, device=image.device)
glimpse = sample_at_viewpoint(spatial=image, viewpoint=vp, glimpse_size_px=128)

with torch.inference_mode():
    out = model(glimpse=glimpse, state=state, viewpoint=vp)

canvas_spatial = model.get_spatial(out.state.canvas)  # [1, 1024, 1024]
canvas_spatial = canvas_spatial.unflatten(1, (32, 32))  # [1, 32, 32, 1024] — spatial feature map
out.state.recurrent_cls  # [1, 1, 768] — global CLS token
out.local_patches        # [1, 64, 768] — glimpse patch features

# Second glimpse: zoom into the top-left quadrant
vp2 = Viewpoint(centers=torch.tensor([[-.5, -.5]]), scales=torch.tensor([.5]))
glimpse2 = sample_at_viewpoint(spatial=image, viewpoint=vp2, glimpse_size_px=128)
with torch.inference_mode():
    out2 = model(glimpse=glimpse2, state=out.state, viewpoint=vp2)
```

For a full demo with classification and PCA visualization:

```bash
git clone https://github.com/m2b3/CanViT-PyTorch.git
cd CanViT-PyTorch
uv run --extra demo python demos/basic.py
```

## Pretrained checkpoints

We release checkpoints on HuggingFace under the [`canvit`](https://huggingface.co/canvit) namespace.

The following checkpoints are currently available:

- [`canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02`](https://huggingface.co/canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02)


## Citation

If you use this work, please cite this repository.
An updated citation will be available upon preprint release.

```bibtex
@misc{berreby2026canvit,
  title={CanViT: Toward Active Vision Foundation Models},
  author={Berreby, Yoha{\"i}-Eliel and Du, Sabrina and Durand, Audrey and Krishna, Suresh},
  year={2026},
  howpublished={\url{https://github.com/m2b3/CanViT-PyTorch}}
}
```

## License

MIT. See [LICENSE.md](LICENSE.md) for details.
