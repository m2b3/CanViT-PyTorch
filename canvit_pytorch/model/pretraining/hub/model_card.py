"""README.md generation for CanViTForPretraining Hub repos: one template fed by
checkpoint metadata, so the data-specific facts track the weights they describe."""

_DATASET_LABEL = {
    "in21k": "ImageNet-21k",
    "in1k": "ImageNet-1k",
    "sa1b": "SA-1B",
}

_CITATION = r"""```bibtex
@article{berreby2026canvit,
  title={CanViT: Toward Active-Vision Foundation Models},
  author={Berreby, Yoha{\"i}-Eliel and Du, Sabrina and Durand, Audrey and Krishna, B. Suresh},
  year={2026},
  eprint={2603.22570},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2603.22570}
}
```"""


def pretrain_model_card(
    repo_id: str,
    *,
    dataset: str,
    teacher_name: str,
    scene_px: int,
    glimpse_px: int,
    canvas_grid: int,
    step: int | None = None,
) -> str:
    dataset_label = _DATASET_LABEL.get(dataset, dataset)
    step_clause = f" for {step:,} steps" if step is not None else ""
    return f"""---
license: mit
pipeline_tag: image-feature-extraction
---

# CanViT: Toward Active-Vision Foundation Models

CanViT (Canvas Vision Transformer) is a scalable recurrent architecture for fine-grained vision and the first **Active-Vision Foundation Model (AVFM)**. It processes scenes through sequences of localized glimpses, integrating observations over time into a persistent scene-wide latent workspace — the **canvas** — via **Canvas Attention**, an efficient asymmetric cross-attention mechanism.

- **Paper:** [CanViT: Toward Active-Vision Foundation Models](https://arxiv.org/abs/2603.22570)
- **Code:** [https://github.com/m2b3/CanViT-PyTorch](https://github.com/m2b3/CanViT-PyTorch)

## Model Description

CanViT decouples "thinking" (backbone-level) and "memory" (canvas-level), eliminating canvas-side self-attention to achieve low-latency sequential inference. The model was pretrained via policy-agnostic passive-to-active dense latent distillation, reconstructing scene-wide DINOv3 embeddings from sequences of randomized glimpses. It demonstrates strong performance on ADE20K semantic segmentation and ImageNet-1k classification, outperforming prior active vision models.

This checkpoint was pretrained on **{dataset_label}**{step_clause}, distilling a `{teacher_name}` teacher at {scene_px}px scene resolution with {glimpse_px}px glimpses onto a {canvas_grid}×{canvas_grid} canvas.

## Sample Usage

We recommend installing via `uv`:

```bash
uv add canvit-pytorch
```

Or via `pip`:

```bash
pip install canvit-pytorch
```

Then, you can run inference using the following snippet:

```python
from canvit_pytorch import CanViTForPretrainingHFHub, Viewpoint, sample_at_viewpoint
from canvit_pytorch.preprocess import preprocess
from PIL import Image
import torch

# CanViT is integrated with the HuggingFace Hub.
model = CanViTForPretrainingHFHub.from_pretrained(
    "{repo_id}"
).eval()

# Replace with the image of your choice
image = Image.open("test_data/Cat03.jpg").convert("RGB")
image = preprocess({scene_px})(image)
image = image.unsqueeze(0)  # [1, 3, {scene_px}, {scene_px}]

# CanViT is a recurrent model.
state = model.init_state(batch_size=1, canvas_grid_size={canvas_grid})

# Let's process a first glimpse: centered, zoomed-out.
with torch.inference_mode():
    vp = Viewpoint.full_scene(batch_size=1, device=image.device)
    glimpse = sample_at_viewpoint(spatial=image, viewpoint=vp, glimpse_size_px={glimpse_px})
    out = model(glimpse=glimpse, state=state, viewpoint=vp)

# Inspect the canvas structure
# The canvas contains the model's working understanding of the scene
canvas_spatial = model.get_spatial(out.state.canvas)
canvas_spatial = canvas_spatial.unflatten(1, ({canvas_grid}, {canvas_grid}))  # spatial feature map
print(out.state.recurrent_cls.shape)  # global CLS token
```

## Citation

{_CITATION}
"""
