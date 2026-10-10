# InternVideo

This repository is a fork of [OpenGVLab's InternVideo repository](https://github.com/OpenGVLab/InternVideo).

This fork adds separate vision and text encoder classes for inference with the InternVideo2 Stage 2 checkpoint. These additions are located in `InternVideo2/multi_modality`.

## Installation

The package has been tested with:

- Python 3.11
- CUDA 11.8
- PyTorch 2.4.1
- FlashAttention 2.6.3

The FlashAttention wheel below is specific to Python 3.11, Linux x86-64, and the CUDA/PyTorch/ABI combination encoded in its filename. Install a compatible wheel or build from source if your environment differs.

Install PyTorch before building the extensions:

```bash
python3 -m pip install torch==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu118
```

Compiling the fused dense and layer normalization extensions requires a compatible CUDA toolkit, including `nvcc`, and a C++ compiler. The PyTorch wheel alone does not provide the full build toolchain. Ensure the Python build dependencies are installed:

```bash
python3 -m pip install setuptools wheel packaging ninja
```

Install FlashAttention and the two extensions:

```bash
python3 -m pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu118torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"

git clone --branch v2.6.3 https://github.com/Dao-AILab/flash-attention.git
python3 -m pip install --no-build-isolation ./flash-attention/csrc/fused_dense_lib
python3 -m pip install --no-build-isolation ./flash-attention/csrc/layer_norm
```

For the remaining commands, run from the directory containing the `InternVideo` folder:

```bash
python3 -m pip install --no-build-isolation ./InternVideo/InternVideo2/multi_modality
```

## Model weights

The supported inference checkpoint is [OpenGVLab/InternVideo2-Stage2_1B-224p-f4](https://huggingface.co/OpenGVLab/InternVideo2-Stage2_1B-224p-f4), which uses four frames at a resolution of 224 × 224 pixels.

This [checkpoint repository](https://huggingface.co/loreleva/InternVideo2-Stage2_1B-224p-f4) provides its vision and text encoder weights in separate files. The original model was developed by OpenGVLab.

The download commands use the Hugging Face CLI (`hf`). If it is not available, install a version compatible with the package's pinned Transformers version:

```bash
python3 -m pip install "huggingface_hub>=0.34,<1.0"
```

### Download BERT files

Download the BERT configuration, tokenizer files, and original BERT weights:

```bash
hf download google-bert/bert-large-uncased \
    config.json \
    model.safetensors \
    special_tokens_map.json \
    tokenizer_config.json \
    vocab.txt \
    --local-dir ./InternVideo/model_weights/bert-large-uncased
```

### Download InternVideo2 encoder weights

```bash
hf download loreleva/InternVideo2-Stage2_1B-224p-f4 \
    InternVideo2-stage2_1b-224p-f4-vision_encoder.pt \
    InternVideo2-stage2_1b-224p-f4-text_encoder.pt \
    --local-dir ./InternVideo/model_weights
```

## Usage

Both loaders require a configuration file. A configuration is provided at `InternVideo/internvideo2_config.py`. Update its BERT directory and encoder weight paths to match the downloaded files. Run the example from the same working directory used above.

```python
from internvideo2_multi_modality.inference import (
    load_text_encoder,
    load_vision_encoder,
)
import torch
import torch.nn.functional as F

config_path = "InternVideo/internvideo2_config.py"
text_encoder = load_text_encoder(config_path)
vision_encoder = load_vision_encoder(config_path)

# Expected input shape: (batch_size, num_frames, channels, height, width).
# For this checkpoint: (B, 4, 3, 224, 224).
videos_tensor = ...

with torch.inference_mode():
    text_feat = text_encoder(["A cat playing", "A dog playing"])
    text_feat = F.normalize(text_feat, dim=-1)

    video_feat = vision_encoder(videos_tensor)
    video_feat = F.normalize(video_feat, dim=-1)

    similarities = video_feat @ text_feat.T
```

The video tensor must contain RGB frames sampled, resized, and normalized according to the original model's preprocessing. Its dtype and device must match the vision encoder. The example leaves video loading and preprocessing to the caller; `videos_tensor = ...` must be replaced with an actual tensor.

## Attribution and license

The original InternVideo2 model and code were developed by OpenGVLab. Refer to the [upstream repository](https://github.com/OpenGVLab/InternVideo) for the original implementation and citation information.

The source checkpoint is published under Apache-2.0. Preserve its applicable license and attribution notices when redistributing the extracted weights, and observe any additional access terms accepted when obtaining the original checkpoint.