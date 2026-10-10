from functools import partial
import math
import os
import torch
import torch.nn as nn

from ..models.backbones.internvideo2.internvideo2 import (
    RMSNorm,
    PatchEmbed,
    Block,
    AttentionPoolingBlock,
    trunc_normal_,
    get_3d_sincos_pos_embed,
) 
from ..utils.config import Config, eval_dict_leaf



try:
    from flash_attn.ops.rms_norm import DropoutAddRMSNorm
except ImportError as exc:
    raise ImportError("DropoutAddRMSNorm from flash_attn is not installed.") from exc


def load_vision_encoder(filepath: str):
    """Load vision encoder weights and configure precision, device, and evaluation mode.
    
    Args:
        filepath: Path to the configuration file.
    
    Returns:
        A configured VisionEncoderStage2 ready for inference.
    
    Raises:
        RuntimeError: If the vision encoder weights path is empty or is not a file.
    """    
    config = Config.from_file(filepath)
    config = eval_dict_leaf(config)

    vision_encoder = VisionEncoderStage2(
        use_flash_attn=config.model.get('use_flash_attn', True),
        use_fused_rmsnorm=config.model.get('use_fused_rmsnorm', True),
        use_fused_mlp=config.model.get('use_fused_mlp', True)
    )


    if config.vision_encoder_weights_path.strip() and os.path.isfile(config.vision_encoder_weights_path):
        vision_encoder.load_state_dict(torch.load(config.vision_encoder_weights_path, weights_only=False), strict=False)
    else:
        raise RuntimeError(
            f"Couldn't load vision model weights from {config.vision_encoder_weights_path}"
        )
    if config.get('use_bf16', False):
        vision_encoder = vision_encoder.to(torch.bfloat16)
    elif config.get('use_half_precision', False):
        vision_encoder = vision_encoder.to(torch.float16)
    else:
        vision_encoder = vision_encoder.to(torch.float32)

    vision_encoder.eval()
    vision_encoder = vision_encoder.to(config.device)
    return vision_encoder

class VisionEncoderStage2(nn.Module):
    """Encode video clips using InternVideo2 and project them into the embedding space."""
    
    def __init__(
                self,
                in_chans: int = 3,
                patch_size: int = 14,
                img_size: int = 224,
                qkv_bias: bool = False,
                drop_path_rate: float = 0.25,
                embed_dim: int = 1408,
                proj_embed_dim: int = 512,
                num_heads: int = 16,
                mlp_ratio: float = 48/11,
                init_values: float = 1e-5,
                qk_normalization: bool = True,
                depth: int = 40,
                use_flash_attn: bool = True,
                use_fused_rmsnorm: bool = True,
                use_fused_mlp: bool = True,
                attn_pool_num_heads: int = 16,
                clip_embed_dim: int = 768,
                num_frames: int = 4,
                tubelet_size: int = 1
            ):
        super().__init__()
        
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        assert use_flash_attn == use_fused_rmsnorm == use_fused_mlp, 'use_flash_attn, use_fused_rmsnorm and use_fused_mlp should be consistent'
        
        self.use_flash_attn = use_flash_attn
        self.embed_dim = embed_dim
        self.depth = depth
        self.img_size = img_size
        
        if use_fused_rmsnorm:
            norm_layer_for_blocks = partial(DropoutAddRMSNorm, eps=1e-6, prenorm=True)
        else:
            norm_layer_for_blocks = partial(RMSNorm, eps=1e-6)
        self.norm_layer_for_blocks = norm_layer_for_blocks
        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_chans, embed_dim,
            num_frames=num_frames, tubelet_size=tubelet_size,
        )
        num_patches = self.patch_embed.num_patches
        num_img_patches = self.patch_embed.num_img_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.img_pos_embed = nn.Parameter(torch.zeros(1, num_img_patches + 1, embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        self.blocks = nn.ModuleList([
            Block(
                embed_dim, 
                num_heads, 
                mlp_ratio, 
                qkv_bias=qkv_bias,
                norm_layer=norm_layer_for_blocks,
                drop_path=dpr[i], 
                init_values=init_values, 
                attn_drop=0.,
                use_flash_attn=use_flash_attn, 
                use_fused_mlp=use_fused_mlp,
                fused_mlp_heuristic=1,
                with_cp=True,
                qk_normalization=qk_normalization,
                layerscale_no_force_fp32=False,
                use_fused_rmsnorm=use_fused_rmsnorm
            ) for i in range(depth)]
        )
        
        self.clip_projector = AttentionPoolingBlock(
            dim=embed_dim, 
            num_heads=attn_pool_num_heads, 
            qkv_bias=True, 
            qk_scale=None,
            drop=0., 
            attn_drop=0., 
            norm_layer=partial(nn.LayerNorm, eps=1e-5), 
            out_dim=clip_embed_dim
        )

        self.vision_proj = nn.Linear(clip_embed_dim, proj_embed_dim)
        
        self.init_pos_embed()
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def init_pos_embed(self):
        """Initialize video and image positional embeddings with sine-cosine values."""
        pos_embed = get_3d_sincos_pos_embed(
            self.pos_embed.shape[-1], 
            self.patch_embed.grid_size[1], # height & weight
            self.patch_embed.grid_size[0], # t_size
            cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        img_pos_embed = get_3d_sincos_pos_embed(
            self.pos_embed.shape[-1], 
            self.patch_embed.grid_size[1], # height & weight
            1,
            cls_token=True
        )
        self.img_pos_embed.data.copy_(torch.from_numpy(img_pos_embed).float().unsqueeze(0))

    def _init_weights(self, m):
        """Initialize linear and LayerNorm parameters."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def fix_init_weight(self):
        """Rescale attention and MLP output weights by transformer layer depth."""
        def rescale(param, layer_id):
            """Divide a parameter in place by the square root of twice its layer index."""
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)
    
    @property
    def dtype(self):
        """Return the data type of the patch embedding weights."""
        return self.patch_embed.proj.weight.dtype

    def get_num_layers(self):
        """Return the number of transformer blocks."""
        return len(self.blocks)

    @torch.inference_mode()
    def forward(self, x):
        """Return projected video embeddings without computing gradients.
        
        Args:
            x: Preprocessed video clips with shape (B, T, C, H, W), on the
                same device as the model.
        
        Returns:
            Unnormalized embeddings with shape (B, proj_embed_dim).
        """
        # [B,T,C,H,W] -> [B,C,T,H,W]
        x = x.permute(0, 2, 1, 3, 4).to(self.dtype)
        x = self.patch_embed(x)
        # T: temporal; L: spatial
        B, T, L, C = x.shape 
        pos_embed = self.img_pos_embed if T == 1 else self.pos_embed
        x = x.view([B, T * L, C])

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + pos_embed
        x = x.reshape(B, -1, C) 

        residual = None
        for idx, blk in enumerate(self.blocks):
            if isinstance(x, tuple) and len(x) == 2:
                x, residual = x
            x = blk(x, residual=residual)
            if idx == self.depth-1:
                break

        if isinstance(x, tuple) and len(x) == 2:
            x, residual = x
            if residual is not None:
                x = x + residual
        
        x = self.clip_projector(x)
        x = self.vision_proj(x)
        return x