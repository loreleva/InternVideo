import os
import torch
import torch.nn as nn

from ..models.backbones.bert.xbert import BertConfig, BertForMaskedLM
from ..models.backbones.bert.tokenization_bert import BertTokenizer
from ..utils.config import Config, eval_dict_leaf


def load_text_encoder(filepath: str):
    """Load text encoder weights and configure precision, device, and evaluation mode.
    
    Args:
        filepath: Path to the configuration file.
    
    Returns:
        A configured BertTextEncoder ready for inference.
    """
    config = Config.from_file(filepath)
    config = eval_dict_leaf(config)

    text_encoder = BertTextEncoder(config)

    if config.text_encoder_weights_path.strip() and os.path.isfile(config.text_encoder_weights_path):
        text_encoder.load_state_dict(torch.load(config.text_encoder_weights_path, weights_only=False), strict=False)
    else:
        raise RuntimeError("Couldn't load text model weights from {config.text_encoder_weights_path}")

    if config.get('use_bf16', False):
        text_encoder = text_encoder.to(torch.bfloat16)
    elif config.get('use_half_precision', False):
        text_encoder = text_encoder.to(torch.float16)
    else:
        text_encoder = text_encoder.to(torch.float32)

    text_encoder.eval()
    text_encoder = text_encoder.to(config.device)
    return text_encoder


class BertTextEncoder(nn.Module):
    """Encode text using BERT and project its CLS features into the embedding space."""
    
    def __init__(self, config):
        super().__init__()
        bert_config = BertConfig.from_json_file(config.model.text_encoder.config)
        bert_config.encoder_width = config.model.vision_encoder.d_model                
        bert_config.gradient_checkpointing = True
        bert_config.fusion_layer = config.model.text_encoder.fusion_layer
        self.device = config.device
        self.max_txt_l = config.max_txt_l

        self.tokenizer = BertTokenizer.from_pretrained(
            config.model.text_encoder.pretrained, 
            local_files_only=True
        )
        self.text_encoder, _ = BertForMaskedLM.from_pretrained(
            config.model.text_encoder.pretrained,
            config=bert_config,
            output_loading_info=True, 
            local_files_only=True
        )
        self.text_proj = nn.Linear(
            config.model.text_encoder.d_model, 
            config.model.embed_dim
        )

    @torch.inference_mode()
    def forward(self, text):
        """Return projected CLS embeddings without computing gradients.
        
        Args:
            text: A string or a list of strings to tokenize, pad, and truncate.
        
        Returns:
            An unnormalized tensor with shape (N, D), where N is the number
            of texts and D is the configured embedding dimension.
        """
        text_tokens = self.tokenizer(
            text, 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_txt_l,
            return_tensors="pt"
        ).to(self.device)
        
        text_output = self.text_encoder.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
            mode="text"
        )
        text_embeds = text_output.last_hidden_state
        tfeat = text_embeds[:, 0]
        tfeat = self.text_proj(tfeat)
        return tfeat