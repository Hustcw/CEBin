import logging
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn

from transformers.models.roformer.modeling_roformer import (
    RoFormerEmbeddings,
    RoFormerModel,
    RoFormerEncoder,
    RoFormerOnlyMLMHead,
    RoFormerForMaskedLM,
    RoFormerLayer,
    RoFormerAttention,
    RoFormerIntermediate,
    RoFormerOutput,
    RoFormerSelfAttention,
)

from accelerate.logging import get_logger


logger = get_logger(__name__)

class JRoFormerEmbeddings(RoFormerEmbeddings):
    """Construct the embeddings from word and token_type embeddings."""

    def __init__(self, config):
        super().__init__(config)
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id
        )
        self.token_type_embeddings = self.word_embeddings

class JRoFormerSelfAttention(RoFormerSelfAttention):
    def __init__(self, config):
        super().__init__(config)
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias)

class JRoFormerAttention(RoFormerAttention):
    def __init__(self, config):
        super().__init__(config)
        self.self = JRoFormerSelfAttention(config)

class JRoFormerLayer(RoFormerLayer):
    def __init__(self, config):
        super().__init__(config)
        self.attention = JRoFormerAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = RoFormerAttention(config)
        self.intermediate = RoFormerIntermediate(config)
        self.output = RoFormerOutput(config)

class JRoFormerEncoder(RoFormerEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([JRoFormerLayer(config) for _ in range(config.num_hidden_layers)])

class JRoFormerModel(RoFormerModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embeddings = JRoFormerEmbeddings(config)

        if config.embedding_size != config.hidden_size:
            self.embeddings_project = nn.Linear(
                config.embedding_size, config.hidden_size
            )

        self.encoder = JRoFormerEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

class JRoFormerForMaskedLM(RoFormerForMaskedLM):
    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `RoFormerForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.roformer = JRoFormerModel(config)
        self.cls = RoFormerOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

class CEBinEncoder(nn.Module):
    def __init__(self, dim, pretrain_path="../models/cebin"):
        super().__init__()
        self.encoder = JRoFormerModel.from_pretrained(pretrain_path)
        self.config = self.encoder.config
        self.fc1 = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.fc2 = nn.Linear(self.config.hidden_size, dim)
        self.activation = nn.ReLU()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).last_hidden_state

        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(output.size()).half()
            masked_output = output * mask_expanded
            sum_masked_output = torch.sum(masked_output, dim=1)
            sum_attention_mask = torch.sum(mask_expanded, dim=1)
            pooled_output = sum_masked_output / sum_attention_mask
        else:
            pooled_output = torch.mean(output, dim=1)

        output = self.fc1(pooled_output.to(self.fc1.weight.dtype))
        output = self.activation(output)
        output = self.fc2(output)
        return output

class RetrivalEncoder(nn.Module):
    def __init__(self, encoder_q, encoder_k):
        super(RetrivalEncoder, self).__init__()
        self.encoder_q = encoder_q
        self.encoder_k = encoder_k

    def forward(self, func_q, func_k):
        """
        Input:
            func_q: a batch of query functions
            func_k: a batch of key funtions
        Output:
            logits
        """

        q = self.encoder_q(**func_q)  # queries: NxC
        k = self.encoder_k(**func_k)  # keys: NxC
        logits = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        return logits
    
class CEBinPairEncoder(nn.Module):
    def __init__(self, pretrain_path="../models/cebin"):
        super().__init__()
        self.encoder = JRoFormerModel.from_pretrained(pretrain_path)
        self.config = self.encoder.config
        self.fc = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.cls = nn.Linear(self.config.hidden_size, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).last_hidden_state

        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(output.size()).half()
            masked_output = output * mask_expanded
            sum_masked_output = torch.sum(masked_output, dim=1)
            sum_attention_mask = torch.sum(mask_expanded, dim=1)
            pooled_output = sum_masked_output / sum_attention_mask
        else:
            pooled_output = torch.mean(output, dim=1)

        output = self.fc(pooled_output.to(self.fc.weight.dtype))
        output = self.relu(output)
        output = self.cls(output)
        output = self.sigmoid(output)
        return output



