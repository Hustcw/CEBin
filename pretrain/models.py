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
    RoFormerSinusoidalPositionalEmbedding,
    RoFormerLayer,
    RoFormerAttention,
    RoFormerIntermediate,
    RoFormerOutput,
    RoFormerSelfAttention,
    RoFormerSelfOutput
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
        self.query = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.use_bias
        )
        self.key = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.use_bias
        )
        self.value = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.use_bias
        )


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
                raise ValueError(
                    f"{self} should be used as a decoder model if cross attention is added"
                )
            self.crossattention = RoFormerAttention(config)
        self.intermediate = RoFormerIntermediate(config)
        self.output = RoFormerOutput(config)


class JRoFormerEncoder(RoFormerEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList(
            [JRoFormerLayer(config) for _ in range(config.num_hidden_layers)]
        )


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
