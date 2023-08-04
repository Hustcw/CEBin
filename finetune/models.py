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

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, accelerator, pretrain_path="../models/cebin", freeze_layers=-1, dim=128, K=65536, m=0.999, T=0.5):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(dim=dim, pretrain_path=pretrain_path)
        self.encoder_k = base_encoder(dim=dim, pretrain_path=pretrain_path)
        self.accelerator = accelerator

        # if mlp:  # hack: brute-force replacement
        #     dim_mlp = self.encoder_q.fc.weight.shape[1]
        #     self.encoder_q.fc = nn.Sequential(
        #         nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc
        #     )
        #     self.encoder_k.fc = nn.Sequential(
        #         nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc
        #     )

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param in self.encoder_q.encoder.embeddings.parameters():
            param.requires_grad = False

        if freeze_layers != -1:
            for layer in self.encoder_q.encoder.encoder.layer[:freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)
        with torch.no_grad():  # no gradient to keys
            # gather keys before updating queue
            keys = self.accelerator.gather(keys)

            batch_size = keys.shape[0]

            ptr = int(self.queue_ptr)
            assert self.K % batch_size == 0  # for simplicity

            # replace the keys at ptr (dequeue and enqueue)
            self.queue[:, ptr : ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K  # move pointer

            self.queue_ptr[0] = ptr

    def forward(self, func_q, func_k):
        """
        Input:
            func_q: a batch of query functions
            func_k: a batch of key funtions
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(**func_q)  # queries: NxC

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            k = self.encoder_k(**func_k)  # keys: NxC

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels

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



