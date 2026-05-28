from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * x / rms


class RNNTextClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_classes: int = 4,
        hidden_size: int = 128,
        num_layers: int = 1,
        model_type: str = "lstm",
        bidirectional: bool = True,
        dropout: float = 0.35,
        pad_idx: int = 0,
        pooling: str = "last",
        embedding_weights=None,
        freeze_embeddings: bool = False,
        use_rms_norm: bool = False,
        use_layer_norm: bool = False,
        l2_lambda: float = 0.0,
    ):
        super().__init__()
        if model_type not in {"lstm", "gru"}:
            raise ValueError("model_type must be 'lstm' or 'gru'")
        if pooling not in {"last", "meanmax"}:
            raise ValueError("pooling must be 'last' or 'meanmax'")
        if use_rms_norm and use_layer_norm:
            raise ValueError("use_rms_norm and use_layer_norm are mutually exclusive")

        self.model_type = model_type
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.pooling = pooling
        self.num_directions = 2 if bidirectional else 1
        self.pad_idx = pad_idx
        self.l2_lambda = l2_lambda

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        if embedding_weights is not None:
            weights = torch.as_tensor(embedding_weights, dtype=torch.float32)
            if weights.shape != (vocab_size, embedding_dim):
                raise ValueError(
                    f"embedding_weights shape must be {(vocab_size, embedding_dim)}, got {tuple(weights.shape)}"
                )
            with torch.no_grad():
                self.embedding.weight.copy_(weights)
        self.embedding.weight.requires_grad = not freeze_embeddings

        norm_cls = RMSNorm if use_rms_norm else nn.LayerNorm if use_layer_norm else None
        self.embedding_norm = norm_cls(embedding_dim) if norm_cls is not None else nn.Identity()

        rnn_cls = nn.LSTM if model_type == "lstm" else nn.GRU
        rnn_dropout = dropout if num_layers > 1 else 0.0
        self.rnn = rnn_cls(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=rnn_dropout,
        )
        self.dropout = nn.Dropout(dropout)

        feature_dim = hidden_size * self.num_directions
        if pooling == "meanmax":
            feature_dim *= 2
        self.feature_norm = norm_cls(feature_dim) if norm_cls is not None else nn.Identity()
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.embedding_norm(self.embedding(input_ids))
        emb = self.dropout(emb)
        lengths_cpu = lengths.detach().cpu().clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        packed_output, hidden = self.rnn(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=input_ids.size(1)
        )

        if self.pooling == "meanmax":
            mask = (input_ids != self.pad_idx).unsqueeze(-1)
            masked_output = output.masked_fill(~mask, 0.0)
            denom = mask.sum(dim=1).clamp(min=1)
            mean_pool = masked_output.sum(dim=1) / denom
            max_pool = output.masked_fill(~mask, torch.finfo(output.dtype).min).max(dim=1).values
            no_tokens = mask.sum(dim=1).eq(0)
            max_pool = torch.where(no_tokens, torch.zeros_like(max_pool), max_pool)
            features = torch.cat([mean_pool, max_pool], dim=1)
        else:
            if self.model_type == "lstm":
                h_n, _ = hidden
            else:
                h_n = hidden
            if self.bidirectional:
                forward_last = h_n[-2]
                backward_last = h_n[-1]
                features = torch.cat([forward_last, backward_last], dim=1)
            else:
                features = h_n[-1]

        features = self.feature_norm(features)
        logits = self.classifier(self.dropout(features))
        return logits
