import torch
import torch.nn as nn


class KeystrokeTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        vocab_size: int,
        d_model: int = 160,
        nhead: int = 4,
        num_layers: int = 3,
        dim_ff: int = 320,
        dropout: float = 0.1,
        use_cls_token: bool = True,
    ) -> None:
        super().__init__()
        self.use_cls_token = use_cls_token
        self.d_model = d_model

        self.key_emb = nn.Embedding(vocab_size, d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.ln = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

        if use_cls_token:
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        else:
            self.cls = None

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

        nn.init.normal_(self.key_emb.weight, mean=0.0, std=0.02)
        if self.cls is not None:
            nn.init.normal_(self.cls, mean=0.0, std=0.02)

    def encode(
        self,
        keysym_ids: torch.Tensor,
        dwell: torch.Tensor,
        flight: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _ = keysym_ids.shape

        key_features = self.key_emb(keysym_ids)
        time_features = torch.stack([dwell, flight], dim=-1)
        time_embeddings = self.time_mlp(time_features)

        x = self.ln(key_features + time_embeddings)
        x = self.drop(x)

        if self.use_cls_token:
            cls = self.cls.expand(batch_size, 1, self.d_model)
            x = torch.cat([cls, x], dim=1)
            attn_mask = torch.cat(
                [torch.ones(batch_size, 1, device=attn_mask.device, dtype=torch.bool), attn_mask],
                dim=1,
            )

        src_key_padding_mask = ~attn_mask
        hidden = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        if self.use_cls_token:
            pooled = hidden[:, 0, :]
        else:
            mask = attn_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        return pooled

    def forward(
        self,
        keysym_ids: torch.Tensor,
        dwell: torch.Tensor,
        flight: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        pooled = self.encode(keysym_ids, dwell, flight, attn_mask)
        return self.head(pooled)
