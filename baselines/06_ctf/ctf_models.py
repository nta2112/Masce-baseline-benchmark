#!/usr/bin/env python3
"""
CTF (Coarse-to-Fine) Model Architectures for Autoregressive Image Generation.
References:
  - "Improving Autoregressive Image Generation through Coarse-to-Fine Token Prediction"
    (Guo, Zhang, Shieh, CVPR 2025 / arXiv:2503.16194)
  - "A Flat Vocabulary or a Rich Hierarchy? Re-introducing Intrinsic Structure
     Transforms the Autoregressive Image Generation" (ICML 2026 Table 3, Table 9, Appendix C.2)

Architecture Overview:
  1. Primary Coarse AR Backbone (GPT-B / GPT-L):
     - Predicts the sequence of coarse cluster indices (k-way classification, e.g. k=4,096).
     - Outputs coarse logits [B, L, k] and final hidden representations [B, L, hidden_dim].
  2. Hierarchical Fine Refiner (Stage 2):
     - Lightweight refinement head / single-layer decoder conditioned on [hidden_states, coarse_tokens].
     - Predicts the exact fine-grained codebook index (N-way classification, N=16,384).
     - Supports intra-cluster candidate masking for restricted high-precision prediction.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CTFFineRefiner(nn.Module):
    """
    Fine-stage refinement head for CTF framework.
    Takes the main AR transformer's hidden state and the coarse cluster index,
    and predicts the fine-grained token index in [0, N-1].
    """
    def __init__(self, hidden_dim: int, n_coarse: int, n_fine: int, embed_dim: int = 512, n_heads: int = 4, use_transformer: bool = False):
        super().__init__()
        self.n_coarse = n_coarse
        self.n_fine = n_fine
        self.hidden_dim = hidden_dim
        self.use_transformer = use_transformer
        
        self.coarse_embed = nn.Embedding(n_coarse, embed_dim)
        self.in_proj = nn.Linear(hidden_dim, embed_dim)
        
        if use_transformer:
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=n_heads,
                dim_feedforward=embed_dim * 2,
                dropout=0.1,
                batch_first=True,
                norm_first=True
            )
            self.refiner = nn.TransformerDecoder(decoder_layer, num_layers=1)
        else:
            self.refiner = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.GELU(),
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim)
            )
            
        self.norm = nn.LayerNorm(embed_dim)
        self.fine_head = nn.Linear(embed_dim, n_fine, bias=False)

    def forward(self, hidden_state: torch.Tensor, coarse_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_state: [B, L, hidden_dim] from primary coarse AR backbone.
            coarse_indices: [B, L] coarse cluster labels.
        Returns:
            fine_logits: [B, L, n_fine] logits over all fine-grained tokens.
        """
        c_emb = self.coarse_embed(coarse_indices)  # [B, L, embed_dim]
        h_proj = self.in_proj(hidden_state)        # [B, L, embed_dim]
        
        if self.use_transformer:
            refined = self.refiner(tgt=c_emb, memory=h_proj)
        else:
            combined = torch.cat([h_proj, c_emb], dim=-1)  # [B, L, embed_dim * 2]
            refined = self.refiner(combined)
            
        refined = self.norm(refined)
        fine_logits = self.fine_head(refined)  # [B, L, n_fine]
        return fine_logits

class CTFAutoregressiveModel(nn.Module):
    """
    Combined Coarse-to-Fine (CTF) Autoregressive System.
    Wraps the primary LlamaGen backbone with the CTFFineRefiner.
    """
    def __init__(self, backbone: nn.Module, k_coarse: int = 4096, n_fine: int = 16384, refiner_dim: int = 512):
        super().__init__()
        self.backbone = backbone
        self.k_coarse = k_coarse
        self.n_fine = n_fine
        
        hidden_dim = getattr(backbone, "dim", getattr(backbone, "embed_dim", 768))
        self.refiner = CTFFineRefiner(
            hidden_dim=hidden_dim,
            n_coarse=k_coarse,
            n_fine=n_fine,
            embed_dim=refiner_dim
        )

    def forward_train(self, coarse_indices: torch.Tensor, cond: torch.Tensor, fine_targets: torch.Tensor, beta: float = 1.0):
        """
        Joint CTF training forward pass.
        Args:
            coarse_indices: [B, L] input coarse tokens (shifted or teacher-forced).
            cond: [B] class labels.
            fine_targets: [B, L] ground-truth fine tokens.
            beta: weighting factor for fine-stage loss.
        Returns:
            total_loss, coarse_loss, fine_loss, coarse_logits, fine_logits
        """
        # Primary Coarse Stage Forward
        # Most LlamaGen models return (logits, loss) or logits given input tokens
        coarse_logits, coarse_loss, hidden_states = self._forward_backbone(coarse_indices, cond)
        
        # Secondary Fine Stage Forward
        fine_logits = self.refiner(hidden_states, coarse_indices)
        
        fine_loss = F.cross_entropy(
            fine_logits.view(-1, self.n_fine),
            fine_targets.view(-1),
            label_smoothing=0.0
        )
        
        total_loss = coarse_loss + beta * fine_loss
        return total_loss, coarse_loss, fine_loss, coarse_logits, fine_logits

    def _forward_backbone(self, x: torch.Tensor, cond: torch.Tensor):
        # Extract features from LlamaGen backbone
        # Hooks or layer extraction for hidden representations
        model = self.backbone
        bs = x.shape[0]
        
        # Embeddings
        token_emb = model.tok_embeddings(x)
        if hasattr(model, "tok_dropout"):
            token_emb = model.tok_dropout(token_emb)
            
        cond_emb = model.cls_embedding(cond, train=model.training)[:, :model.cls_token_num]
        h = torch.cat([cond_emb, token_emb], dim=1)
        
        # Forward through transformer layers
        for layer in model.layers:
            h = layer(h)
            
        h = model.norm(h)
        # Token-level hidden states (skipping cls tokens)
        token_h = h[:, model.cls_token_num:]
        coarse_logits = model.output(token_h)
        
        # Compute coarse next-token loss
        targets = x.clone()
        targets = targets.contiguous()
        coarse_loss = F.cross_entropy(coarse_logits.view(-1, self.k_coarse), targets.view(-1))
        
        return coarse_logits, coarse_loss, token_h
