import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer_expert import TransformerExpert
from .abmil_expert import ABMILExpert



class MixtureOfAggregators(nn.Module):
    """
    Container for multiple experts.
    router_style:
      - "dense" : soft mixture of ALL experts
      - "topk"  : sparse mixture of only k experts
    """
    def __init__(self,
                 num_classes,
                 expert_arch,
                 input_dim=2048,
                 dim=512,
                 depth=2,
                 heads=8,
                 mlp_dim=512,
                 dim_head=64,
                 dropout=0.1,
                 emb_dropout=0.1,
                 pool="cls",
                 pos_enc=None,
                 mode="separate",         # "separate", "shared", "shared_adapter"
                 num_experts=4,
                 router_style="dense",    # "dense" or "topk"
                 k_active=2,
                 router_type="linear",    # "linear" or "mlp"
                 experts_use_local_head=True,
                 ):  
        super().__init__()
        self.num_experts = num_experts
        self.router_style = router_style
        self.k_active = k_active
        self.mode = mode
        self.experts_use_local_head = experts_use_local_head
        self.router_type = router_type

        # === shared projection if needed ===
        shared_proj = None
        if mode in ["shared", "shared_adapter"]:
            # For Transformer experts, the projection output must match heads*dim_head
            # Ensure consistency or raise a clear error
            if expert_arch == 'Transformer' and dim != heads * dim_head:
                raise ValueError(f"For Transformer experts, require dim == heads*dim_head; got dim={dim}, heads*dim_head={heads*dim_head}")
            proj_dim = heads * dim_head if expert_arch == 'Transformer' else 512
            shared_proj = nn.Sequential(nn.Linear(input_dim, proj_dim, bias=True), nn.ReLU())

        # === experts ===
        if expert_arch == 'Transformer':
            self.experts = nn.ModuleList([
                TransformerExpert(
                    num_classes=num_classes,
                    input_dim=input_dim,
                    dim=dim,
                    depth=depth,
                    heads=heads,
                    mlp_dim=mlp_dim,
                    dim_head=dim_head,
                    dropout=dropout,
                    emb_dropout=emb_dropout,
                    pool=pool,
                    pos_enc=pos_enc,
                    mode=mode,
                    shared_proj=shared_proj,
                    use_local_head=experts_use_local_head
                )
                for _ in range(num_experts)
            ])
        else:
            # ABMIL experts use a simpler interface; pass only supported args
            self.experts = nn.ModuleList([
                ABMILExpert(
                    input_dim=input_dim,
                    size_arg="big",
                    dropout=dropout,
                    num_classes=num_classes,
                    mode=mode,
                    shared_proj=shared_proj,
                    use_local_head=experts_use_local_head
                )
                for _ in range(num_experts)
            ])


        # === router ===
        proj_dim = heads * dim_head
        self.router_proj = nn.Sequential(nn.Linear(input_dim, proj_dim, bias=True), nn.ReLU())
        if self.router_type == "mlp":
            self.router_fc = nn.Sequential(
                nn.Linear(proj_dim, 256), nn.ReLU(),
                nn.Linear(256, num_experts)
            )
        elif self.router_type == "linear":
            self.router_fc = nn.Linear(proj_dim, num_experts)
        elif self.router_type == "transformer":
            self.router_fc = TransformerExpert(
            num_classes=num_experts,
            input_dim=input_dim,
            dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim,
            dim_head=dim_head, dropout=dropout, emb_dropout=emb_dropout,
            pool=pool, pos_enc=pos_enc, mode='separate',
            shared_proj=None,               # decouple from experts
            use_local_head=True
        )

        else:
            raise ValueError(f"Unknown router_type {self.router_type}")

        # === global head (for topk or for dense+no-local-head) ===
        if not experts_use_local_head or router_style == "topk":
            if expert_arch =='Transformer':
                self.head = nn.Sequential(
                    nn.Linear(dim, 64), nn.ReLU(),
                    nn.Linear(64, num_classes)
                )
            if expert_arch == 'ABMIL':
                self.head = nn.Sequential(nn.Linear(64, num_classes))

        else:
            self.head = None

    def forward(self, x, temp=1.0, k=None, use_gumbel=False):
        B, N, _ = x.shape
        k = k or self.k_active
        if self.router_type == "transformer":
            router_in=x
            _,router_logits=self.router_fc(router_in)  # [B, E]
        else:
            router_in = self.router_proj(x).mean(dim=1)  # [B, proj_dim]
            router_logits = self.router_fc(router_in)    # [B, E]

        # Optional Gumbel-softmax noise during training to encourage exploration
        if use_gumbel and self.training:
            eps = 1e-9
            U = torch.rand_like(router_logits)
            gumbel = -torch.log(-torch.log(U + eps) + eps)
            router_logits = router_logits + gumbel

        g_soft = F.softmax(router_logits / temp, dim=-1)  # [B, E]

        if self.router_style == "dense":
            # All experts contribute
            latents, logits = [], []
            for expert in self.experts:
                latent, logit = expert(x)
                latents.append(latent)
                logits.append(logit)

            latents = torch.stack(latents, dim=1)  # [B, E, D]

            if self.experts_use_local_head:
                logits = torch.stack(logits, dim=1)    # [B, E, C]
                gates = g_soft.unsqueeze(-1)
                latent = (latents * gates).sum(dim=1)
                logits = (logits * gates).sum(dim=1)
            else:
                gates = g_soft.unsqueeze(-1)
                latent = (latents * gates).sum(dim=1)
                logits = self.head(latent)

            return latent, logits, g_soft

        elif self.router_style == "topk":
            k = min(k, self.num_experts)

            # Pick top-k experts per sample
            topk = g_soft.topk(k, dim=-1)
            idx = topk.indices                  # [B, k]
            weights = topk.values               # [B, k]
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

            z_list = []
            if self.experts_use_local_head:
                logit_list = []

            for b in range(B):
                z_b = 0
                if self.experts_use_local_head:
                    logits_b = 0

                for j in range(k):
                    e_idx = idx[b, j].item()
                    w = weights[b, j]

                    # Run only the chosen experts for this sample
                    latent_bj, logit_bj = self.experts[e_idx](x[b].unsqueeze(0))  # latent: [1, D], logit: [1, C] (if local head)
                    # Normalize shapes defensively in case an expert returns [D]
                    if latent_bj.dim() == 1:
                        latent_bj = latent_bj.unsqueeze(0)
                    if self.experts_use_local_head and (logit_bj is not None) and logit_bj.dim() == 1:
                        logit_bj = logit_bj.unsqueeze(0)

                    # Always blend latents to return a representative embedding
                    z_b = z_b + w * latent_bj

                    # If local heads are enabled, blend logits directly
                    if self.experts_use_local_head:
                        if logit_bj is None:
                            raise RuntimeError("experts_use_local_head=True but expert returned logits=None.")
                        logits_b = logits_b + w * logit_bj

                z_list.append(z_b)
                if self.experts_use_local_head:
                    logit_list.append(logits_b)

            latent = torch.stack(z_list, dim=0).squeeze(1)   # [B, D]

            if self.experts_use_local_head:
                logits = torch.cat(logit_list, dim=0)        # [B, C] (weighted sum of top-k expert logits)
            else:
                logits = self.head(latent)                   # [B, C] (global head as before)

            return latent, logits, g_soft
