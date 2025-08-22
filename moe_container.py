import torch
import torch.nn as nn
from .transformer_aggregator import TransformerAggregator, Projection


class MoA(nn.Module):
    def __init__(self,
                 num_classes,
                 K=4,
                 input_dim=2048,
                 dim=512,
                 depth=2,
                 heads=8,
                 mlp_dim=512,
                 dim_head=64,
                 mode="shared",     # "shared", "separate", "shared+adapter"
                 k_active=2,
                 router_type="mlp"):
        super().__init__()
        self.K = K
        self.k_active = k_active
        self.mode = mode

        # === shared projection if needed ===
        if mode in ["shared", "shared+adapter"]:
            self.shared_proj = Projection(input_dim, dim, heads, dim_head)
        else:
            self.shared_proj = None

        # === experts ===
        self.experts = nn.ModuleList([
            TransformerAggregator(num_classes=num_classes,
                                  input_dim=input_dim,
                                  dim=dim,
                                  depth=depth,
                                  heads=heads,
                                  mlp_dim=mlp_dim,
                                  dim_head=dim_head,
                                  mode=mode,
                                  shared_proj=self.shared_proj)
            for _ in range(K)
        ])

        # === router ===
        if router_type == "mlp":
            self.router_fc = nn.Sequential(
                nn.Linear(dim, 256), nn.ReLU(),
                nn.Linear(256, K)
            )
        elif router_type == "linear":
            self.router_fc = nn.Linear(dim, K)
        else:
            raise ValueError("Unknown router_type")

        # === final classifier ===
        self.head = nn.Sequential(
            nn.Linear(dim, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, temp=1.0, k=None):
        B, N, _ = x.shape
        k = k or self.k_active

        # Router uses mean of projected tokens (shared if available)
        if self.shared_proj is not None:
            H = self.shared_proj(x)
            bag_repr = H.mean(dim=1)   # [B, dim]
        else:
            bag_repr = x.mean(dim=1)   # [B, input_dim]

        router_logits = self.router_fc(bag_repr)   # [B, K]
        g_soft = torch.softmax(router_logits / temp, dim=-1)

        # Top-k routing
        topk = g_soft.topk(k, dim=-1)
        idx = topk.indices
        weights = topk.values / (topk.values.sum(dim=-1, keepdim=True) + 1e-8)

        z_list = []
        for b in range(B):
            z_b = 0
            for j in range(k):
                e_idx = idx[b, j].item()
                latent, _ = self.experts[e_idx](x[b].unsqueeze(0))
                z_b += weights[b, j] * latent
            z_list.append(z_b)
        z = torch.stack(z_list, dim=0)  # [B, dim]

        logits = self.head(z)
        return z, logits, g_soft
