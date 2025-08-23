import torch.nn as nn
import models as models

class cAItomorph(nn.Module):
    def __init__(self, class_count, arch, embedding_dim=768,
                 expert_mode="shared", router_style="topk", topk=1,
                 use_local_head=False, save_gates=False, num_expert=1, router_type="linear"):
        super(cAItomorph, self).__init__()

        if arch not in models.__dict__:
            raise ValueError(f"Unknown model architecture '{arch}'")
        
        if expert_mode is None :
            self.model = models.__dict__[arch](
                input_dim=embedding_dim,
                num_classes=class_count
            )
        else:
            self.model = models.__dict__[arch](
                input_dim=embedding_dim,
                num_classes=class_count,
                mode=expert_mode,
                router_style=router_style,
                k_active=topk,
                experts_use_local_head=use_local_head,
                num_experts=num_expert,
                router_type=router_type

            )
        self.save_gates = save_gates
 
    def forward(self, embeddings, return_latent=False, return_gates=False):
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0).unsqueeze(0)
        elif embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)
        elif embeddings.dim() != 3:
            raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")

        out = self.model(embeddings)
        if len(out) == 3:
            latent, logits, gates = out
        else:
            latent, logits = out
            gates = None

        if return_latent and return_gates:
            return latent, logits, gates
        elif return_latent:
            return latent, logits
        elif return_gates:
            return logits, gates
        return logits

    def __repr__(self):
        return f"cAItomorph(model={self.model})"
