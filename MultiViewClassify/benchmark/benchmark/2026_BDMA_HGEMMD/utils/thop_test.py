from thop import profile
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=64, num_heads=8, batch_first=True)
    def forward(self, x):
        return self.attn(x, x, x)[0]

model = MyModel()
x = torch.randn(10, 8, 64)  # (seq_len, batch, embed_dim)

def count_mha(module, input, output):
    # print(input[0].shape); print(input[1].shape); print(input[2].shape) # input: (B, L, E) * 3
    # print(output[0].shape); print(output[1].shape) # output, attention: (B, L, E), (B, L, L)
    q, k, v = input[0], input[1], input[2]
    B, L_q, E = q.shape; H = module.num_heads
    L_k, L_v = k.shape[1], v.shape[1]
    flops, params = 0, 0
    # Q, K, V projections: B * L_q * E * E + B * L_k * E * E + B * L_v * E * E
    flops += B * L_q * (E * E + E) + B * L_k * (E * E + E) + B * L_v * (E * E + E)
    params += E * E + E
    # Q × K^T: B * H * L_q * L_k * D
    flops += B * H * L_q * L_k * (E // H)  # Q × K^T
    # Softmax: B * H * L_q * L_k * 3 (exp + sum + div)
    flops += B * H * L_q * L_k * 3
    # Attention × V: B * H * L_q * L_v * D
    flops += B * H * L_q * L_v * (E // H)
    # Output projection: B * L * E * E
    flops += B * L_q * (E * E + E)
    params += E * E + E
    module.total_ops += flops

custom_ops = {nn.MultiheadAttention: count_mha}
flops, params = profile(model, inputs=(x,), verbose=True, custom_ops=custom_ops)
print(f"thop FLOPs: {flops}")
print(f"thop Params: {params}")