from torch.nn import Sequential, Linear, LayerNorm, GELU

def build_projection(input_dim, output_dim):
    """Build projection layer for LayoutLMv3 to T5."""
    return Sequential(
        Linear(input_dim, output_dim),
        LayerNorm(output_dim),
        GELU()
    )