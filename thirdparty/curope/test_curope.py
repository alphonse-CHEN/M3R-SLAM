"""Quick smoke test for curope CUDA kernel."""
import torch

def test_curope():
    import curope as _kernels
    print(f"curope module loaded: {_kernels}")

    B, N, H, D = 2, 16, 8, 64
    tokens = torch.randn(B, N, H, D, device='cuda', dtype=torch.float32)
    positions = torch.randint(0, 16, (B, N, 2), device='cuda', dtype=torch.int64)

    tokens_clone = tokens.clone()
    _kernels.rope_2d(tokens_clone, positions, 100.0, 1.0)

    # Verify it modified the tensor in-place
    assert not torch.equal(tokens, tokens_clone), "rope_2d should modify tokens in-place"
    print("PASSED: curope smoke test")

if __name__ == "__main__":
    test_curope()
