"""Unit tests for CustomCNN (baseline).

Run with:  pytest models/tests/test_custom_cnn.py -v

Covers:
    1. Forward pass output shape (batch dims, num_classes)
    2. Parameter count within expected baseline range (~1.8M)
    3. Single-channel input requirement / configurable channels
    4. Gradient flow: every trainable parameter receives a gradient
    5. Configurable num_classes
    6. eval() vs train() mode (BatchNorm/Dropout behaviour differs)
    7. Logits (not probabilities) are returned
    8. Reproducibility under a fixed manual seed
"""

from __future__ import annotations

import torch

try:
    # When run as part of the package (pytest from repo root):
    #   pytest models/tests/test_custom_cnn.py
    from models.custom_cnn import CustomCNN, count_parameters
except ModuleNotFoundError:
    # Fallback when run directly from inside models/ or models/tests/.
    from custom_cnn import CustomCNN, count_parameters


# ---------------------------------------------------------------------
# 1. Forward pass shape
# ---------------------------------------------------------------------
def test_forward_output_shape():
    model = CustomCNN()
    x = torch.randn(4, 1, 224, 224)
    out = model(x)
    assert out.shape == (4, 8), f"Expected (4, 8), got {tuple(out.shape)}"


def test_forward_batch_size_one():
    model = CustomCNN()
    model.eval()  # BatchNorm needs eval() for batch size 1
    x = torch.randn(1, 1, 224, 224)
    out = model(x)
    assert out.shape == (1, 8)


# ---------------------------------------------------------------------
# 2. Parameter count
# ---------------------------------------------------------------------
def test_parameter_count_in_baseline_range():
    model = CustomCNN()
    n = count_parameters(model)
    # Baseline target ~1.8M; allow a generous window so minor tweaks
    # (e.g. dropout location) don't break the test.
    assert 1_000_000 <= n <= 3_000_000, f"Param count {n:,} out of range"


def test_all_parameters_trainable_by_default():
    model = CustomCNN()
    total = count_parameters(model, trainable_only=False)
    trainable = count_parameters(model, trainable_only=True)
    assert total == trainable


# ---------------------------------------------------------------------
# 3. Channel configuration
# ---------------------------------------------------------------------
def test_default_single_channel():
    model = CustomCNN()
    assert model.in_channels == 1
    x = torch.randn(2, 1, 224, 224)
    assert model(x).shape == (2, 8)


def test_configurable_in_channels():
    model = CustomCNN(in_channels=3)
    x = torch.randn(2, 3, 224, 224)
    assert model(x).shape == (2, 8)


# ---------------------------------------------------------------------
# 4. Gradient flow
# ---------------------------------------------------------------------
def test_gradient_flow_reaches_all_params():
    model = CustomCNN()
    model.train()
    x = torch.randn(8, 1, 224, 224)
    target = torch.randint(0, 8, (8,))

    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, target)
    loss.backward()

    missing = [
        name
        for name, p in model.named_parameters()
        if p.requires_grad and (p.grad is None or torch.all(p.grad == 0))
    ]
    assert not missing, f"Params with no/zero gradient: {missing}"


def test_loss_is_finite():
    model = CustomCNN()
    model.train()
    x = torch.randn(8, 1, 224, 224)
    target = torch.randint(0, 8, (8,))
    loss = torch.nn.functional.cross_entropy(model(x), target)
    assert torch.isfinite(loss), "Loss is not finite"


# ---------------------------------------------------------------------
# 5. Configurable num_classes
# ---------------------------------------------------------------------
def test_configurable_num_classes():
    model = CustomCNN(num_classes=5)
    x = torch.randn(2, 1, 224, 224)
    assert model(x).shape == (2, 5)


# ---------------------------------------------------------------------
# 6. train / eval mode behaviour
# ---------------------------------------------------------------------
def test_eval_mode_is_deterministic():
    model = CustomCNN()
    model.eval()
    x = torch.randn(2, 1, 224, 224)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    # In eval mode dropout is off and BN uses running stats -> identical.
    assert torch.allclose(out1, out2), "eval() forward not deterministic"


# ---------------------------------------------------------------------
# 7. Returns raw logits (not a probability distribution)
# ---------------------------------------------------------------------
def test_returns_logits_not_probabilities():
    model = CustomCNN()
    model.eval()
    x = torch.randn(4, 1, 224, 224)
    with torch.no_grad():
        out = model(x)
    row_sums = out.sum(dim=1)
    # Probabilities would each sum to 1.0; logits will not (generically).
    assert not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3), (
        "Output rows sum to 1 -- looks like softmax was applied"
    )


# ---------------------------------------------------------------------
# 8. Reproducibility under a fixed seed
# ---------------------------------------------------------------------
def test_reproducible_init_under_seed():
    torch.manual_seed(42)
    m1 = CustomCNN()
    torch.manual_seed(42)
    m2 = CustomCNN()
    for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1, p2), f"Param {n1} differs under same seed"


if __name__ == "__main__":
    # Allow running directly without pytest.
    import sys
    import traceback

    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL  {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
