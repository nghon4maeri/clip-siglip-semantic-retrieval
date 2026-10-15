#!/usr/bin/env python3
"""Smoke test cho ``compute_recall_at_k`` bằng ma trận tổng hợp (không cần data).

Chạy: python tests/test_metrics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from src.evaluation.metrics import compute_recall_at_k  # noqa: E402


def _check(name: str, actual: float, expected: float, tol: float = 1e-6) -> bool:
    ok = abs(actual - expected) < tol
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: {actual:.2f} (kỳ vọng {expected:.2f})")
    return ok


def test_perfect_block_diagonal() -> bool:
    """Ma trận block-diagonal (mỗi ảnh khớp hoàn hảo với captions của nó) => 100%."""
    n_images, n_captions = 4, 5
    sim = torch.zeros(n_images, n_images * n_captions)
    for i in range(n_images):
        sim[i, i * n_captions : (i + 1) * n_captions] = 1.0
    image_ids = torch.arange(n_images).repeat_interleave(n_captions)

    res = compute_recall_at_k(sim, image_ids=image_ids)
    ok = True
    for key, value in res.items():
        if key.startswith(("i2t", "t2i")):
            ok &= _check(key, value, 100.0)
    return ok


def test_known_counts() -> bool:
    """Ca có đáp án tính tay để kiểm tra đếm đúng.

    N=2 ảnh, C=2 captions/ảnh (M=4). Ma trận:
        img0: [0.9, 0.8, 0.3, 0.2]
        img1: [0.85, 0.7, 0.6, 0.1]
    i2t: R@1 = 50 (img1 lấy nhầm caption của img0), R@2 = 50, R@4 = 100.
    t2i: R@1 = 75 (caption 3 xếp nhầm), R@2 = 100.
    """
    sim = torch.tensor([[0.9, 0.8, 0.3, 0.2], [0.85, 0.7, 0.6, 0.1]])
    image_ids = torch.tensor([0, 0, 1, 1])

    res = compute_recall_at_k(sim, image_ids=image_ids, k_values=(1, 2, 4))
    checks = [
        ("i2t_R@1", 50.0),
        ("i2t_R@2", 50.0),
        ("i2t_R@4", 100.0),
        ("t2i_R@1", 75.0),
        ("t2i_R@2", 100.0),
    ]
    ok = all(_check(key, res[key], expected) for key, expected in checks)
    # Chỉ có 2 ảnh nên K=4 không tính được cho t2i — phải bị bỏ qua, không crash.
    skipped = "t2i_R@4" not in res
    print(f"  {'PASS' if skipped else 'FAIL'}  t2i_R@4 bị bỏ qua khi chỉ có 2 ảnh")
    return ok and skipped


def test_one_to_one_protocol() -> bool:
    """Protocol 1:1 (image_ids=None) vẫn phải hoạt động.

    i2t: ảnh 0 lấy đúng caption 0, ảnh 1 lấy nhầm caption 0 => 50%.
    t2i: cả hai cột đều có ảnh gốc đứng đầu => 100%.
    """
    sim = torch.tensor([[0.9, 0.1], [0.8, 0.2]])
    res = compute_recall_at_k(sim, k_values=(1, 2))
    return _check("i2t_R@1", res["i2t_R@1"], 50.0) & _check("t2i_R@1", res["t2i_R@1"], 100.0) & _check(
        "i2t_R@2", res["i2t_R@2"], 100.0
    )


def test_small_dataset_no_crash() -> bool:
    """Với < 10 mẫu, các K lớn hơn kích thước phải bị bỏ qua thay vì crash."""
    sim = torch.eye(3)
    res = compute_recall_at_k(sim, k_values=(1, 5, 10))
    ok = (
        "i2t_R@1" in res
        and "t2i_R@1" in res
        and "i2t_R@5" not in res
        and "i2t_R@10" not in res
        and res["i2t_R@1"] == 100.0
    )
    print(f"  {'PASS' if ok else 'FAIL'}  small dataset: keys={sorted(res)}")
    return ok


def test_invalid_inputs() -> bool:
    ok = True
    try:
        compute_recall_at_k(torch.rand(4, 6))  # M != N nhưng thiếu image_ids
        print("  FAIL  thiếu image_ids nhưng không raise")
        ok = False
    except ValueError:
        print("  PASS  raise khi thiếu image_ids")

    try:
        compute_recall_at_k(torch.rand(2, 4), image_ids=torch.tensor([0, 0]))  # sai độ dài
        print("  FAIL  image_ids sai độ dài nhưng không raise")
        ok = False
    except ValueError:
        print("  PASS  raise khi image_ids sai độ dài")
    return ok


def main() -> int:
    tests = [
        ("perfect block diagonal", test_perfect_block_diagonal),
        ("known counts", test_known_counts),
        ("1:1 protocol", test_one_to_one_protocol),
        ("small dataset", test_small_dataset_no_crash),
        ("invalid inputs", test_invalid_inputs),
    ]
    print("Chạy test cho compute_recall_at_k...")
    results = []
    for name, fn in tests:
        print(f"\n[{name}]")
        results.append(fn())

    passed = sum(results)
    print(f"\nKết quả: {passed}/{len(results)} test PASS")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
