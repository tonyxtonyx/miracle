"""A toy task used to exercise (and, if ever needed, tune) the diagnosis architecture WITHOUT touching SWE-bench tasks.

Planted bug: `total()` applies tax twice when a discount is given. Plausible competing explanations: the discount is applied wrongly,
the tax rate is wrong, rounding. The gold patch / tests exist only for the oracle judge and the toy pass/fail check."""
from __future__ import annotations

from pathlib import Path

INSTANCE_ID = "toy__shop-1"
ISSUE = """total() returns too much when a discount is applied

`shop.pricing.total(100, discount=10)` returns 108.9 but I expected 99.0 (10% off 100 is 90, plus 10% tax is 99).
Without a discount `total(100)` correctly returns 110.0. The `apply_discount` helper on its own looks fine to me.
"""

FILES = {
    "shop/__init__.py": "from .pricing import apply_discount, add_tax, total\n",
    "shop/config.py": "TAX_RATE = 0.10\n",
    "shop/pricing.py": '''from .config import TAX_RATE


def apply_discount(price, percent):
    """Reduce `price` by `percent` percent."""
    return price * (1 - percent / 100)


def add_tax(amount):
    """Add sales tax."""
    return amount * (1 + TAX_RATE)


def total(price, discount=0):
    """Final price: discount first, then tax once."""
    if discount:
        price = add_tax(apply_discount(price, discount))
    return add_tax(price)
''',
    "tests/test_pricing.py": '''import pytest
from shop.pricing import add_tax, apply_discount, total


def test_apply_discount():
    assert apply_discount(100, 10) == pytest.approx(90.0)


def test_add_tax():
    assert add_tax(100) == pytest.approx(110.0)


def test_total_without_discount():
    assert total(100) == pytest.approx(110.0)


def test_total_with_discount_applies_tax_once():
    assert total(100, discount=10) == pytest.approx(99.0)
''',
    "conftest.py": "import sys, pathlib\nsys.path.insert(0, str(pathlib.Path(__file__).parent))\n",
}
GOLD_PATCH = '''diff --git a/shop/pricing.py b/shop/pricing.py
--- a/shop/pricing.py
+++ b/shop/pricing.py
@@ -13,5 +13,5 @@ def add_tax(amount):
 def total(price, discount=0):
     """Final price: discount first, then tax once."""
     if discount:
-        price = add_tax(apply_discount(price, discount))
+        price = apply_discount(price, discount)
     return add_tax(price)
'''
FAIL_TO_PASS = ["tests/test_pricing.py::test_total_with_discount_applies_tax_once"]
PASS_TO_PASS = ["tests/test_pricing.py::test_apply_discount", "tests/test_pricing.py::test_add_tax", "tests/test_pricing.py::test_total_without_discount"]
TEST_COMMAND = "python -m pytest -q -x tests/ 2>&1 | tail -n 5"


def write_repo(dest: Path) -> Path:
    for rel, text in FILES.items():
        p = Path(dest) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return Path(dest)


def instance_record() -> dict:
    """An instance record in the SWE-bench shape (gold fields included: they are for the judges only)."""
    return {"instance_id": INSTANCE_ID, "repo": "toy/shop", "base_commit": "toy", "problem_statement": ISSUE, "patch": GOLD_PATCH,
            "test_patch": "", "FAIL_TO_PASS": FAIL_TO_PASS, "PASS_TO_PASS": PASS_TO_PASS, "hints_text": "", "version": "1", "difficulty": "toy"}
