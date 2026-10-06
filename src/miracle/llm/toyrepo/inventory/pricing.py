from .config import TAX_RATE


def apply_discount(price: float, percent: float) -> float:
    """Reduce `price` by `percent` percent."""
    return price * (1 - percent / 100)


def final_price(price: float, percent: float = 0.0) -> float:
    """Discounted price including tax, rounded to cents."""
    return round(apply_discount(price, percent) * (1 + TAX_RATE), 2)
