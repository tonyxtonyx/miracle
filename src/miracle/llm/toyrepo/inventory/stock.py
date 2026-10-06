LOW_STOCK_THRESHOLD = 7


class Stock:
    def __init__(self):
        self.items: dict[str, int] = {}

    def add(self, sku: str, qty: int) -> None:
        self.items[sku] = self.items.get(sku, 0) + qty

    def is_low(self, sku: str) -> bool:
        return self.items.get(sku, 0) < LOW_STOCK_THRESHOLD
