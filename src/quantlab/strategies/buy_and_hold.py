"""Constant long-signal reference strategy."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from quantlab.features._validation import choice
from quantlab.strategies.base import (
    PRICE_TYPES,
    BaseStrategy,
    SignalReasons,
    register_strategy,
)


@register_strategy("buy_and_hold")
class BuyAndHoldStrategy(BaseStrategy):
    """Emit a long signal wherever an asset has a valid price.

    The allocator and rebalance schedule determine the resulting portfolio;
    for multiple assets this is not necessarily a literal buy-once portfolio.
    """

    def __init__(self, price_type: str = "adjusted_close") -> None:
        values = self.validate_parameters({"price_type": price_type})
        self.price_type = values["price_type"]
        self._freeze_parameters()

    @classmethod
    def validate_parameters(cls, parameters: Mapping[str, Any]) -> dict[str, Any]:
        """Validate the price-type choice."""
        values = dict(parameters)
        values["price_type"] = choice(
            values["price_type"], name="price_type", options=PRICE_TYPES
        )
        return values

    def generate_signals(
        self, data: pd.DataFrame, features: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Return an all-long signal matrix aligned to prices."""
        prices = self._prices(data)
        signals = prices.notna().astype(float)
        return self._validate_signals(signals, prices)

    def explain_signals(
        self, data: pd.DataFrame, features: pd.DataFrame | None = None
    ) -> SignalReasons:
        """Explain each 0<->1 transition as a change in price availability.

        Price availability is the ONLY thing this strategy's signal
        depends on (see ``generate_signals``), so every transition is
        either the first date a symbol's price becomes valid, or a later
        date it stops being valid (a gap/delisting in the underlying
        data).
        """
        prices = self._prices(data)
        available = prices.notna()
        previous_available = available.shift(1, fill_value=False)
        became_available = (available & ~previous_available).to_numpy()
        became_unavailable = (~available & previous_available).to_numpy()

        detail_code = np.full(became_available.shape, None, dtype=object)
        details = np.full(became_available.shape, None, dtype=object)
        detail_code[became_available] = "price_became_available"
        details[became_available] = "price became available"
        detail_code[became_unavailable] = "price_became_unavailable"
        details[became_unavailable] = "price became unavailable"

        return self._validate_signal_reasons(
            pd.DataFrame(
                detail_code, index=prices.index, columns=prices.columns, dtype=object
            ),
            pd.DataFrame(
                details, index=prices.index, columns=prices.columns, dtype=object
            ),
            prices,
        )
