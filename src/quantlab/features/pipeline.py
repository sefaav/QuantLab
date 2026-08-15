"""Composable feature pipeline with reproducibility metadata."""

from __future__ import annotations

import json
from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
from typing import Any, Self, cast

import pandas as pd

from quantlab.features._validation import numeric_pandas, positive_int
from quantlab.logging_config import get_logger

logger = get_logger(__name__)

Transformer = Callable[[pd.DataFrame], pd.Series | pd.DataFrame]


@dataclass
class FeatureSpec:
    """Registration and fitted metadata for one feature."""

    name: str
    transformer: Transformer
    params: dict[str, Any] = field(default_factory=dict)
    window: int | None = None
    generated_nans: int = 0
    first_valid: Hashable | None = None


class FeaturePipeline:
    """Register, fit and apply feature transformers to a wide data matrix."""

    def __init__(self) -> None:
        self._specs: list[FeatureSpec] = []
        self._fitted = False

    def add(
        self,
        name: str,
        transformer: Transformer,
        *,
        params: dict[str, Any] | None = None,
        window: int | None = None,
    ) -> Self:
        """Register a uniquely named transformer and invalidate prior fitting."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Feature name must be a non-empty string.")
        clean_name = name.strip()
        if not callable(transformer):
            raise TypeError("transformer must be callable.")
        if any(spec.name == clean_name for spec in self._specs):
            raise ValueError(f"Feature '{clean_name}' is already registered.")
        if params is not None and not isinstance(params, dict):
            raise TypeError("params must be a dictionary or None.")
        copied_params = dict(params or {})
        try:
            json.dumps(copied_params, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "params must contain only JSON-serialisable values."
            ) from exc
        valid_window = None if window is None else positive_int(window, name="window")
        self._specs.append(
            FeatureSpec(
                name=clean_name,
                transformer=transformer,
                params=copied_params,
                window=valid_window,
            )
        )
        self._fitted = False
        return self

    def fit(self, data: pd.DataFrame) -> Self:
        """Run each transformer once and record its fitted metadata."""
        validated = self._validate_input(data)
        self._fitted = False
        self._compute(validated, record_metadata=True)
        self._fitted = True
        logger.info("Fitted feature pipeline with %d features.", len(self._specs))
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply a fitted pipeline and return ``(feature, symbol)`` columns."""
        if not self._fitted:
            raise RuntimeError("FeaturePipeline must be fitted before transform().")
        validated = self._validate_input(data)
        return self._combine(validated, self._compute(validated, record_metadata=False))

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform while executing each transformer exactly once."""
        validated = self._validate_input(data)
        self._fitted = False
        pieces = self._compute(validated, record_metadata=True)
        self._fitted = True
        logger.info("Fitted feature pipeline with %d features.", len(self._specs))
        return self._combine(validated, pieces)

    @staticmethod
    def _validate_input(data: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        return numeric_pandas(data, name="data")

    def _apply(self, spec: FeatureSpec, data: pd.DataFrame) -> pd.DataFrame:
        output = spec.transformer(data)
        if isinstance(output, pd.Series):
            if len(data.columns) != 1:
                raise ValueError(
                    f"Feature '{spec.name}' returned a Series for multi-asset input."
                )
            output = output.to_frame(name=data.columns[0])
        elif not isinstance(output, pd.DataFrame):
            raise TypeError(
                f"Feature '{spec.name}' must return a pandas Series or DataFrame."
            )
        numeric_pandas(output, name=f"output from feature '{spec.name}'")
        if not output.index.equals(data.index):
            raise ValueError(f"Feature '{spec.name}' changed or reordered the index.")
        if not output.columns.equals(data.columns):
            raise ValueError(f"Feature '{spec.name}' changed or reordered the columns.")
        return output

    def _compute(
        self, data: pd.DataFrame, *, record_metadata: bool
    ) -> dict[str, pd.DataFrame]:
        pieces: dict[str, pd.DataFrame] = {}
        for spec in self._specs:
            output = self._apply(spec, data)
            pieces[spec.name] = output
            if record_metadata:
                spec.generated_nans = int(output.isna().to_numpy().sum())
                valid_rows = output.dropna(how="any")
                spec.first_valid = valid_rows.index[0] if not valid_rows.empty else None
        return pieces

    @staticmethod
    def _combine(data: pd.DataFrame, pieces: dict[str, pd.DataFrame]) -> pd.DataFrame:
        if not pieces:
            return pd.DataFrame(index=data.index)
        combined = pd.concat(pieces, axis=1)
        combined.columns.names = ["feature", "symbol"]
        return combined

    @property
    def feature_names(self) -> list[str]:
        """Return registered feature names in registration order."""
        return [spec.name for spec in self._specs]

    @property
    def min_usable_date(self) -> Hashable | None:
        """Return the first index label where every fitted feature is defined."""
        if not self._specs:
            return None
        if not self._fitted:
            raise RuntimeError("FeaturePipeline must be fitted before introspection.")
        firsts = [spec.first_valid for spec in self._specs]
        if any(first is None for first in firsts):
            return None
        defined = [first for first in firsts if first is not None]
        return cast(Hashable, pd.Index(defined).max())

    def metadata(self) -> list[dict[str, Any]]:
        """Return JSON-serialisable fitted metadata for each feature."""
        if self._specs and not self._fitted:
            raise RuntimeError("FeaturePipeline must be fitted before introspection.")
        return [
            {
                "name": spec.name,
                "params": dict(spec.params),
                "window": spec.window,
                "generated_nans": spec.generated_nans,
                "first_valid": (
                    str(spec.first_valid) if spec.first_valid is not None else None
                ),
            }
            for spec in self._specs
        ]
