import contextlib
from datetime import datetime
from typing import Any

import polars as pl
from finnhub import Client

from ..core import BaseProvider

__all__ = ["Finnhub"]


class Finnhub(BaseProvider):
    """
    A class for interacting with the Finnhub API to fetch financial data.

    This class provides methods to call various endpoints of the Finnhub API,
    handling both tabular data (which can be converted to a Polars DataFrame)
    and non-tabular data (which is returned as raw results).

    Attributes:
        finnhub_api_key (str): API key for authenticating with the Finnhub API.
        client (Client): Instance of the Finnhub API client.

    Methods:
        fetch(endpoint: str, *args, **kwargs) -> pl.DataFrame | object:
            Calls a specified Finnhub API endpoint with given arguments.
            Returns a Polars DataFrame if the result is tabular, otherwise returns the raw result.
    """

    def __init__(self, finnhub_api_key: str) -> None:
        super().__init__()
        self.finnhub_api_key = finnhub_api_key
        self.client = Client(api_key=self.finnhub_api_key)

    def fetch(
        self,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> pl.DataFrame | None:
        """
        Call a finnhub endpoint specified via the `endpoint` kwarg.

        The finnhub client method name must be provided as keyword `endpoint`.
        If the response contains tabular data (a dict with one or more lists of
        dicts), the largest such list is converted to a Polars DataFrame, with a
        best-effort parse of a 'date' column. On error or non-tabular result,
        None is returned and self._dataframe is cleared.
        """
        endpoint = kwargs.pop("endpoint", None)
        if not isinstance(endpoint, str):
            raise ValueError(
                "The finnhub endpoint must be provided via the 'endpoint' keyword argument"
            )

        if symbol and "symbol" not in kwargs:
            kwargs.setdefault("symbol", symbol)

        try:
            method = getattr(self.client, endpoint)
        except AttributeError as exc:
            raise AttributeError(f"Finnhub client has no endpoint '{endpoint}'") from exc

        try:
            result = method(*args, **kwargs)
        except Exception:
            self._dataframe = None
            return None

        # If result is a dict, find the largest value that is a list of dicts
        if isinstance(result, dict):
            candidate_lists = [
                v for v in result.values()
                if isinstance(v, list) and v and all(isinstance(i, dict) for i in v)
            ]
            if candidate_lists:
                rows = max(candidate_lists, key=len)
                df = pl.DataFrame(rows)

                if "date" in df.columns:
                    df = df.with_columns(
                        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False)
                    )

                self._dataframe = df
                return self._dataframe

        self._dataframe = None

        return None
