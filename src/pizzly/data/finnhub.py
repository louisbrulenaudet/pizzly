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

    def fetch(self, endpoint: str, *args: object, **kwargs: object) -> pl.DataFrame | object:
        """
        Call any finnhub endpoint with given arguments.
        If the result is tabular (dict of lists), convert to polars.DataFrame.
        Otherwise, store and return the raw result.

        Args:
            endpoint (str): Name of the finnhub client method to call.
            *args: Positional arguments for the endpoint.
            **kwargs: Keyword arguments for the endpoint.

        Returns:
            pl.DataFrame | object: DataFrame if possible, else raw result.

        Raises:
            AttributeError: If the specified endpoint does not exist on the finnhub client.

        Example:
            >>> provider = Finnhub("your_api_key")
            >>> df = provider.fetch("stock_candles", symbol="AAPL", resolution="D", from_=1609459200, to=1612137600)
            >>> print(df)
        """
        method = getattr(self.client, endpoint)
        result = method(*args, **kwargs)

        lists = [v for v in result.values() if isinstance(v, list)][0]

        if isinstance(lists, list) and all(isinstance(item, dict) for item in lists):
            result = pl.DataFrame(lists)

        # Cast 'date' column to date type if present
        if isinstance(result, pl.DataFrame) and "date" in result.columns:
            result = result.with_columns(
                pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False)
            )

        self._dataframe = result

        return self._dataframe
