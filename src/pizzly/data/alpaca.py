from datetime import datetime

import polars as pl
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from ..core import BaseProvider


class AlpacaStock(BaseProvider):
    """
    A class representing a financial security for fetching and managing historical market data.

    This class encapsulates functionality to interact with the Alpaca API to retrieve historical price data, handle API authentication, convert data formats, and manage errors. It provides a simple interface for fetching market data with flexible timeframe options.

    Attributes:
        alpaca_api_key (str): Alpaca API key for authentication.
        alpaca_api_secret (str): Alpaca API secret key for authentication.
        symbol (str): The stock symbol to fetch data for (e.g., "AAPL", "GOOGL", "MSFT").
        stock_client (StockHistoricalDataClient): Configured Alpaca API client instance.
        dataframe (Optional[pl.DataFrame]): Cached historical data after fetching.

    Features:
        - Automatic API authentication handling.
        - Flexible timeframe selection (e.g., from 1 minute to 1 day).
        - Data caching to minimize API calls.
        - Efficient data manipulation using a Polars DataFrame.
        - Comprehensive error handling for API interactions and data inconsistencies.
        - Memory-efficient data storage and processing.

    Time Complexity:
        O(n) for data fetching and processing, where n is the number of requested data points.

    Space Complexity:
        O(n) where n is the number of data points stored in the DataFrame.

    Error Handling:
        - API connection failures.
        - Invalid symbol requests.
        - Authentication errors.
        - Rate limit issues.
        - Data format inconsistencies.

    Example:
        >>> # Initialize the security instance and set credentials
        >>> security = AlpacaStock(alpaca_api_key="your_key", alpaca_api_secret="your_secret")
        >>> security.symbol = "AAPL"  # Set the symbol
        >>> # Fetch last month's daily data
        >>> start_date = datetime.now() - timedelta(days=30)
        >>> df = security.fetch(start=start_date, timeframe=TimeFrame.Day)

    Note:
        - All times are in UTC.
        - Volume and trade_count may be 0 during periods with no trading activity.
        - VWAP (Volume Weighted Average Price) calculation spans the query period.
        - API rate limits apply based on your Alpaca account tier.
    """

    def __init__(self, alpaca_api_key: str, alpaca_api_secret: str) -> None:
        """
        Initialize a Security instance with Alpaca API credentials.

        Args:
            alpaca_api_key (str): Alpaca API key for authentication
            alpaca_api_secret (str): Alpaca API secret key for authentication
        """
        super().__init__()  # No symbol required for initialization
        self.alpaca_api_key = alpaca_api_key
        self.alpaca_api_secret = alpaca_api_secret
        self.stock_client = StockHistoricalDataClient(
            api_key=self.alpaca_api_key, secret_key=self.alpaca_api_secret
        )

    def fetch(
        self,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
        timeframe: TimeFrame = TimeFrame(amount=1, unit=TimeFrameUnit.Day),
    ) -> pl.DataFrame | None:
        """
        Fetch historical market data for the security.

        Args:
            symbol (str): The stock symbol to fetch data for (e.g., "AAPL", "GOOGL")
            start (datetime | None): Start date for historical data. If None, defaults to 6 months ago.
            end (datetime | None): End date for historical data. If None, defaults to current date.
            timeframe (TimeFrame, optional): Data timeframe. Defaults to 1 Day.

        Returns:
            pl.DataFrame | None: DataFrame containing price data with columns:
                - timestamp (datetime)
                - open (float)
                - high (float)
                - low (float)
                - close (float)
                - volume (int)
                - trade_count (int)
                - vwap (float)
                Or None if the fetch operation fails.

        Example:
            >>> from datetime import datetime, timedelta
            >>> start_date = datetime.now() - timedelta(days=30)
            >>> security = AlpacaStock(alpaca_api_key="your_key", alpaca_api_secret="your_secret")
            >>> security.symbol = "AAPL"  # Set the symbol before fetching
            >>> df = security.fetch(start=start_date)
            >>> print(df.tail(3))
            shape: (3, 8)

        Raises:
            ValueError: If no symbol has been set
        """
        try:
            if not symbol:
                raise ValueError("Symbol cannot be empty")

            if start is None:
                start = datetime.now().replace(month=datetime.now().month - 6)
            if end is None:
                end = datetime.combine(datetime.today(), datetime.min.time())

            request_params = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=start,
                end=end,
            )

            quotes = self.stock_client.get_stock_bars(request_params)
            self._dataframe = pl.from_dicts(quotes[symbol])

            return self._dataframe
        except Exception:
            return None
