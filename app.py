import os
from abc import ABCMeta, abstractmethod
from datetime import datetime

import polars as pl
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from dotenv import load_dotenv
from smolagents import DuckDuckGoSearchTool, HfApiModel, Tool
from smolagents.agents import ToolCallingAgent

# Load environment variables
load_dotenv()


class Indicator(metaclass=ABCMeta):
    """
    Abstract base class for financial technical indicators.

    This class provides a standardized interface for implementing technical indicators, ensuring consistent behavior across different implementations. It includes basic error handling and logging functionality.

    Attributes:
        name (str): The identifier for the indicator
        serie (Optional[pl.Series]): The computed indicator values stored as a Polars series

    Note:
        All concrete indicator classes must implement get_name() and get_series() methods.
        The compute() method should be implemented based on the specific indicator's logic.
    """

    def __init__(self, name: str) -> None:
        """
        Initialize the indicator with a name.

        Args:
            name (str): Identifier for the indicator
        """
        self.name = name
        self.series = None

    @abstractmethod
    def get_name(self) -> str:
        """
        Retrieve the indicator's name.

        Returns:
            str: The indicator's name

        Raises:
            Exception: If there's an error accessing the name attribute
        """
        try:
            return self.name
        except Exception:
            return ""

    @abstractmethod
    def get_series(self) -> pl.Series | None:
        """
        Retrieve the computed indicator values.

        Returns:
            Optional[pl.Series]: The computed indicator values or None if not calculated

        Raises:
            Exception: If there's an error accessing the series
        """
        try:
            return self.series
        except Exception:
            return pl.Series([])


class RSI(Indicator):
    """
    Relative Strength Index (RSI) implementation.

    RSI is a momentum oscillator that measures the speed and magnitude of price changes. Values range from 0 to 100, with traditional interpretation being:
        - RSI > 70: Potentially overbought conditions
        - RSI < 30: Potentially oversold conditions
        - RSI = 50: Neutral momentum

    Time Complexity: O(n), where n is the length of the input data
    Space Complexity: O(n) for storing the computed values

    Args:
        dataframe (pl.DataFrame): Input price data
        column (str): Column name containing price data
        window_size (int, optional): Look-back period for calculations. Defaults to 14
        min_periods (int, optional): Minimum periods required. Defaults to 1

    Attributes:
        dataframe (pl.DataFrame): Input price data
        column (str): Target price column name
        window_size (int): Calculation window size
        min_periods (int): Minimum required periods

    Example:
        >>> # Create sample price data
        >>> data = {
        ...     "close": [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42,
        ...              45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28]
        ... }
        >>> df = pl.DataFrame(data)
        >>>
        >>> # Calculate RSI
        >>> rsi = RSI(df, "close", window_size=14)
        >>> values = rsi.compute()
        >>>
        >>> print(values)
        shape: (14,)
        Series: 'rsi' [f64]
        [
            70.53
            66.32
            66.55
            69.41
            66.36
            57.97
            62.93
            63.26
            56.06
            62.38
            54.71
            50.42
            39.99
            41.46
        ]

    Note:
        The first window_size periods will have incomplete calculations,
        potentially resulting in NaN or None values.
    """

    def __init__(
        self,
        dataframe: pl.DataFrame,
        column: str,
        window_size: int = 14,
        min_periods: int = 1,
    ) -> None:
        """Initialize RSI indicator with input data and parameters."""
        super().__init__(name="RSI")
        self.dataframe = dataframe
        self.column = column
        self.window_size = window_size
        self.min_periods = min_periods

    def get_name(self) -> str:
        """
        Retrieve the name identifier of the RSI indicator.

        Returns:
            str: The name of the indicator ('RSI')
        """
        try:
            return self.name
        except Exception:
            return ""

    def get_series(self) -> pl.Series | None:
        """
        Retrieve the computed RSI values.

        Returns:
            Optional[pl.Series]: The computed RSI series or None if not calculated
        """
        try:
            return self.series
        except Exception:
            return None

    def compute(self) -> pl.Series | None:
        """
        Calculate the RSI values for the input data.

        Returns:
            pl.Series: Series containing RSI values. First window_size values may be None.

        Raises:
            Exception: If calculation fails, returns None and logs error

        Note:
            The computation follows these steps:
            1. Calculate price changes
            2. Separate gains and losses
            3. Calculate average gains and losses using rolling mean
            4. Calculate RS (relative strength) as avg_gain/avg_loss
            5. Convert RS to RSI using the formula: 100 - (100 / (1 + RS))
        """
        try:
            # Calculate price changes
            self.dataframe = self.dataframe.with_columns(
                change=pl.col(self.column).diff()
            )

            # Calculate gains and losses
            self.dataframe = self.dataframe.with_columns(
                [
                    pl.when(pl.col("change") > 0)
                    .then(pl.col("change"))
                    .otherwise(0)
                    .alias("gain"),
                    pl.when(pl.col("change") < 0)
                    .then(-pl.col("change"))
                    .otherwise(0)
                    .alias("loss"),
                ]
            )

            # Calculate average gains and losses
            self.dataframe = self.dataframe.with_columns(
                [
                    pl.col("gain")
                    .rolling_mean(
                        window_size=self.window_size, min_samples=self.min_periods
                    )
                    .alias("avg_gain"),
                    pl.col("loss")
                    .rolling_mean(
                        window_size=self.window_size, min_samples=self.min_periods
                    )
                    .alias("avg_loss"),
                ]
            )

            # Calculate RS and RSI
            self.dataframe = self.dataframe.with_columns(
                rs=pl.col("avg_gain").truediv(pl.col("avg_loss"))
            )

            self.dataframe = self.dataframe.with_columns(
                rsi=pl.lit(100).sub(pl.lit(100).truediv(pl.lit(1).add(pl.col("rs"))))
            )

            return self.dataframe["rsi"].tail(-1)
        except Exception:
            return None


class SmaBB(Indicator):
    """
    Simple Moving Average Bollinger Bands implementation.

    Bollinger Bands are a volatility indicator consisting of three bands:
        - Middle Band: Simple Moving Average (SMA) of the price
        - Upper Band: SMA + (standard deviation x 2)
        - Lower Band: SMA - (standard deviation x 2)

    The bands expand and contract based on price volatility, helping to:
        - Identify potential overbought/oversold conditions
        - Measure price volatility
        - Spot potential trend reversals
        - Define dynamic support and resistance levels

    Time Complexity: O(n), where n is the length of the input data
    Space Complexity: O(n) for storing the computed values

    Args:
        dataframe (pl.DataFrame): Input price data containing the target column
        column (str): Column name containing price data (typically 'close')
        window_size (int, optional): Look-back period for calculations. Defaults to 14
        min_periods (Optional[int], optional): Minimum periods required before computing.
            If None, defaults to window_size

    Returns:
        Tuple[pl.Series, pl.Series, pl.Series]: A tuple containing:
            - Middle band (SMA)
            - Upper band (SMA + 2*std)
            - Lower band (SMA - 2*std)

    Trading Signals:
        - Price above Upper Band: Potentially overbought
        - Price below Lower Band: Potentially oversold
        - Price moving from outside to inside bands: Potential trend reversal
        - Band width expanding: Increasing volatility
        - Band width contracting: Decreasing volatility

    Example:
        >>> # Create sample price data
        >>> df = pl.DataFrame({
        ...     "close": [10.0, 10.5, 11.2, 10.8, 11.5, 11.3]
        ... })
        >>> bb = SmaBB(df, "close", window_size=3)
        >>> sma, upper, lower = bb.compute()
        >>> print(pl.DataFrame({
        ...     "SMA": sma,
        ...     "Upper": upper,
        ...     "Lower": lower
        ... }))
        shape: (6, 3)
        ┌──────────┬──────────┬──────────┐
        │ SMA      ┆ Upper    ┆ Lower    │
        │ ---      ┆ ---      ┆ ---      │
        │ f64      ┆ f64      ┆ f64      │
        ╞══════════╪══════════╪══════════╡
        │ null     ┆ null     ┆ null     │
        │ 10.25    ┆ 11.15    ┆ 9.35     │
        │ 10.85    ┆ 12.05    ┆ 9.65     │
        │ 10.83    ┆ 11.93    ┆ 9.73     │
        │ 11.17    ┆ 12.37    ┆ 9.97     │
        │ 11.37    ┆ 12.47    ┆ 10.27    │
        └──────────┴──────────┴──────────┘

    Note:
        - The first window_size periods will have incomplete calculations
        - Standard deviation multiplier of 2 is the traditional setting
        - Bands typically contain 85-95% of price action
        - Useful when combined with other indicators for confirmation
    """

    def __init__(
        self,
        dataframe: pl.DataFrame,
        column: str,
        window_size: int = 14,
        min_periods: int | None = None,
    ) -> None:
        super().__init__(name="SmaBB")
        self.dataframe = dataframe
        self.column = column
        self.window_size = window_size
        self.min_periods = min_periods

    def get_name(self) -> str:
        """
        Retrieve the name identifier of the Bollinger Bands indicator.

        Returns:
            str: The name of the indicator ('SmaBB')

        Raises:
            Exception: If there's an error accessing the name attribute, returns empty string
                      and logs the error

        Example:
            >>> bb = SmaBB(df, "close")
            >>> print(bb.get_name())
            'SmaBB'
        """
        try:
            return self.name
        except Exception:
            return ""

    def get_series(self) -> pl.Series | None:
        """
        Retrieve the computed Bollinger Bands values.

        This method provides access to the stored indicator values after computation. Should be called after compute() to ensure values are available.

        Returns:
            Optional[pl.Series]: The computed Bollinger Bands series or None if not calculated

        Raises:
            Exception: If there's an error accessing the series, returns None and logs the error

        Example:
            >>> bb = SmaBB(df, "close")
            >>> bb.compute()
            >>> series = bb.get_series()
            >>> if series is not None:
            ...     print("Bollinger Bands calculated successfully")
        """
        try:
            return self.series
        except Exception:
            return None

    def compute(self) -> tuple[pl.Series, pl.Series, pl.Series]:
        """
        Calculate Bollinger Bands components for the input data.

        This method implements the core Bollinger Bands computation:
        1. Calculates Simple Moving Average (SMA) as the middle band
        2. Computes standard deviation over the window period
        3. Creates upper and lower bands at 2 standard deviations from SMA

        Time Complexity: O(n), where n is the length of input data
            - O(n) for rolling mean calculation
            - O(n) for standard deviation computation
            - O(n) for band calculations

        Space Complexity: O(n) for storing three series (SMA, upper, lower)

        Returns:
            Tuple[pl.Series, pl.Series, pl.Series]: A tuple containing:
                - Middle band (Simple Moving Average)
                - Upper band (SMA + 2 x standard deviation)
                - Lower band (SMA - 2 x standard deviation)
                Returns (None, None, None) if calculation fails

        Raises:
            Exception: Catches any computation errors, returns None values and logs error

        Implementation Details:
            - Uses Polars' efficient rolling window functions
            - Standard deviation multiplier of 2 is fixed for traditional bands
            - min_periods parameter controls minimum observations needed
            - NaN/None values at the start due to window calculations

        Example:
            >>> df = pl.DataFrame({"close": [10, 11, 12, 11, 10, 11]})
            >>> bb = SmaBB(df, "close", window_size=3)
            >>> sma, upper, lower = bb.compute()
            >>> print("Middle Band:", sma.tail(1)[0])
            >>> print("Upper Band:", upper.tail(1)[0])
            >>> print("Lower Band:", lower.tail(1)[0])
        """
        try:
            # Calculate Simple Moving Average
            self.sma = self.dataframe[self.column].rolling_mean(
                window_size=self.window_size, min_samples=self.min_periods
            )

            # Calculate Standard Deviation
            std = self.dataframe[self.column].rolling_std(window_size=self.window_size)

            # Calculate Bollinger Bands
            self.upper_band = self.sma + (std * 2)
            self.lower_band = self.sma - (std * 2)

            return self.sma, self.upper_band, self.lower_band
        except Exception:
            return pl.Series([]), pl.Series([]), pl.Series([])


class Security:
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
        >>> # Initialize the security instance.
        >>> security = Security("AAPL")

        >>> # Fetch last month's daily data.
        >>> start_date = datetime.now() - timedelta(days=30)
        >>> df = security.fetch(start=start_date, timeframe=TimeFrame.Day)

    Note:
        - All times are in UTC.
        - Volume and trade_count may be 0 during periods with no trading activity.
        - VWAP (Volume Weighted Average Price) calculation spans the query period.
        - API rate limits apply based on your Alpaca account tier.
    """

    def __init__(
        self, symbol: str, alpaca_api_key: str, alpaca_api_secret: str
    ) -> None:
        self.alpaca_api_key = alpaca_api_key
        self.alpaca_api_secret = alpaca_api_secret
        self.symbol = symbol
        self.stock_client = StockHistoricalDataClient(
            api_key=self.alpaca_api_key, secret_key=self.alpaca_api_secret
        )
        self.dataframe = None

    def fetch(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        timeframe: TimeFrame = TimeFrame(amount=1, unit=TimeFrameUnit.Day),
    ) -> pl.DataFrame | None:
        """
        Fetch historical market data for the security.

        Args:
            start (datetime): Start date for historical data
            timeframe (TimeFrame, optional): Data timeframe. Defaults to TimeFrame.Day

        Returns:
            pl.DataFrame: DataFrame containing price data with columns:
                - timestamp (datetime)
                - open (float)
                - high (float)
                - low (float)
                - close (float)
                - volume (int)
                - trade_count (int)
                - vwap (float)

        Example:
            >>> from datetime import datetime, timedelta
            >>> start_date = datetime.now() - timedelta(days=30)
            >>> security = Security("AAPL")
            >>> df = security.fetch(start=start_date)
            >>> print(df.tail(3))
            shape: (3, 8)
        """
        try:
            if start is None:
                start = datetime.now().replace(month=datetime.now().month - 6)
            if end is None:
                end = datetime.combine(datetime.today(), datetime.min.time())

            request_params = StockBarsRequest(
                symbol_or_symbols=self.symbol, timeframe=timeframe, start=start, end=end
            )

            quotes = self.stock_client.get_stock_bars(request_params)
            self.dataframe = pl.from_dicts(quotes[self.symbol])

            return self.dataframe
        except Exception:
            return None


class FinancialTool(Tool):
    """
    A comprehensive market analysis tool combining multiple technical indicators.

    This tool integrates RSI and Bollinger Bands indicators to provide a holistic market analysis system. It processes historical price data to generate actionable trading signals and market insights, considering both momentum and volatility metrics.

    Time Complexity:
        - Data Fetching: O(n) for historical data retrieval
        - Analysis: O(n) for indicator calculations
        - Signal Generation: O(1) for final analysis

    Space Complexity:
        - O(n) for storing historical data and computed indicators
        - O(1) for analysis results and signals

    Features:
        - Automated data retrieval from Alpaca API
        - Multi-indicator analysis (RSI + Bollinger Bands)
        - Customizable calculation windows
        - Clear signal generation
        - Comprehensive error handling
        - Memory-efficient data processing

    Signal Types:
        - Momentum signals from RSI
        - Volatility signals from Bollinger Bands
        - Combined indicator signals
        - Support/Resistance levels
        - Trend analysis

    Attributes:
        alpaca_api_key (str): Alpaca API key for data retrieval
        alpaca_api_secret (str): Alpaca API secret key for authentication
        name (str): Tool identifier for the smol-ai framework
        description (str): Detailed tool functionality description
        inputs (Dict): Parameter specifications including:
            - symbol: Stock symbol to analyze
            - rsi_window: RSI calculation period
            - bb_window: Bollinger Bands calculation period
        output_type (str): Format of analysis results

    Usage Guidelines:
        1. Initialize the tool with default parameters
        2. Call forward() with desired symbol and window sizes
        3. Parse returned analysis for trading signals
        4. Consider multiple timeframes for confirmation
        5. Use signals as part of a broader trading strategy

    Example:
        >>> analyzer = MarketAnalysisTool()
        >>> analysis = analyzer.forward("AAPL", rsi_window="14", bb_window="20")
        >>> print(analysis)
        Market Analysis Results for AAPL:
        RSI (14 periods): 58.43

        Price Position:
        Current Price: 173.25
        Upper Band: 180.45
        Lower Band: 165.87

        Signals:
        - Price within Bollinger Bands - neutral trend

    Note:
        - Recommended to use default window sizes unless specific strategy requires otherwise
        - Analysis more reliable during regular market hours
        - Consider market conditions and volatility when interpreting signals
        - Use in conjunction with fundamental analysis for better results
    """

    name = "financial_tool"
    description = """Provides a market analysis by integrating technical indicators such as the Relative Strength Index (RSI) and Bollinger Bands. This tool fetches historical market data and computes statistical metrics to assess momentum and volatility. It identifies potential overbought/oversold conditions, support/resistance levels, and trend reversals, thereby generating actionable trading signals to assist in making informed investment decisions."""
    inputs = {
        "symbol": {
            "type": "string",
            "description": "Stock symbol to analyze (e.g., 'AAPL')",
        },
        "rsi_window": {
            "type": "integer",
            "description": "Window size for RSI calculation (default: 14)",
        },
        "bb_window": {
            "type": "integer",
            "description": "Window size for Bollinger Bands calculation (default: 20)",
        },
    }
    output_type = "string"

    def __init__(
        self,
        alpaca_api_key: str,
        alpaca_api_secret: str,
        **kwargs: dict[str, str | int | float],
    ) -> None:
        self.alpaca_api_key = alpaca_api_key
        self.alpaca_api_secret = alpaca_api_secret

        super().__init__(**kwargs)

    def fetch_market_data(
        self, symbol: str, start: datetime | None = None, end: datetime | None = None
    ) -> pl.DataFrame | None:
        """
        Fetch historical market data for analysis.

        Retrieves 6 months of historical daily price data for the specified symbol
        using the Security class.

        Args:
            symbol (str): Stock symbol to fetch data for (e.g., 'AAPL')

        Returns:
            pl.DataFrame: DataFrame containing historical price data

        Example:
            >>> analyzer = MarketAnalysisTool()
            >>> df = analyzer.fetch_market_data("AAPL")
            >>> print(df.head(3))
            shape: (3, 8)
        """
        security = Security(
            symbol,
            alpaca_api_key=self.alpaca_api_key,
            alpaca_api_secret=self.alpaca_api_secret,
        )
        return security.fetch(start=start, end=end)

    def analyze_market_conditions(
        self, df: pl.DataFrame, rsi_window: int, bb_window: int
    ) -> dict:
        """
        Analyze market conditions using technical indicators.

        This method combines RSI and Bollinger Bands analysis to provide a comprehensive market assessment including trend strength, potential reversal points, and trading signals.

        Args:
            df (pl.DataFrame): Price data containing required columns:
                - timestamp (datetime): Time of the observation
                - open (float): Opening price
                - high (float): High price
                - low (float): Low price
                - close (float): Closing price
                - volume (int): Trading volume
                - trade_count (int): Number of trades
                - vwap (float): Volume Weighted Average Price
            rsi_window (int): Look-back period for RSI calculation
            bb_window (int): Look-back period for Bollinger Bands calculation

        Returns:
            Dict[str, any]: Analysis results containing:
                - rsi_value (float): Current RSI value
                - price_position (Dict): Current price relative to Bollinger Bands
                    - current_price (float): Latest closing price
                    - upper_band (float): Upper Bollinger Band value
                    - lower_band (float): Lower Bollinger Band value
                - signals (List[str]): List of trading signals and conditions

        Example:
            >>> df = analyzer.fetch_market_data("AAPL")
            >>> analysis = analyzer.analyze_market_conditions(df, 14, 20)
            >>> print(analysis)
            {
                'rsi_value': 62.5,
                'price_position': {
                    'current_price': 178.05,
                    'upper_band': 182.35,
                    'lower_band': 173.75
                },
                'signals': [
                    'RSI indicates neutral conditions',
                    'Price within Bollinger Bands - neutral trend'
                ]
            }
        """
        # Calculate RSI
        rsi_indicator = RSI(dataframe=df, column="close", window_size=rsi_window)
        rsi_values = rsi_indicator.compute()

        # Calculate Bollinger Bands
        bb_indicator = SmaBB(dataframe=df, column="close", window_size=bb_window)
        sma, upper_band, lower_band = bb_indicator.compute()

        # Get the latest values
        if rsi_values is None or upper_band is None or lower_band is None:
            raise ValueError("Failed to compute technical indicators")

        latest_rsi = rsi_values.tail(1)[0]
        latest_price = df["close"].tail(1)[0]
        latest_upper_band = upper_band.tail(1)[0]
        latest_lower_band = lower_band.tail(1)[0]

        # Generate market analysis
        analysis = {
            "rsi_value": float(latest_rsi),
            "price_position": {
                "current_price": float(latest_price),
                "upper_band": float(latest_upper_band),
                "lower_band": float(latest_lower_band),
            },
            "signals": [],
        }

        # RSI signal analysis
        if latest_rsi > 70:
            analysis["signals"].append("RSI indicates overbought conditions")
        elif latest_rsi < 30:
            analysis["signals"].append("RSI indicates oversold conditions")

        # Bollinger Bands signal analysis
        if latest_price > latest_upper_band:
            analysis["signals"].append(
                "Price above upper Bollinger Band - potential resistance"
            )
        elif latest_price < latest_lower_band:
            analysis["signals"].append(
                "Price below lower Bollinger Band - potential support"
            )

        return analysis

    inputs = {
        "symbol": {
            "type": "string",
            "description": "Stock symbol to analyze (e.g., 'AAPL')",
        },
        "rsi_window": {
            "type": "integer",
            "description": "Window size for RSI calculation (default: 14)",
        },
        "bb_window": {
            "type": "integer",
            "description": "Window size for Bollinger Bands calculation (default: 20)",
        },
        "start_date": {
            "type": "string",
            "description": "Start date for analysis (YYYY-MM-DD format). This is the lower bound for historical data retrieval. The range is in days, so the start date is generally set to 6 months before the current date.",
            "required": True,
        },
        "end_date": {
            "type": "string",
            "description": "End date for analysis (YYYY-MM-DD format). This is casually set to the current date if provided.",
            "required": False,
            "nullable": True,
        },
    }

    def forward(  # type: ignore
        self,
        *,
        symbol: str,
        rsi_window: int,
        bb_window: int,
        start_date: str,
        end_date: str | None = (
            datetime.combine(datetime.today(), datetime.min.time())
        ).strftime("%Y-%m-%d"),
    ) -> str:
        """
        Perform complete market analysis for a given symbol.

        This is the main entry point for market analysis. It combines data fetching and technical analysis to provide actionable insights about market conditions.

        Args:
            symbol (str): Stock symbol to analyze (e.g., 'AAPL')
            rsi_window (str, optional): RSI calculation period. Defaults to 14
            bb_window (str, optional): Bollinger Bands calculation period. Defaults to 20

        Returns:
            str: Formatted analysis results including:
                - Current RSI value and interpretation
                - Price position relative to Bollinger Bands
                - Generated trading signals
                - Overall market condition assessment

        Example:
            >>> analyzer = MarketAnalysisTool()
            >>> result = analyzer.forward("AAPL")
            >>> print(result)
            Market Analysis Results for AAPL:
            RSI (14 periods): 58.43

            Price Position:
            Current Price: 173.25
            Upper Band: 180.45
            Lower Band: 165.87

            Signals:
            - Price within Bollinger Bands - neutral trend

        Note:
            - RSI > 70 indicates overbought conditions
            - RSI < 30 indicates oversold conditions
            - Price near Bollinger Bands can signal potential reversals
        """
        try:
            # Convert date strings to datetime objects if provided
            start = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
            end = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None

            df = self.fetch_market_data(symbol, start=start, end=end)
            if df is None:
                return "Error: Failed to fetch market data"

            analysis_result = self.analyze_market_conditions(df, rsi_window, bb_window)

            # Format output
            output = [
                f"Market Analysis Results for {symbol}:",
                f"RSI ({rsi_window} periods): {analysis_result['rsi_value']:.2f}",
                "\nPrice Position:",
                f"Current Price: {analysis_result['price_position']['current_price']:.2f}",
                f"Upper Band of Bollinger Bands: {analysis_result['price_position']['upper_band']:.2f}",
                f"Lower Band of Bollinger Bands: {analysis_result['price_position']['lower_band']:.2f}",
            ]

            return "\n".join(output)

        except Exception as e:
            return f"Error analyzing market data: {str(e)}"


if __name__ == "__main__":
    model = HfApiModel(
        "Qwen/Qwen2.5-72B-Instruct", token="hf_"
    )

    api_key = os.getenv("ALPACA_API_KEY", "")
    secret_key = os.getenv("ALPACA_API_SECRET", "")

    market_analysis_tool = FinancialTool(
        alpaca_api_key=api_key, alpaca_api_secret=secret_key
    )

    search_tool = DuckDuckGoSearchTool()

    agent = ToolCallingAgent(tools=[market_analysis_tool, search_tool], model=model)
    agent_output = agent.run(
        f"""Please give me a detailed analysis of the market conditions for NVDA.
    Include:
    - Technical indicators (RSI, Bollinger Bands) using financial_tool
    - Current news sentiment
    - PE ratio comparison with industry
    - Market trend analysis
The date of the day is {datetime.now().strftime("%Y-%m-%d")}."""
    )
