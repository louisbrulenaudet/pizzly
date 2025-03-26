from datetime import datetime

import polars as pl
from smolagents import Tool

from .._enums import Type
from ..core.processor import Processor
from ..core.provider import BaseProvider
from ..indicators import RSI, SmaBB
from ..models.tools import IndicatorOutput

__all__ = ["FinancialTool"]


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
        - Parallel processing of indicators
        - Detailed interpretations for each indicator

    Signal Types:
        - Momentum signals from RSI
        - Volatility signals from Bollinger Bands
        - Combined indicator signals
        - Support/Resistance levels
        - Trend analysis

    Attributes:
        data_provider (BaseProvider): Data provider instance for fetching market data
        processor (Processor): Processor for parallel computation of indicators
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
        >>> data_provider = AlpacaStock("AAPL", api_key, secret_key)
        >>> analyzer = FinancialTool(data_provider)
        >>> analysis = analyzer.forward("AAPL", rsi_window=14, bb_window=20, start_date="2023-01-01")
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
            "type": Type.STRING,
            "description": "Stock symbol to analyze (e.g., 'AAPL')",
        },
        "rsi_window": {
            "type": Type.INTEGER,
            "description": "Window size for RSI calculation (default: 14)",
        },
        "bb_window": {
            "type": Type.INTEGER,
            "description": "Window size for Bollinger Bands calculation (default: 20)",
        },
        "start_date": {
            "type": Type.STRING,
            "description": "Start date for analysis (YYYY-MM-DD format). This is the lower bound for historical data retrieval. The range is in days, so the start date is generally set to 6 months before the current date.",
            "required": True,
        },
        "end_date": {
            "type": Type.STRING,
            "description": "End date for analysis (YYYY-MM-DD format). This is casually set to the current date if provided.",
            "required": False,
            "nullable": True,
        },
    }
    output_type = Type.STRING

    def __init__(
        self,
        data_provider: BaseProvider,
        **kwargs: dict[str, str | int | float],
    ) -> None:
        """
        Initialize the financial analysis tool with a data provider.

        Args:
            data_provider (BaseProvider): Provider instance for fetching market data
            **kwargs: Additional arguments to pass to the Tool constructor

        Example:
            >>> from src.pizzly.data.alpaca import AlpacaStock
            >>> data_provider = AlpacaStock(api_key, secret_key)
            >>> tool = FinancialTool(data_provider=data_provider)
        """
        self.data_provider = data_provider
        self.processor = Processor(max_workers=4)
        super().__init__(**kwargs)

    def fetch_market_data(
        self, symbol: str, start: datetime | None = None, end: datetime | None = None
    ) -> pl.DataFrame | None:
        """
        Fetch historical market data for analysis.

        Retrieves historical price data for the specified symbol using the provided data provider.

        Args:
            symbol (str): Stock symbol to fetch data for (e.g., 'AAPL')
            start (datetime | None): Start date for historical data
            end (datetime | None): End date for historical data

        Returns:
            pl.DataFrame: DataFrame containing historical price data

        Example:
            >>> analyzer = FinancialTool(data_provider)
            >>> df = analyzer.fetch_market_data("AAPL")
            >>> print(df.head(3))
            shape: (3, 8)
        """
        return self.data_provider.fetch(symbol=symbol, start=start, end=end)

    def analyze_market_conditions(
        self, df: pl.DataFrame, rsi_window: int, bb_window: int
    ) -> str:
        """
        Analyze market conditions using technical indicators.

        This method combines RSI and Bollinger Bands analysis to provide a comprehensive market assessment including trend strength, potential reversal points, and trading signals. It uses parallel processing to compute indicators efficiently and gathers their interpretations.

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
                - interpretations (Dict[str, str]): Human-readable interpretations from each indicator

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
                ],
                'interpretations': {
                    'RSI': 'RSI (14 periods) = 62.50 - The market is showing bullish momentum, but not yet at extreme levels.',
                    'SmaBB': 'Bollinger Bands (20 periods): Price ($178.05) is between the middle ($177.45) and upper band ($182.35), suggesting bullish price action within normal volatility range.'
                }
            }
        """
        if len(df) < max(rsi_window, bb_window):
            raise ValueError(
                f"Insufficient data: need at least {max(rsi_window, bb_window)} data points"
            )

        rsi_indicator = RSI(dataframe=df, column="close", window_size=rsi_window)
        bb_indicator = SmaBB(dataframe=df, column="close", window_size=bb_window)

        indicators = [rsi_indicator, bb_indicator]

        self.processor.process_parallel(indicators, "compute")

        interpretations = self.processor.process_parallel(indicators, "to_string")

        return IndicatorOutput.format(interpretations)

    def forward(  # type: ignore[override]
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
            rsi_window (int): RSI calculation period. Defaults to 14
            bb_window (int): Bollinger Bands calculation period. Defaults to 20
            start_date (str): Start date for analysis in YYYY-MM-DD format.
            end_date (str, optional): End date for analysis in YYYY-MM-DD format. Defaults to current date.

        Returns:
            str: Formatted analysis results including:
                - Current RSI value and interpretation
                - Price position relative to Bollinger Bands
                - Generated trading signals
                - Overall market condition assessment
                - Detailed interpretations from each indicator

        Example:
            >>> data_provider = AlpacaStock("AAPL", api_key, secret_key)
            >>> analyzer = FinancialTool(data_provider=data_provider)
            >>> result = analyzer.forward(
            ...     symbol="AAPL",
            ...     rsi_window=14,
            ...     bb_window=20,
            ...     start_date="2024-01-01"
            ... )
            >>> print(result)
            Market Analysis Results for AAPL:
            RSI (14 periods): 58.43

            Price Position:
            Current Price: 173.25
            Upper Band: 180.45
            Lower Band: 165.87

            Detailed Analysis:
            - RSI (14 periods) = 58.43 - The market is showing bullish momentum, but not yet at extreme levels.
            - Bollinger Bands (20 periods): Price ($173.25) is between the middle ($172.15) and upper band ($180.45), suggesting bullish price action within normal volatility range.

        Note:
            - RSI > 70 indicates overbought conditions
            - RSI < 30 indicates oversold conditions
            - Price near Bollinger Bands can signal potential reversals
        """
        try:
            try:
                if start_date:
                    datetime.strptime(start_date, "%Y-%m-%d")
                if end_date:
                    datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                return "Error: Date format must be YYYY-MM-DD"

            if rsi_window <= 0 or bb_window <= 0:
                return "Error: Window sizes must be positive integers"

            # Convert date strings to datetime objects if provided
            start = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
            end = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None

            df = self.fetch_market_data(symbol, start=start, end=end)
            if df is None:
                return "Error: Failed to fetch market data"

            analysis = self.analyze_market_conditions(df, rsi_window, bb_window)

            return analysis

        except Exception as e:
            return f"Error: {str(e)}"
