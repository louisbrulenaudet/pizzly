import polars as pl

from ..core import BaseIndicator


class SmaBB(BaseIndicator):
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
        window_size: int = 20,
        min_periods: int = 1,
    ) -> None:
        """Initialize Bollinger Bands indicator with input data and parameters."""
        super().__init__(
            name="SmaBB", description="Simple Moving Average Bollinger Bands"
        )
        self.dataframe = dataframe
        self.column = column
        self.window_size = window_size
        self.min_periods = min_periods
        self.sma = None
        self.upper_band = None
        self.lower_band = None
        self.latest_price = None
        self.latest_upper = None
        self.latest_lower = None
        self.latest_sma = None

    def get_series(self) -> pl.Series | None:
        """
        Retrieve the computed Bollinger Bands values.

        This method provides access to the stored indicator values after computation. Should be called after compute() to ensure values are available.

        Returns:
            Optional[tuple[pl.Series, pl.Series, pl.Series]]: A tuple containing the computed Bollinger Bands series or None if not calculated

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

            # Store latest values for interpretation
            if len(self.dataframe) > 0:
                self.latest_price = self.dataframe[self.column].tail(1)[0]

            if len(self.sma) > 0:
                self.latest_sma = self.sma.tail(1)[0]

            if len(self.upper_band) > 0:
                self.latest_upper = self.upper_band.tail(1)[0]

            if len(self.lower_band) > 0:
                self.latest_lower = self.lower_band.tail(1)[0]

            return self.sma, self.upper_band, self.lower_band

        except Exception:
            self.latest_price = None
            self.latest_sma = None
            self.latest_upper = None
            self.latest_lower = None
            return pl.Series([]), pl.Series([]), pl.Series([])

    def to_string(self) -> str:
        """
        Interpret the latest Bollinger Bands values.

        This method provides a human-readable interpretation of the relationship between the current price and the Bollinger Bands, offering insights into volatility, potential overbought/oversold conditions, and trend strength.

        Returns:
            str: Textual interpretation of the current price position relative to Bollinger Bands
                and its implications for market conditions. If values are not available, returns
                an error message.

        Example:
            >>> bb = SmaBB(df, "close", window_size=20)
            >>> bb.compute()
            >>> interpretation = bb.interpret()
            >>> print(interpretation)
            'Bollinger Bands (20 periods): Price ($172.55) is above the upper band ($171.25),
            indicating potential overbought conditions or strong upward momentum.'
        """
        if (
            self.latest_price is None
            or self.latest_sma is None
            or self.latest_upper is None
            or self.latest_lower is None
        ):
            return "Bollinger Bands values are not available. Please compute the indicator first."

        # Format values with 2 decimal places
        price = float(self.latest_price)
        sma = float(self.latest_sma)
        upper = float(self.latest_upper)
        lower = float(self.latest_lower)

        # Base information
        interpretation = (
            f"Bollinger Bands ({self.window_size} periods): Price (${price:.2f})"
        )

        # Calculate bandwidth for volatility assessment
        bandwidth = (upper - lower) / sma * 100

        # Position relative to bands
        if price > upper:
            interpretation += f" is above the upper band (${upper:.2f}), indicating potential overbought conditions or strong upward momentum."
        elif price < lower:
            interpretation += f" is below the lower band (${lower:.2f}), indicating potential oversold conditions or strong downward momentum."
        elif price > sma:
            interpretation += f" is between the middle (${sma:.2f}) and upper band (${upper:.2f}), suggesting bullish price action within normal volatility range."
        elif price < sma:
            interpretation += f" is between the middle (${sma:.2f}) and lower band (${lower:.2f}), suggesting bearish price action within normal volatility range."
        else:
            interpretation += (
                f" is at the middle band (${sma:.2f}), indicating neutral momentum."
            )

        # Add volatility assessment
        if bandwidth > 4:
            interpretation += (
                f" Bandwidth of {bandwidth:.2f}% indicates high volatility."
            )
        elif bandwidth < 2:
            interpretation += f" Bandwidth of {bandwidth:.2f}% indicates low volatility, suggesting potential for increased volatility soon."

        return interpretation
