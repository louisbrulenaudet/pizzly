import polars as pl

from ..core import BaseIndicator


class RSI(BaseIndicator):
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
        super().__init__(name="RSI", description="Relative Strength Index")
        self.dataframe = dataframe
        self.column = column
        self.window_size = window_size
        self.min_periods = min_periods
        self.latest_value = None

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

            # Store the latest RSI value for interpretation
            rsi_series = self.dataframe["rsi"].tail(-1)
            if len(rsi_series) > 0:
                self.latest_value = rsi_series.tail(1)[0]

            return rsi_series
        except Exception:
            self.latest_value = None
            return None

    def to_string(self) -> str:
        """
        Interpret the latest RSI value.

        This method provides a human-readable interpretation of the RSI indicator based on the most recent calculated value. It analyzes the momentum conditions and provides appropriate trading signals.

        Returns:
            str: Textual interpretation of the current RSI value and its implications
                for market conditions. If the RSI value is not available, returns
                an error message.

        Example:
            >>> rsi = RSI(df, "close", window_size=14)
            >>> rsi.compute()
            >>> interpretation = rsi.interpret()
            >>> print(interpretation)
            'RSI (14 periods) = 72.50 - The market is showing overbought conditions, suggesting potential reversal or correction.'
        """
        if self.latest_value is None:
            return "RSI indicator value is not available. Please compute the indicator first."

        rsi_value = float(self.latest_value)

        # Base interpretation text
        interpretation = f"RSI ({self.window_size} periods) = {rsi_value:.2f}"

        # Add condition-specific interpretation
        if rsi_value > 70:
            interpretation += " - The market is showing overbought conditions, suggesting potential reversal or correction."
        elif rsi_value < 30:
            interpretation += " - The market is showing oversold conditions, suggesting potential reversal or buying opportunity."
        elif rsi_value > 60:
            interpretation += " - The market is showing bullish momentum, but not yet at extreme levels."
        elif rsi_value < 40:
            interpretation += " - The market is showing bearish momentum, but not yet at extreme levels."
        else:
            interpretation += " - The market is in a neutral momentum state with no clear directional bias."

        return interpretation
