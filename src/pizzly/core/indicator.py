from abc import ABCMeta, abstractmethod

import polars as pl
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class BaseIndicator(metaclass=ABCMeta):
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

    def __init__(self, name: str, description: str) -> None:
        """
        Initialize the indicator with a name and description.

        Args:
            name (str): Identifier for the indicator
        """
        self.name = name
        self.description = description
        self.series = None

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

    def get_description(self) -> str:
        """
        Retrieve the indicator's description.

        Returns:
            str: The indicator's description

        Raises:
            Exception: If there's an error accessing the description attribute
        """
        try:
            return self.description
        except Exception:
            return ""

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

    @abstractmethod
    def compute(self) -> pl.Series | pl.DataFrame | None | tuple | list:
        """
        Compute the indicator values.

        This method implements the core logic for calculating the indicator values based on the input data. Each indicator subclass must provide its specific implementation.

        Returns:
            Optional[pl.Series | pl.DataFrame ]: The calculated indicator values or None if calculation fails

        Raises:
            NotImplementedError: If not implemented by the subclass

        Example:
            >>> indicator = ConcreteIndicator(df, "close", window_size=14)
            >>> result = indicator.compute()
            >>> print(result)
            shape: (100,)
            Series: 'indicator_name' [f64]
            [
                45.23,
                46.78,
                ...
            ]
        """
        raise NotImplementedError("Each indicator must implement its computation logic")

    def to_string(self) -> str:
        """
        Generate a human-readable interpretation of the indicator values.

        This method provides a default implementation that returns a simple message. Indicator subclasses should override this method to provide specific, meaningful interpretations of their calculated values.

        Returns:
            str: A text interpretation of the indicator values

        Example:
            >>> indicator = ConcreteIndicator(df, "close", window_size=14)
            >>> indicator.compute()
            >>> print(indicator.to_string())
            'ConcreteIndicator value is 45.67, indicating neutral market conditions.'
        """
        return f"No specific interpretation available for {self.get_name()} indicator. Consider implementing a custom to_string() method."
