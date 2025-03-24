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
