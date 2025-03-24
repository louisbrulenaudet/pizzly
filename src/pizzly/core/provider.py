from abc import ABC, abstractmethod
from datetime import datetime

import polars as pl


class BaseProvider(ABC):
    """Abstract base class for data providers.

    This class provides a standardized interface for implementing market data sources for different types of securities (stocks, crypto, etc).
    """

    def __init__(self) -> None:
        self._dataframe = None

    @property
    def dataframe(self) -> pl.DataFrame | None:
        """Get security data.

        Returns:
            pl.DataFrame | None: The security data or None if not fetched
        """
        return self._dataframe

    @abstractmethod
    def fetch(
        self, symbol: str, start: datetime | None = None, end: datetime | None = None
    ) -> pl.DataFrame | None:
        """Fetch market data for the security.

        Args:
            start (datetime | None): Start date for data fetch
            end (datetime | None): End date for data fetch

        Returns:
            pl.DataFrame | None: Fetched market data or None if fetch fails
        """
        pass
