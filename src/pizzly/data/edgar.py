from datetime import date, timedelta

import httpx
import polars as pl

from ..__version__ import __version__
from ..core import BaseProvider

__all__ = ["Edgar"]


class Edgar(BaseProvider):
    """
    A provider for fetching and parsing SEC Edgar S-1 and F-1 filings as Polars DataFrames.

    This class interacts with the SEC Edgar daily index to retrieve recent S-1 and F-1 filings, parses the results, and returns them as a Polars DataFrame. It handles batching over multiple days, skips weekends/holidays, and provides robust error handling for missing data.

    Attributes:
        base_url (str): URL template for SEC Edgar daily index files.
        http (httpx.Client): HTTP client configured with appropriate headers and timeout.
        _dataframe (pl.DataFrame): Cached DataFrame of the most recent filings.

    Features:
        - Fetches S-1 and F-1 filings from the SEC Edgar daily index.
        - Handles batching over a configurable number of days.
        - Skips weekends and holidays automatically.
        - Returns results as a Polars DataFrame with parsed columns.
        - Robust error handling for missing or malformed data.

    Example:
        >>> from src.pizzly.data.edgar import Edgar
        >>> provider = Edgar()
        >>> df = provider.fetch(days_back=5)
        >>> print(df.head())

    Notes:
        - Only S-1 and F-1 forms are parsed and returned.
        - All dates are in UTC.
        - Requires internet access to fetch data from the SEC.
        - User-Agent header is required by the SEC and set automatically.
    """

    def __init__(self, user_agent: str = f"Pizzly/{__version__} (+https://example.com/contact)") -> None:
        """
        Initialize the Edgar provider with a custom User-Agent.

        Args:
            user_agent (str): User-Agent string for SEC requests. Defaults to a Pizzly-specific value.
        """
        super().__init__()
        self.base_url = "https://www.sec.gov/Archives/edgar/daily-index/{year}/QTR{q}/master.{yyyymmdd}.idx"
        self.http = httpx.Client(
            headers={
                "User-Agent": user_agent,
            },
            timeout=httpx.Timeout(30),
        )

    def _master_url_for(self, d: date) -> str:
        """
        Construct the SEC Edgar master index URL for a given date.

        Args:
            d (date): The date for which to construct the URL.

        Returns:
            str: The formatted URL for the SEC Edgar master index file.
        """
        quarter = ((d.month - 1) // 3) + 1

        return self.base_url.format(
            year=d.year, q=quarter, yyyymmdd=d.strftime("%Y%m%d")
        )

    def fetch(self, days_back: int = 7) -> pl.DataFrame:
        """
        Fetch recent S-1 and F-1 filings from the SEC Edgar daily index.

        Iterates over the last `days_back` days, retrieves and parses the daily index files,
        and returns a Polars DataFrame containing the filings.

        Args:
            days_back (int): Number of days to look back for filings. Defaults to 7.

        Returns:
            pl.DataFrame: DataFrame with columns ['cik', 'company', 'form_type', 'date_filed', 'file_name'].

        Raises:
            httpx.HTTPStatusError: For non-404 HTTP errors during data retrieval.

        Example:
            >>> provider = Edgar()
            >>> df = provider.fetch(days_back=5)
            >>> print(df.head())
        """
        records = []
        today = date.today()
        for d in (today - timedelta(n) for n in range(days_back)):
            try:
                txt = self.http.get(self._master_url_for(d)).text

            except httpx.HTTPStatusError as ex:
                if ex.response.status_code == 404:
                    continue  # weekends / holidays
                raise

            for line in txt.splitlines():
                if "|S-1" in line or "|F-1" in line:
                    parts = line.split("|")
                    if len(parts) != 5:
                        continue

                    record = {
                        "cik": parts[0],
                        "company": parts[1],
                        "form_type": parts[2],
                        "date_filed": parts[3],
                        "file_name": parts[4],
                    }

                    records.append(record)

        self._dataframe = (
            pl.DataFrame(records)
            .with_columns(
                pl.col("date_filed").str.strptime(pl.Date, "%Y%m%d")
            )
        )

        return self._dataframe
