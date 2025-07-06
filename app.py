from polars import DataFrame

from pizzly.data import Edgar, Finnhub

if __name__ == "__main__":
    finnhub_provider = Finnhub(
        finnhub_api_key="d1ci5l9r01qvlf605p10d1ci5l9r01qvlf605p1g"
    )

    ipo_calendar: DataFrame | object = finnhub_provider.fetch(
        "ipo_calendar", _from="2025-07-01", to="2025-08-01"
    )

    print(ipo_calendar.to_dict())

    # edgar_provider = Edgar()
    # edgar_filings = edgar_provider.fetch(days_back=7)
    # print(edgar_filings)
