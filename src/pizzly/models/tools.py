from pydantic import BaseModel

__all__ = ["FinancialAnalysisModel"]


class PricePositionModel(BaseModel):
    """
    Pydantic model for price position analysis.

    This model represents the structured output of price position analysis,
    including the current price and its relation to Bollinger Bands.
    """

    current_price: float
    upper_band: float
    lower_band: float


class FinancialAnalysisModel(BaseModel):
    """
    Pydantic model for financial analysis results.

    This model represents the structured output of market analysis,
    including RSI values, price positions relative to Bollinger Bands,
    and generated trading signals.
    """

    # RSI Analysis
    rsi_value: float | None

    # Price Position Analysis
    price_position: PricePositionModel | None

    # Trading Signals
    signals: list[str] | None = []
