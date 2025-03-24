from .bollinger import SmaBB
from .registry import IndicatorRegistry
from .rsi import RSI

# Register the indicators
IndicatorRegistry.register(RSI)
IndicatorRegistry.register(SmaBB)

__all__ = ["IndicatorRegistry", "RSI", "SmaBB"]
