from .alpaca import AlpacaStock
from .edgar import Edgar
from .finnhub import Finnhub
from .registry import DataSourceRegistry

# Register the Alpaca data source
DataSourceRegistry.register(AlpacaStock)
DataSourceRegistry.register(Finnhub)
DataSourceRegistry.register(Edgar)

__all__ = ["DataSourceRegistry", "AlpacaStock", "Finnhub", "Edgar"]
