from .alpaca import AlpacaStock
from .registry import DataSourceRegistry

# Register the Alpaca data source
DataSourceRegistry.register(AlpacaStock)

__all__ = ["DataSourceRegistry", "AlpacaStock"]
