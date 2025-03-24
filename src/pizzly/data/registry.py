from ..core import BaseProvider
from ..core.registry import BaseRegistry


class DataSourceRegistry(BaseRegistry[BaseProvider]):
    """Registry for managing market data sources.

    This class provides a centralized registry for registering and retrieving
    market data source implementations.
    """

    _base_type = BaseProvider

    @classmethod
    def list_sources(cls) -> list[str]:
        """Get list of registered data sources.

        Returns:
            list[str]: Names of registered data sources
        """
        return cls.list_items()
