from ..core import BaseIndicator
from ..core.registry import BaseRegistry


class IndicatorRegistry(BaseRegistry[BaseIndicator]):
    """Registry for managing and accessing technical indicators.

    This class provides a centralized registry for registering and retrieving
    technical indicator implementations.
    """

    _base_type = BaseIndicator

    @classmethod
    def list_indicators(cls) -> list[str]:
        """Get list of registered indicators.

        Returns:
            list[str]: Names of registered indicators
        """
        return cls.list_items()
