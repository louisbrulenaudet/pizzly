from smolagents import Tool

from ..core.registry import BaseRegistry


class ToolRegistry(BaseRegistry[Tool]):
    """Registry for managing market analysis tools.

    This class provides a centralized registry for registering and retrieving
    market analysis tool implementations.
    """

    _base_type = Tool

    @classmethod
    def list_tools(cls) -> list[str]:
        """Get list of registered tools.

        Returns:
            list[str]: Names of registered tools
        """
        return cls.list_items()
