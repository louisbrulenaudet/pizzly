from typing import Generic, TypeVar

T = TypeVar("T")


class BaseRegistry(Generic[T]):
    """Base class for registry implementations.

    This class provides common registry functionality that can be inherited
    by specific registry implementations.
    """

    _registry: dict[str, type[T]] = {}
    _base_type: type[T] | None = None

    @classmethod
    def register(
        cls, item: type[T] | list[type[T]] | tuple[type[T], ...] | dict[str, type[T]]
    ) -> None:
        """
        Register a new item or multiple items.

        Args:
            item: Single item or collection of items to register

        Raises:
            typeError: If item is not a class or doesn't inherit from required base type
            ValueError: If item is already registered
        """
        if isinstance(item, list | tuple):
            for i in item:
                cls.register(i)
            return

        if isinstance(item, dict):
            for i in item.values():
                cls.register(i)
            return

        if not isinstance(item, type):
            raise TypeError(f"{item.__name__} must be a class, not an instance")

        if cls._base_type and not issubclass(item, cls._base_type):
            raise TypeError(
                f"{item.__name__} must inherit from {cls._base_type.__name__}"
            )

        if item.__name__ in cls._registry:
            raise ValueError(f"{item.__name__} is already registered")

        cls._registry[item.__name__] = item

    @classmethod
    def get(cls, name: str) -> type[T]:
        """
        Get an item by name.

        Args:
            name: Name of the item to retrieve

        Returns:
            The requested item

        Raises:
            KeyError: If item is not found in registry
        """
        if name not in cls._registry:
            raise KeyError(f"{name} not found in registry")
        return cls._registry[name]

    @classmethod
    def list_items(cls) -> list[str]:
        """
        Get list of registered items.

        Returns:
            list of registered item names
        """
        return list(cls._registry.keys())
