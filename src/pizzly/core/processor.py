import concurrent.futures
from collections.abc import Callable
from typing import Any, Generic, ParamSpec, TypeVar

from .indicator import BaseIndicator

T = TypeVar("T")
P = ParamSpec("P")  # To represent positional parameter types


class Processor(Generic[T]):
    """
    A general-purpose processor for parallel execution of operations on indicators.

    This class provides functionality to efficiently process multiple technical indicators in parallel, improving performance for computationally intensive tasks while maintaining a clean, modular design.

    Time Complexity:
        - Parallel processing: O(max(n)), where n is the time for each individual operation
        - Sequential processing: O(sum(n)), cumulative time for all operations

    Space Complexity:
        - O(m), where m is the number of indicators/operations being processed

    Attributes:
        max_workers (int): Maximum number of concurrent workers for parallel processing

    Example:
        >>> rsi_indicator = RSI(dataframe=df, column="close", window_size=14)
        >>> bb_indicator = SmaBB(dataframe=df, column="close", window_size=20)
        >>> processor = Processor(max_workers=2)
        >>> results = processor.process_parallel([rsi_indicator, bb_indicator], 'compute')
        >>> interpretations = processor.process_parallel([rsi_indicator, bb_indicator], 'interpret')
    """

    def __init__(self, max_workers: int | None = None) -> None:
        """
        Initialize the processor with specified worker configuration.

        Args:
            max_workers (int, optional): Maximum number of concurrent workers.
                If None, uses the default from concurrent.futures. Defaults to None.
        """
        self.max_workers = max_workers

    def process_parallel(
        self,
        items: list[BaseIndicator],
        method_name: str,
        *args: object,
        **kwargs: dict[str, Any],
    ) -> dict[str, T]:
        """
        Process multiple indicators in parallel by calling a specified method on each.

        This method efficiently distributes the processing of indicators across multiple
        workers, maximizing CPU utilization for computationally intensive tasks.

        Args:
            items (List[BaseIndicator]): List of indicator objects to process
            method_name (str): Name of the method to call on each indicator
            *args: Positional arguments to pass to the called method (any object type)
            **kwargs: Keyword arguments to pass to the called method

        Returns:
            Dict[str, T]: Dictionary mapping indicator names to their results

        Example:
            >>> processor = Processor(max_workers=4)
            >>> indicators = [RSI(df, "close"), SmaBB(df, "close")]
            >>> # Compute all indicators in parallel
            >>> computed = processor.process_parallel(indicators, "compute")
            >>> # Get interpretations in parallel
            >>> interpretations = processor.process_parallel(indicators, "interpret")
        """
        results = {}

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            future_to_indicator = {
                executor.submit(
                    self._call_method, item, method_name, *args, **kwargs
                ): item
                for item in items
            }

            for future in concurrent.futures.as_completed(future_to_indicator):
                indicator = future_to_indicator[future]
                try:
                    result = future.result()
                    results[indicator.get_name()] = result
                except Exception as exc:
                    results[indicator.get_name()] = None
                    raise RuntimeError(
                        f"Error processing {indicator.get_name()}: {exc}"
                    ) from exc

        return results

    def _call_method(
        self, obj: object, method_name: str, *args: object, **kwargs: dict[str, Any]
    ) -> object:
        """
        Call a method on an object with the provided arguments.

        Args:
            obj (Any): Object instance to call method on
            method_name (str): Name of the method to call
            *args: Positional arguments to pass to the method (any object type)
            **kwargs: Keyword arguments to pass to the method

        Returns:
            Any: Result of the method call

        Raises:
            AttributeError: If the method doesn't exist on the object
        """
        method = getattr(obj, method_name)
        return method(*args, **kwargs)

    def map_sequential(
        self,
        items: list[Any],
        func: Callable[[Any], T],
        *args: object,
        **kwargs: dict[str, Any],
    ) -> dict[str, T]:
        """
        Map a function to multiple items sequentially.

        Args:
            items (List[Any]): List of items to process
            func (Callable): Function to apply to each item
            *args: Additional positional arguments to pass to the function (any object type)
            **kwargs: Additional keyword arguments to pass to the function

        Returns:
            Dict[str, T]: Dictionary mapping item names to their results

        Example:
            >>> processor = Processor()
            >>> results = processor.map_sequential(indicators, lambda x: x.calculate_value())
        """
        results = {}
        for item in items:
            try:
                result = func(item, *args, **kwargs)
                results[item.get_name() if hasattr(item, "get_name") else str(item)] = (
                    result
                )
            except Exception:
                name = item.get_name() if hasattr(item, "get_name") else str(item)
                results[name] = None
        return results
