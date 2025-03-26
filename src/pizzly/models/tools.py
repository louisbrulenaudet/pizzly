from pydantic import BaseModel

__all__ = ["IndicatorOutput"]


class IndicatorOutput(BaseModel):
    """
    A class that transforms a dictionary of technical indicator interpretations into formatted text.

    This class is designed to take technical indicator analysis results in dictionary form and convert them into a readable and structured text format suitable for presentation to users or for use by an LLM.

    Attributes:
        None

    Methods:
        format_interpretations: Converts a dictionary of interpretations into formatted text

    Examples:
        >>> interpretations = {
        ...     'RSI': 'RSI (14 periods) = 53.84 - The market is in a neutral momentum state with no clear directional bias.',
        ...     'SmaBB': 'Bollinger Bands (20 periods): Price ($120.69) is between the middle ($117.33) and upper band ($128.40), suggesting bullish price action within normal volatility range.'
        ... }
        >>> formatter = IndicatorOutput()
        >>> result = formatter.format_interpretations(interpretations)
        >>> print(result)
        Detailed Analysis:
        - RSI (14 periods) = 53.84 - The market is in a neutral momentum state with no clear directional bias.
        - Bollinger Bands (20 periods): Price ($120.69) is between the middle ($117.33) and upper band ($128.40), suggesting bullish price action within normal volatility range.
    """

    text: str | None = None

    @staticmethod
    def format(interpretations: dict[str, str]) -> str:
        """
        Converts a dictionary of indicator interpretations into formatted text.

        This method takes a dictionary where the keys are indicator names and
        values are textual interpretations of those indicators, then formats them into structured text with a header and bullet points for each indicator.

        Args:
            interpretations (dict[str, str]): Dictionary containing indicator
                interpretations, where keys are indicator names and values
                are textual descriptions.

        Returns:
            str: Formatted text containing all indicator interpretations
                with a header and readable structure.

        Example:
            >>> interpretations = {
            ...     'RSI': 'RSI (14 periods) = 62.50 - The market is showing bullish momentum, but not yet at extreme levels.',
            ...     'SmaBB': 'Bollinger Bands (20 periods): Price ($178.05) is between the middle ($177.45) and upper band ($182.35).'
            ... }
            >>> IndicatorOutput.format_interpretations(interpretations)
            'Detailed Analysis:\n- RSI (14 periods) = 62.50 - The market is showing bullish momentum, but not yet at extreme levels.\n- Bollinger Bands (20 periods): Price ($178.05) is between the middle ($177.45) and upper band ($182.35).'
        """
        if not interpretations:
            return "No indicator analysis available."

        result = "Detailed Analysis:\n"
        for _, interpretation in interpretations.items():
            result += f"- {interpretation}\n"

        return result.strip()
