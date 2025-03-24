![Plot](https://github.com/louisbrulenaudet/pizzly/blob/main/assets/thumbnail.png?raw=true)

# Pizzly, financial market analysis combining technical indicators with LLMs, featuring real-time data processing and AI-powered market insights ⚡️
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) ![Maintainer](https://img.shields.io/badge/maintainer-@louisbrulenaudet-blue) ![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg) ![Code Style](https://img.shields.io/badge/code%20style-ruff-000000.svg) ![Package Manager](https://img.shields.io/badge/package%20manager-uv-purple.svg)

This agentic market analysis system is a Python-based framework that combines technical analysis with artificial intelligence to provide comprehensive market insights. At its core, the system implements a modular architecture that seamlessly integrates statistical analysis methods with natural language processing capabilities.

The system's foundation is built upon two primary technical indicators: the Relative Strength Index (RSI) and Bollinger Bands. The RSI implementation provides momentum analysis through a configurable calculation window (default: 14 periods), employing dynamic gain/loss computation and rolling averages to measure the velocity and magnitude of price movements. This is complemented by a Bollinger Bands implementation that utilizes Simple Moving Averages (SMA) and dynamic standard deviation calculations to create adaptive volatility bands that automatically adjust to market conditions.

Market data acquisition is handled through an integration with the Alpaca API, providing access to historical price data across various timeframes. The system employs Polars for high-performance data manipulation, leveraging its columnar storage format and lazy evaluation capabilities to efficiently process large datasets.

The AI integration layer bridges technical analysis with natural language processing using the Qwen2.5-72B-Instruct model via the Hugging Face API. This enables sophisticated market analysis by combining traditional technical indicators with real-time news sentiment analysis through DuckDuckGo search integration.

## Implementation Guide

### Installation

Install the package using pip:

```bash
pip install pizzly
```

# Usage

First, set up your API credentials as environment variables:

```bash
export ALPACA_API_KEY=your_alpaca_api_token
export ALPACA_API_SECRET=your_alpaca_api_secret
export HF_TOKEN=your_hf_token
```

Then use Pizzly like this:

```python
from smolagents import DuckDuckGoSearchTool, HfApiModel
from smolagents.agents import ToolCallingAgent

from pizzly.data.alpaca import AlpacaStock
from pizzly.tools import FinancialTool

model = HfApiModel(
    "meta-llama/Llama-3.3-70B-Instruct",
    token=hf_token
)

data_provider = AlpacaStock(api_key, secret_key)
market_analysis_tool = FinancialTool(data_provider=data_provider)

search_tool = DuckDuckGoSearchTool()

agent = ToolCallingAgent(
    tools=[
        market_analysis_tool,
        search_tool
    ],
    model=model
)

prompt = f"""Please give me a detailed analysis of the market conditions for NVDA.
Include:
- Technical indicators (RSI, Bollinger Bands) using financial_tool
- Current news sentiment
- PE ratio comparison with industry
- Market trend analysis
The date of the day is {datetime.now().strftime("%Y-%m-%d")}."""

agent_output = agent.run(
    prompt
)
```

## Development
### Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) for package management

### Setting up the development environment

1. Clone the repository:
```bash
git clone https://github.com/louisbrulenaudet/pizzly
cd pizzly
```

2. Initialize the development environment:
```bash
make init
```

## Citing this project
If you use this code in your research, please use the following BibTeX entry.

```BibTeX
@misc{louisbrulenaudet2025,
	author = {Louis Brulé Naudet},
	title = {Pizzly, financial market analysis combining technical indicators with LLMs, featuring real-time data processing and AI-powered market insights ⚡️},
	howpublished = {\url{https://github.com/louisbrulenaudet/pizzly}},
	year = {2025}
}
```

## Feedback
If you have any feedback, please reach out at [louisbrulenaudet@icloud.com](mailto:louisbrulenaudet@icloud.com).
