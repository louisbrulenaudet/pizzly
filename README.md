
<div align="center">
    <img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python Version">
    <img src="https://img.shields.io/badge/code%20style-ruff-000000.svg" alt="Code Style">
    <img src="https://img.shields.io/badge/type%20checker-pyright-yellowgreen.svg" alt="Type Checker">
    <img src="https://img.shields.io/badge/package%20manager-uv-purple.svg" alt="Package Manager">
    <img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen.svg" alt="Pre-commit">
</div>

<h3>
    <div style="display:flex;flex-direction:row;justify-content: center;align-items: center;">
        <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/mascot.png" alt="Hugging Face mascot as James Bond" width="100px">
    </div>
</h3>

# Barebones tool built upon Hugging Face smolagents and Alpaca for financial analysis automation 🤗

## Overview

This agentic market analysis system is a Python-based framework that combines technical analysis with artificial intelligence to provide comprehensive market insights. At its core, the system implements a modular architecture that seamlessly integrates statistical analysis methods with natural language processing capabilities.

The system's foundation is built upon two primary technical indicators: the Relative Strength Index (RSI) and Bollinger Bands. The RSI implementation provides momentum analysis through a configurable calculation window (default: 14 periods), employing dynamic gain/loss computation and rolling averages to measure the velocity and magnitude of price movements. This is complemented by a Bollinger Bands implementation that utilizes Simple Moving Averages (SMA) and dynamic standard deviation calculations to create adaptive volatility bands that automatically adjust to market conditions.

Market data acquisition is handled through an integration with the Alpaca API, providing access to historical price data across various timeframes. The system employs Polars for high-performance data manipulation, leveraging its columnar storage format and lazy evaluation capabilities to efficiently process large datasets.

The AI integration layer bridges technical analysis with natural language processing using the Qwen2.5-72B-Instruct model via the Hugging Face API. This enables sophisticated market analysis by combining traditional technical indicators with real-time news sentiment analysis through DuckDuckGo search integration.

## Implementation Guide

### Installation

The project uses a Makefile system for setup and development:

1. **Initial Setup**
```bash
make init
```

2. **Development Environment**
```bash
make run
```

4. **Code Quality Checks**
```bash
make check
```

[![Code execution](https://img.youtube.com/vi/FXY-WNnp2oE/0.jpg)](https://www.youtube.com/watch?v=FXY-WNnp2oE)

### Configuration

Required environment variables in `.env`:
```
ALPACA_API_KEY=your_api_key
ALPACA_API_SECRET=your_api_secret
```

### Usage Example

The following example demonstrates comprehensive market analysis combining technical indicators with AI-powered insights:

```python
import os
from datetime import datetime
from smolagents import DuckDuckGoSearchTool, HfApiModel
from smolagents.agents import ToolCallingAgent

# Initialize AI model
model = HfApiModel(
    "Qwen/Qwen2.5-72B-Instruct",
    token="hf_"
)

# Set up market analysis tools
api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_API_SECRET")

market_analysis_tool = FinancialTool(
    alpaca_api_key=api_key,
    alpaca_api_secret=secret_key
)

# Configure search capability
search_tool = DuckDuckGoSearchTool()

# Create analysis agent
agent = ToolCallingAgent(
    tools=[market_analysis_tool, search_tool],
    model=model
)

# Execute comprehensive analysis
analysis = agent.run(
    """Please give me a detailed analysis of the market conditions for NVDA.
    Include:
    - Technical indicators (RSI, Bollinger Bands)
    - Current news sentiment
    - PE ratio comparison with industry
    - Market trend analysis
    """
)
```

## Citing this project
If you use this code in your research, please use the following BibTeX entry.

```BibTeX
@misc{louisbrulenaudet2025,
	author = {Louis Brulé Naudet},
	title = {Barebones tool built upon Hugging Face smolagents and Alpaca for financial analysis automation},
	howpublished = {\url{https://github.com/louisbrulenaudet/agentic-market-tool}},
	year = {2025}
}
```

## Feedback
If you have any feedback, please reach out at [louisbrulenaudet@icloud.com](mailto:louisbrulenaudet@icloud.com).
