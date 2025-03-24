import os
from datetime import datetime

from dotenv import load_dotenv
from smolagents import DuckDuckGoSearchTool, HfApiModel
from smolagents.agents import ToolCallingAgent

from src.pizzly.data.alpaca import AlpacaStock
from src.pizzly.tools import FinancialTool

# Load environment variables
load_dotenv()


if __name__ == "__main__":
    hf_token = os.getenv("HF_TOKEN", "hf_")
    api_key = os.getenv("ALPACA_API_KEY", "")
    secret_key = os.getenv("ALPACA_API_SECRET", "")

    model = HfApiModel("meta-llama/Llama-3.3-70B-Instruct", token=hf_token)
    data_provider = AlpacaStock(api_key, secret_key)

    market_analysis_tool = FinancialTool(data_provider=data_provider)

    search_tool = DuckDuckGoSearchTool()

    agent = ToolCallingAgent(tools=[market_analysis_tool, search_tool], model=model)

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
