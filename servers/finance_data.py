"""MCP server for finance data tools (yfinance + BSE/NSE reports). Port 8011."""

import logging

import yfinance as yf
from dotenv import load_dotenv
from fastmcp import FastMCP

load_dotenv()

logger = logging.getLogger("mcp_tool_servers.finance_data")

mcp = FastMCP("finance-data", instructions="Financial market data and reports tools.")


@mcp.tool()
def get_ticker_data(ticker: str) -> str:
    """Get basic market data and company information for a given ticker symbol.
    For Indian stocks use .NS (NSE) or .BO (BSE) suffix, e.g., 'RELIANCE.NS'.
    Returns current price, market cap, P/E ratios, and a business summary."""
    logger.info("Fetching market data for ticker='%s'", ticker)
    try:
        t = yf.Ticker(ticker)
        info = t.info
        relevant_keys = [
            "shortName", "symbol", "currentPrice", "marketCap", "sector", "industry",
            "trailingPE", "forwardPE", "dividendYield", "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
            "longBusinessSummary",
        ]
        data = {k: info.get(k) for k in relevant_keys}
        lines = [f"**{k}:** {v}" for k, v in data.items() if v is not None]
        return "\n".join(lines) if lines else f"No data found for {ticker}"
    except Exception as e:
        logger.error("Error fetching data for ticker='%s': %s", ticker, e)
        return f"Failed to fetch data for {ticker}: {e}"


@mcp.tool()
def get_bse_nse_reports(ticker: str) -> str:
    """Fetch raw quarterly and yearly financial reports (Income Statement, Balance Sheet, Cash Flow)
    for a given ticker (use .NS or .BO suffix for Indian stocks).
    Returns markdown-formatted tabular data."""
    logger.info("Fetching financial reports for ticker='%s'", ticker)
    results = []
    try:
        t = yf.Ticker(ticker)

        report_sources = [
            ("Yearly Income Statement", lambda: t.income_stmt),
            ("Quarterly Income Statement", lambda: t.quarterly_income_stmt),
            ("Yearly Balance Sheet", lambda: t.balance_sheet),
            ("Quarterly Balance Sheet", lambda: t.quarterly_balance_sheet),
            ("Yearly Cash Flow", lambda: t.cashflow),
            ("Quarterly Cash Flow", lambda: t.quarterly_cashflow),
        ]

        for title, fetch_fn in report_sources:
            try:
                df = fetch_fn()
                if not df.empty:
                    results.append(f"## {ticker} {title}\n\n{df.to_markdown()}")
            except Exception as e:
                logger.warning("Failed to fetch %s for %s: %s", title, ticker, e)

        return "\n\n---\n\n".join(results) if results else f"No financial reports found for {ticker}"
    except Exception as e:
        logger.error("Error fetching financial reports for ticker='%s': %s", ticker, e)
        return f"Failed to fetch reports for {ticker}: {e}"


if __name__ == "__main__":
    from shared.config import PORTS
    mcp.run(transport="streamable-http", host="0.0.0.0", port=PORTS["finance-data"])
