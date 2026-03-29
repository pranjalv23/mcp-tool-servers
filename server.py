"""Combined MCP server — all tools on a single port."""

import json
import logging
import os
import sys
from pathlib import Path

import arxiv
import yfinance as yf
from dotenv import load_dotenv
from fastmcp import FastMCP
from firecrawl import FirecrawlApp
from tavily import TavilyClient

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.vector_db import VectorDB

logger = logging.getLogger("mcp_tool_servers")

mcp = FastMCP("mcp-tool-servers", instructions="Web search, finance data, and vector DB tools.")


# ── Lazy clients ──────────────────────────────────────────────

_tavily_client = None
_firecrawl_app = None
_arxiv_client = arxiv.Client()
_db_instances = {}


def _get_tavily() -> TavilyClient:
    global _tavily_client
    if _tavily_client is None:
        _tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    return _tavily_client


def _get_firecrawl() -> FirecrawlApp:
    global _firecrawl_app
    if _firecrawl_app is None:
        _firecrawl_app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
    return _firecrawl_app


def _get_db(index_name: str) -> VectorDB:
    if index_name not in _db_instances:
        _db_instances[index_name] = VectorDB(index_name=index_name)
    return _db_instances[index_name]


# ── Web Search Tools ──────────────────────────────────────────

@mcp.tool()
def tavily_quick_search(query: str, max_results: int = 3) -> str:
    """Perform a quick web search across the internet. Returns synthesized answers and snippets.
    Ideal for news, quick fact-checking, and broad questions."""
    logger.info("Tavily search — query='%s', max_results=%d", query, max_results)
    try:
        client = _get_tavily()
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_answer=True,
        )
        results = []
        if response.get("answer"):
            results.append(f"**AI Answer:** {response['answer']}")
        for r in response.get("results", []):
            results.append(f"**{r['title']}** ({r['url']})\n{r['content']}")
        return "\n\n---\n\n".join(results) if results else "No results found."
    except Exception as e:
        logger.error("Tavily search error: %s", e)
        return f"Search failed: {e}"


@mcp.tool()
def firecrawl_deep_scrape(url: str) -> str:
    """Deep scrape a specific URL to extract its full markdown content.
    Use when you need to read a long-form article, report, or earnings transcript."""
    logger.info("Firecrawl scraping — url='%s'", url)
    try:
        app = _get_firecrawl()
        scrape_result = app.scrape(url, formats=["markdown"])
        markdown = scrape_result.get("markdown", "")
        metadata = scrape_result.get("metadata", {})
        title = metadata.get("title", "")
        return f"# {title}\n\n{markdown}" if title else markdown
    except Exception as e:
        logger.error("Firecrawl scrape error: %s", e)
        return f"Scrape failed for {url}: {e}"


# ── Finance Data Tools ────────────────────────────────────────

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


# ── Vector DB Tools ───────────────────────────────────────────

@mcp.tool()
def check_in_vector_db(identifier: str, index_name: str) -> str:
    """Check if documents with a given identifier exist in the vector DB.
    For financial reports, identifier is a ticker (e.g., 'RELIANCE.NS') and index_name is 'financial-reports'.
    For research papers, use a query string and index_name 'research-papers'."""
    logger.info("Checking vector DB — identifier='%s', index='%s'", identifier, index_name)
    try:
        db = _get_db(index_name)
        if index_name == "research-papers":
            exists = db.papers_exist(identifier)
        else:
            exists = db.reports_exist(identifier)
        status = "found" if exists else "not found"
        return f"Documents for '{identifier}' in index '{index_name}': {status}"
    except Exception as e:
        logger.error("Error checking vector DB: %s", e)
        return f"Error checking vector DB: {e}"


@mcp.tool()
def upsert_to_vector_db(data: str, metadata_json: str, index_name: str) -> str:
    """Upsert a document to the vector DB. data is the text content, metadata_json is a JSON string
    of metadata fields, and index_name is the target Pinecone index."""
    logger.info("Upserting to vector DB — index='%s'", index_name)
    try:
        metadata = json.loads(metadata_json)
        db = _get_db(index_name)
        doc_id = metadata.get("doc_id", "doc")
        num_chunks = db.upsert_chunks(doc_id, data, metadata)
        return f"Successfully upserted {num_chunks} chunks to index '{index_name}'"
    except Exception as e:
        logger.error("Error upserting to vector DB: %s", e)
        return f"Upsert failed: {e}"


@mcp.tool()
def retrieve_from_vector_db(query: str, index_name: str, filter_key: str = "",
                            filter_value: str = "", top_k: int = 5) -> str:
    """Retrieve relevant document chunks from the vector DB using semantic search.
    Args:
        query: Search query
        index_name: Pinecone index name (e.g., 'financial-reports' or 'research-papers')
        filter_key: Optional metadata key to filter by (e.g., 'ticker')
        filter_value: Optional metadata value to filter by (e.g., 'RELIANCE.NS')
        top_k: Number of results to return"""
    logger.info("Retrieving from vector DB — query='%s', index='%s'", query[:80], index_name)
    try:
        db = _get_db(index_name)
        results = db.retrieve(query, top_k=top_k, filter_key=filter_key, filter_value=filter_value)
        if not results:
            return "No relevant results found."
        chunks = []
        for r in results:
            score = r.pop("score", 0)
            text = r.pop("text", "")
            title = r.get("title", "Unknown")
            chunks.append(f"**{title}** (score: {score:.3f})\n{text}")
        return "\n\n---\n\n".join(chunks)
    except Exception as e:
        logger.error("Error retrieving from vector DB: %s", e)
        return f"Retrieval failed: {e}"


@mcp.tool()
def add_financial_reports_to_db(ticker: str) -> str:
    """Fetch quarterly and yearly financial reports for a ticker and store them in the vector DB.
    Always check if reports exist first with check_in_vector_db."""
    logger.info("Adding financial reports for ticker='%s'", ticker)
    try:
        db = _get_db("financial-reports")
        if db.reports_exist(ticker):
            return f"Reports for {ticker} already exist in the database."

        t = yf.Ticker(ticker)
        reports = []

        report_sources = [
            ("Yearly Income Statement", "Yearly", "Income Statement", lambda: t.income_stmt),
            ("Quarterly Income Statement", "Quarterly", "Income Statement", lambda: t.quarterly_income_stmt),
            ("Yearly Balance Sheet", "Yearly", "Balance Sheet", lambda: t.balance_sheet),
            ("Quarterly Balance Sheet", "Quarterly", "Balance Sheet", lambda: t.quarterly_balance_sheet),
            ("Yearly Cash Flow", "Yearly", "Cash Flow", lambda: t.cashflow),
            ("Quarterly Cash Flow", "Quarterly", "Cash Flow", lambda: t.quarterly_cashflow),
        ]

        for title_suffix, period, report_type, fetch_fn in report_sources:
            try:
                df = fetch_fn()
                if not df.empty:
                    reports.append({
                        "title": f"{ticker} {title_suffix}",
                        "content": f"{title_suffix} for {ticker}:\n\n{df.to_markdown()}",
                        "period": period,
                        "type": report_type,
                    })
            except Exception as e:
                logger.warning("Failed to fetch %s for %s: %s", title_suffix, ticker, e)

        if not reports:
            return f"No financial reports found for {ticker}."

        db.upsert_reports(ticker, reports)
        return f"Successfully loaded {len(reports)} reports into the vector DB for {ticker}."
    except Exception as e:
        logger.error("Error adding reports: %s", e)
        return f"Failed to add reports: {e}"


@mcp.tool()
def check_papers_in_db(query: str) -> str:
    """Check if relevant research papers exist in the vector DB for a given query."""
    logger.info("Checking papers in DB — query='%s'", query[:80])
    try:
        db = _get_db("research-papers")
        exists = db.papers_exist(query)
        if exists:
            return f"Relevant papers found in the database for query: '{query}'"
        return f"No relevant papers found in the database for query: '{query}'"
    except Exception as e:
        logger.error("Error checking papers: %s", e)
        return f"Error checking papers: {e}"


@mcp.tool()
def retrieve_papers(query: str, top_k: int = 5) -> str:
    """Retrieve relevant research paper chunks from the vector DB using semantic search."""
    logger.info("Retrieving papers — query='%s', top_k=%d", query[:80], top_k)
    try:
        db = _get_db("research-papers")
        results = db.retrieve(query, top_k=top_k)
        if not results:
            return "No relevant papers found."
        chunks = []
        for r in results:
            score = r.pop("score", 0)
            text = r.pop("text", "")
            title = r.get("title", "Unknown")
            authors = r.get("authors", "")
            chunks.append(f"**{title}** by {authors} (score: {score:.3f})\n{text}")
        return "\n\n---\n\n".join(chunks)
    except Exception as e:
        logger.error("Error retrieving papers: %s", e)
        return f"Retrieval failed: {e}"


def _rerank_candidates(query: str, candidates: list[dict], top_n: int) -> list[dict]:
    """Rerank arxiv candidates by embedding similarity to the query."""
    if len(candidates) <= top_n:
        return candidates

    db = _get_db("research-papers")
    candidate_texts = [f"{c['title']}. {c['summary']}" for c in candidates]
    query_vector = db.embeddings.embed_query(query)
    candidate_vectors = db.embeddings.embed_documents(candidate_texts)

    similarities = []
    for i, cvec in enumerate(candidate_vectors):
        dot = sum(q * c for q, c in zip(query_vector, cvec))
        similarities.append((i, dot))

    similarities.sort(key=lambda x: x[1], reverse=True)
    reranked = []
    for idx, score in similarities[:top_n]:
        candidates[idx]["_relevance_score"] = round(score, 4)
        reranked.append(candidates[idx])

    logger.info("Reranked %d candidates → top %d (scores: %.3f – %.3f)",
                len(candidates), len(reranked),
                reranked[0].get("_relevance_score", 0),
                reranked[-1].get("_relevance_score", 0))
    return reranked


@mcp.tool()
def download_and_store_arxiv_papers(query: str, max_results: int = 5,
                                     sort_by: str = "relevance",
                                     categories: str = "") -> str:
    """Search arXiv, download PDFs, convert to markdown, and store in the vector DB.
    Fetches a larger pool of candidates from arXiv, then reranks them by semantic
    similarity to the query and only downloads the most relevant ones.

    Args:
        query: Search query — use precise academic terms for best results.
        max_results: Max papers to download (default 5). Keep small for targeted searches.
        sort_by: Sort order — 'relevance' (default), 'submitted' (newest first), or 'updated'.
        categories: Optional comma-separated arXiv categories to filter by
                    (e.g., 'cs.AI,cs.CL,cs.LG'). Leave empty for all categories.
    """
    logger.info("Downloading arXiv papers — query='%s', max_results=%d, sort='%s', categories='%s'",
                query, max_results, sort_by, categories)
    try:
        sort_map = {
            "relevance": arxiv.SortCriterion.Relevance,
            "submitted": arxiv.SortCriterion.SubmittedDate,
            "updated": arxiv.SortCriterion.LastUpdatedDate,
        }
        sort_criterion = sort_map.get(sort_by, arxiv.SortCriterion.Relevance)

        full_query = query
        if categories:
            cat_filter = " OR ".join(f"cat:{c.strip()}" for c in categories.split(",") if c.strip())
            full_query = f"({query}) AND ({cat_filter})"

        fetch_count = max_results * 3
        search = arxiv.Search(
            query=full_query,
            max_results=fetch_count,
            sort_by=sort_criterion,
            sort_order=arxiv.SortOrder.Descending,
        )

        candidates = []
        for paper in _arxiv_client.results(search):
            candidates.append({
                "title": paper.title,
                "authors": [a.name for a in paper.authors],
                "summary": paper.summary,
                "pdf_url": paper.pdf_url,
                "short_id": paper.get_short_id(),
                "categories": list(paper.categories),
                "published": paper.published.strftime("%Y-%m-%d") if paper.published else "",
            })

        if not candidates:
            return "No papers found on arXiv for this query."

        logger.info("Fetched %d candidates from arXiv, reranking to select top %d",
                     len(candidates), max_results)

        top_candidates = _rerank_candidates(query, candidates, max_results)

        download_dir = Path("papers")
        download_dir.mkdir(exist_ok=True)

        papers = []
        for candidate in top_candidates:
            file_name = f"{candidate['short_id']}.pdf"
            file_path = download_dir / file_name

            logger.info("Downloading paper: '%s' (relevance: %.3f) → %s",
                        candidate["title"],
                        candidate.get("_relevance_score", 0),
                        file_path)

            paper_search = arxiv.Search(id_list=[candidate["short_id"]])
            paper_obj = next(_arxiv_client.results(paper_search), None)
            if paper_obj is None:
                logger.warning("Could not re-fetch paper '%s', skipping", candidate["short_id"])
                continue
            paper_obj.download_pdf(dirpath=str(download_dir), filename=file_name)

            papers.append({
                "title": candidate["title"],
                "authors": candidate["authors"],
                "summary": candidate["summary"],
                "pdf_path": str(file_path),
                "pdf_url": candidate["pdf_url"],
                "categories": candidate["categories"],
                "published": candidate["published"],
            })

        if not papers:
            return "No papers could be downloaded."

        db = _get_db("research-papers")
        db.upsert_papers(papers)

        summaries = []
        for i, p in enumerate(papers):
            authors = ", ".join(p["authors"][:3])
            if len(p["authors"]) > 3:
                authors += " et al."
            date_str = f" ({p['published']})" if p.get("published") else ""
            score_str = f" [relevance: {top_candidates[i].get('_relevance_score', 'N/A')}]"
            summaries.append(f"- **{p['title']}** by {authors}{date_str}{score_str}")

        return f"Downloaded and stored {len(papers)} papers (reranked from {len(candidates)} candidates):\n" + "\n".join(summaries)
    except Exception as e:
        logger.error("Error downloading arXiv papers: %s", e)
        return f"Failed to download papers: {e}"


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8010))
    mcp.run(transport="streamable-http", host="0.0.0.0", port=port)
