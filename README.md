# mcp-tool-servers

Three FastMCP HTTP servers exposing shared tools as MCP endpoints. Each runs on its own port with streamable HTTP transport.

## Servers

### web-search (port 8010)

| Tool | Description |
|------|-------------|
| `tavily_quick_search(query, max_results)` | Quick web search via Tavily API. Returns synthesized answers and snippets. |
| `firecrawl_deep_scrape(url)` | Deep scrape a specific URL for full markdown content via Firecrawl. |

### finance-data (port 8011)

| Tool | Description |
|------|-------------|
| `get_ticker_data(ticker)` | Market data, P/E ratios, company summary via yfinance. Supports .NS/.BO for Indian stocks. |
| `get_bse_nse_reports(ticker)` | Raw quarterly/yearly income statements, balance sheets, cash flow in markdown. |

### vector-db (port 8012)

| Tool | Description |
|------|-------------|
| `check_in_vector_db(identifier, index_name)` | Check if documents exist in a Pinecone index. |
| `upsert_to_vector_db(data, metadata_json, index_name)` | Upsert a document to the vector DB. |
| `retrieve_from_vector_db(query, index_name, filter_key, filter_value, top_k)` | Semantic search over a Pinecone index. |
| `add_financial_reports_to_db(ticker)` | Fetch financial reports and store in the `financial-reports` index. |
| `check_papers_in_db(query)` | Check if relevant papers exist in the `research-papers` index. |
| `retrieve_papers(query, top_k)` | Semantic search over stored research papers. |
| `download_and_store_arxiv_papers(query, max_results)` | Search arXiv, download PDFs, convert to markdown, upsert to vector DB. |

## Structure

```
mcp-tool-servers/
├── pyproject.toml
├── servers/
│   ├── web_search.py           # port 8010
│   ├── finance_data.py         # port 8011
│   └── vector_db_server.py     # port 8012
├── shared/
│   ├── config.py               # Port assignments
│   └── vector_db.py            # Consolidated VectorDB class
└── scripts/
    └── run_all.py              # Launch all servers for dev
```

## Running

All servers:
```bash
infisical run -- python scripts/run_all.py
```

Individual server:
```bash
infisical run -- python servers/web_search.py
```

## Environment Variables

| Variable | Used by | Description |
|----------|---------|-------------|
| `TAVILY_API_KEY` | web-search | Tavily API key |
| `FIRECRAWL_API_KEY` | web-search | Firecrawl API key |
| `PINECONE_API_KEY` | vector-db | Pinecone API key |
| `NVIDIA_API_KEY` | vector-db | NVIDIA embeddings API key |
| `GEMINI_API_KEY` | vector-db (if using gemini provider) | Google AI API key |

## Dependencies

`fastmcp>=3.0`, `tavily-python`, `firecrawl-py`, `yfinance`, `arxiv`, `pymupdf4llm`, `pinecone`, `langchain-google-genai`, `langchain-nvidia-ai-endpoints`, `langchain-text-splitters`, `python-dotenv`
