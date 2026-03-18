PORTS = {
    "web-search": 8010,
    "finance-data": 8011,
    "vector-db": 8012,
}

def server_url(name: str) -> str:
    return f"http://localhost:{PORTS[name]}"
