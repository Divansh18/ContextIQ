"""External dependency readiness checks."""

from elasticsearch import Elasticsearch


def is_elasticsearch_ready(client: Elasticsearch) -> bool:
    """Return whether the configured Elasticsearch cluster is reachable."""
    return bool(client.ping())
