"""Create the ContextIQ Elasticsearch index when it does not exist."""

from app.core.config import get_settings
from app.core.elasticsearch import create_elasticsearch_client
from app.rag.vector_store import (
    IncompatibleIndexMappingError,
    ensure_document_chunk_index,
)
from app.services.health_service import is_elasticsearch_ready


def main() -> None:
    """Verify Elasticsearch and initialize the non-destructive chunk index."""
    settings = get_settings()
    client = create_elasticsearch_client(settings.elasticsearch_url)

    try:
        if not is_elasticsearch_ready(client):
            raise SystemExit(
                f"Elasticsearch is unavailable at {settings.elasticsearch_url}."
            )

        created = ensure_document_chunk_index(
            client,
            index_name=settings.elasticsearch_index,
            embedding_dimensions=settings.embedding_dimensions,
        )
    except IncompatibleIndexMappingError as exc:
        raise SystemExit(str(exc)) from None
    finally:
        client.close()

    action = "Created" if created else "Already exists"
    print(f"{action}: {settings.elasticsearch_index}")


if __name__ == "__main__":
    main()
