"""OpenAI embedding generation with batching and response validation."""

from collections.abc import Sequence

from openai import OpenAI, OpenAIError


class EmbeddingError(RuntimeError):
    """Base error for embedding generation failures."""


class MissingOpenAIAPIKeyError(EmbeddingError):
    """Raised when embedding generation is requested without an API key."""


class EmbeddingAPIError(EmbeddingError):
    """Raised when the OpenAI embeddings request fails."""


class UnexpectedEmbeddingResponseError(EmbeddingError):
    """Raised when OpenAI returns incomplete or malformed embedding data."""


class UnexpectedEmbeddingDimensionError(EmbeddingError):
    """Raised when a returned vector has an unexpected dimension."""


def validate_embedding_configuration(
    *,
    api_key: str | None,
    model: str,
    expected_dimensions: int,
) -> None:
    """Validate required embedding configuration without making an API call."""
    if not api_key:
        raise MissingOpenAIAPIKeyError(
            "OPENAI_API_KEY is required to generate embeddings."
        )
    if not model.strip():
        raise ValueError("embedding model must not be empty")
    if expected_dimensions <= 0:
        raise ValueError("expected embedding dimensions must be greater than zero")


def embed_text(
    text: str,
    *,
    api_key: str | None,
    model: str,
    expected_dimensions: int,
) -> list[float]:
    """Generate and validate one text embedding."""
    if not text.strip():
        raise ValueError("text to embed must not be empty")

    return embed_texts(
        [text],
        api_key=api_key,
        model=model,
        expected_dimensions=expected_dimensions,
    )[0]


def embed_texts(
    texts: Sequence[str],
    *,
    api_key: str | None,
    model: str,
    expected_dimensions: int,
) -> list[list[float]]:
    """Generate embeddings for multiple texts in one OpenAI API request."""
    input_texts = list(texts)
    if not input_texts:
        return []
    if any(not text.strip() for text in input_texts):
        raise ValueError("texts to embed must not contain empty values")

    validate_embedding_configuration(
        api_key=api_key,
        model=model,
        expected_dimensions=expected_dimensions,
    )

    client: OpenAI | None = None
    try:
        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(
            model=model,
            input=input_texts,
            dimensions=expected_dimensions,
            encoding_format="float",
        )
    except OpenAIError as exc:
        raise EmbeddingAPIError("OpenAI embedding generation failed.") from exc
    finally:
        if client is not None:
            client.close()

    response_data = sorted(response.data, key=lambda item: item.index)
    expected_indices = list(range(len(input_texts)))
    if [item.index for item in response_data] != expected_indices:
        raise UnexpectedEmbeddingResponseError(
            "OpenAI returned an unexpected number or ordering of embeddings."
        )

    embeddings: list[list[float]] = []
    for item in response_data:
        embedding = list(item.embedding)
        if len(embedding) != expected_dimensions:
            raise UnexpectedEmbeddingDimensionError(
                "OpenAI returned an embedding with "
                f"{len(embedding)} dimensions; expected {expected_dimensions}."
            )
        embeddings.append(embedding)

    return embeddings
