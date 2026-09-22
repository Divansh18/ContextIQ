"""Unit tests for framework-independent text chunking."""

import pytest

from app.rag.chunking import chunk_text


def test_short_text_produces_one_chunk() -> None:
    text = "ContextIQ keeps chunking understandable."

    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)

    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].text == text
    assert chunks[0].character_count == len(text)


def test_default_configuration_uses_1000_characters_with_200_overlap() -> None:
    text = "x" * 1_200

    chunks = chunk_text(text)

    assert [chunk.character_count for chunk in chunks] == [1_000, 400]
    assert chunks[0].text[-200:] == chunks[1].text[:200]


def test_long_text_produces_multiple_chunks() -> None:
    text = "0123456789" * 25

    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)

    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]
    assert [chunk.text for chunk in chunks] == [
        text[0:100],
        text[80:180],
        text[160:250],
    ]
    assert [chunk.character_count for chunk in chunks] == [100, 100, 90]


def test_chunks_repeat_the_configured_overlap() -> None:
    chunks = chunk_text("abcdefghij", chunk_size=5, chunk_overlap=2)

    assert [chunk.text for chunk in chunks] == ["abcde", "defgh", "ghij"]
    assert chunks[0].text[-2:] == chunks[1].text[:2]
    assert chunks[1].text[-2:] == chunks[2].text[:2]


@pytest.mark.parametrize("text", ["", "   \n\t"])
def test_empty_or_whitespace_text_produces_no_chunks(text: str) -> None:
    assert chunk_text(text) == []


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [
        (0, 0),
        (-1, 0),
        (10, -1),
        (10, 10),
        (10, 11),
    ],
)
def test_invalid_chunk_configuration_raises_value_error(
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    with pytest.raises(ValueError):
        chunk_text(
            "some text",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
