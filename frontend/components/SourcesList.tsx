import type { AnswerSource } from "@/types/api";

interface SourcesListProps {
  sources: AnswerSource[];
}

export function SourcesList({ sources }: SourcesListProps) {
  if (sources.length === 0) {
    return <p className="no-sources">No source passages were returned.</p>;
  }

  return (
    <ul className="sources-list">
      {sources.map((source) => (
        <li key={`${source.document_id}-${source.chunk_index}`}>
          <span className="source-file-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24" role="img">
              <path d="M7 3h7l4 4v14H7z" />
              <path d="M14 3v5h5" />
            </svg>
          </span>
          <span className="source-details">
            <strong>{source.filename}</strong>
            <small>Chunk {source.chunk_index}</small>
          </span>
          <span className="score-pill" title="Elasticsearch retrieval score">
            {source.score.toFixed(3)}
          </span>
        </li>
      ))}
    </ul>
  );
}
