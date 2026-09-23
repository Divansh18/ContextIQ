import { SourcesList } from "@/components/SourcesList";
import type { AskQuestionResponse } from "@/types/api";

interface AnswerCardProps {
  answer: AskQuestionResponse | null;
  isLoading: boolean;
}

export function AnswerCard({ answer, isLoading }: AnswerCardProps) {
  return (
    <section className={`answer-card${answer ? " has-answer" : ""}`} aria-live="polite">
      <div className="answer-heading">
        <div>
          <p className="section-kicker">Grounded response</p>
          <h2>Answer</h2>
        </div>
        {answer && <span className="grounded-badge"><span>✓</span> Source grounded</span>}
      </div>

      {isLoading ? (
        <div className="answer-loading">
          <span className="thinking-mark" aria-hidden="true">
            <i /><i /><i />
          </span>
          <div>
            <strong>Searching your document</strong>
            <p>Retrieving evidence and generating a grounded answer…</p>
          </div>
        </div>
      ) : answer ? (
        <>
          <p className="answer-question">“{answer.question}”</p>
          <p className="answer-text">{answer.answer}</p>
          <div className="sources-section">
            <div className="sources-heading">
              <h3>Sources</h3>
              <span>{answer.sources.length} retrieved</span>
            </div>
            <SourcesList sources={answer.sources} />
          </div>
        </>
      ) : (
        <div className="answer-empty">
          <span className="answer-empty-icon" aria-hidden="true">✦</span>
          <strong>Your grounded answer will appear here</strong>
          <p>Upload a PDF and ask a question to see the supporting sources.</p>
        </div>
      )}
    </section>
  );
}
