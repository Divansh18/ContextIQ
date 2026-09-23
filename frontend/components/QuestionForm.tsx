"use client";

import { useState, type FormEvent } from "react";

interface QuestionFormProps {
  documentName: string | null;
  error: string | null;
  isAsking: boolean;
  onAsk: (question: string) => Promise<void>;
}

export function QuestionForm({
  documentName,
  error,
  isAsking,
  onAsk,
}: QuestionFormProps) {
  const [question, setQuestion] = useState("");
  const normalizedQuestion = question.trim();
  const canSubmit = Boolean(documentName && normalizedQuestion && !isAsking);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!canSubmit) {
      return;
    }
    await onAsk(normalizedQuestion);
  }

  return (
    <article className="panel question-panel">
      <div className="panel-heading">
        <span className="step-number">2</span>
        <div>
          <p className="section-kicker">Question</p>
          <h2>Ask your document</h2>
        </div>
      </div>
      <p className="section-description">
        ContextIQ retrieves the most relevant passages before composing an answer.
      </p>

      {documentName ? (
        <div className="active-document">
          <span className="active-dot" aria-hidden="true" />
          Asking <strong>{documentName}</strong>
        </div>
      ) : (
        <div className="active-document waiting">
          <span className="lock-icon" aria-hidden="true">○</span>
          Upload a document to enable questions
        </div>
      )}

      <form onSubmit={handleSubmit}>
        <label className="question-label" htmlFor="question">
          What would you like to know?
        </label>
        <textarea
          id="question"
          rows={5}
          value={question}
          disabled={!documentName || isAsking}
          placeholder="e.g. Which service lets me run containers without managing virtual machines?"
          onChange={(event) => setQuestion(event.target.value)}
        />

        {error && (
          <div className="message error-message" role="alert">
            <span aria-hidden="true">!</span>
            {error}
          </div>
        )}

        <button
          className="button dark-button full-width"
          type="submit"
          disabled={!canSubmit}
        >
          {isAsking ? (
            <><span className="spinner light" aria-hidden="true" /> Searching and generating…</>
          ) : (
            <>Ask ContextIQ <span aria-hidden="true">→</span></>
          )}
        </button>
      </form>
    </article>
  );
}
