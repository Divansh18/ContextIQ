"use client";

import { useState } from "react";

import { AnswerCard } from "@/components/AnswerCard";
import { DocumentUpload } from "@/components/DocumentUpload";
import { QuestionForm } from "@/components/QuestionForm";
import { ApiError, askQuestion, uploadDocument } from "@/lib/api";
import type {
  AskQuestionResponse,
  DocumentUploadResponse,
} from "@/types/api";

function readableError(error: unknown, fallback: string): string {
  if (error instanceof ApiError || error instanceof Error) {
    return error.message;
  }
  return fallback;
}

export function KnowledgeWorkspace() {
  const [document, setDocument] = useState<DocumentUploadResponse | null>(null);
  const [answer, setAnswer] = useState<AskQuestionResponse | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isAsking, setIsAsking] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [questionError, setQuestionError] = useState<string | null>(null);

  async function handleUpload(file: File) {
    setIsUploading(true);
    setUploadError(null);
    setQuestionError(null);
    setAnswer(null);

    try {
      const uploadedDocument = await uploadDocument(file);
      setDocument(uploadedDocument);
    } catch (error) {
      setUploadError(
        readableError(error, "The PDF could not be uploaded. Please try again."),
      );
    } finally {
      setIsUploading(false);
    }
  }

  async function handleQuestion(question: string) {
    if (!document) {
      setQuestionError("Upload and index a PDF before asking a question.");
      return;
    }

    setIsAsking(true);
    setQuestionError(null);
    setAnswer(null);

    try {
      const response = await askQuestion({
        question,
        top_k: 3,
        document_id: document.document_id,
      });
      setAnswer(response);
    } catch (error) {
      setQuestionError(
        readableError(error, "ContextIQ could not answer that question."),
      );
    } finally {
      setIsAsking(false);
    }
  }

  return (
    <main className="page-shell">
      <header className="site-header">
        <a className="brand" href="#top" aria-label="ContextIQ home">
          <span className="brand-mark" aria-hidden="true">
            C
          </span>
          <span>ContextIQ</span>
        </a>
        <span className="header-label">Grounded document intelligence</span>
      </header>

      <section className="hero" id="top">
        <div className="eyebrow">
          <span className="eyebrow-dot" aria-hidden="true" />
          Retrieval-augmented knowledge assistant
        </div>
        <h1>Ask better questions.<br />Get answers grounded in your documents.</h1>
        <p>
          Upload a PDF, let ContextIQ index its meaning, and ask focused questions
          with source-backed answers you can verify.
        </p>

        <ol className="workflow-steps" aria-label="ContextIQ workflow">
          <li><span>01</span> Upload PDF</li>
          <li><span>02</span> Ask a question</li>
          <li><span>03</span> Verify sources</li>
        </ol>
      </section>

      <section className="workspace-grid" aria-label="ContextIQ workspace">
        <DocumentUpload
          document={document}
          error={uploadError}
          isUploading={isUploading}
          onUpload={handleUpload}
        />
        <QuestionForm
          key={document?.document_id ?? "no-document"}
          documentName={document?.filename ?? null}
          error={questionError}
          isAsking={isAsking}
          onAsk={handleQuestion}
        />
      </section>

      <AnswerCard answer={answer} isLoading={isAsking} />

      <footer className="site-footer">
        <span>ContextIQ</span>
        <span>Answers grounded in the documents you provide.</span>
      </footer>
    </main>
  );
}
