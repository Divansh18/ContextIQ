"use client";

import { useRef, useState, type ChangeEvent, type FormEvent } from "react";

import type { DocumentUploadResponse } from "@/types/api";

interface DocumentUploadProps {
  document: DocumentUploadResponse | null;
  error: string | null;
  isUploading: boolean;
  onUpload: (file: File) => Promise<void>;
}

function isPdf(file: File): boolean {
  return file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf");
}

export function DocumentUpload({
  document,
  error,
  isUploading,
  onUpload,
}: DocumentUploadProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectionError, setSelectionError] = useState<string | null>(null);

  function handleSelection(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0] ?? null;
    setSelectionError(null);

    if (file && !isPdf(file)) {
      setSelectedFile(null);
      setSelectionError("Choose a PDF file to continue.");
      event.target.value = "";
      return;
    }

    setSelectedFile(file);
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selectedFile) {
      setSelectionError("Choose a PDF file before uploading.");
      inputRef.current?.focus();
      return;
    }

    await onUpload(selectedFile);
  }

  return (
    <article className="panel upload-panel">
      <div className="panel-heading">
        <span className="step-number">1</span>
        <div>
          <p className="section-kicker">Document</p>
          <h2>Upload your PDF</h2>
        </div>
      </div>
      <p className="section-description">
        Your document is extracted, chunked, embedded, and indexed for retrieval.
      </p>

      <form onSubmit={handleSubmit}>
        <label className={`file-picker${selectedFile ? " has-file" : ""}`}>
          <input
            ref={inputRef}
            type="file"
            accept="application/pdf,.pdf"
            disabled={isUploading}
            onChange={handleSelection}
          />
          <span className="file-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24" role="img">
              <path d="M7 3h7l4 4v14H7z" />
              <path d="M14 3v5h5M9.5 13h5M9.5 16h5" />
            </svg>
          </span>
          <span className="file-copy">
            <strong>{selectedFile ? selectedFile.name : "Choose a PDF document"}</strong>
            <small>
              {selectedFile
                ? `${(selectedFile.size / 1024).toFixed(1)} KB · Ready to upload`
                : "PDF files only · Click to browse"}
            </small>
          </span>
          <span className="browse-label">Browse</span>
        </label>

        {(selectionError || error) && (
          <div className="message error-message" role="alert">
            <span aria-hidden="true">!</span>
            {selectionError ?? error}
          </div>
        )}

        <button
          className="button primary-button full-width"
          type="submit"
          disabled={isUploading}
        >
          {isUploading ? (
            <><span className="spinner" aria-hidden="true" /> Uploading and indexing…</>
          ) : (
            <><span aria-hidden="true">↑</span> Upload &amp; index</>
          )}
        </button>
      </form>

      {document && !isUploading && (
        <div className="indexed-document" aria-live="polite">
          <span className="success-icon" aria-hidden="true">✓</span>
          <div>
            <strong>{document.filename}</strong>
            <p>
              {document.chunk_count} {document.chunk_count === 1 ? "chunk" : "chunks"}
              {" · "}{document.page_count} {document.page_count === 1 ? "page" : "pages"}
            </p>
          </div>
          <span className="status-pill">Indexed</span>
        </div>
      )}
    </article>
  );
}
