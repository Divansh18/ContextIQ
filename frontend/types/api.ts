export interface DocumentChunk {
  chunk_index: number;
  text: string;
  character_count: number;
}

export interface DocumentUploadResponse {
  document_id: string;
  filename: string;
  page_count: number;
  character_count: number;
  preview: string;
  chunk_count: number;
  chunks: DocumentChunk[];
}

export interface AskQuestionRequest {
  question: string;
  top_k?: number;
  document_id?: string;
}

export interface AnswerSource {
  document_id: string;
  filename: string;
  chunk_index: number;
  score: number;
}

export interface AskQuestionResponse {
  question: string;
  answer: string;
  sources: AnswerSource[];
}
