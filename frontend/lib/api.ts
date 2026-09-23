import type {
  AskQuestionRequest,
  AskQuestionResponse,
  DocumentUploadResponse,
} from "@/types/api";

const API_BASE_URL = (
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000"
).replace(/\/$/, "");

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status?: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

function extractErrorMessage(payload: unknown): string | null {
  if (typeof payload !== "object" || payload === null || !("detail" in payload)) {
    return null;
  }

  const detail = payload.detail;
  if (typeof detail === "string") {
    return detail;
  }

  if (Array.isArray(detail)) {
    const messages = detail.flatMap((issue) => {
      if (
        typeof issue === "object" &&
        issue !== null &&
        "msg" in issue &&
        typeof issue.msg === "string"
      ) {
        return [issue.msg];
      }
      return [];
    });
    return messages.length > 0 ? messages.join(" ") : null;
  }

  return null;
}

async function request<T>(path: string, init: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, init);
  } catch {
    throw new ApiError(
      "ContextIQ could not reach the backend. Confirm FastAPI is running and try again.",
    );
  }

  let payload: unknown = null;
  try {
    payload = await response.json();
  } catch {
    if (response.ok) {
      throw new ApiError("The backend returned an unreadable response.", response.status);
    }
  }

  if (!response.ok) {
    throw new ApiError(
      extractErrorMessage(payload) ?? "The request could not be completed.",
      response.status,
    );
  }

  return payload as T;
}

export async function uploadDocument(
  file: File,
): Promise<DocumentUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  return request<DocumentUploadResponse>("/documents/upload", {
    method: "POST",
    body: formData,
  });
}

export async function askQuestion(
  requestBody: AskQuestionRequest,
): Promise<AskQuestionResponse> {
  return request<AskQuestionResponse>("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(requestBody),
  });
}
