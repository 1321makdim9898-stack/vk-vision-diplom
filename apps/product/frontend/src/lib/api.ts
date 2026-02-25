import type { HistoryRecord } from "../types/api";

const API_BASE =
  (import.meta as any).env?.VITE_API_BASE?.toString() || "/api";

function toUrl(path: string) {
  // supports absolute and relative
  if (/^https?:\/\//i.test(path)) return path;
  return `${API_BASE.replace(/\/$/, "")}/${path.replace(/^\//, "")}`;
}

async function parseJsonOrText(r: Response): Promise<any> {
  const text = await r.text();
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

export async function fetchHistory(limit = 20): Promise<HistoryRecord[]> {
  const r = await fetch(toUrl(`/history?limit=${encodeURIComponent(limit)}`));

  const data = await parseJsonOrText(r);
  if (!r.ok) throw new Error(typeof data === "string" ? data : `History failed: ${r.status}`);
  if (!data?.ok) throw new Error(data?.error || "History failed");

  return (data.items || []) as HistoryRecord[];
}

export async function analyzeImage(file: File): Promise<HistoryRecord> {
  const fd = new FormData();

  /**
   * ВАЖНО:
   * Твой backend (Express + multer) по логу падает "Unexpected field" —
   * это почти всегда означает, что multer настроен на .single('file') или поля через .fields([{name:'file'}])
   * Поэтому сюда отправляем "file".
   *
   * А уже backend может прокинуть дальше в ML сервис как "image" или "file" (у ML оба поддерживаются).
   */
  fd.append("file", file);

  const r = await fetch(toUrl("/analyze"), {
    method: "POST",
    body: fd,
  });

  const data = await parseJsonOrText(r);

  if (!r.ok) {
    throw new Error(typeof data === "string" ? data : data?.error || `Analyze failed: ${r.status}`);
  }
  if (!data?.ok) {
    throw new Error(data?.error || "Analyze failed");
  }

  return data.record as HistoryRecord;
}
