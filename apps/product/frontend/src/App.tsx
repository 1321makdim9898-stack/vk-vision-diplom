import React, { useEffect, useMemo, useRef, useState } from 'react';
import './styles.css';

import { analyzeImage, fetchHistory, type HistoryRecord } from './api';
import UploadCard from './components/UploadCard';
import ResultCards from './components/ResultCards';
import HistoryPanel from './components/HistoryPanel';

function clamp(n: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, n));
}

/**
 * bbox может прийти:
 * 1) как [x, y, w, h]  (твой кейс из ML)
 * 2) или как [x1, y1, x2, y2] (на будущее/другие детекторы)
 * 3) или объектом
 */
function normalizeBBox(
  bbox: any,
  img?: { w: number; h: number }
): { x: number; y: number; w: number; h: number } | null {
  // массив
  if (Array.isArray(bbox) && bbox.length >= 4) {
    const [a, b, c, d] = bbox;
    if (![a, b, c, d].every((v) => typeof v === 'number' && Number.isFinite(v))) return null;

    // Попытка понять: это [x,y,w,h] или [x1,y1,x2,y2]
    if (img && img.w > 0 && img.h > 0) {
      // если это [x,y,w,h], то (x+w) и (y+h) должны укладываться в картинку
      const looksLikeXYWH = a + c <= img.w + 1 && b + d <= img.h + 1 && c > 0 && d > 0;
      if (looksLikeXYWH) return { x: a, y: b, w: c, h: d };

      // иначе пробуем как [x1,y1,x2,y2]
      const w = c - a;
      const h = d - b;
      if (w > 0 && h > 0) return { x: a, y: b, w, h };

      return null;
    }

    // Если размеры картинки ещё неизвестны — по умолчанию считаем XYWH (т.к. твой ML так отдаёт)
    if (c > 0 && d > 0) return { x: a, y: b, w: c, h: d };

    // fallback как x2,y2
    const w = c - a;
    const h = d - b;
    if (w > 0 && h > 0) return { x: a, y: b, w, h };

    return null;
  }

  // объект
  if (bbox && typeof bbox === 'object') {
    // варианты: x,y,w,h
    const x = bbox.x ?? bbox.left ?? bbox.x1;
    const y = bbox.y ?? bbox.top ?? bbox.y1;
    const w = bbox.w ?? (typeof bbox.x2 === 'number' && typeof x === 'number' ? bbox.x2 - x : undefined);
    const h = bbox.h ?? (typeof bbox.y2 === 'number' && typeof y === 'number' ? bbox.y2 - y : undefined);

    if ([x, y, w, h].every((v) => typeof v === 'number' && Number.isFinite(v)) && w > 0 && h > 0) {
      return { x, y, w, h };
    }
  }

  return null;
}

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');

  const [history, setHistory] = useState<HistoryRecord[]>([]);
  const [activeRecord, setActiveRecord] = useState<HistoryRecord | null>(null);

  // ✅ отличаем "выбрано из истории" от "выбрано с ПК"
  const [selectedFromHistory, setSelectedFromHistory] = useState(false);

  // ✅ активное лицо для табов + подсветки рамки
  const [activeFaceIdx, setActiveFaceIdx] = useState(0);

  // --- Import by URL (MVP: no auth) ---
  const [urlInput, setUrlInput] = useState<string>('');
  const [urlLoading, setUrlLoading] = useState(false);
  const [urlError, setUrlError] = useState<string>('');

  // refs для bbox overlay
  const selectedImgRef = useRef<HTMLImageElement | null>(null);
  const [selectedImgSize, setSelectedImgSize] = useState<{
    cw: number;
    ch: number;
    nw: number;
    nh: number;
  } | null>(null);

  useEffect(() => {
    if (!file) return;
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  async function loadHistory() {
    try {
      setError('');
      const items = await fetchHistory(20);
      setHistory(items);
    } catch (e: any) {
      setError(e?.message || 'Не удалось загрузить историю');
    }
  }

  // UI-only clear: ничего не трогаем на диске
  function clearHistoryUi() {
    setHistory([]);
    setActiveRecord(null);
    setSelectedFromHistory(false);
    setActiveFaceIdx(0);
  }

  useEffect(() => {
    loadHistory();
  }, []);

  async function onAnalyze() {
    if (!file) return;
    try {
      setError('');
      setLoading(true);

      setSelectedFromHistory(false);
      setActiveFaceIdx(0);

      const rec = await analyzeImage(file);

      setActiveRecord(rec);
      setHistory((prev) => [rec, ...prev.filter((x) => x.id !== rec.id)]);
    } catch (e: any) {
      setError(e?.message || 'Ошибка анализа');
    } finally {
      setLoading(false);
    }
  }

  function onClear() {
    setError('');
    setFile(null);
    setPreviewUrl('');
    setActiveRecord(null);
    setSelectedFromHistory(false);
    setActiveFaceIdx(0);
  }

  // url крупной карточки показываем только когда кликнули историю (или пришло из URL)
  const selectedFromHistoryUrl = selectedFromHistory ? activeRecord?.image?.url || '' : '';

  async function analyzeByUrl() {
    const url = urlInput.trim();
    if (!url) return;

    try {
      setUrlError('');
      setUrlLoading(true);

      const r = await fetch('/api/analyze_url', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url }),
      });

      const j = await r.json();
      if (!r.ok || !j?.ok) throw new Error(j?.error || 'Ошибка загрузки по ссылке');

      const rec: HistoryRecord = j.record;
      setActiveRecord(rec);
      setSelectedFromHistory(true);
      setActiveFaceIdx(0);

      setHistory((prev) => [rec, ...prev.filter((x) => x.id !== rec.id)]);

      // чтобы не путалось с локальным выбором
      setFile(null);
      setPreviewUrl('');
    } catch (e: any) {
      setUrlError(e?.message || 'Ошибка анализа по ссылке');
    } finally {
      setUrlLoading(false);
    }
  }

  // --- данные для bbox overlay ---
  const faces = useMemo(() => {
    const r: any = activeRecord?.result;
    const arr = r?.faces;
    return Array.isArray(arr) ? arr : [];
  }, [activeRecord]);

  useEffect(() => {
    setActiveFaceIdx(0);
  }, [activeRecord?.id]);

  function updateSelectedImgSize() {
    const img = selectedImgRef.current;
    if (!img) return;

    const cw = img.clientWidth;
    const ch = img.clientHeight;
    const nw = img.naturalWidth || 0;
    const nh = img.naturalHeight || 0;

    if (cw && ch && nw && nh) {
      setSelectedImgSize({ cw, ch, nw, nh });
    }
  }

  useEffect(() => {
    updateSelectedImgSize();
    const onResize = () => updateSelectedImgSize();
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedFromHistoryUrl]);

  const overlays = useMemo(() => {
    if (!selectedImgSize) return [];
    const { cw, ch, nw, nh } = selectedImgSize;

    const sx = cw / nw;
    const sy = ch / nh;

    return faces
      .map((f: any, idx: number) => {
        const bb = normalizeBBox(f?.bbox, { w: nw, h: nh });
        if (!bb) return null;

        const left = clamp(bb.x * sx, 0, cw);
        const top = clamp(bb.y * sy, 0, ch);
        const width = clamp(bb.w * sx, 2, cw - left);
        const height = clamp(bb.h * sy, 2, ch - top);

        return { idx, left, top, width, height };
      })
      .filter(Boolean) as Array<{ idx: number; left: number; top: number; width: number; height: number }>;
  }, [faces, selectedImgSize]);

  return (
    <div className="page">
      <header className="topbar">
        <div>
          <div className="title">VK Vision — Visual Content Analyzer</div>
          <div className="subtitle">React+TS • Express • ML service • История и карточки</div>
        </div>

        <div className="links">
          <a className="pill" href="/api/health" target="_blank" rel="noreferrer">
            API: /api
          </a>
          <a className="pill" href="/uploads" target="_blank" rel="noreferrer">
            Uploads: /uploads
          </a>
        </div>
      </header>

      {error ? <div className="error">{error}</div> : null}

      <main className="grid">
        <section className="col">
          <UploadCard
            file={file}
            previewUrl={previewUrl}
            loading={loading}
            onPickFile={(f) => {
              setFile(f);
              setSelectedFromHistory(false);
              setActiveRecord(null);
              setActiveFaceIdx(0);
            }}
            onAnalyze={onAnalyze}
            onClear={onClear}
          />

          {/* ✅ Импорт изображения по ссылке (без авторизации) */}
          <div className="card">
            <div className="cardTitle">Импорт по ссылке (VK/URL)</div>
            <div className="cardBody">
              <div className="muted" style={{ marginBottom: 10 }}>
                Вставь прямую ссылку на изображение (например, из VK CDN <code>userapi.com</code>) или с любого сайта.
              </div>

              {urlError ? <div className="error">{urlError}</div> : null}

              <div className="row" style={{ gap: 10, flexWrap: 'wrap' }}>
                <input
                  type="url"
                  value={urlInput}
                  onChange={(e) => setUrlInput(e.target.value)}
                  placeholder="https://.../image.jpg"
                  style={{
                    flex: '1 1 360px',
                    minWidth: 260,
                    padding: '8px 10px',
                    borderRadius: 10,
                    border: '1px solid rgba(255,255,255,0.10)',
                    background: 'rgba(0,0,0,0.18)',
                    color: 'rgba(255,255,255,0.92)',
                    outline: 'none',
                  }}
                />

                <button
                  className="btn"
                  onClick={analyzeByUrl}
                  disabled={!urlInput.trim() || urlLoading || loading}
                  title="Скачать изображение по URL и выполнить анализ"
                >
                  {urlLoading ? 'Загрузка…' : 'Анализировать по ссылке'}
                </button>
              </div>

              <div className="muted" style={{ marginTop: 10 }}>
                ⚠️ Если ссылка ведёт на страницу <code>vk.com/photo...</code>, её нужно заменить на прямую ссылку на файл
                (которая заканчивается на <code>.jpg/.png/.webp</code>).
              </div>
            </div>
          </div>

          {/* ✅ Крупное изображение + bbox overlay */}
          {selectedFromHistoryUrl ? (
            <div className="card" style={{ marginTop: 12 }}>
              <div className="cardTitle">Выбранное изображение</div>
              <div className="cardBody">
                <div className="selectedImageWrap">
                  <img
                    ref={selectedImgRef}
                    className="selectedImage"
                    src={selectedFromHistoryUrl}
                    alt="selected"
                    onLoad={() => updateSelectedImgSize()}
                  />

                  {selectedImgSize && overlays.length ? (
                    <div className="bboxLayer" aria-hidden="true">
                      {overlays.map((b) => {
                        const isActive = b.idx === activeFaceIdx;
                        return (
                          <div
                            key={b.idx}
                            className={`bbox ${isActive ? 'active' : ''}`}
                            style={{
                              left: `${b.left}px`,
                              top: `${b.top}px`,
                              width: `${b.width}px`,
                              height: `${b.height}px`,
                            }}
                            title={`Лицо ${b.idx + 1}`}
                            onClick={() => setActiveFaceIdx(b.idx)}
                          >
                            <div className="bboxBadge">{b.idx + 1}</div>
                          </div>
                        );
                      })}
                    </div>
                  ) : null}
                </div>

                {faces.length ? (
                  <div className="muted" style={{ marginTop: 10 }}>
                    Рамки показывают, где детектор нашёл лица. Активная рамка соответствует выбранному лицу в результатах ниже.
                  </div>
                ) : null}
              </div>
            </div>
          ) : null}

          <ResultCards
            record={activeRecord}
            loading={loading}
            activeFaceIdx={activeFaceIdx}
            onActiveFaceIdxChange={setActiveFaceIdx}
          />
        </section>

        <section className="col">
          <HistoryPanel
            items={history}
            activeId={activeRecord?.id || null}
            onRefresh={loadHistory}
            onClearUi={clearHistoryUi}
            onSelect={(rec) => {
              setActiveRecord(rec);
              setSelectedFromHistory(true);
              setActiveFaceIdx(0);

              setFile(null);
              setPreviewUrl('');
            }}
          />
        </section>
      </main>
    </div>
  );
}