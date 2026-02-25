import React, { useEffect, useMemo, useState } from 'react';
import './styles.css';

import { analyzeImage, fetchHistory, type HistoryRecord } from './api';
import UploadCard from './components/UploadCard';
import ResultCards from './components/ResultCards';
import HistoryPanel from './components/HistoryPanel';

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');

  const [history, setHistory] = useState<HistoryRecord[]>([]);
  const [activeRecord, setActiveRecord] = useState<HistoryRecord | null>(null);

  // local preview
  useEffect(() => {
    if (!file) return;
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const activeImageUrl = useMemo(() => {
    if (activeRecord?.image?.url) return activeRecord.image.url;
    return previewUrl;
  }, [activeRecord, previewUrl]);

  async function loadHistory() {
    try {
      setError('');
      const items = await fetchHistory(20);
      setHistory(items);
    } catch (e: any) {
      setError(e?.message || 'Не удалось загрузить историю');
    }
  }

  useEffect(() => {
    loadHistory();
  }, []);

  async function onAnalyze() {
    if (!file) return;
    try {
      setError('');
      setLoading(true);

      const rec = await analyzeImage(file);

      setActiveRecord(rec);
      // prepend to history
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
  }

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
            onPickFile={setFile}
            onAnalyze={onAnalyze}
            onClear={onClear}
          />

          <ResultCards record={activeRecord} loading={loading} />
        </section>

        <section className="col">
          <HistoryPanel
            items={history}
            activeId={activeRecord?.id || null}
            onRefresh={loadHistory}
            onSelect={(rec) => setActiveRecord(rec)}
          />
        </section>
      </main>

      <div className="previewWrap">
        {activeImageUrl ? (
          <img className="bigPreview" src={activeImageUrl} alt="preview" />
        ) : null}
      </div>
    </div>
  );
}
