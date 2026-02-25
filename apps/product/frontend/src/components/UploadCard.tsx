import React, { useRef } from 'react';

type Props = {
  file: File | null;
  previewUrl: string;
  loading: boolean;
  onPickFile: (f: File | null) => void;
  onAnalyze: () => void;
  onClear: () => void;
};

export default function UploadCard({
  file,
  previewUrl,
  loading,
  onPickFile,
  onAnalyze,
  onClear,
}: Props) {
  const inputRef = useRef<HTMLInputElement | null>(null);

  function onChange(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0] || null;
    onPickFile(f);
  }

  function openPicker() {
    inputRef.current?.click();
  }

  return (
    <div className="card">
      <div className="cardTitle">1) Загрузка изображения</div>
      <div className="cardBody">
        <div className="hint">
          Перетащите фото сюда или кликните<br />
          <span className="muted">Совет: лицо/человек должны быть достаточно крупно, без сильного размытия.</span>
        </div>

        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          onChange={onChange}
          style={{ display: 'none' }}
        />

        <div className="row">
          <button className="btn" onClick={openPicker} disabled={loading}>
            {file ? 'Заменить файл' : 'Выбрать файл'}
          </button>

          <button className="btn primary" onClick={onAnalyze} disabled={!file || loading}>
            {loading ? 'Обработка…' : 'Анализировать'}
          </button>

          <button className="btn" onClick={onClear} disabled={loading}>
            Очистить
          </button>
        </div>

        <div className="muted" style={{ marginTop: 8 }}>
          {file ? `Выбран: ${file.name}` : 'Файл не выбран'}
        </div>

        {previewUrl ? (
          <div className="thumbWrap">
            <img className="thumb" src={previewUrl} alt="thumb" />
          </div>
        ) : null}
      </div>
    </div>
  );
}
