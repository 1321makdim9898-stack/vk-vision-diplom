import React from 'react';
import type { HistoryRecord } from '../api';

type Props = {
  items: HistoryRecord[];
  activeId: string | null;
  onRefresh: () => void;
  onClearUi: () => void;
  onSelect: (rec: HistoryRecord) => void;
};

export default function HistoryPanel({ items, activeId, onRefresh, onClearUi, onSelect }: Props) {
  return (
    <div className="card">
      <div className="cardTitle row spaceBetween">
        <span>3) История анализов</span>
        <div className="row" style={{ gap: 8 }}>
          <button type="button" className="btn" onClick={onRefresh}>
            Обновить
          </button>
          <button
            type="button"
            className="btn"
            onClick={onClearUi}
            title="Очистить историю только на странице"
          >
            Очистить
          </button>
        </div>
      </div>

      <div className="cardBody">
        {!items?.length ? (
          <div className="muted">История пустая. Сделайте первый анализ.</div>
        ) : (
          <div className="historyList">
            {items.map((it) => (
              <button
                key={it.id}
                type="button"
                className={`historyItem ${activeId === it.id ? 'active' : ''}`}
                onClick={() => onSelect(it)}
              >
                <img className="historyThumb" src={it.image.url} alt="thumb" />
                <div className="historyMeta">
                  <div className="historyName">{it.image.originalname || it.image.filename}</div>
                  <div className="muted">{new Date(it.createdAt).toLocaleString()}</div>
                </div>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}