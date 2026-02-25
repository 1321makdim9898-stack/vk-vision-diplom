import React, { useMemo } from 'react';
import type { HistoryRecord, FaceResult } from '../api';

type Props = {
  record: HistoryRecord | null;
  loading: boolean;
};

function normalizeFaces(result: any): FaceResult[] {
  if (!result) return [];
  if (Array.isArray(result)) return result as FaceResult[];
  if (Array.isArray(result.faces)) return result.faces as FaceResult[];
  return [];
}

export default function ResultCards({ record, loading }: Props) {
  const faces = useMemo(() => normalizeFaces(record?.result), [record]);

  const first = faces[0];

  const ageText = first?.age_gender?.age_est;
  const ageBin = first?.age_gender?.age_bin;
  const gender = first?.age_gender?.gender;

  const emotionLabel = first?.emotion?.label;
  const emotionProb = first?.emotion?.prob;

  return (
    <div className="card">
      <div className="cardTitle">2) Результаты анализа</div>
      <div className="cardBody">
        {loading ? <div className="muted">Обработка…</div> : null}

        {!loading && !record ? (
          <div className="muted">Пока нет результатов. Загрузите фото и нажмите «Анализировать».</div>
        ) : null}

        {!loading && record ? (
          <div className="resultGrid">
            <div className="resultItem">
              <div className="label">Статус</div>
              <div className="value">OK</div>
            </div>

            <div className="resultItem">
              <div className="label">Лиц</div>
              <div className="value">{faces.length}</div>
            </div>

            <div className="resultItem">
              <div className="label">Возраст</div>
              <div className="value">
                {typeof ageText === 'number' && Number.isFinite(ageText) ? `${Math.round(ageText)}` : '—'}
                {ageBin ? ` (${ageBin})` : ''}
              </div>
            </div>

            <div className="resultItem">
              <div className="label">Пол</div>
              <div className="value">{gender || '—'}</div>
            </div>

            <div className="resultItem">
              <div className="label">Эмоция</div>
              <div className="value">
                {emotionLabel || '—'}
                {typeof emotionProb === 'number' ? ` (${(emotionProb * 100).toFixed(1)}%)` : ''}
              </div>
            </div>

            <details className="details">
              <summary>RAW JSON</summary>
              <pre>{JSON.stringify(record.result, null, 2)}</pre>
            </details>
          </div>
        ) : null}
      </div>
    </div>
  );
}
