import React, { useEffect, useMemo } from 'react';
import type { HistoryRecord } from '../api';

function safeRound(n: unknown): number | null {
  if (typeof n !== 'number' || !Number.isFinite(n)) return null;
  return Math.round(n);
}

function genderRu(g: unknown): string {
  if (typeof g !== 'string') return '—';
  const v = g.toLowerCase();
  if (v === 'male' || v === 'm') return 'мужчина';
  if (v === 'female' || v === 'f') return 'женщина';
  return '—';
}

function emotionRu(label: unknown): string {
  if (typeof label !== 'string') return '—';
  const v = label.toLowerCase();
  const map: Record<string, string> = {
    angry: 'злость',
    disgust: 'отвращение',
    fear: 'страх',
    happy: 'радость',
    neutral: 'нейтральная',
    sad: 'грусть',
    surprise: 'удивление',
  };
  return map[v] ?? label;
}

function ageBinRu(bin: unknown): string {
  if (typeof bin !== 'string') return '';
  return bin;
}

export default function ResultCards({
  record,
  loading,
  activeFaceIdx,
  onActiveFaceIdxChange,
}: {
  record: HistoryRecord | null;
  loading: boolean;
  activeFaceIdx: number;
  onActiveFaceIdxChange: (idx: number) => void;
}) {
  const faces = (record?.result as any)?.faces ?? [];
  const facesCount = (record?.result as any)?.faces_count ?? faces?.length ?? 0;

  // при новом результате — сбрасываем активное лицо на 0
  useEffect(() => {
    onActiveFaceIdxChange(0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [record?.id]);

  const face = useMemo(() => {
    if (!Array.isArray(faces) || faces.length === 0) return null;
    const idx = Math.max(0, Math.min(activeFaceIdx, faces.length - 1));
    return faces[idx] ?? null;
  }, [faces, activeFaceIdx]);

  const ageGender = face?.age_gender ?? null;
  const emotion = face?.emotion ?? null;

  const ageEst = safeRound(ageGender?.age_est);
  const ageBin = ageBinRu(ageGender?.age_bin);
  const ageText = ageEst !== null ? `${ageEst}${ageBin ? ` (${ageBin})` : ''}` : ageBin ? ageBin : '—';

  const genderText = genderRu(ageGender?.gender);
  const emotionText = emotionRu(emotion?.label);

  return (
    <div className="card" style={{ marginTop: 12 }}>
      <div className="cardTitle">2) Результаты анализа</div>
      <div className="cardBody">
        <div className="resultGrid">
          <div className="statCard">
            <div className="statLabel">Статус</div>
            <div className="statValue">{record ? 'OK' : loading ? '…' : '—'}</div>
          </div>

          <div className="statCard">
            <div className="statLabel">Найдено лиц</div>
            <div className="statValue">{record ? String(facesCount) : loading ? '…' : '—'}</div>
          </div>
        </div>

        <details style={{ marginTop: 10 }}>
          <summary style={{ cursor: 'pointer', userSelect: 'none' }}>RAW JSON</summary>
          <pre className="jsonBox">{JSON.stringify(record?.result ?? null, null, 2)}</pre>
        </details>

        {record && Array.isArray(faces) && faces.length > 0 ? (
          <div style={{ marginTop: 14 }}>
            <div className="muted" style={{ marginBottom: 10 }}>
              Лица
            </div>

            <div className="faceTabs">
              {faces.map((_: any, idx: number) => {
                const active = idx === activeFaceIdx;
                return (
                  <button
                    key={idx}
                    className={`faceTab ${active ? 'active' : ''}`}
                    onClick={() => onActiveFaceIdxChange(idx)}
                    type="button"
                    title={`Лицо ${idx + 1}`}
                  >
                    Лицо {idx + 1}
                  </button>
                );
              })}
            </div>

            <div className="faceCard" style={{ marginTop: 12 }}>
              <div className="faceCardTitle">Лицо {activeFaceIdx + 1}</div>

              <div className="resultGrid" style={{ marginTop: 10 }}>
                <div className="statCard">
                  <div className="statLabel">Возраст</div>
                  <div className="statValue">{ageText}</div>
                </div>

                <div className="statCard">
                  <div className="statLabel">Пол</div>
                  <div className="statValue">{genderText}</div>
                </div>
              </div>

              <div className="statCard" style={{ marginTop: 10 }}>
                <div className="statLabel">Эмоция</div>
                <div className="statValue">{emotionText}</div>
              </div>

              <div className="muted" style={{ marginTop: 10 }}>
                Порядок лиц — слева направо (по координатам детектора). Поле <b>BBox</b> скрыто: это координаты прямоугольника лица.
              </div>
            </div>
          </div>
        ) : (
          <div className="muted" style={{ marginTop: 12 }}>
            Пока нет результатов. Загрузите фото и нажмите «Анализировать».
          </div>
        )}
      </div>
    </div>
  );
}