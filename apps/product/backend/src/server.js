'use strict';

const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs');

const analyzeRouter = require('./routes/analyze');
const historyRouter = require('./routes/history');
const vkRouter = require('./routes/vk');

const app = express();

// Отключаем ETag, чтобы браузер не получал 304 на /api/history (кнопка "Обновить")
app.set('etag', false);

// --- config ---
const PORT = Number(process.env.PORT || 3001);
const HOST = process.env.HOST || '127.0.0.1';

const uploadsDir = process.env.UPLOADS_DIR
  ? path.resolve(process.env.UPLOADS_DIR)
  : path.resolve(__dirname, '..', 'uploads');

const storageDir = process.env.STORAGE_DIR
  ? path.resolve(process.env.STORAGE_DIR)
  : path.resolve(__dirname, 'storage');

const mlServiceUrl = process.env.ML_SERVICE_URL || 'http://127.0.0.1:8001';

app.locals.uploadsDir = uploadsDir;
app.locals.storageDir = storageDir;
app.locals.mlServiceUrl = mlServiceUrl;

app.use(cors());
app.use(express.json({ limit: '2mb' }));

// --- Uploads: удобная страница вместо "Cannot GET /uploads/" ---
app.get(['/uploads', '/uploads/'], (req, res) => {
  let files = [];
  try {
    files = fs
      .readdirSync(uploadsDir, { withFileTypes: true })
      .filter((d) => d.isFile())
      .map((d) => d.name)
      .sort((a, b) => a.localeCompare(b));
  } catch (e) {
    files = [];
  }

  const rows = files
    .map((name) => {
      const href = `/uploads/${encodeURIComponent(name)}`;
      return `<li><a href="${href}" target="_blank" rel="noreferrer">${name}</a></li>`;
    })
    .join('\n');

  res.status(200).type('html').send(`<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Uploads</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }
    code { background:#f3f3f3; padding:2px 6px; border-radius:6px; }
  </style>
</head>
<body>
  <h1>Uploaded files</h1>
  <p>Папка: <code>${uploadsDir.replace(/</g, '&lt;')}</code></p>
  ${files.length ? `<ul>${rows}</ul>` : `<p>Пока нет файлов. Сначала загрузи изображение через UI.</p>`}
</body>
</html>`);
});

// static uploads (файлы по /uploads/<filename>)
app.use(
  '/uploads',
  express.static(uploadsDir, {
    etag: false,
    maxAge: 0,
    setHeaders(res) {
      res.setHeader('Cache-Control', 'no-store');
    },
  })
);

// health
app.get('/api/health', (req, res) => {
  res.set('Cache-Control', 'no-store');
  res.json({
    ok: true,
    service: 'vk-vision-product-backend',
    uploads: uploadsDir,
    storage: storageDir,
    ml: mlServiceUrl,
  });
});

// routes
app.use('/api', analyzeRouter);
app.use('/api', historyRouter);
app.use('/api', vkRouter);

// error fallback
app.use((err, req, res, next) => {
  const status = err?.statusCode || err?.status || 500;
  const message = err?.message || 'Internal Server Error';
  res.status(status).json({ ok: false, error: message });
});

app.listen(PORT, HOST, () => {
  console.log(`[backend] listening on http://${HOST}:${PORT}`);
  console.log(`[backend] ML_SERVICE_URL=${mlServiceUrl}`);
  console.log(`[backend] uploads=${uploadsDir}`);
  console.log(`[backend] storage=${storageDir}`);
});