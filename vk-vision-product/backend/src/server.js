'use strict';

const express = require('express');
const cors = require('cors');
const path = require('path');

const analyzeRouter = require('./routes/analyze');
const historyRouter = require('./routes/history');

const app = express();

// --- config ---
const PORT = Number(process.env.PORT || 3001);
const HOST = process.env.HOST || '127.0.0.1';

const uploadsDir = process.env.UPLOADS_DIR
  ? path.resolve(process.env.UPLOADS_DIR)
  : path.resolve(__dirname, '..', 'uploads');

const storageDir = process.env.STORAGE_DIR
  ? path.resolve(process.env.STORAGE_DIR)
  : path.resolve(__dirname, 'storage');

app.locals.uploadsDir = uploadsDir;
app.locals.storageDir = storageDir;

app.use(cors());
app.use(express.json({ limit: '2mb' }));

// static uploads
app.use('/uploads', express.static(uploadsDir));

// health
app.get('/api/health', (req, res) => {
  res.json({
    ok: true,
    service: 'vk-vision-product-backend',
    uploads: uploadsDir,
    storage: storageDir,
    ml: process.env.ML_SERVICE_URL || 'http://127.0.0.1:8001',
  });
});

// routes
app.use('/api', analyzeRouter);
app.use('/api', historyRouter);

// error fallback
app.use((err, req, res, next) => {
  // multer errors come here too
  const status = err?.statusCode || err?.status || 500;
  const message = err?.message || 'Internal Server Error';
  res.status(status).json({ ok: false, error: message });
});

app.listen(PORT, HOST, () => {
  // eslint-disable-next-line no-console
  console.log(`[backend] listening on http://${HOST}:${PORT}`);
  // eslint-disable-next-line no-console
  console.log(`[backend] ML_SERVICE_URL=${process.env.ML_SERVICE_URL || 'http://127.0.0.1:8001'}`);
  // eslint-disable-next-line no-console
  console.log(`[backend] uploads=${uploadsDir}`);
  // eslint-disable-next-line no-console
  console.log(`[backend] storage=${storageDir}`);
});
