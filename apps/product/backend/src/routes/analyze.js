'use strict';

const express = require('express');
const path = require('path');
const fs = require('fs');
const crypto = require('crypto');

const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');

const { appendHistoryRecord } = require('../storage/historyStore');

const router = express.Router();

function ensureDir(dir) {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

function safeExt(originalName) {
  const ext = path.extname(originalName || '').toLowerCase();
  // allow common image formats; fallback to .bin
  if (['.jpg', '.jpeg', '.png', '.webp'].includes(ext)) return ext;
  return ext || '.bin';
}

function makeFilename(originalName) {
  const id = crypto.randomBytes(10).toString('hex');
  return `${Date.now()}_${id}${safeExt(originalName)}`;
}

function buildPublicUrl(req, fileName) {
  return `${req.protocol}://${req.get('host')}/uploads/${encodeURIComponent(fileName)}`;
}

// Multer storage (disk)
function makeUploader(uploadsDir) {
  ensureDir(uploadsDir);

  const storage = multer.diskStorage({
    destination: (req, file, cb) => cb(null, uploadsDir),
    filename: (req, file, cb) => cb(null, makeFilename(file.originalname)),
  });

  return multer({
    storage,
    limits: { fileSize: 10 * 1024 * 1024 }, // 10MB
  });
}

function extFromContentType(ct) {
  const t = String(ct || '').toLowerCase();
  if (t.includes('image/jpeg')) return '.jpg';
  if (t.includes('image/png')) return '.png';
  if (t.includes('image/webp')) return '.webp';
  return '';
}

function looksLikeImageUrl(url) {
  return /\.(jpg|jpeg|png|webp)(\?|$)/i.test(String(url || ''));
}

async function downloadUrlToFile(inputUrl, outPath) {
  // ВАЖНО: VK CDN иногда режет “ботов”. Эти заголовки помогают.
  const resp = await axios.get(inputUrl, {
    responseType: 'stream',
    timeout: 30000,
    maxRedirects: 5,
    validateStatus: () => true,
    headers: {
      'User-Agent':
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
      Accept: 'image/avif,image/webp,image/apng,image/*,*/*;q=0.8',
      Referer: 'https://vk.com/',
    },
  });

  if (resp.status < 200 || resp.status >= 300) {
    const ct = String(resp.headers['content-type'] || '');
    throw new Error(`Download failed: HTTP ${resp.status} (content-type: ${ct || 'unknown'})`);
  }

  // Пытаемся понять, что это не HTML
  const contentType = String(resp.headers['content-type'] || '').toLowerCase();
  const isHtml = contentType.includes('text/html') || contentType.includes('text/plain');
  if (contentType && isHtml && !looksLikeImageUrl(inputUrl)) {
    throw new Error(`URL returned non-image content-type: ${contentType}`);
  }

  await new Promise((resolve, reject) => {
    const ws = fs.createWriteStream(outPath);
    resp.data.pipe(ws);
    ws.on('finish', resolve);
    ws.on('error', reject);
  });

  return { contentType };
}

async function inferWithMl(mlBase, filePath, originalName, mimetype) {
  const mlUrl = `${mlBase.replace(/\/$/, '')}/infer`;

  const form = new FormData();
  form.append('image', fs.createReadStream(filePath), {
    filename: originalName || path.basename(filePath),
    contentType: mimetype || 'application/octet-stream',
  });

  const mlResp = await axios.post(mlUrl, form, {
    headers: form.getHeaders(),
    timeout: 120000,
    maxBodyLength: Infinity,
    maxContentLength: Infinity,
    validateStatus: () => true,
  });

  if (mlResp.status < 200 || mlResp.status >= 300) {
    const err = new Error(`ML service error: ${mlResp.status}`);
    err.details = mlResp.data;
    err.statusCode = 502;
    throw err;
  }

  return mlResp.data;
}

// ==========================
// 1) Анализ загруженного файла
// ==========================
router.post(
  '/analyze',
  (req, res, next) => {
    // inject uploader with correct uploadsDir
    const uploadsDir = req.app.locals.uploadsDir;
    const upload = makeUploader(uploadsDir);

    // IMPORTANT: accept BOTH names to avoid "Unexpected field"
    const handler = upload.fields([
      { name: 'image', maxCount: 1 },
      { name: 'file', maxCount: 1 },
    ]);

    handler(req, res, (err) => {
      if (err) return next(err);
      return next();
    });
  },
  async (req, res) => {
    try {
      const mlBase = process.env.ML_SERVICE_URL || req.app.locals.mlServiceUrl || 'http://127.0.0.1:8001';

      const files = req.files || {};
      const picked = (files.image && files.image[0]) || (files.file && files.file[0]) || null;

      if (!picked) {
        return res.status(400).json({
          ok: false,
          error: 'No file uploaded. Send multipart/form-data with field "image" (or "file").',
        });
      }

      const result = await inferWithMl(mlBase, picked.path, picked.originalname, picked.mimetype);

      const imageUrl = buildPublicUrl(req, picked.filename);

      const record = {
        id: crypto.randomBytes(10).toString('hex'),
        createdAt: new Date().toISOString(),
        image: {
          filename: picked.filename,
          originalname: picked.originalname,
          mimetype: picked.mimetype,
          size: picked.size,
          url: imageUrl,
        },
        result,
      };

      appendHistoryRecord(req.app.locals.storageDir, record);

      return res.json({ ok: true, record });
    } catch (e) {
      const status = e?.statusCode || 500;
      return res.status(status).json({
        ok: false,
        error: e?.message || 'Analyze failed',
        details: e?.details || null,
      });
    }
  }
);

// ==========================
// 2) Анализ по ссылке (VK CDN / любой URL)
// POST /api/analyze_url { url: "https://..." }
// ==========================
router.post('/analyze_url', async (req, res) => {
  try {
    const mlBase = process.env.ML_SERVICE_URL || req.app.locals.mlServiceUrl || 'http://127.0.0.1:8001';
    const uploadsDir = req.app.locals.uploadsDir;
    ensureDir(uploadsDir);

    const inputUrl = String(req.body?.url || req.body?.imageUrl || '').trim();
    if (!inputUrl) {
      return res.status(400).json({ ok: false, error: 'Missing "url" in JSON body.' });
    }

    let urlObj;
    try {
      urlObj = new URL(inputUrl);
    } catch {
      return res.status(400).json({ ok: false, error: 'Invalid URL.' });
    }

    // Сгенерим имя файла. Расширение: либо по content-type, либо из URL.
    const baseNameFromUrl = path.basename(urlObj.pathname || 'image');
    let tmpName = makeFilename(baseNameFromUrl);
    let outPath = path.join(uploadsDir, tmpName);

    // Скачиваем
    const { contentType } = await downloadUrlToFile(inputUrl, outPath);

    // Если content-type дал лучшее расширение — переименуем файл
    const extByCt = extFromContentType(contentType);
    if (extByCt && path.extname(outPath).toLowerCase() !== extByCt) {
      const renamed = outPath.replace(path.extname(outPath), extByCt);
      fs.renameSync(outPath, renamed);
      outPath = renamed;
      tmpName = path.basename(outPath);
    }

    // Минимальная проверка: файл реально не пустой
    const st = fs.statSync(outPath);
    if (!st.size || st.size < 20) {
      throw new Error('Downloaded file looks empty.');
    }

    // Инференс
    const result = await inferWithMl(mlBase, outPath, baseNameFromUrl, contentType || 'application/octet-stream');

    const imageUrl = buildPublicUrl(req, tmpName);

    const record = {
      id: crypto.randomBytes(10).toString('hex'),
      createdAt: new Date().toISOString(),
      image: {
        filename: tmpName,
        originalname: baseNameFromUrl,
        mimetype: contentType || 'application/octet-stream',
        size: st.size,
        url: imageUrl,
        sourceUrl: inputUrl,
      },
      result,
    };

    appendHistoryRecord(req.app.locals.storageDir, record);

    return res.json({ ok: true, record });
  } catch (e) {
    const status = e?.statusCode || 500;
    return res.status(status).json({
      ok: false,
      error: e?.message || 'Analyze URL failed',
      details: e?.details || null,
    });
  }
});

module.exports = router;