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
      const mlBase = process.env.ML_SERVICE_URL || 'http://127.0.0.1:8001';
      const mlUrl = `${mlBase.replace(/\/$/, '')}/infer`;

      const files = req.files || {};
      const picked = (files.image && files.image[0]) || (files.file && files.file[0]) || null;

      if (!picked) {
        return res.status(400).json({
          ok: false,
          error: 'No file uploaded. Send multipart/form-data with field "image" (or "file").',
        });
      }

      // Forward to ML via multipart
      const form = new FormData();
      // ML accepts both "image" and "file"; we always send as "image"
      form.append('image', fs.createReadStream(picked.path), {
        filename: picked.originalname || path.basename(picked.path),
        contentType: picked.mimetype || 'application/octet-stream',
      });

      const mlResp = await axios.post(mlUrl, form, {
        headers: form.getHeaders(),
        timeout: 120000,
        maxBodyLength: Infinity,
        maxContentLength: Infinity,
        validateStatus: () => true,
      });

      if (mlResp.status < 200 || mlResp.status >= 300) {
        return res.status(502).json({
          ok: false,
          error: `ML service error: ${mlResp.status}`,
          details: mlResp.data,
        });
      }

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
        result: mlResp.data,
      };

      appendHistoryRecord(req.app.locals.storageDir, record);

      return res.json({ ok: true, record });
    } catch (e) {
      return res.status(500).json({
        ok: false,
        error: e?.message || 'Analyze failed',
      });
    }
  }
);

module.exports = router;
