'use strict';

const express = require('express');
const path = require('path');
const fs = require('fs');
const { appendHistoryRecord } = require('../storage/historyStore');

const router = express.Router();

const VK_API_VERSION = process.env.VK_API_VERSION || '5.199';

function pickBestSize(sizes = []) {
  if (!Array.isArray(sizes) || sizes.length === 0) return null;
  return sizes.reduce((best, s) => {
    const bestArea = (best?.width || 0) * (best?.height || 0);
    const area = (s?.width || 0) * (s?.height || 0);
    return area > bestArea ? s : best;
  }, sizes[0]);
}

function ensureDir(dir) {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

async function vkCall(method, params) {
  const qs = new URLSearchParams({ ...params, v: VK_API_VERSION });
  const url = `https://api.vk.com/method/${method}?${qs.toString()}`;

  const r = await fetch(url, { method: 'GET' });
  const data = await r.json();

  if (!r.ok) throw new Error(`VK HTTP ${r.status}`);
  if (data?.error) throw new Error(`VK: ${data.error.error_msg || 'VK error'}`);

  return data.response;
}

function getToken(req) {
  const token = req.header('X-VK-Token') || '';
  if (!token) throw new Error('Missing X-VK-Token header');
  return token;
}

// GET /api/vk/photos?count=30  (фото со страницы: album_id=profile)
router.get('/vk/photos', async (req, res) => {
  try {
    const token = getToken(req);

    const count = Math.min(Math.max(Number(req.query.count || 20), 1), 200);

    const me = await vkCall('users.get', { access_token: token });
    const ownerId = me?.[0]?.id;
    if (!ownerId) throw new Error('Cannot detect VK user id from token');

    const photos = await vkCall('photos.get', {
      owner_id: String(ownerId),
      album_id: 'profile',
      count: String(count),
      rev: '1',
      photo_sizes: '1',
      access_token: token,
    });

    const items = (photos?.items || [])
      .map((p) => {
        const best = pickBestSize(p.sizes);
        const thumb =
          pickBestSize((p.sizes || []).filter((s) => (s.width || 0) <= 200)) || best;

        return {
          id: p.id,
          date: p.date,
          text: p.text || '',
          thumbUrl: thumb?.url || best?.url || null,
          url: best?.url || null,
        };
      })
      .filter((x) => x.url);

    res.json({ ok: true, ownerId, items });
  } catch (e) {
    res.status(400).json({ ok: false, error: e?.message || 'VK photos failed' });
  }
});

// POST /api/vk/analyze  body: { url: "https://..." }
router.post('/vk/analyze', express.json({ limit: '1mb' }), async (req, res) => {
  try {
    // токен нужен как минимум для /vk/photos; тут проверим тоже, чтобы не забыли
    getToken(req);

    const photoUrl = req.body?.url;
    if (!photoUrl) throw new Error('Missing body.url');

    const imgResp = await fetch(photoUrl, { method: 'GET' });
    if (!imgResp.ok) throw new Error(`Failed to download photo: HTTP ${imgResp.status}`);

    const contentType = imgResp.headers.get('content-type') || 'image/jpeg';
    const buf = Buffer.from(await imgResp.arrayBuffer());

    const ext =
      contentType.includes('png') ? 'png' :
      contentType.includes('webp') ? 'webp' :
      contentType.includes('avif') ? 'avif' :
      'jpg';

    const filename = `vk_${Date.now()}.${ext}`;

    const uploadsDir = req.app.locals.uploadsDir;
    ensureDir(uploadsDir);
    const savePath = path.join(uploadsDir, filename);
    fs.writeFileSync(savePath, buf);

    // ML infer
    const mlUrl = String(req.app.locals.mlServiceUrl || 'http://127.0.0.1:8001').replace(/\/$/, '');
    const form = new FormData();
    const blob = new Blob([buf], { type: contentType });
    form.append('file', blob, filename);

    const mlResp = await fetch(`${mlUrl}/infer`, { method: 'POST', body: form });
    const mlJson = await mlResp.json();
    if (!mlResp.ok || !mlJson?.ok) {
      throw new Error(mlJson?.error || `ML infer failed (HTTP ${mlResp.status})`);
    }

    const record = {
      id: String(Date.now()),
      createdAt: new Date().toISOString(),
      image: {
        filename,
        originalname: filename,
        mimetype: contentType,
        size: buf.length,
        url: `/uploads/${filename}`,
      },
      result: mlJson,
      source: { type: 'vk', url: photoUrl },
    };

    appendHistoryRecord(req.app.locals.storageDir, record);

    res.json({ ok: true, record });
  } catch (e) {
    res.status(400).json({ ok: false, error: e?.message || 'VK analyze failed' });
  }
});

module.exports = router;