'use strict';

const express = require('express');
const { readHistory } = require('../storage/historyStore');

const router = express.Router();

router.get('/history', (req, res) => {
  // Важно для кнопки "Обновить": не даём браузеру отдавать 304 из кеша
  res.set('Cache-Control', 'no-store');
  res.set('Pragma', 'no-cache');

  const limitRaw = req.query.limit;
  const limit = Math.min(Math.max(Number(limitRaw || 20), 1), 200);

  const items = readHistory(req.app.locals.storageDir, limit);

  res.json({ ok: true, items });
});

module.exports = router;