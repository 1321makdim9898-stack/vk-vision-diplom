'use strict';

const fs = require('fs');
const path = require('path');

function ensureDir(dir) {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

function historyPath(storageDir) {
  return path.join(storageDir, 'history.json');
}

function readAll(storageDir) {
  ensureDir(storageDir);
  const p = historyPath(storageDir);
  if (!fs.existsSync(p)) return [];
  try {
    const raw = fs.readFileSync(p, 'utf-8');
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function writeAll(storageDir, items) {
  ensureDir(storageDir);
  const p = historyPath(storageDir);
  fs.writeFileSync(p, JSON.stringify(items, null, 2), 'utf-8');
}

function appendHistoryRecord(storageDir, record) {
  const items = readAll(storageDir);
  items.unshift(record);
  // keep last 500
  writeAll(storageDir, items.slice(0, 500));
}

function readHistory(storageDir, limit) {
  const items = readAll(storageDir);
  return items.slice(0, limit);
}

module.exports = {
  appendHistoryRecord,
  readHistory,
};
