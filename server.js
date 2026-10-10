'use strict';
// Small local server for the training app. No dependencies.
//   npm start            start the server and open the browser
//   npm start -- --no-open   start without opening the browser
// PORT=5000 npm start    use another start port
const http = require('http');
const fs = require('fs');
const path = require('path');
const { exec } = require('child_process');

const ROOT = __dirname;
const DATA_DIR = path.join(ROOT, 'data');
const DATA_FILE = path.join(DATA_DIR, 'progress.json');
const START_PORT = Number(process.env.PORT) || 4318;
const PUBLIC_TOP = new Set(['index.html', 'src', 'content', 'assets']);
const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.ico': 'image/x-icon',
};

function send(res, code, body, type) {
  res.writeHead(code, {
    'Content-Type': type || 'text/plain; charset=utf-8',
    'Cache-Control': 'no-store',
    'X-Content-Type-Options': 'nosniff',
  });
  res.end(body);
}

// Only answer requests that address this server by a local name (blocks DNS rebinding).
function hostOk(req) {
  const host = String(req.headers.host || '').toLowerCase();
  const port = req.socket.localPort;
  return host === `localhost:${port}` || host === `127.0.0.1:${port}` || host === `[::1]:${port}`;
}

function serveStatic(res, pathname) {
  let rel;
  try { rel = decodeURIComponent(pathname); } catch (e) { return send(res, 400, 'Bad request'); }
  if (rel.includes('\0')) return send(res, 400, 'Bad request');
  if (rel === '/') rel = '/index.html';
  const file = path.resolve(ROOT, '.' + rel);
  const fromRoot = path.relative(ROOT, file);
  if (fromRoot.startsWith('..') || path.isAbsolute(fromRoot)) return send(res, 403, 'Forbidden');
  if (!PUBLIC_TOP.has(fromRoot.split(path.sep)[0])) return send(res, 404, 'Not found');
  fs.stat(file, (err, st) => {
    if (err || !st.isFile()) return send(res, 404, 'Not found');
    res.writeHead(200, {
      'Content-Type': TYPES[path.extname(file).toLowerCase()] || 'application/octet-stream',
      'Cache-Control': 'no-store',
      'X-Content-Type-Options': 'nosniff',
    });
    fs.createReadStream(file).pipe(res);
  });
}

function handleState(req, res) {
  const json = 'application/json; charset=utf-8';
  if (req.method === 'GET') {
    return fs.readFile(DATA_FILE, 'utf8', (err, txt) => send(res, 200, err ? '{}' : txt, json));
  }
  if (req.method !== 'PUT') return send(res, 405, 'Method not allowed');
  // A JSON content type forces a CORS preflight, so other web pages cannot write here.
  if (!/^application\/json/i.test(req.headers['content-type'] || '')) return send(res, 415, 'JSON only');
  const chunks = [];
  let size = 0, aborted = false;
  req.on('data', (c) => {
    size += c.length;
    if (size > 1000000) { aborted = true; send(res, 413, 'Too large'); req.destroy(); } else chunks.push(c);
  });
  req.on('end', () => {
    if (aborted) return;
    let data;
    try { data = JSON.parse(Buffer.concat(chunks).toString('utf8')); } catch (e) { return send(res, 400, 'Bad JSON'); }
    if (!data || typeof data !== 'object' || Array.isArray(data)) return send(res, 400, 'Object expected');
    try {
      fs.mkdirSync(DATA_DIR, { recursive: true });
      const tmp = DATA_FILE + '.tmp';
      fs.writeFileSync(tmp, JSON.stringify(data, null, 2));
      fs.renameSync(tmp, DATA_FILE);
    } catch (e) { return send(res, 500, 'Could not save'); }
    res.writeHead(204, { 'Cache-Control': 'no-store' });
    res.end();
  });
}

function handler(req, res) {
  if (!hostOk(req)) return send(res, 403, 'Forbidden');
  const pathname = (req.url || '/').split('?')[0];
  if (pathname === '/api/state') return handleState(req, res);
  if (req.method !== 'GET' && req.method !== 'HEAD') return send(res, 405, 'Method not allowed');
  return serveStatic(res, pathname);
}

function openBrowser(url) {
  const cmd = process.platform === 'win32' ? `start "" "${url}"` : process.platform === 'darwin' ? `open "${url}"` : `xdg-open "${url}"`;
  exec(cmd, () => { /* if no browser opens, the URL is in the console */ });
}

function listen(port, retries) {
  const server = http.createServer(handler);
  server.once('error', (err) => {
    if (err.code === 'EADDRINUSE' && retries > 0) return listen(port + 1, retries - 1);
    console.error('Could not start the server: ' + err.message);
    process.exit(1);
  });
  server.listen(port, '127.0.0.1', () => {
    const url = `http://localhost:${port}/`;
    console.log(`MLIR and LLVM training app: ${url}`);
    console.log('Progress is saved in data/progress.json. Press Ctrl+C to stop.');
    if (!process.argv.includes('--no-open') && !process.env.NO_OPEN) openBrowser(url);
  });
}

listen(START_PORT, 20);
