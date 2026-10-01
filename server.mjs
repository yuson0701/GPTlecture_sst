import http from 'node:http';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { LocalTranscriber } from './local/transcriber.mjs';
import { localSummary } from './local/summary.mjs';

const files = new Map([['/', ['index.html', 'text/html']], ...['app.js', 'transcript.js', 'audio.js', 'capture-worklet.js'].map(file => ['/' + file, [file, 'text/javascript']]), ['/style.css', ['style.css', 'text/css']]]);
export function createApp({ env = process.env, fetcher = fetch, transcriber = new LocalTranscriber(env) } = {}) {
  const json = (res, status, value) => { res.writeHead(status, { 'Content-Type': 'application/json' }); res.end(JSON.stringify(value)); };
  let sttBusy = false, summaryBusy = false;
  const server = http.createServer(async (req, res) => {
    res.setHeader('Cache-Control', 'no-store');
    res.setHeader('X-Content-Type-Options', 'nosniff');
    res.setHeader('Content-Security-Policy', "default-src 'self'; connect-src 'self'; style-src 'self'; script-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'");
    try {
      const host = req.headers.host;
      if (!/^(localhost|127\.0\.0\.1|\[::1\])(?::\d+)?$/.test(host || '')) return json(res, 403, { error: 'Local access only.' });
      if (req.method === 'GET' && req.url === '/api/config') {
        let status; try { status = await transcriber.request('status'); } catch { status = { available: false }; }
        return json(res, 200, { configured: status.available, reason: status.reason, model: `faster-whisper · ${env.WHISPER_MODEL || 'large-v3-turbo'}`, summary: env.SUMMARY_MODE === 'extractive' ? 'extractive' : 'ollama', local: true });
      }
      if (['GET', 'HEAD'].includes(req.method) && files.has(req.url)) {
        const [file, mime] = files.get(req.url), content = await readFile(new URL(`./public/${file}`, import.meta.url));
        res.writeHead(200, { 'Content-Type': `${mime}; charset=utf-8` });
        return res.end(req.method === 'HEAD' ? undefined : content);
      }
      if (req.method !== 'POST' || !['/api/session', '/api/transcribe', '/api/summary'].includes(req.url)) return json(res, 404, { error: 'Not found' });
      if (req.headers.origin !== `http://${host}` && req.headers.origin !== `https://${host}`) return json(res, 403, { error: 'Origin rejected' });
      if (!req.headers['content-type']?.startsWith('application/json')) return json(res, 415, { error: 'JSON required' });
      let raw = ''; for await (const chunk of req) { raw += chunk; if (Buffer.byteLength(raw) > 700000) return json(res, 413, { error: 'Request too large' }); }
      let body; try { body = JSON.parse(raw); } catch { return json(res, 400, { error: 'Invalid JSON' }); }
      if (!body || typeof body !== 'object') return json(res, 400, { error: 'Invalid request' });
      if (req.url === '/api/session' || req.url === '/api/transcribe') {
        if (typeof body.glossary !== 'string' || body.glossary.length > 1500) return json(res, 400, { error: '용어는 1,500자 이내로 입력해 주세요.' });
        if (req.url === '/api/transcribe') {
          if (typeof body.audio !== 'string' || body.audio.length > 640000 || !/^[A-Za-z0-9+/]+={0,2}$/.test(body.audio) || body.audio.length % 4) return json(res, 400, { error: 'Invalid PCM audio' });
          const audio = Buffer.from(body.audio, 'base64');
          if (audio.length < 3200 || audio.length > 480000 || audio.length % 2) return json(res, 400, { error: 'Invalid PCM audio length' });
        }
        if (sttBusy) return json(res, 429, { error: '음성 처리 중입니다. 다른 강의 탭을 종료해 주세요.' });
        sttBusy = true;
        try { const result = await transcriber.request(req.url === '/api/session' ? 'load' : 'transcribe', { audio: body.audio, glossary: body.glossary }); return json(res, 200, result); }
        catch (error) { return json(res, 503, { error: error.message }); }
        finally { sttBusy = false; }
      }
      if (typeof body.transcript !== 'string' || !body.transcript.trim() || body.transcript.length > 7000 || typeof body.previous !== 'string' || body.previous.length > 6000) return json(res, 400, { error: 'Invalid summary input' });
      if (summaryBusy) return json(res, 429, { error: '요약 처리 중입니다. 잠시 후 다시 시도해 주세요.' });
      summaryBusy = true;
      try { json(res, 200, await localSummary(body, { env, fetcher })); } finally { summaryBusy = false; }
    } catch { if (!res.headersSent) json(res, 500, { error: '로컬 처리에 실패했습니다. 설정을 확인한 후 다시 시도해 주세요.' }); else res.end(); }
  });
  server.on('close', () => transcriber.close());
  return server;
}
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const server = createApp();
  server.listen(Number(process.env.PORT || 3000), '127.0.0.1', () => console.log(`Lecture Note (local): http://localhost:${process.env.PORT || 3000}`));
  for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => { server.close(); server.closeAllConnections(); });
}
