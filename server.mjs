import http from 'node:http';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { LocalTranscriber } from './local/transcriber.mjs';
import { RecordStore } from './local/records.mjs';
import { localSummary } from './local/summary.mjs';

const files = new Map([['/', ['index.html', 'text/html']], ...['app.js', 'transcript.js', 'audio.js', 'capture-worklet.js', 'streaming.js'].map(file => ['/' + file, [file, 'text/javascript']]), ['/style.css', ['style.css', 'text/css']]]);
export function createApp({ env = process.env, fetcher = fetch, transcriber = new LocalTranscriber(env) } = {}) {
  const json = (res, status, value) => { res.writeHead(status, { 'Content-Type': 'application/json' }); res.end(JSON.stringify(value)); };
  const records = new RecordStore(env.LECTURE_DATA_DIR || fileURLToPath(new URL('./data/lectures', import.meta.url)));
  let sttBusy = false, summaryBusy = false;
  const qwen = (env.STT_BACKEND || env.WHISPER_BACKEND) === 'qwen-mlx';
  const streaming = (env.STT_BACKEND || env.WHISPER_BACKEND) === 'sherpa';
  const server = http.createServer(async (req, res) => {
    res.setHeader('Cache-Control', 'no-store');
    res.setHeader('X-Content-Type-Options', 'nosniff');
    res.setHeader('Content-Security-Policy', "default-src 'self'; connect-src 'self'; style-src 'self'; script-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'");
    try {
      const host = req.headers.host;
      if (!/^(localhost|127\.0\.0\.1|\[::1\])(?::\d+)?$/.test(host || '')) return json(res, 403, { error: 'Local access only.' });
      const recordRoute = req.url.match(/^\/api\/records\/([a-f0-9-]+)$/);
      if (req.method === 'GET' && (req.url === '/api/records' || recordRoute)) {
        if (req.headers.origin && ![`http://${host}`, `https://${host}`].includes(req.headers.origin)) return json(res, 403, { error: 'Origin rejected' });
        try { return json(res, 200, recordRoute ? await records.get(recordRoute[1]) : { records: await records.list() }); }
        catch (e) { return json(res, e.status || 500, { error: e.status ? e.message : '저장된 기록을 읽지 못했습니다.' }); }
      }
      if (req.method === 'GET' && req.url === '/api/config') {
        let status; try { status = await transcriber.request('status'); } catch { status = { available: false }; }
        return json(res, 200, { configured: status.available, reason: status.reason, streaming, backend: qwen ? 'qwen-mlx' : undefined, model: qwen ? `Qwen3-ASR · Apple GPU · ${env.QWEN_ASR_MODEL?.match(/Qwen3-ASR-(0\.6B|1\.7B)/)?.[1] || '1.7B'}` : streaming ? 'Korean Zipformer · 실시간 스트리밍' : (env.STT_BACKEND || env.WHISPER_BACKEND) === 'mlx' ? `MLX · Apple GPU · ${env.MLX_WHISPER_MODEL || 'whisper-turbo'}` : `faster-whisper · ${env.WHISPER_MODEL || 'large-v3-turbo'}`, summary: env.SUMMARY_MODE === 'extractive' ? 'extractive' : 'ollama', local: true });
      }
      if (['GET', 'HEAD'].includes(req.method) && files.has(req.url)) {
        const [file, mime] = files.get(req.url), content = await readFile(new URL(`./public/${file}`, import.meta.url));
        res.writeHead(200, { 'Content-Type': `${mime}; charset=utf-8` });
        return res.end(req.method === 'HEAD' ? undefined : content);
      }
      if (req.method !== 'POST' || !recordRoute && !['/api/session', '/api/transcribe', '/api/summary'].includes(req.url)) return json(res, 404, { error: 'Not found' });
      if (req.headers.origin !== `http://${host}` && req.headers.origin !== `https://${host}`) return json(res, 403, { error: 'Origin rejected' });
      if (!req.headers['content-type']?.startsWith('application/json')) return json(res, 415, { error: 'JSON required' });
      let raw = ''; for await (const chunk of req) { raw += chunk; if (Buffer.byteLength(raw) > (recordRoute ? 10000000 : 700000)) return json(res, 413, { error: 'Request too large' }); }
      let body; try { body = JSON.parse(raw); } catch { return json(res, 400, { error: 'Invalid JSON' }); }
      if (!body || typeof body !== 'object') return json(res, 400, { error: 'Invalid request' });
      if (recordRoute) {
        try { return json(res, 200, await records.save(recordRoute[1], body)); }
        catch (e) { return json(res, e.status || 500, { error: e.status ? e.message : '기록을 저장하지 못했습니다. 디스크 공간과 권한을 확인하세요.' }); }
      }
      if (req.url === '/api/session' || req.url === '/api/transcribe') {
        if (typeof body.glossary !== 'string' || body.glossary.length > 1500) return json(res, 400, { error: '용어는 1,500자 이내로 입력해 주세요.' });
        if (req.url === '/api/transcribe') {
          if (body.final !== undefined && typeof body.final !== 'boolean') return json(res, 400, { error: 'Invalid transcription mode' });
          if (typeof body.audio !== 'string' || body.audio.length > 640000 || !/^[A-Za-z0-9+/]*={0,2}$/.test(body.audio) || body.audio.length % 4) return json(res, 400, { error: 'Invalid PCM audio' });
          const audio = Buffer.from(body.audio, 'base64');
          if (streaming) {
            if (typeof body.session !== 'string' || !Number.isSafeInteger(body.sequence) || body.sequence < 0 || audio.length > 64000 || audio.length % 2 || (!audio.length && !body.final)) return json(res, 400, { error: 'Invalid streaming input' });
          } else if (audio.length < 3200 || audio.length > 480000 || audio.length % 2) return json(res, 400, { error: 'Invalid PCM audio length' });
        }
        if (sttBusy) return json(res, 429, { error: '음성 처리 중입니다. 다른 강의 탭을 종료해 주세요.' });
        sttBusy = true;
        try { const result = await transcriber.request(req.url === '/api/session' ? 'load' : 'transcribe', { audio: body.audio, glossary: body.glossary, final: body.final !== false, session: body.session, sequence: body.sequence }); return json(res, 200, result); }
        catch (error) { return json(res, 503, { error: error.message }); }
        finally { sttBusy = false; }
      }
      if (typeof body.transcript !== 'string' || !body.transcript.trim() || body.transcript.length > 7000 || typeof body.previous !== 'string' || body.previous.length > 6000) return json(res, 400, { error: 'Invalid summary input' });
      if (summaryBusy) return json(res, 429, { error: '요약 처리 중입니다. 잠시 후 다시 시도해 주세요.' });
      summaryBusy = true;
      try { json(res, 200, await localSummary(body, { env: { ...env, SUMMARY_MODE: body.block === true ? 'ollama' : body.live && streaming ? (env.LECTURE_SUMMARY_MODE || 'extractive') : env.SUMMARY_MODE }, fetcher })); } finally { summaryBusy = false; }
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
