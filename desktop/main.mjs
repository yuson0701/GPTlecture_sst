import { app, BrowserWindow, dialog, ipcMain, Menu, session, shell, systemPreferences } from 'electron';
import { spawn } from 'node:child_process';
import { mkdir, readFile, readdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { join } from 'node:path';
import { createApp } from '../server.mjs';
import { RecordStore } from '../local/records.mjs';
import { runtimePaths, runtimeEnv, trustedPage, usableModel } from './runtime.mjs';
const root = fileURLToPath(new URL('..', import.meta.url));
const setupUrl = pathToFileURL(join(root, 'desktop/setup.html')).href;
let window, server, origin = '', installer = null, quitting = false;
app.setName('Lecture Note');
app.setPath('userData', join(app.getPath('appData'), 'Lecture Note'));
if (!app.requestSingleInstanceLock()) app.quit();
else {
  app.on('second-instance', () => { if (window?.isMinimized()) window.restore(); window?.focus(); });
  await app.whenReady();
  const paths = runtimePaths({ packaged: app.isPackaged, resources: process.resourcesPath, root, userData: app.getPath('userData') });
  await mkdir(app.getPath('userData'), { recursive: true });
  const checkSender = event => {
    if (event.sender !== window?.webContents || !trustedPage(event.senderFrame?.url, origin, setupUrl)) throw Error('Untrusted page');
  };
  async function openNotes() {
    const config = JSON.parse(await readFile(paths.config, 'utf8'));
    if (!usableModel(config)) throw Error('모델 파일을 찾을 수 없습니다. 모델 설치를 다시 실행하세요.');
    if (!server) {
      server = createApp({ env: runtimeEnv(paths, config) });
      await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
      origin = `http://127.0.0.1:${server.address().port}`;
    }
    await window.loadURL(origin);
  }
  ipcMain.handle('desktop:setup', async event => {
    checkSender(event);
    if (event.senderFrame.url !== setupUrl || installer) throw Error('Setup is already running or unavailable.');
    if (process.platform !== 'darwin' || process.arch !== 'arm64') throw Error('이 앱은 Apple Silicon Mac용입니다.');
    if (!existsSync(paths.python)) throw Error('앱에 Python 실행 환경이 없습니다. 앱을 다시 다운로드하세요.');
    const env = runtimeEnv(paths); delete env.HF_HUB_OFFLINE;
    await new Promise((resolve, reject) => {
      const child = spawn(paths.python, ['-u', join(root, 'desktop/setup_model.py'), '--config', paths.config], { cwd: root, env, stdio: ['ignore', 'pipe', 'pipe'] });
      installer = child;
      const output = data => { if (!window?.isDestroyed()) window.webContents.send('desktop:progress', data.toString().slice(-4000)); };
      child.stdout.on('data', output); child.stderr.on('data', output);
      child.once('error', e => { installer = null; reject(e); });
      child.once('exit', code => { installer = null; code === 0 ? resolve() : reject(Error('모델 설치에 실패했습니다. 아래 로그를 확인하고 다시 시도하세요. 다운로드한 파일은 재사용됩니다.')); });
    });
    await openNotes(); return true;
  });
  ipcMain.handle('desktop:notes', async event => {
    checkSender(event);
    if (installer) throw Error('설치가 끝날 때까지 기다려 주세요.');
    if (!server) {
      // Allow the library/demo without model downloads; recording stays unavailable.
      server = createApp({ env: runtimeEnv(paths) });
      await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
      origin = `http://127.0.0.1:${server.address().port}`;
    }
    await window.loadURL(origin);
  });
  session.defaultSession.setPermissionCheckHandler((contents, permission, requestingOrigin, details = {}) =>
    contents === window?.webContents && permission === 'media' && details.mediaType !== 'video' && requestingOrigin === origin);
  session.defaultSession.setPermissionRequestHandler(async (contents, permission, callback, details) => {
    try {
      if (contents !== window?.webContents || permission !== 'media' || !origin || new URL(details.requestingUrl).origin !== origin || details.mediaTypes?.includes('video')) return callback(false);
      callback(process.platform !== 'darwin' || await systemPreferences.askForMediaAccess('microphone'));
    } catch { callback(false); }
  });
  const makeWindow = async () => {
    window = new BrowserWindow({ width: 1320, height: 900, minWidth: 390, minHeight: 600, title: 'Lecture Note', backgroundColor: '#f7f7f8',
      webPreferences: { preload: join(root, 'desktop/preload.cjs'), nodeIntegration: false, contextIsolation: true, sandbox: true } });
    window.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));
    window.webContents.on('will-navigate', (event, url) => { if (!trustedPage(url, origin, setupUrl)) event.preventDefault(); });
    window.webContents.on('will-prevent-unload', async event => {
      const { response } = await dialog.showMessageBox(window, { type: 'question', buttons: ['계속 기록하기', '종료'], defaultId: 0, cancelId: 0,
        message: '녹음 중이거나 저장하지 못한 내용이 있습니다.', detail: '종료하면 처리하지 못한 음성과 저장 전 변경 사항이 사라질 수 있습니다.' });
      // Electron requires synchronous preventDefault to allow an unload; use destroy only after explicit confirmation.
      if (response === 1) { quitting = true; window.destroy(); app.quit(); }
    });
    if (process.env.LECTURE_DESKTOP_SMOKE === '1') {
      await window.loadURL(setupUrl);
      const title = await window.webContents.executeJavaScript('document.title');
      console.log(`DESKTOP_SMOKE_OK ${title}`); quitting = true; app.quit(); return;
    }
    try { if (!existsSync(paths.python)) throw Error('Missing Python'); await openNotes(); }
    catch { await window.loadURL(setupUrl); }
  };
  Menu.setApplicationMenu(Menu.buildFromTemplate([
    { label: 'Lecture Note', submenu: [{ role: 'about' }, { type: 'separator' }, { role: 'quit' }] },
    { label: '파일', submenu: [
      { label: '모델 설치 / 복구', click: async () => {
        if (origin && !await window.webContents.executeJavaScript('Boolean(window.lectureCanLeave?.())')) {
          await dialog.showMessageBox(window, { message: '먼저 녹음을 종료하고 저장이 끝날 때까지 기다려 주세요.' }); return;
        }
        const { response } = await dialog.showMessageBox(window, { buttons: ['취소', '설치 화면 열기'], defaultId: 0, cancelId: 0, message: '녹음을 종료하고 저장한 뒤 설치 화면을 여세요.' });
        if (response === 1) { server?.close(); server?.closeAllConnections(); server = null; origin = ''; await window.loadURL(setupUrl); }
      } },
      { label: '저장된 강의 폴더 열기', click: async () => { await mkdir(paths.records, { recursive: true }); await shell.openPath(paths.records); } },
      { label: '기존 강의 기록 가져오기…', click: async () => {
        const selection = await dialog.showOpenDialog(window, { title: '기존 프로젝트의 data/lectures 폴더를 선택하세요', properties: ['openDirectory'] });
        if (selection.canceled) return;
        const store = new RecordStore(paths.records); let added = 0, skipped = 0;
        try {
          for (const name of await readdir(selection.filePaths[0])) {
            if (!/^[a-f0-9-]{36}\.json$/.test(name)) continue;
            const id = name.slice(0, -5);
            try { await store.get(id); skipped++; continue; } catch (e) { if (e.status !== 404) throw e; }
            const record = JSON.parse(await readFile(join(selection.filePaths[0], name), 'utf8'));
            await store.save(id, { ...record, revision: 0 }); added++;
          }
          await dialog.showMessageBox(window, { message: `${added}개 기록을 가져왔습니다. 기존 기록 ${skipped}개는 유지했습니다.`, detail: '나의 강의 기록 버튼을 눌러 목록을 새로 불러오세요.' });
        } catch (e) { await dialog.showMessageBox(window, { type: 'error', message: '일부 기록을 가져오지 못했습니다.', detail: `${added}개 가져오기 완료. ${e.message}` }); }
      } },
      { label: 'Ollama 다운로드', click: () => shell.openExternal('https://ollama.com/download/mac') },
    ] }, { role: 'editMenu' }, { role: 'viewMenu' }, { role: 'windowMenu' },
  ]));
  await makeWindow();
  app.on('activate', () => { if (BrowserWindow.getAllWindows().length === 0) void makeWindow(); });
  app.on('window-all-closed', () => app.quit());
  app.on('before-quit', event => {
    if (!quitting && installer) {
      event.preventDefault();
      void dialog.showMessageBox(window, { buttons: ['계속 설치', '설치 중단 후 종료'], defaultId: 0, cancelId: 0, message: '모델을 설치하고 있습니다.' }).then(({ response }) => {
        if (response === 1) { quitting = true; installer?.kill(); app.quit(); }
      });
    }
  });
  app.on('will-quit', () => { installer?.kill(); server?.close(); server?.closeAllConnections(); });
}
