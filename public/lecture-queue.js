// Final audio lives in IndexedDB until its transcript has been saved.
// Only metadata and the current decode are kept in memory.
export class AudioSpool {
  static async open(recordId) {
    const db = await new Promise((resolve, reject) => {
      const request = indexedDB.open('lecture-audio-recovery', 1);
      request.onupgradeneeded = () => {
        const store = request.result.createObjectStore('clips', { keyPath: ['recordId', 'id'] });
        store.createIndex('recordId', 'recordId');
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
      request.onblocked = () => reject(new Error('음성 저장소를 열 수 없습니다. 다른 강의 탭을 닫아 주세요.'));
    });
    db.onversionchange = () => db.close();
    return new AudioSpool(db, recordId);
  }
  constructor(db, recordId) { this.db = db; this.recordId = recordId; }
  async transaction(mode, action) {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('clips', mode);
      let result;
      tx.oncomplete = () => resolve(result);
      tx.onerror = tx.onabort = () => reject(tx.error || new Error('음성 저장 실패'));
      action(tx.objectStore('clips'), value => { result = value; });
    });
  }
  put(item) { return this.transaction('readwrite', store => store.put({ ...item, recordId: this.recordId })); }
  get(id) {
    return this.transaction('readonly', (store, done) => {
      const request = store.get([this.recordId, id]); request.onsuccess = () => done(request.result);
    });
  }
  delete(id) { return this.transaction('readwrite', store => store.delete([this.recordId, id])); }
  list() {
    return this.transaction('readonly', (store, done) => {
      const items = [];
      const cursor = store.index('recordId').openCursor(IDBKeyRange.only(this.recordId));
      cursor.onsuccess = () => {
        if (!cursor.result) { done(items.sort((a, b) => a.seconds - b.seconds)); return; }
        const { audio, recordId, ...meta } = cursor.result.value;
        items.push(meta); cursor.result.continue();
      };
    });
  }
}

export class LectureQueue {
  constructor(process, onChange, storage) {
    this.process = process; this.onChange = onChange; this.storage = storage;
    this.items = []; this.failures = []; this.current = null;
    this.running = false; this.failed = null; this.writes = Promise.resolve(); this.task = null;
  }
  get waiting() { return this.items.filter(item => item !== this.current); }
  async restore() {
    // An interrupted decode is offered for explicit retry, never silently lost.
    this.failures = await this.storage.list(); this.onChange();
  }
  enqueue(item) {
    if (item.final === false) {
      if (this.failed || this.failures.length || this.items.some(x => x.final !== false)) return true;
      this.items = this.items.filter(x => x === this.current);
      this.items.push(item); this.onChange(); void this.pump(); return true;
    }
    this.items = this.items.filter(x => x === this.current || x.final !== false);
    const { audio, ...meta } = item;
    // ready=false prevents a silence summary from overtaking an uncommitted write.
    const entry = { ...meta, ready: false };
    this.items.push(entry);
    this.writes = this.writes.then(async () => {
      try { await this.storage.put(item); entry.ready = true; }
      catch (error) {
        entry.audio = audio; // Retain this one clip if disk/quota fails.
        this.failed = new Error('음성 임시 저장 실패: ' + error.message);
      }
      this.onChange(); void this.pump();
    });
    this.onChange(); return true;
  }
  async pump() {
    if (this.running || this.failed || !this.items.length || this.items[0].ready === false) return;
    this.running = true;
    this.task = (async () => {
      try {
        while (this.items.length && !this.failed && this.items[0].ready !== false) {
          const entry = this.current = this.items[0];
          if (entry.final === false) {
            try { await this.process(entry); } catch { /* Drafts are disposable. */ }
          } else {
            let clip;
            try {
              clip = await this.storage.get(entry.id);
              if (!clip) throw new Error('보관한 음성 구간을 찾을 수 없습니다.');
            } catch (error) { this.failed = error; break; }
            try { await this.process(clip); }
            catch (error) {
              this.failures.push({ ...entry, error: error.message });
              this.items.shift(); this.current = null; this.onChange();
              continue; // One bad decode must not stop capture or block later clips.
            }
            try { await this.storage.delete(entry.id); }
            catch (error) { this.failed = error; break; }
          }
          this.items.shift(); this.current = null; this.onChange();
        }
      } finally { this.current = null; this.running = false; this.onChange(); }
    })();
    return this.task;
  }
  async settle() {
    // A final chunk may still be committing while microphone.stop() returns.
    do {
      const writes = this.writes; await writes;
      await this.pump(); if (this.task) await this.task;
      if (writes === this.writes && !this.running) break;
    } while (true);
  }
  async retry() {
    await this.settle();
    this.failed = null;
    for (const entry of this.items.filter(x => x.ready === false)) {
      try {
        await this.storage.put(entry); delete entry.audio; entry.ready = true;
      } catch (error) { this.failed = error; this.onChange(); return; }
    }
    this.items.push(...this.failures.splice(0).map(({ error, ...item }) => ({ ...item, ready: true })));
    this.items.sort((a, b) => a.seconds - b.seconds);
    await this.pump(); this.onChange();
  }
}
