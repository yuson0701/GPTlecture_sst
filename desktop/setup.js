const install = document.getElementById('install'), notes = document.getElementById('notes'), status = document.getElementById('status'), log = document.getElementById('log');
window.lectureDesktop.onProgress(text => { log.textContent = (log.textContent + text).slice(-12000); log.scrollTop = log.scrollHeight; });
install.onclick = async () => {
  install.disabled = notes.disabled = true; status.textContent = '모델 설치 중입니다. 처음에는 시간이 걸릴 수 있어요.';
  try { await window.lectureDesktop.setup(); }
  catch (e) { status.textContent = e.message; install.disabled = notes.disabled = false; install.textContent = '설치 다시 시도'; }
};
notes.onclick = async () => {
  notes.disabled = install.disabled = true;
  try { await window.lectureDesktop.notes(); } catch (e) { status.textContent = e.message; notes.disabled = install.disabled = false; }
};
