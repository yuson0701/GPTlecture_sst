const { contextBridge, ipcRenderer } = require('electron');
contextBridge.exposeInMainWorld('lectureDesktop', {
  setup: () => ipcRenderer.invoke('desktop:setup'),
  notes: () => ipcRenderer.invoke('desktop:notes'),
  onProgress: callback => { ipcRenderer.on('desktop:progress', (_event, text) => callback(text)); },
});
