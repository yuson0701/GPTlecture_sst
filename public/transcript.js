export class Transcript {
  constructor() { this.items = new Map(); }
  set(id, values) { this.items.set(id, { id, text: '', final: false, seconds: 0, ...this.items.get(id), ...values }); }
  ordered() { return [...this.items.values()].sort((a, b) => a.seconds - b.seconds); }
}
export const timestamp = seconds => `${Math.floor(seconds / 60).toString().padStart(2, '0')}:${Math.floor(seconds % 60).toString().padStart(2, '0')}`;
