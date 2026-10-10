import { test } from 'node:test';
import assert from 'node:assert/strict';
import { nearestLecture } from '../public/lecture-picker.js';
const rows = Array.from({ length: 8 }, (_, i) => ({ id: String(i + 1), period: String(i + 1), date: '2026-10-12' }));
const at = time => new Date(`2026-10-12T${time}:00+09:00`);
test('all eight Korean class start times select their scheduled lecture', () => {
  ['08:30', '09:30', '10:40', '11:40', '13:30', '14:30', '15:30', '16:30'].forEach((time, i) => {
    assert.equal(nearestLecture(rows, at(time)).id, String(i + 1));
  });
});
test('closest start uses actual gaps, earlier ties, and Seoul dates across midnight', () => {
  assert.equal(nearestLecture(rows, at('10:04')).id, '2');
  assert.equal(nearestLecture(rows, at('10:05')).id, '2');
  assert.equal(nearestLecture(rows, at('10:06')).id, '3');
  assert.equal(nearestLecture(rows, at('12:36')).id, '5');
  assert.equal(nearestLecture(rows, at('06:00')).id, '1');
  assert.equal(nearestLecture(rows, at('22:00')).id, '8');
  assert.equal(nearestLecture(rows, new Date('2026-10-11T23:30:00Z')).id, '1');
  assert.equal(nearestLecture(rows, new Date('2026-10-12T15:00:00Z')).id, '8');
  assert.equal(nearestLecture([], at('09:30')), null);
});
test('multi-period classes match each covered start; malformed periods are ignored', () => {
  const ranged = [{ ...rows[0], period: '1-3' }, rows[3], { ...rows[4], period: '제5교시, 6교시' }];
  assert.equal(nearestLecture(ranged, at('10:40')).id, '1');
  assert.equal(nearestLecture(ranged, at('14:30')).id, '5');
  for (const period of ['0', '9', '1-10', '3-1', 'unknown', '']) {
    assert.equal(nearestLecture([{ ...rows[0], period }], at('08:30')), null);
  }
});

test('falls back across days and compares tomorrow against past lectures', () => {
  const lectures = [rows[0], { ...rows[0], id: 'tomorrow', date: '2026-10-13' }];
  assert.equal(nearestLecture(lectures, at('23:00')).id, 'tomorrow');
  assert.equal(nearestLecture(lectures, new Date('2026-10-11T12:00:00+09:00')).id, '1');
  assert.equal(nearestLecture(lectures, new Date('2026-10-14T12:00:00+09:00')).id, 'tomorrow');
  assert.equal(nearestLecture([{ ...rows[0], date: '2026-02-30' }], at('09:00')), null);
});
