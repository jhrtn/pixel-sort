import test from 'node:test';
import assert from 'node:assert/strict';
import { CameraCycle } from './camera-cycle.mjs';

test('live motion alternates with held frames on an irregular clock', () => {
  const cycle = new CameraCycle(() => .5);
  cycle.reset(100);
  assert.equal(cycle.tick(2999), false);
  assert.equal(cycle.tick(3000), true);
  assert.equal(cycle.tick(7299), true);
  assert.equal(cycle.tick(7300), false);
});
test('a drag protects the captured frame and leaves time to look after release', () => {
  const cycle = new CameraCycle(() => 0);
  cycle.grab();
  assert.equal(cycle.tick(30000), true);
  cycle.release(30000);
  assert.equal(cycle.tick(32399), true);
  assert.equal(cycle.tick(32400), false);
});
test('tap pins a frame until a second tap, including across a drag', () => {
  const cycle = new CameraCycle(() => 0);
  cycle.toggle(0);
  assert.equal(cycle.tick(90000), true);
  cycle.grab(); cycle.release(90000);
  assert.equal(cycle.tick(100000), true);
  cycle.toggle(100000);
  assert.equal(cycle.tick(100001), false);
  assert.equal(cycle.tick(102000), true);
});
test('reset clears a pinned or interrupted session', () => {
  const cycle = new CameraCycle(() => 0);
  cycle.toggle(0); cycle.grab(); cycle.reset(1000);
  assert.equal(cycle.pinned, false);
  assert.equal(cycle.dragging, false);
  assert.equal(cycle.tick(1000), false);
});
