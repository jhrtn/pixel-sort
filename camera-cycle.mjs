// Pure timing state: media capture happens in the renderer, not in this clock.
export class CameraCycle {
  constructor(random = Math.random) { this.random = random; this.reset(0); }
  reset(now) {
    this.held = false;
    this.pinned = false;
    this.dragging = false;
    this.next = now + this.duration(false);
  }
  duration(held) { return held ? 3200 + this.random() * 2200 : 2000 + this.random() * 1800; }
  tick(now) {
    if (!this.dragging && !this.pinned && now >= this.next) {
      this.held = !this.held;
      this.next = now + this.duration(this.held);
    }
    return this.held;
  }
  grab() { this.dragging = true; this.held = true; }
  release(now) { this.dragging = false; this.next = now + 2400; }
  toggle(now) {
    this.pinned = !this.pinned;
    this.held = this.pinned;
    this.next = now + this.duration(false);
  }
}
