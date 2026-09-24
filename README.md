# Pixel Sort

Full-screen camera pixel sorting. One start button, then only the image.

```sh
python3 -m http.server 8081 --bind 127.0.0.1
```

Open http://localhost:8081 in a WebGPU browser. No dependencies or build step.

The camera alternates between 2–3.8 seconds of live motion and 3.2–5.4 seconds holding a captured frame. Every automatic capture chooses a new sort. The effect grows into the held frame and recedes when motion returns.

- **Drag:** hold the current frame and pull its pixels. Horizontal and vertical movement affect threshold, span length, direction and effect strength. The frame stays held for 2.4 seconds after release.
- **Tap / Space:** pin the frame indefinitely; tap again to resume the cycle. With an image, tap changes the effect.
- **R:** new effect. **S:** save PNG. **Z / Cmd/Ctrl Z:** undo parameters.
- **O:** open an image. **C:** return to the camera. Dropped images work too.

A short interaction hint disappears after startup. There are no panels, labels or persistent controls on the image. A generated signal is available before camera access. Camera access is explicit, requires localhost or HTTPS, and is released when switching to an image. Processing stays in the browser.

Exports capture the viewport's cover-fit crop, without UI. Resolution follows viewport size and device pixel ratio (capped at 2). The sorting work texture caps line length at 2048 pixels.

## Checks

```sh
node --check app.js
node --test camera-cycle.test.mjs
```

The clock tests cover automatic cycling, dragging, pinned frames and reset. Browser checked: GPU startup, minimal UI and interaction with the generated source. Live camera capture and downloaded PNG contents need a device check.
