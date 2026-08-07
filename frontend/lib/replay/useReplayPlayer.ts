"use client";

import { useEffect, useRef, useState } from "react";
import { computeView, drawScene, drawStrip } from "./draw";
import type { Frame, Replay, View } from "./types";

/**
 * Owns the replay's animation state (refs, not React state, for the 60fps-adjacent hot
 * path - see the original single-file implementation this was extracted from). Returns
 * DOM refs to attach to the canvas/strip/scrubber elements plus a small imperative API;
 * StageCanvas/PlaybackControls/RewardStrip are purely presentational around this.
 *
 * DPR-aware + ResizeObserver-driven: the canvas backing store is sized to
 * cssSize*devicePixelRatio and the 2D context is scaled once, so drawing code in draw.ts
 * always works in plain CSS pixels - fixes the original's fixed 900x584 bitmap (soft/blurry
 * on HiDPI, and never recomputed on resize).
 */
export function useReplayPlayer(replay: Replay | null) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stripRef = useRef<HTMLCanvasElement>(null);
  const scrubRef = useRef<HTMLInputElement>(null);
  const frameLabelRef = useRef<HTMLSpanElement>(null);

  const tlRef = useRef<Frame[]>([]);
  const curRef = useRef(0);
  const playingRef = useRef(false);
  const speedRef = useRef(2);
  const labelsRef = useRef(false);
  const viewRef = useRef<View | null>(null);
  const cssSizeRef = useRef({ w: 0, h: 0 });

  const [playing, setPlaying] = useState(false);
  const [frameCount, setFrameCount] = useState(0);

  // rebuild the flattened tick timeline whenever a new replay arrives
  useEffect(() => {
    if (!replay) {
      tlRef.current = [];
      curRef.current = 0;
      setFrameCount(0);
      return;
    }
    let gi = 0;
    tlRef.current = replay.steps.flatMap((st) =>
      st.ticks.map((bricks, j) => ({ st, bricks, base: st.base_ticks?.[j], gi: gi++ })),
    );
    curRef.current = 0;
    playingRef.current = false;
    setPlaying(false);
    setFrameCount(tlRef.current.length);
  }, [replay]);

  // DPR + resize: recompute the backing store and the world->pixel transform whenever the
  // canvas's CSS size changes (window resize, panel collapse, sidebar toggle, ...)
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !replay) return;
    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) return;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.round(rect.width * dpr);
      canvas.height = Math.round(rect.height * dpr);
      const ctx = canvas.getContext("2d");
      if (ctx) ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      cssSizeRef.current = { w: rect.width, h: rect.height };
      viewRef.current = computeView(replay.length, replay.n_courses, rect.width, rect.height);
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);
    return () => ro.disconnect();
  }, [replay]);

  // the strip canvas is simpler (no world transform) but still needs DPR + resize
  useEffect(() => {
    const strip = stripRef.current;
    if (!strip) return;
    const resize = () => {
      const rect = strip.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) return;
      const dpr = window.devicePixelRatio || 1;
      strip.width = Math.round(rect.width * dpr);
      strip.height = Math.round(rect.height * dpr);
      const ctx = strip.getContext("2d");
      if (ctx) ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(strip);
    return () => ro.disconnect();
  }, [replay]);

  // the single animation loop, mounted once - advances curRef when playing, always redraws
  // the current frame, and writes the scrubber/frame-label via refs (not state) to avoid a
  // React re-render on every one of 60 frames/sec
  //
  // The playhead must advance INDEPENDENTLY of viewRef (the 2D world->pixel transform).
  // viewRef is only set by the resize effect above, which bails on a 0x0 canvas - and the 2D
  // canvas IS 0x0 whenever ReplayViewer is in "3d" mode, since it hides it with `display:none`
  // rather than unmounting it. Gating this whole block on `view` used to freeze playback in
  // 3D: Play flipped the button to "Pause" but nothing moved, the frame label stayed "-", and
  // the reward strip never drew. `view` now guards ONLY the 2D scene draw; SceneCanvas's
  // useFrame reads the same curRef this loop advances, so both renderers share one playhead.
  useEffect(() => {
    let raf = 0;
    let lastT = 0;
    let lastStripStep = -1;
    const loop = (ts: number) => {
      const tl = tlRef.current;
      const dt = lastT ? Math.min(0.1, (ts - lastT) / 1000) : 0;
      lastT = ts;
      if (replay && tl.length) {
        if (playingRef.current) {
          curRef.current += 30 * speedRef.current * dt;
          if (curRef.current >= tl.length - 1) {
            curRef.current = tl.length - 1;
            playingRef.current = false;
            setPlaying(false);
          }
        }
        const ci = Math.min(Math.max(0, Math.floor(curRef.current)), tl.length - 1);

        // --- 2D stage: only when the transform exists (i.e. the canvas is actually visible)
        const view = viewRef.current;
        const canvas = canvasRef.current;
        if (view && canvas) {
          const ctx = canvas.getContext("2d");
          const { w, h } = cssSizeRef.current;
          if (ctx && w > 0) drawScene(ctx, w, h, view, replay, tl[ci], { labels: labelsRef.current });
        }

        // --- renderer-agnostic chrome (no world transform): strip, scrubber, frame label.
        // The strip only changes when the STEP changes, not every tick, so skip the redraw
        // (and its forced layout read) otherwise - this now runs in 3D mode too, where the
        // WebGL renderer is already competing for the main thread.
        const st = tl[ci].st.i;
        if (st !== lastStripStep) {
          const strip = stripRef.current;
          if (strip) {
            const ctx = strip.getContext("2d");
            const rect = strip.getBoundingClientRect();
            if (ctx && rect.width > 0) {
              drawStrip(ctx, rect.width, rect.height, replay, st);
              lastStripStep = st;
            }
          }
        }
        if (scrubRef.current && document.activeElement !== scrubRef.current) {
          scrubRef.current.value = String(ci);
        }
        if (frameLabelRef.current) frameLabelRef.current.textContent = `${ci + 1}/${tl.length}`;
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [replay]);

  const togglePlay = () => {
    const tl = tlRef.current;
    if (!tl.length) return;
    if (curRef.current >= tl.length - 1) curRef.current = 0;
    const np = !playingRef.current;
    playingRef.current = np;
    setPlaying(np);
  };

  const seek = (frameIndex: number) => {
    curRef.current = frameIndex;
    playingRef.current = false;
    setPlaying(false);
  };

  const setSpeed = (v: number) => {
    speedRef.current = v;
  };

  const setLabels = (v: boolean) => {
    labelsRef.current = v;
  };

  return {
    canvasRef, stripRef, scrubRef, frameLabelRef,
    playing, frameCount, togglePlay, seek, setSpeed, setLabels,
    // exposed for SceneCanvas (the 3D renderer): it reads the SAME playhead this hook
    // advances (tlRef/curRef) via its own useFrame instead of a second requestAnimationFrame
    // loop - only one visual is ever mounted at a time (see the 2D/3D toggle in
    // ReplayViewer), but the playhead is one shared source of truth either way.
    tlRef, curRef, labelsRef,
  };
}
