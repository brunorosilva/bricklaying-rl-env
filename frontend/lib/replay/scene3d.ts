// Geometry helpers for the 3D replay scene (SceneCanvas). World units stay in mm, exactly
// like the 2D renderer's View/px/py transform - this is a rendering-layer swap, not a new
// coordinate system: three.js X = world x, three.js Y = world y (height), three.js Z is the
// new depth axis a 2D sim never had, given a fixed thickness so bricks read as solids.

import * as THREE from "three";

export const BRICK_DEPTH_MM = 100; // a real brick's depth - the wall's "into the screen" extent
export const PERSPECTIVE_FOV_DEG = 42; // vertical FOV - matches the camera's old fixed fov

let sharedUnitBox: THREE.BoxGeometry | null = null;

/** A TRUE unit box (1x1x1) - every dimension, including depth, is applied entirely via
 * mesh.scale at the call site (e.g. `.scale.set(w, h, BRICK_DEPTH_MM)`), so this stays a
 * single shared geometry reused for every flat brick/gantry part rather than one allocation
 * per shape. (A previous version baked BRICK_DEPTH_MM into the geometry itself AND scaled
 * by it again at the call site, squaring the depth - fixed by keeping this genuinely unit.) */
export function unitBox(): THREE.BoxGeometry {
  if (!sharedUnitBox) sharedUnitBox = new THREE.BoxGeometry(1, 1, 1);
  return sharedUnitBox;
}

const extrudeCache = new Map<string, THREE.ExtrudeGeometry>();

/** Extrude an arbitrary 2D polygon (local, centroid-relative points, as sent for voussoirs
 * and hard bodies) into a solid with depth. Cached by its own point data + depth: the same
 * wedge shape recurs every tick of an episode (only its transform moves), and there are at
 * most a couple dozen distinct shapes in any one replay (voussoirs in a ring, a handful of
 * hard bodies) - well worth memoizing rather than rebuilding 30x/sec. */
export function extrudedPolygon(points: [number, number][], depth: number = BRICK_DEPTH_MM): THREE.ExtrudeGeometry {
  const key = `${depth}:${points.map(([x, y]) => `${x},${y}`).join(";")}`;
  let g = extrudeCache.get(key);
  if (g) return g;
  const shape = new THREE.Shape(points.map(([x, y]) => new THREE.Vector2(x, y)));
  g = new THREE.ExtrudeGeometry(shape, { depth, bevelEnabled: false, curveSegments: 1 });
  g.translate(0, 0, -depth / 2); // center on z=0, matching unitBox's centered depth
  extrudeCache.set(key, g);
  return g;
}

/** Default camera framing for a wall of the given length/height - a three-quarter
 * architectural-rendering angle, close enough to be legible, far enough to see the whole
 * build. Aspect-aware: a 16x16 facade in a wide container needs a very different pull-back
 * distance than the same facade in a narrow one, so the distance is derived from the live
 * canvas aspect + the camera's own (vertical) FOV, not a fixed heuristic that assumed some
 * average aspect ratio. OrbitControls take it from here after the initial frame. */
export function defaultCameraPosition(
  lengthMm: number, nCoursesOrHeightMm: number, aspect: number = 16 / 9, isHeightMm = false,
  centerXMm?: number,
): {
  position: [number, number, number];
  target: [number, number, number];
} {
  const heightMm = isHeightMm ? nCoursesOrHeightMm : nCoursesOrHeightMm * 60;
  // centerXMm lets a caller frame a SUB-region (e.g. the Strike page cropping in on one
  // arch's opening) without moving or resizing anything actually rendered - only the camera
  // changes; `lengthMm` still controls the fitted SPAN in that case, not the true wall length.
  const cx = centerXMm ?? lengthMm / 2;
  const cy = heightMm / 2;
  const vFov = (PERSPECTIVE_FOV_DEG * Math.PI) / 180;
  const hFov = 2 * Math.atan(Math.tan(vFov / 2) * aspect);
  const margin = 1.2; // headroom so the wall doesn't touch the frame edges
  // distance needed to fit the whole length horizontally, and the whole height vertically -
  // the three-quarter angle foreshortens length a little, which `margin` covers rather than
  // a precise projection (OrbitControls lets a visitor correct the last few percent anyway).
  const distForWidth = (lengthMm * margin) / 2 / Math.tan(hFov / 2);
  const distForHeight = (heightMm * margin) / 2 / Math.tan(vFov / 2);
  const dist = Math.max(distForWidth, distForHeight, 800);
  return {
    position: [cx - dist * 0.18, heightMm * 0.7 + dist * 0.28, dist * 0.92],
    target: [cx, cy * 0.6, 0],
  };
}

/** Frustum for the "drawing" mode's orthographic elevation camera - the honest
 * architectural view (a true elevation, dead-on, not a 3/4 render), sized to fit the wall
 * plus a margin at the given aspect. Position the camera on +Z looking at -Z (see
 * SceneCanvas's DrawingCamera) and these become its left/right/top/bottom frustum planes. */
export function orthographicFraming(
  lengthMm: number, heightMm: number, aspect: number, marginMm = 400,
): { left: number; right: number; top: number; bottom: number } {
  const worldW = lengthMm + marginMm * 2;
  const worldH = heightMm + marginMm * 2;
  let halfW = worldW / 2;
  let halfH = worldH / 2;
  if (halfW / halfH > aspect) halfH = halfW / aspect;
  else halfW = halfH * aspect;
  return { left: -halfW, right: halfW, top: halfH, bottom: -halfH };
}
