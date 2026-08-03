// Geometry helpers for the 3D replay scene (SceneCanvas). World units stay in mm, exactly
// like the 2D renderer's View/px/py transform - this is a rendering-layer swap, not a new
// coordinate system: three.js X = world x, three.js Y = world y (height), three.js Z is the
// new depth axis a 2D sim never had, given a fixed thickness so bricks read as solids.

import * as THREE from "three";

export const BRICK_DEPTH_MM = 100; // a real brick's depth - the wall's "into the screen" extent

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
 * build; OrbitControls take it from here. */
export function defaultCameraPosition(lengthMm: number, nCoursesOrHeightMm: number, isHeightMm = false): {
  position: [number, number, number];
  target: [number, number, number];
} {
  const heightMm = isHeightMm ? nCoursesOrHeightMm : nCoursesOrHeightMm * 60;
  const cx = lengthMm / 2;
  const cy = heightMm / 2;
  const span = Math.max(lengthMm, heightMm * 2, 800);
  return {
    position: [cx - span * 0.15, heightMm * 0.75 + span * 0.35, span * 0.9],
    target: [cx, cy * 0.6, 0],
  };
}
