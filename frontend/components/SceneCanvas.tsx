"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import {
  ContactShadows, Environment, Html, Lightformer, OrbitControls, OrthographicCamera,
  PerspectiveCamera,
} from "@react-three/drei";
import * as THREE from "three";
import type { RefObject } from "react";
import type { Frame as ReplayFrame, Match, Replay, Target } from "@/lib/replay/types";
import {
  BRICK_DEPTH_MM, PERSPECTIVE_FOV_DEG, defaultCameraPosition, extrudedPolygon,
  orthographicFraming, unitBox,
} from "@/lib/replay/scene3d";
import {
  ARM_MARGIN_MM, COURSE_MM, GHOST_OPACITY, HARD_BODY_COLORS, NEXT_SLOT_OPACITY, PALETTE,
  brickColorRgb, brickFace, foldDeg, jitterSigned, type ViewMode,
} from "@/lib/replay/shared";
import { GrainOverlay } from "./GrainOverlay";

type PlayerRefs = {
  tlRef: RefObject<ReplayFrame[]>;
  curRef: RefObject<number>;
  labelsRef: RefObject<boolean>;
};

const CHALK_COLOR = new THREE.Color(PALETTE.chalk);

/** react-three-fiber replay renderer: extrudes the same 2D poses draw.ts consumes into
 * real 3D solids (a wall you can orbit, not a flat sprite), with lighting/shadows so
 * depth actually reads. Data contract is identical to the 2D canvas renderer - this is a
 * rendering-layer swap, not a new episode format; see draw.ts for the 2D counterpart and
 * lib/replay/shared.ts for the palette/color math both share.
 *
 * `mode` (as-built/inspect/drawing) is a SEPARATE axis from the 2D/3D renderer choice in
 * ReplayViewer - it swaps the color function (see shared.ts's brickColorRgb) and, in
 * "drawing", the camera projection (see CameraRig), but has nothing to do with which
 * renderer is mounted.
 *
 * `active` is false while the 2D stage is the visible one (ReplayViewer hides this with
 * `display:none` rather than unmounting it, so the R3F root - and its useFrame loops - would
 * otherwise keep rendering an invisible canvas at 60fps). Passed straight through as the
 * Canvas frameloop mode: "always" while visible, "never" while hidden. Don't use "demand" -
 * that needs explicit invalidate() calls this scene never makes. */
const DEFAULT_HEIGHT_CLASS = "h-[60vh] min-h-[360px] w-full md:h-[65vh]";

export function SceneCanvas({
  replay, tlRef, curRef, labelsRef, mode, active = true, autoRotate = false, focus, heightClassName,
}: {
  replay: Replay | null; mode: ViewMode; active?: boolean; autoRotate?: boolean;
  /** Crop the CAMERA to a sub-region (e.g. the Strike page zooming in on one arch's
   * opening) without touching anything actually rendered - the whole replay still draws,
   * only the framing changes. See scene3d.ts's defaultCameraPosition. */
  focus?: { centerX: number; span: number };
  /** Overrides the default full-viewer sizing (60-65vh) - for contexts that AREN'T the main
   * /replay stage, e.g. the Strike page's one-third-width grid panels or the Compare page's
   * side-by-side pair, where a viewport-height panel would be absurdly tall. */
  heightClassName?: string;
} & PlayerRefs) {
  const length = replay?.length ?? 3000;
  const nCourses = replay?.n_courses ?? 6;

  return (
    <div className={`relative overflow-hidden rounded-md ${heightClassName ?? DEFAULT_HEIGHT_CLASS}`}>
      <Canvas
        shadows
        frameloop={active ? "always" : "never"}
        gl={{ toneMapping: THREE.ACESFilmicToneMapping, toneMappingExposure: 1.15 }}
        className="!absolute inset-0"
      >
        <color attach="background" args={[PALETTE.bg]} />
        <fog attach="fog" args={[PALETTE.bg, 2000, 9000]} />
        <ambientLight intensity={0.25} />
        <directionalLight
          position={[length * 0.3, 2200, 1400]}
          intensity={1.1}
          castShadow
          shadow-mapSize={[1024, 1024]}
          shadow-radius={4}
          shadow-camera-left={-3000}
          shadow-camera-right={3000}
          shadow-camera-top={3000}
          shadow-camera-bottom={-500}
          shadow-camera-far={6000}
        />
        <hemisphereLight args={[PALETTE.bg, PALETTE.ground, 0.3]} />
        {/* Softened via the shadow's own `radius` (a standard PCF blur), NOT drei's
            SoftShadows - that patches THREE.ShaderChunk with a PCSS shader that assumes a
            shadow-map chunk shape THREE r185 changed (`unpackRGBAToDepth` no longer resolves
            at that injection point), producing a hard fragment-shader compile failure in
            every browser, not a headless-testing artifact - confirmed via a real GLSL
            compile error, not a driver quirk. */}
        {/* A virtual light rig baked into a cheap cubemap for image-based lighting - no HDR
            file (keeps the static export self-contained), just two soft panels: a big warm
            key from front-above (where clay picks up its specular falloff) and a cool dim
            rim from behind-left (edge definition against the dark ground). */}
        <Environment resolution={256} background={false}>
          <Lightformer form="rect" intensity={2.4} color="#fff2df" position={[length * 0.35, 2600, 2200]} scale={[3200, 2200, 1]} target={[length * 0.35, 0, 0]} />
          <Lightformer form="rect" intensity={0.5} color="#7fa0c9" position={[-1600, 900, -2600]} scale={[1800, 1800, 1]} />
        </Environment>
        <Ground length={length} />
        <ContactShadows position={[length / 2, 0.5, 0]} opacity={0.45} width={length + 1200} height={1400} blur={2.2} far={500} resolution={512} color="#000000" />
        {replay && <SceneContents replay={replay} tlRef={tlRef} curRef={curRef} labelsRef={labelsRef} mode={mode} />}
        {replay && <OutlierLabels tlRef={tlRef} curRef={curRef} labelsRef={labelsRef} mode={mode} />}
        <CameraRig length={length} nCourses={nCourses} mode={mode} autoRotate={autoRotate} focus={focus} />
      </Canvas>
      <GrainOverlay />
      <SceneHud replay={replay} tlRef={tlRef} curRef={curRef} active={active} mode={mode} />
    </div>
  );
}

/** Both cameras + OrbitControls, all derived from the same (length, nCourses, aspect, mode)
 * inputs, declared once here rather than split across the Canvas prop and an imperative
 * effect. This is a genuine simplification over the old approach (a manual useEffect
 * correcting a camera position the Canvas's `camera` prop could only set once): `position`,
 * `target`, `left/right/top/bottom` etc. are all normal reactive R3F props on Object3D-
 * derived elements, so they're re-applied every render exactly like `<mesh position={...}>`
 * is - no remount, no stale value, no manual `.update()` call (drei's OrbitControls already
 * calls `controls.update()` every frame on its own).
 *
 * "drawing" mode swaps to the orthographic elevation camera (see scene3d.ts's
 * orthographicFraming) - the honest architectural view this project's own plans are drawn
 * in - and disables orbit rotation (pan/zoom only), so it can't be nudged into a 3/4 angle
 * and stop being an elevation.
 *
 * `autoRotate` (used by the home page hero) drifts the camera azimuth continuously via
 * OrbitControls' own autoRotate, but yields the moment a visitor actually touches the
 * controls (paused on `onStart`, resumed a few seconds after `onEnd` - not "canceled
 * forever", since a visitor who lets go probably wants the ambient motion back) and is
 * disabled outright under `prefers-reduced-motion` or in "drawing" mode. */
export function CameraRig({
  length, nCourses, mode, autoRotate = false, focus,
}: {
  length: number; nCourses: number; mode: ViewMode; autoRotate?: boolean;
  focus?: { centerX: number; span: number };
}) {
  const { width, height } = useThree((s) => s.size);
  const aspect = width / Math.max(1, height);
  const heightMm = nCourses * COURSE_MM;
  const persp = focus
    ? defaultCameraPosition(focus.span, nCourses, aspect, false, focus.centerX)
    : defaultCameraPosition(length, nCourses, aspect);
  const ortho = orthographicFraming(length, heightMm, aspect);
  const orthoCenterX = focus?.centerX ?? length / 2;
  const target: [number, number, number] = mode === "drawing" ? [orthoCenterX, heightMm / 2, 0] : persp.target;

  const [reducedMotion, setReducedMotion] = useState(false);
  const [userActive, setUserActive] = useState(false);
  const resumeTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReducedMotion(mq.matches);
    const onChange = () => setReducedMotion(mq.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);

  const effectiveAutoRotate = autoRotate && !reducedMotion && !userActive && mode !== "drawing";

  return (
    <>
      <PerspectiveCamera makeDefault={mode !== "drawing"} position={persp.position} fov={PERSPECTIVE_FOV_DEG} near={10} far={40000} />
      <OrthographicCamera
        makeDefault={mode === "drawing"}
        manual
        position={[orthoCenterX, heightMm / 2, 3000]}
        left={ortho.left}
        right={ortho.right}
        top={ortho.top}
        bottom={ortho.bottom}
        near={10}
        far={6000}
      />
      <OrbitControls
        makeDefault
        target={target}
        enableRotate={mode !== "drawing"}
        autoRotate={effectiveAutoRotate}
        autoRotateSpeed={0.5}
        maxPolarAngle={Math.PI / 2 - 0.01}
        minDistance={200}
        maxDistance={8000}
        onStart={() => {
          clearTimeout(resumeTimer.current);
          setUserActive(true);
        }}
        onEnd={() => {
          resumeTimer.current = setTimeout(() => setUserActive(false), 4000);
        }}
      />
    </>
  );
}

export function Ground({ length }: { length: number }) {
  const w = length + 2000;
  return (
    <mesh position={[length / 2, -100, 0]} receiveShadow>
      <boxGeometry args={[w, 200, 800]} />
      <meshStandardMaterial color={PALETTE.ground} roughness={1} />
    </mesh>
  );
}

/** A small HTML overlay (not WebGL text) for the HUD strip - matches the 2D renderer's HUD
 * bar, updated imperatively via refs so it doesn't fight the canvas for 30fps re-renders.
 * Runs its own rAF (independent of the Canvas's frameloop, which `active` also gates) - kept
 * cheap enough that it just early-returns rather than being torn down while inactive. */
function SceneHud({
  replay, tlRef, curRef, active = true, mode,
}: { replay: Replay | null; active?: boolean; mode: ViewMode } & Pick<PlayerRefs, "tlRef" | "curRef">) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    let raf = 0;
    const tick = () => {
      const tl = tlRef.current;
      if (active && replay && ref.current && tl.length) {
        const ci = Math.min(Math.max(0, Math.floor(curRef.current)), tl.length - 1);
        const st = tl[ci].st;
        const isRobot = !!replay.robot;
        const tail = isRobot ? `moves ${st.moves ?? 0}   placed ${st.placements ?? 0}` : `waste ${st.waste ?? 0}`;
        const modeLabel = isRobot ? `${["place", "move ←", "move →"][st.mode ?? 0]}   ` : "";
        ref.current.textContent =
          `${replay._policy ?? ""}   ${replay.spec.n_modules}m × ${replay.spec.n_courses}c   ` +
          `step ${st.i + 1}/${replay.steps.length}   ${modeLabel}in-tol ${(st.frac_in_tol * 100).toFixed(0)}%   ` +
          `filled ${(st.frac_filled * 100).toFixed(0)}%   ${tail}   return ${st.return.toFixed(2)}   view ${mode}`;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [replay, tlRef, curRef, active, mode]);

  return (
    <div
      className="pointer-events-none absolute inset-x-0 top-0 flex h-8 items-center gap-3 px-3 font-mono text-[12px]"
      style={{ background: PALETTE.hudBg + "e6" }}
    >
      <span style={{ color: PALETTE.accent }}>Bricklaying with RL</span>
      <span ref={ref} style={{ color: PALETTE.hudText }} />
    </div>
  );
}

type OutlierLabel = { key: string; x: number; y: number; text: string };

/** mm deviation callouts, drawn ONLY for outliers (|dx| > TOL_MM) and only in "inspect" mode
 * - the old 2D-renderer-derived behavior labeled every matched brick (all 600 of them on a
 * full facade), which is exactly the clutter the rest of this redesign is undoing. There was
 * never a 3D equivalent before this (labelsRef was plumbed through but unused here) - this
 * adds one, deliberately narrow in scope.
 *
 * Driven by a rAF poll + a signature comparison (not a per-tick React re-render): the label
 * SET only changes when a brick is placed/matched, far less often than the 30fps playhead
 * advances, so `setLabels` only fires when the outlier set actually differs from last frame -
 * everything else in this scene stays ref-only for exactly this reason. */
function OutlierLabels({
  tlRef, curRef, labelsRef, mode,
}: { mode: ViewMode } & Pick<PlayerRefs, "tlRef" | "curRef" | "labelsRef">) {
  const [labels, setLabels] = useState<OutlierLabel[]>([]);

  useEffect(() => {
    let raf = 0;
    let lastSig = "";
    const tick = () => {
      const tl = tlRef.current;
      const showing = mode === "inspect" && labelsRef.current && tl.length > 0;
      if (showing) {
        const ci = Math.min(Math.max(0, Math.floor(curRef.current)), tl.length - 1);
        const frame = tl[ci];
        const byId = new Map(frame.bricks.map((b) => [b[4], b] as const));
        const next: OutlierLabel[] = [];
        for (const m of frame.st.matches ?? []) {
          if (m.in_tol) continue;
          const b = byId.get(m.brick_id);
          if (!b) continue;
          const [, h] = brickFace(b[3]);
          next.push({
            key: String(m.brick_id),
            x: b[0],
            y: b[1] + h / 2 + 14,
            text: `${m.dx >= 0 ? "+" : ""}${m.dx.toFixed(1)}`,
          });
        }
        const sig = next.map((l) => `${l.key}:${l.text}`).join("|");
        if (sig !== lastSig) {
          lastSig = sig;
          setLabels(next);
        }
      } else if (lastSig !== "") {
        lastSig = "";
        setLabels([]);
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [mode, labelsRef, tlRef, curRef]);

  return (
    <>
      {labels.map((l) => (
        <Html key={l.key} position={[l.x, l.y, 0]} center distanceFactor={800} style={{ pointerEvents: "none" }}>
          <span className="font-mono text-[11px]" style={{ color: PALETTE.label }}>
            {l.text}
          </span>
        </Html>
      ))}
    </>
  );
}

export function SceneContents({
  replay, tlRef, curRef, labelsRef, mode,
}: { replay: Replay; mode: ViewMode } & PlayerRefs) {
  const maxFlat = useMemo(
    () => Math.max(1, ...replay.steps.flatMap((st) => st.ticks.map((t) => t.filter((b) => b[3] !== 2).length))),
    [replay],
  );
  const maxVoussoir = useMemo(
    () => Math.max(0, ...replay.steps.flatMap((st) => st.ticks.map((t) => t.filter((b) => b[3] === 2).length))),
    [replay],
  );
  const maxHard = (replay.hard_bodies ?? []).length;

  const flatMeshRef = useRef<THREE.InstancedMesh>(null);
  const voussoirRefs = useRef<(THREE.Mesh | null)[]>([]);
  const hardRefs = useRef<(THREE.Mesh | null)[]>([]);
  const ghostRefs = useRef<(THREE.LineSegments | null)[]>([]);
  const gantryRef = useRef<THREE.Group>(null);
  const mastRef = useRef<THREE.Mesh>(null);
  const beamRef = useRef<THREE.Mesh>(null);
  const chassisRef = useRef<THREE.Mesh>(null);
  const wheelLRef = useRef<THREE.Mesh>(null);
  const wheelRRef = useRef<THREE.Mesh>(null);
  const toolRef = useRef<THREE.Mesh>(null);
  const gripperRef = useRef<THREE.Mesh>(null);
  const reachBandRef = useRef<THREE.Mesh>(null);

  const flatGeom = useMemo(() => unitBox(), []);
  const ghostGeom = useMemo(() => new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)), []);
  const wheelGeom = useMemo(() => new THREE.CylinderGeometry(30, 30, 14, 20), []);

  // Scratch objects reused every brick, every frame - up to ~600 bricks x 30fps, so a fresh
  // Matrix4/Quaternion/Vector3/Color per brick per frame would be a meaningful GC pressure
  // source; these are mutated in place instead.
  const scratch = useMemo(
    () => ({
      matrix: new THREE.Matrix4(),
      quat: new THREE.Quaternion(),
      euler: new THREE.Euler(),
      pos: new THREE.Vector3(),
      scale: new THREE.Vector3(),
      color: new THREE.Color(),
    }),
    [],
  );

  useFrame(() => {
    const tl = tlRef.current;
    if (!tl.length) return;
    const ci = Math.min(Math.max(0, Math.floor(curRef.current)), tl.length - 1);
    const frame = tl[ci];
    const bricks = frame.bricks;
    const isRobot = !!replay.robot;

    const matches: Match[] = frame.st.matches ?? [];
    const matchByTarget = new Map<number, Match>();
    const matchByBrick = new Map<number, Match>();
    for (const m of matches) {
      matchByTarget.set(m.target_id, m);
      matchByBrick.set(m.brick_id, m);
    }

    // --- ghost targets + next-slot highlight - always chalk, differing only in opacity, so
    // "not yet real" reads as one consistent hue rather than competing with the wall's own
    // material or measurement colors ---
    const cursor = frame.st.cursor;
    let nextTarget: Target | null = null;
    if (cursor != null) {
      for (const t of replay.targets) {
        if (t.course !== cursor || matchByTarget.has(t.tid)) continue;
        if (!nextTarget || t.x < nextTarget.x) nextTarget = t;
      }
    }
    replay.targets.forEach((t, i) => {
      const seg = ghostRefs.current[i];
      if (!seg) return;
      const filled = matchByTarget.has(t.tid);
      seg.visible = !filled;
      if (!filled) {
        seg.position.set(t.x, t.y, 0);
        seg.scale.set(t.w, t.h, BRICK_DEPTH_MM);
        const isNext = t === nextTarget;
        const mat = seg.material as THREE.LineBasicMaterial;
        mat.color.copy(CHALK_COLOR);
        mat.opacity = isNext ? NEXT_SLOT_OPACITY : GHOST_OPACITY;
        mat.linewidth = isNext ? 2 : 1;
      }
    });

    // --- static hard bodies: fade in at `appear`, and (a struck arch's centering only) fade
    // back out at `disappear` - see webviz/trajectory.py's _capture_hard for why this needed
    // an upper bound at all: without it, a struck centering that's genuinely gone from the
    // physics world kept rendering under every arch for the rest of the episode. ---
    (replay.hard_bodies ?? []).forEach((hb, i) => {
      const mesh = hardRefs.current[i];
      if (!mesh) return;
      const disappear = hb.disappear ?? Infinity;
      mesh.visible = hb.appear <= frame.gi && frame.gi < disappear;
      mesh.geometry = extrudedPolygon(hb.verts, BRICK_DEPTH_MM);
      const fill = mode === "drawing" ? PALETTE.chalk : HARD_BODY_COLORS[hb.kind]?.[0] ?? PALETTE.stone;
      (mesh.material as THREE.MeshStandardMaterial).color.set(fill);
    });

    // --- bricks: one InstancedMesh for the flat pool (600 draw calls -> 1), plus a small
    // pool of individually-shaped voussoir meshes (each ring has at most a couple dozen
    // distinct wedge shapes - not worth instancing). Color comes from shared.ts's
    // brickColorRgb, the single place all three view modes' rules live. ---
    let flatI = 0;
    let voussoirI = 0;
    const flatMesh = flatMeshRef.current;
    for (const b of bricks) {
      const [x, y, theta, kind, brickId] = b;
      if (kind === 2 && b.length === 6) {
        const mesh = voussoirRefs.current[voussoirI++];
        if (!mesh) continue;
        mesh.visible = true;
        mesh.geometry = extrudedPolygon(b[5], BRICK_DEPTH_MM);
        mesh.position.set(x, y, 0);
        mesh.rotation.set(0, 0, theta);
        mesh.scale.set(1, 1, 1);
        // a real structural voussoir is never scored by the flat-wall audit - "flight"
        // status is the honest read in every mode: as-built shows jittered clay, inspect
        // shows the neutral "no reading" grey (true both before AND after it settles, not
        // just "not yet audited"), drawing shows flat chalk.
        const [r, g, bl] = brickColorRgb(mode, brickId, "flight", null, null);
        (mesh.material as THREE.MeshStandardMaterial).color.setRGB(r, g, bl);
        continue;
      }
      if (!flatMesh) {
        flatI++;
        continue;
      }
      const [w, h] = brickFace(kind);
      const m = matchByBrick.get(brickId);
      const upright = foldDeg(theta) < 15;
      const status = m ? "matched" : upright ? "flight" : "stray";
      // a subtle per-course z stagger + rotation jitter - every brick sitting at exactly
      // z=0 read as an extruded sprite, not a built wall; real wythes aren't a plane.
      const courseIdx = Math.round((y - 30) / COURSE_MM);
      const zOff = (courseIdx % 2 === 0 ? 1 : -1) * 6;
      const rotJitter = (jitterSigned(brickId * 5 + 1) * 0.15 * Math.PI) / 180;
      scratch.pos.set(x, y, zOff);
      scratch.euler.set(0, 0, theta + rotJitter);
      scratch.quat.setFromEuler(scratch.euler);
      scratch.scale.set(w, h, BRICK_DEPTH_MM);
      scratch.matrix.compose(scratch.pos, scratch.quat, scratch.scale);
      flatMesh.setMatrixAt(flatI, scratch.matrix);
      const [r, g, bl] = brickColorRgb(mode, brickId, status, m?.dx ?? null, m?.in_tol ?? null);
      scratch.color.setRGB(r, g, bl);
      flatMesh.setColorAt(flatI, scratch.color);
      flatI++;
    }
    if (flatMesh) {
      flatMesh.count = flatI;
      flatMesh.instanceMatrix.needsUpdate = true;
      if (flatMesh.instanceColor) flatMesh.instanceColor.needsUpdate = true;
    }
    for (let i = voussoirI; i < voussoirRefs.current.length; i++) {
      const mesh = voussoirRefs.current[i];
      if (mesh) mesh.visible = false;
    }

    // --- mobile gantry - equipment, not a toy: steel body, one amber tool/gripper, the same
    // materials in every view mode (it's site infrastructure, not part of what's measured) ---
    if (isRobot && frame.base != null && gantryRef.current) {
      const reach = replay.robot!.reach;
      const beamY = COURSE_MM * replay.n_courses + ARM_MARGIN_MM;
      gantryRef.current.visible = true;
      gantryRef.current.position.set(frame.base, 0, 0);
      if (beamRef.current) {
        beamRef.current.position.set(0, beamY, 0);
        beamRef.current.scale.set(reach * 2, 12, 40);
      }
      if (mastRef.current) {
        mastRef.current.position.set(0, beamY / 2, 0);
        mastRef.current.scale.set(16, beamY, 16);
      }
      if (chassisRef.current) chassisRef.current.position.set(0, 20, 0);
      if (wheelLRef.current) wheelLRef.current.position.set(-40, 12, 0);
      if (wheelRRef.current) wheelRRef.current.position.set(40, 12, 0);
      if (reachBandRef.current) {
        reachBandRef.current.position.set(0, 400, -60);
        reachBandRef.current.scale.set(reach * 2, 800, 1);
      }
      const last = bricks[bricks.length - 1];
      if (last && toolRef.current && gripperRef.current) {
        const clampedX = Math.min(Math.max(last[0], frame.base - reach), frame.base + reach) - frame.base;
        const [, h] = brickFace(last[3]);
        const topY = last[3] === 2 ? last[1] : last[1] + h / 2;
        const toolHeight = Math.max(1, beamY - topY);
        toolRef.current.position.set(clampedX, (beamY + topY) / 2, 0);
        toolRef.current.scale.set(6, toolHeight, 6);
        gripperRef.current.position.set(clampedX, topY, 0);
        gripperRef.current.visible = true;
      } else if (gripperRef.current) {
        gripperRef.current.visible = false;
      }
    } else if (gantryRef.current) {
      gantryRef.current.visible = false;
    }
  });

  return (
    <group>
      <instancedMesh ref={flatMeshRef} args={[flatGeom, undefined, maxFlat]} castShadow receiveShadow frustumCulled={false}>
        <meshStandardMaterial roughness={0.85} />
      </instancedMesh>
      {Array.from({ length: maxVoussoir }, (_, i) => (
        <mesh key={`vou-${i}`} ref={(el) => (voussoirRefs.current[i] = el)} castShadow receiveShadow>
          <meshStandardMaterial color={PALETTE.clay} roughness={0.8} />
        </mesh>
      ))}
      {Array.from({ length: maxHard }, (_, i) => {
        const hb = replay.hard_bodies![i];
        const [fill] = HARD_BODY_COLORS[hb.kind] ?? [PALETTE.stone];
        return (
          <mesh key={`hard-${i}`} ref={(el) => (hardRefs.current[i] = el)} receiveShadow>
            <meshStandardMaterial color={fill} roughness={0.9} />
          </mesh>
        );
      })}
      {replay.targets.map((_, i) => (
        <lineSegments key={`ghost-${i}`} ref={(el) => (ghostRefs.current[i] = el)} geometry={ghostGeom}>
          <lineBasicMaterial color={PALETTE.chalk} transparent opacity={GHOST_OPACITY} />
        </lineSegments>
      ))}
      {replay.robot && (
        <group ref={gantryRef}>
          <mesh ref={beamRef} geometry={flatGeom} castShadow>
            <meshStandardMaterial color={PALETTE.robot} />
          </mesh>
          <mesh ref={mastRef} geometry={flatGeom}>
            <meshStandardMaterial color={PALETTE.robot} />
          </mesh>
          <mesh ref={chassisRef} castShadow>
            <boxGeometry args={[104, 30, 60]} />
            <meshStandardMaterial color={PALETTE.robotDark} />
          </mesh>
          <mesh ref={wheelLRef} geometry={wheelGeom} rotation={[Math.PI / 2, 0, 0]}>
            <meshStandardMaterial color={PALETTE.robot} />
          </mesh>
          <mesh ref={wheelRRef} geometry={wheelGeom} rotation={[Math.PI / 2, 0, 0]}>
            <meshStandardMaterial color={PALETTE.robot} />
          </mesh>
          <mesh ref={toolRef} geometry={flatGeom}>
            <meshStandardMaterial color={PALETTE.robotTool} />
          </mesh>
          <mesh ref={gripperRef}>
            <sphereGeometry args={[10, 12, 12]} />
            <meshStandardMaterial color={PALETTE.robotTool} />
          </mesh>
          <mesh ref={reachBandRef}>
            <planeGeometry args={[1, 1]} />
            <meshBasicMaterial color={PALETTE.robotTool} transparent opacity={0.08} depthWrite={false} />
          </mesh>
        </group>
      )}
    </group>
  );
}
