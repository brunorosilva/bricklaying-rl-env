"use client";

import { useEffect, useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import type { RefObject } from "react";
import type { Frame as ReplayFrame, Match, Replay, Target } from "@/lib/replay/types";
import { BRICK_DEPTH_MM, defaultCameraPosition, extrudedPolygon, unitBox } from "@/lib/replay/scene3d";
import { ARM_MARGIN_MM, COURSE_MM, HARD_BODY_COLORS, PALETTE, brickFace, foldDeg, qualityColorRgb } from "@/lib/replay/shared";

type PlayerRefs = {
  tlRef: RefObject<ReplayFrame[]>;
  curRef: RefObject<number>;
  labelsRef: RefObject<boolean>;
};

const GHOST_COLOR = new THREE.Color(PALETTE.ghost);
const NEXT_SLOT_COLOR = new THREE.Color(PALETTE.nextSlot);
const MORTAR_COLOR = new THREE.Color(PALETTE.mortar);
const STRAY_COLOR = new THREE.Color(PALETTE.stray);
const TERRACOTTA_COLOR = new THREE.Color(PALETTE.terracotta);
const VOUSSOIR_COLOR = new THREE.Color(PALETTE.voussoirFace);

/** react-three-fiber replay renderer: extrudes the same 2D poses draw.ts consumes into
 * real 3D solids (a wall you can orbit, not a flat sprite), with lighting/shadows so
 * depth actually reads. Data contract is identical to the 2D canvas renderer - this is a
 * rendering-layer swap, not a new episode format; see draw.ts for the 2D counterpart and
 * lib/replay/shared.ts for the palette/color math both share. */
export function SceneCanvas({ replay, tlRef, curRef, labelsRef }: { replay: Replay | null } & PlayerRefs) {
  const cam = useMemo(
    () => defaultCameraPosition(replay?.length ?? 3000, replay?.n_courses ?? 6),
    [replay?.length, replay?.n_courses],
  );

  return (
    <div className="relative h-[60vh] min-h-[360px] w-full overflow-hidden rounded-md md:h-[65vh]">
      <Canvas
        shadows
        camera={{ position: cam.position, fov: 42, near: 10, far: 40000 }}
        className="!absolute inset-0"
      >
        <color attach="background" args={[PALETTE.bg]} />
        <fog attach="fog" args={[PALETTE.bg, 2000, 9000]} />
        <ambientLight intensity={0.65} />
        <directionalLight
          position={[(replay?.length ?? 3000) * 0.3, 2200, 1400]}
          intensity={1.35}
          castShadow
          shadow-mapSize={[2048, 2048]}
          shadow-camera-left={-3000}
          shadow-camera-right={3000}
          shadow-camera-top={3000}
          shadow-camera-bottom={-500}
          shadow-camera-far={6000}
        />
        <hemisphereLight args={[PALETTE.bg, PALETTE.ground, 0.4]} />
        <Ground length={replay?.length ?? 3000} />
        {replay && <SceneContents replay={replay} tlRef={tlRef} curRef={curRef} labelsRef={labelsRef} />}
        <OrbitControls
          target={cam.target}
          maxPolarAngle={Math.PI / 2 - 0.01}
          minDistance={200}
          maxDistance={8000}
        />
      </Canvas>
      <SceneHud replay={replay} tlRef={tlRef} curRef={curRef} />
    </div>
  );
}

function Ground({ length }: { length: number }) {
  const w = length + 2000;
  return (
    <mesh position={[length / 2, -100, 0]} receiveShadow>
      <boxGeometry args={[w, 200, 800]} />
      <meshStandardMaterial color={PALETTE.ground} roughness={1} />
    </mesh>
  );
}

/** A small HTML overlay (not WebGL text) for the HUD strip - matches the 2D renderer's HUD
 * bar, updated imperatively via refs so it doesn't fight the canvas for 30fps re-renders. */
function SceneHud({ replay, tlRef, curRef }: { replay: Replay | null } & Pick<PlayerRefs, "tlRef" | "curRef">) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    let raf = 0;
    const tick = () => {
      const tl = tlRef.current;
      if (replay && ref.current && tl.length) {
        const ci = Math.min(Math.max(0, Math.floor(curRef.current)), tl.length - 1);
        const st = tl[ci].st;
        const isRobot = !!replay.robot;
        const tail = isRobot ? `moves ${st.moves ?? 0}   placed ${st.placements ?? 0}` : `waste ${st.waste ?? 0}`;
        const mode = isRobot ? `${["place", "move ←", "move →"][st.mode ?? 0]}   ` : "";
        ref.current.textContent =
          `${replay._policy ?? ""}   ${replay.spec.n_modules}m × ${replay.spec.n_courses}c   ` +
          `step ${st.i + 1}/${replay.steps.length}   ${mode}in-tol ${(st.frac_in_tol * 100).toFixed(0)}%   ` +
          `filled ${(st.frac_filled * 100).toFixed(0)}%   ${tail}   return ${st.return.toFixed(2)}`;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [replay, tlRef, curRef]);

  return (
    <div className="pointer-events-none absolute inset-x-0 top-0 flex h-8 items-center gap-3 bg-hudBg/90 px-3 font-mono text-[12px]" style={{ background: PALETTE.hudBg + "e6" }}>
      <span style={{ color: PALETTE.nextSlot }}>atrium-sim</span>
      <span ref={ref} style={{ color: PALETTE.hudText }} />
    </div>
  );
}

function SceneContents({ replay, tlRef, curRef, labelsRef }: { replay: Replay } & PlayerRefs) {
  const maxFlat = useMemo(
    () => Math.max(1, ...replay.steps.flatMap((st) => st.ticks.map((t) => t.filter((b) => b[3] !== 2).length))),
    [replay],
  );
  const maxVoussoir = useMemo(
    () => Math.max(0, ...replay.steps.flatMap((st) => st.ticks.map((t) => t.filter((b) => b[3] === 2).length))),
    [replay],
  );
  const maxHard = (replay.hard_bodies ?? []).length;

  const flatRefs = useRef<(THREE.Mesh | null)[]>([]);
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

    // --- ghost targets + next-slot highlight ---
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
        (seg.material as THREE.LineBasicMaterial).color.copy(isNext ? NEXT_SLOT_COLOR : GHOST_COLOR);
        (seg.material as THREE.LineBasicMaterial).linewidth = isNext ? 2 : 1;
      }
    });

    // --- static hard bodies (fade in by frame index) ---
    (replay.hard_bodies ?? []).forEach((hb, i) => {
      const mesh = hardRefs.current[i];
      if (!mesh) return;
      mesh.visible = hb.appear <= frame.gi;
      // cached by content (see extrudedPolygon), so re-assigning the same shape every
      // frame is a cheap Map lookup, not a geometry rebuild - simpler than tracking
      // whether this specific mesh already has it.
      mesh.geometry = extrudedPolygon(hb.verts, BRICK_DEPTH_MM);
    });

    // --- bricks: split into the flat pool (shared box, scaled) and the voussoir pool
    // (per-shape extruded geometry) ---
    let flatI = 0;
    let voussoirI = 0;
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
        (mesh.material as THREE.MeshStandardMaterial).color.copy(VOUSSOIR_COLOR);
        continue;
      }
      const mesh = flatRefs.current[flatI++];
      if (!mesh) continue;
      const [w, h] = brickFace(kind);
      const m = matchByBrick.get(brickId);
      let color = STRAY_COLOR;
      if (m) color = new THREE.Color(...qualityColorRgb(m.d, m.in_tol));
      else if (foldDeg(theta) < 15) color = TERRACOTTA_COLOR;
      mesh.visible = true;
      mesh.position.set(x, y, 0);
      mesh.rotation.set(0, 0, theta);
      mesh.scale.set(w, h, BRICK_DEPTH_MM);
      (mesh.material as THREE.MeshStandardMaterial).color.copy(color);
    }
    for (let i = flatI; i < flatRefs.current.length; i++) {
      const mesh = flatRefs.current[i];
      if (mesh) mesh.visible = false;
    }
    for (let i = voussoirI; i < voussoirRefs.current.length; i++) {
      const mesh = voussoirRefs.current[i];
      if (mesh) mesh.visible = false;
    }

    // --- mobile gantry ---
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
      {Array.from({ length: maxFlat }, (_, i) => (
        <mesh key={`flat-${i}`} ref={(el) => (flatRefs.current[i] = el)} geometry={flatGeom} castShadow receiveShadow>
          <meshStandardMaterial color={PALETTE.terracotta} roughness={0.85} />
        </mesh>
      ))}
      {Array.from({ length: maxVoussoir }, (_, i) => (
        <mesh key={`vou-${i}`} ref={(el) => (voussoirRefs.current[i] = el)} castShadow receiveShadow>
          <meshStandardMaterial color={PALETTE.voussoirFace} roughness={0.8} />
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
          <lineBasicMaterial color={PALETTE.ghost} transparent opacity={0.55} />
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
            <meshBasicMaterial color={PALETTE.nextSlot} transparent opacity={0.08} depthWrite={false} />
          </mesh>
        </group>
      )}
    </group>
  );
}
