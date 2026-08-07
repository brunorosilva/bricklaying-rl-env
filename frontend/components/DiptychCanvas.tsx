"use client";

import { Canvas } from "@react-three/fiber";
import { ContactShadows, Environment, Lightformer } from "@react-three/drei";
import * as THREE from "three";
import type { RefObject } from "react";
import type { Frame, Replay } from "@/lib/replay/types";
import { PALETTE } from "@/lib/replay/shared";
import { CameraRig, Ground, SceneContents } from "./SceneCanvas";
import { GrainOverlay } from "./GrainOverlay";

const GAP_MM = 900;

/** Two walls, ONE canvas, side by side in world space - not two <Canvas>es. A second WebGL
 * context is a real cost (a second GPU context, a second render pass, twice the shader
 * compiles) for no visual gain here, and one scene lets both walls share the instanced brick
 * buffer machinery and a single light rig. Reuses SceneCanvas's own exported pieces
 * (Ground/CameraRig/SceneContents) rather than re-deriving the lighting/instancing setup -
 * each wall is just a <SceneContents> in its own <group> offset along X, exactly like moving
 * any other object in a three.js scene graph. */
export function DiptychCanvas({
  replayA, tlRefA, curRefA, labelsRefA,
  replayB, tlRefB, curRefB, labelsRefB,
}: {
  replayA: Replay; tlRefA: RefObject<Frame[]>; curRefA: RefObject<number>; labelsRefA: RefObject<boolean>;
  replayB: Replay; tlRefB: RefObject<Frame[]>; curRefB: RefObject<number>; labelsRefB: RefObject<boolean>;
}) {
  const offsetX = replayA.length + GAP_MM;
  const totalLength = offsetX + replayB.length;
  const nCourses = Math.max(replayA.n_courses, replayB.n_courses);

  return (
    <div className="relative h-[50vh] min-h-[340px] w-full overflow-hidden rounded-md md:h-[58vh]">
      <Canvas
        shadows
        gl={{ toneMapping: THREE.ACESFilmicToneMapping, toneMappingExposure: 1.15 }}
        className="!absolute inset-0"
      >
        <color attach="background" args={[PALETTE.bg]} />
        <fog attach="fog" args={[PALETTE.bg, 2000, 9000]} />
        <ambientLight intensity={0.25} />
        <directionalLight
          position={[totalLength * 0.3, 2200, 1400]}
          intensity={1.1}
          castShadow
          shadow-mapSize={[1024, 1024]}
          shadow-radius={4}
          shadow-camera-left={-3000}
          shadow-camera-right={totalLength + 3000}
          shadow-camera-top={3000}
          shadow-camera-bottom={-500}
          shadow-camera-far={6000}
        />
        <hemisphereLight args={[PALETTE.bg, PALETTE.ground, 0.3]} />
        {/* See SceneCanvas.tsx's comment on why this uses shadow-radius, not drei's
            SoftShadows (a real shader-compile break against this project's three version). */}
        <Environment resolution={256} background={false}>
          <Lightformer form="rect" intensity={2.4} color="#fff2df" position={[totalLength * 0.35, 2600, 2200]} scale={[3200, 2200, 1]} target={[totalLength * 0.35, 0, 0]} />
          <Lightformer form="rect" intensity={0.5} color="#7fa0c9" position={[-1600, 900, -2600]} scale={[1800, 1800, 1]} />
        </Environment>
        <Ground length={totalLength} />
        <ContactShadows position={[totalLength / 2, 0.5, 0]} opacity={0.45} width={totalLength + 1200} height={1400} blur={2.2} far={500} resolution={512} color="#000000" />

        <group position={[0, 0, 0]}>
          <SceneContents replay={replayA} tlRef={tlRefA} curRef={curRefA} labelsRef={labelsRefA} mode="as-built" />
        </group>
        <group position={[offsetX, 0, 0]}>
          <SceneContents replay={replayB} tlRef={tlRefB} curRef={curRefB} labelsRef={labelsRefB} mode="as-built" />
        </group>

        <CameraRig length={totalLength} nCourses={nCourses} mode="as-built" />
      </Canvas>
      <GrainOverlay />
    </div>
  );
}
