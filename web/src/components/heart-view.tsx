"use client";

import { Html, OrbitControls, useGLTF } from "@react-three/drei";
import { Canvas, useFrame } from "@react-three/fiber";
import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";

import { riskHex } from "@/lib/risk";
import type { VesselId } from "@/lib/types";

type VesselSpec = {
  id: VesselId;
  name: string;
  full: string;
  blurb: string;
  radius: number;
  points: number[][];
  branches: { radius: number; points: number[][] }[];
};

type Anatomy = { note: string; vessels: VesselSpec[] };

const NEUTRAL = "#7f948c";

function tubeGeometry(points: number[][], radius: number) {
  const curve = new THREE.CatmullRomCurve3(
    points.map((point) => new THREE.Vector3(point[0], point[1], point[2])),
  );
  return new THREE.TubeGeometry(curve, 40, radius, 7, false);
}

function VesselMesh({
  points,
  radius,
  color,
  selected,
  onSelect,
}: {
  points: number[][];
  radius: number;
  color: string;
  selected: boolean;
  onSelect: () => void;
}) {
  const geometry = useMemo(() => tubeGeometry(points, radius), [points, radius]);
  const material = useRef<THREE.MeshStandardMaterial>(null);
  const target = useMemo(() => new THREE.Color(color), [color]);

  useEffect(() => {
    return () => geometry.dispose();
  }, [geometry]);

  useFrame(() => {
    const current = material.current;
    if (!current) return;
    current.color.lerp(target, 0.14);
    current.emissive.lerp(target, 0.14);
    const nextIntensity = selected ? 0.55 : 0.16;
    current.emissiveIntensity += (nextIntensity - current.emissiveIntensity) * 0.14;
  });

  return (
    <mesh
      geometry={geometry}
      onClick={(event) => {
        event.stopPropagation();
        onSelect();
      }}
      onPointerOver={(event) => {
        event.stopPropagation();
        document.body.style.cursor = "pointer";
      }}
      onPointerOut={() => {
        document.body.style.cursor = "default";
      }}
    >
      <meshStandardMaterial
        ref={material}
        color={color}
        emissive={color}
        emissiveIntensity={selected ? 0.55 : 0.16}
        roughness={0.38}
        metalness={0.04}
      />
    </mesh>
  );
}

function HeartModel() {
  const gltf = useGLTF("/anatomy/heart.glb");
  useEffect(() => {
    gltf.scene.traverse((object) => {
      if (object instanceof THREE.Mesh) {
        object.material = new THREE.MeshStandardMaterial({
          color: "#8d3b49",
          roughness: 0.64,
          metalness: 0.04,
        });
      }
    });
  }, [gltf]);
  return <primitive object={gltf.scene} />;
}

function Scene({
  vessels,
  probabilities,
  selected,
  onSelect,
  resetToken,
}: {
  vessels: VesselSpec[];
  probabilities: Partial<Record<VesselId, number>>;
  selected: VesselId | null;
  onSelect: (id: VesselId) => void;
  resetToken: number;
}) {
  const group = useRef<THREE.Group>(null);
  const controls = useRef<OrbitControlsImpl>(null);

  useEffect(() => {
    const control = controls.current;
    if (!control) return;
    control.target.set(0.02, 0.08, 0);
    control.object.position.set(0.35, 0.28, 3.55);
    control.update();
  }, [resetToken]);

  useFrame(({ clock }) => {
    const node = group.current;
    if (!node) return;
    const beat = Math.sin(clock.elapsedTime * Math.PI * 1.05);
    const pulse = beat > 0 ? beat ** 10 : 0;
    const scale = 1 + pulse * 0.012;
    node.scale.setScalar(scale);
  });

  return (
    <>
      <color attach="background" args={["#101614"]} />
      <hemisphereLight args={["#f2efe6", "#1c1618", 0.55]} />
      <directionalLight position={[2.4, 2.2, 3.2]} intensity={1.45} />
      <directionalLight position={[-2.2, 0.4, -1.6]} intensity={0.35} color="#9fd0c6" />
      <group ref={group}>
        <HeartModel />
        {vessels.map((vessel) => {
          const probability = probabilities[vessel.id];
          const color = typeof probability === "number" ? riskHex(probability) : NEUTRAL;
          const midpoint = vessel.points[Math.floor(vessel.points.length / 2)];
          return (
            <group key={vessel.id}>
              <VesselMesh
                points={vessel.points}
                radius={vessel.radius}
                color={color}
                selected={selected === vessel.id}
                onSelect={() => onSelect(vessel.id)}
              />
              {vessel.branches.map((branch, index) => (
                <VesselMesh
                  key={index}
                  points={branch.points}
                  radius={branch.radius}
                  color={color}
                  selected={selected === vessel.id}
                  onSelect={() => onSelect(vessel.id)}
                />
              ))}
              <Html position={[midpoint[0], midpoint[1], midpoint[2]]} center distanceFactor={6} zIndexRange={[20, 0]}>
                <div className="pointer-events-none rounded-full border border-white/15 bg-[#101614]/85 px-2 py-0.5 text-[11px] font-medium tracking-wide text-[#f3f7f4]">
                  {vessel.name}
                  {typeof probability === "number" ? ` ${Math.round(probability * 100)}%` : ""}
                </div>
              </Html>
            </group>
          );
        })}
      </group>
      <OrbitControls
        ref={controls}
        enablePan={false}
        minDistance={1.8}
        maxDistance={6.5}
        minPolarAngle={0.25}
        maxPolarAngle={Math.PI - 0.25}
        target={[0.02, 0.08, 0]}
      />
    </>
  );
}

function hasWebGL() {
  try {
    const canvas = document.createElement("canvas");
    return Boolean(canvas.getContext("webgl2") || canvas.getContext("webgl"));
  } catch {
    return false;
  }
}

export default function HeartView({
  probabilities,
  selected,
  onSelect,
  resetToken,
}: {
  probabilities: Partial<Record<VesselId, number>>;
  selected: VesselId | null;
  onSelect: (id: VesselId) => void;
  resetToken: number;
}) {
  const [anatomy, setAnatomy] = useState<Anatomy | null>(null);
  const [webgl, setWebgl] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    setWebgl(hasWebGL());
    let cancel = false;
    fetch("/anatomy/vessels.json")
      .then((response) => {
        if (!response.ok) throw new Error("Could not load vessel paths.");
        return response.json();
      })
      .then((payload: Anatomy) => {
        if (!cancel) setAnatomy(payload);
      })
      .catch((error: Error) => {
        if (!cancel) setLoadError(error.message);
      });
    return () => {
      cancel = true;
    };
  }, []);

  if (!webgl) {
    return (
      <div className="flex h-full items-center justify-center p-6 text-sm text-muted-foreground">
        This browser has no WebGL context, so the heart cannot be drawn. Vessel estimates stay in the side panel.
      </div>
    );
  }

  if (loadError) {
    return <div className="flex h-full items-center justify-center p-6 text-sm text-[#e7b6b2]">{loadError}</div>;
  }

  if (!anatomy) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
        Preparing the heart…
      </div>
    );
  }

  return (
    <Canvas
      camera={{ position: [0.35, 0.28, 3.55], fov: 35, near: 0.1, far: 20 }}
      dpr={[1, 1.5]}
      gl={{ antialias: true, powerPreference: "default", alpha: false }}
    >
      <Suspense fallback={null}>
        <Scene
          vessels={anatomy.vessels}
          probabilities={probabilities}
          selected={selected}
          onSelect={onSelect}
          resetToken={resetToken}
        />
      </Suspense>
    </Canvas>
  );
}

useGLTF.preload("/anatomy/heart.glb");
