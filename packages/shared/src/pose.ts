/**
 * Rotation <-> yaw/pitch/roll in the ENGINE convention (camera x right,
 * y down, z forward): R = Ry(yaw) · Rx(−pitch) · Rz(roll), +pitch up,
 * angles in degrees. Mirrors `prior_rotation` in panoloom-core.
 */

export type Mat3 = [
  [number, number, number],
  [number, number, number],
  [number, number, number],
];

const D2R = Math.PI / 180;
const R2D = 180 / Math.PI;

export function poseToRotation(yaw: number, pitch: number, roll: number): Mat3 {
  const y = yaw * D2R;
  const p = -pitch * D2R;
  const r = roll * D2R;
  const [cy, sy, cp, sp, cr, sr] = [
    Math.cos(y),
    Math.sin(y),
    Math.cos(p),
    Math.sin(p),
    Math.cos(r),
    Math.sin(r),
  ];
  // Ry(y) · Rx(p) · Rz(r), expanded.
  return [
    [cy * cr + sy * sp * sr, -cy * sr + sy * sp * cr, sy * cp],
    [cp * sr, cp * cr, -sp],
    [-sy * cr + cy * sp * sr, sy * sr + cy * sp * cr, cy * cp],
  ];
}

/** Decompose; at gimbal lock (|pitch| = 90°) roll is folded into yaw. */
export function rotationToPose(r: Mat3): {
  yaw: number;
  pitch: number;
  roll: number;
} {
  const sp = Math.max(-1, Math.min(1, r[1][2]));
  const pitch = Math.asin(sp) * R2D;
  if (Math.abs(sp) > 0.999999) {
    // cos(pitch) ~ 0: only yaw+roll (pitch +90°) / yaw−roll (−90°) is
    // observable; fold it into yaw.
    const yaw =
      sp > 0
        ? Math.atan2(-r[0][1], r[0][0]) * R2D
        : Math.atan2(r[0][1], r[0][0]) * R2D;
    return { yaw, pitch, roll: 0 };
  }
  return {
    yaw: Math.atan2(r[0][2], r[2][2]) * R2D,
    pitch,
    roll: Math.atan2(r[1][0], r[1][1]) * R2D,
  };
}
