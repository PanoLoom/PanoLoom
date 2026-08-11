import { describe, expect, it } from "vitest";
import { poseToRotation, rotationToPose, type Mat3 } from "./pose";

function maxAbsDiff(a: Mat3, b: Mat3): number {
  const fa = a.flat();
  const fb = b.flat();
  return Math.max(...fa.map((v, i) => Math.abs(v - (fb[i] ?? 0))));
}

describe("pose <-> rotation", () => {
  it("round-trips a grid of poses", () => {
    for (const yaw of [-170, -90, -30, 0, 45, 120, 179]) {
      for (const pitch of [-80, -45, 0, 30, 85]) {
        for (const roll of [-150, -10, 0, 25, 170]) {
          const r = poseToRotation(yaw, pitch, roll);
          const p = rotationToPose(r);
          expect(p.yaw).toBeCloseTo(yaw, 9);
          expect(p.pitch).toBeCloseTo(pitch, 9);
          expect(p.roll).toBeCloseTo(roll, 9);
        }
      }
    }
  });

  it("round-trips the matrix through decompose at gimbal lock", () => {
    for (const pitch of [90, -90]) {
      for (const yaw of [-120, 0, 60]) {
        for (const roll of [-40, 0, 75]) {
          const r = poseToRotation(yaw, pitch, roll);
          const p = rotationToPose(r);
          // yaw/roll are individually unobservable; the matrix must match.
          expect(maxAbsDiff(r, poseToRotation(p.yaw, p.pitch, p.roll))).toBeLessThan(1e-9);
        }
      }
    }
  });

  it("identity is zero pose", () => {
    const p = rotationToPose([
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ]);
    expect(p).toEqual({ yaw: 0, pitch: 0, roll: 0 });
  });
});
