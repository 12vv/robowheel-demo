"""
Continuous 3-DOF calibration of R_MANO_TO_SPIDER.

Instead of picking from 24 axis-aligned rotations, parametrize R as an axis-angle
rotvec and optimize against multi-frame observable constraints:

  * f0 rest pose (right hand on table): palm_normal in world ≈ (0, 0, -1)
  * grip frames (50, 60, 70 for right; 60, 70 for left if valid):
      palm_normal in bottle frame ≈ -hib_t / |hib_t|
      (i.e., palm faces toward bottle center)

Using bottle-frame targets for grip removes dependence on bottle-tracking drift.
Saves continuous optimum to output/R_MANO_TO_SPIDER.npy.
"""
import os
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import minimize

BASE = os.path.dirname(os.path.abspath(__file__))
CALIB = np.load(f"{BASE}/output/calibrated_trajectory.npz")
FPOSE = np.load(f"{BASE}/output/bottle_6dof_poses.npz")['poses']
R_FPOSE2SIM = Rot.from_euler('x', 90, degrees=True).as_matrix()

palm_normal_spider = np.array([0.0, 0.0, 1.0])   # spider palm body +Z = palm normal
fingers_spider     = np.array([-1.0, 0.0, 0.0])  # spider finger root at -X


def build_loss_right():
    hib_t = CALIB['hand_in_bottle_t']; hib_R = CALIB['hand_in_bottle_R']
    valid = CALIB['valid']; hand_cam_R = CALIB['hand_cam_R']
    R_bc0 = FPOSE[0, :3, :3]
    R_cw = R_FPOSE2SIM @ np.linalg.inv(R_bc0)
    R_hand_world_f0 = R_cw @ hand_cam_R[0, 0]

    grip = []
    for fi in [40, 50, 60, 70]:
        if not valid[fi, 0]: continue
        t = hib_t[fi, 0]; n = np.linalg.norm(t)
        if n < 1e-4: continue
        grip.append((hib_R[fi, 0].copy(), -t / n))

    def loss(rv):
        R = Rot.from_rotvec(rv).as_matrix()
        pn_w = R_hand_world_f0 @ R @ palm_normal_spider
        fn_w = R_hand_world_f0 @ R @ fingers_spider
        rest = (1 - pn_w @ np.array([0, 0, -1])) \
               + 0.5 * (1 - fn_w @ np.array([0, -1, 0]))
        g = sum(1 - (Rh @ R @ palm_normal_spider) @ tgt for Rh, tgt in grip)
        return 2 * rest + g
    return loss, len(grip)


def build_loss_left():
    """Left-hand calibration is grip-only (no rest anchor before f40)."""
    hib_t = CALIB['hand_in_bottle_t']; hib_R = CALIB['hand_in_bottle_R']
    valid = CALIB['valid']

    grip = []
    for fi in [50, 60, 70, 80, 90]:
        if not valid[fi, 1]: continue
        t = hib_t[fi, 1]; n = np.linalg.norm(t)
        if n < 1e-4: continue
        grip.append((hib_R[fi, 1].copy(), -t / n))

    def loss(rv):
        R = Rot.from_rotvec(rv).as_matrix()
        return sum(1 - (Rh @ R @ palm_normal_spider) @ tgt for Rh, tgt in grip)
    return loss, len(grip)


def sweep(loss):
    starts = []
    for axis in ['x', 'y', 'z']:
        for ang in [0, 90, 180, -90]:
            starts.append(Rot.from_euler(axis, ang, degrees=True).as_rotvec())
    for eul in [(90, 90, 0), (180, 90, 0), (0, 0, 180), (90, 0, 90)]:
        starts.append(Rot.from_euler('xyz', eul, degrees=True).as_rotvec())
    best = (np.inf, None)
    for rv0 in starts:
        res = minimize(loss, rv0, method='Powell',
                       options=dict(xtol=1e-6, ftol=1e-6, maxiter=3000))
        if res.fun < best[0]:
            best = (res.fun, res.x)
    return best


def main():
    lossR, nR = build_loss_right()
    lossL, nL = build_loss_left()
    print(f"Right grip anchors: {nR} (+ rest)    Left grip anchors: {nL}")

    costR, rvR = sweep(lossR)
    costL, rvL = sweep(lossL)
    R_right = Rot.from_rotvec(rvR).as_matrix()
    R_left  = Rot.from_rotvec(rvL).as_matrix()

    for lbl, R, cost, rv in [('R', R_right, costR, rvR), ('L', R_left, costL, rvL)]:
        eul = Rot.from_rotvec(rv).as_euler('xyz', degrees=True)
        print(f"[{lbl}] loss={cost:.4f}  euler(xyz)=({eul[0]:+6.1f},{eul[1]:+6.1f},{eul[2]:+6.1f})deg")

    np.savez(f"{BASE}/output/R_MANO_TO_SPIDER.npz",
             right=R_right, left=R_left)
    print(f"\n[save] -> output/R_MANO_TO_SPIDER.npz (right, left)")

    # Diagnostics
    hib_t = CALIB['hand_in_bottle_t']; hib_R = CALIB['hand_in_bottle_R']; valid = CALIB['valid']
    print("\nAlignment (higher = better, max 1.0):")
    for fi, side in [(40,0),(50,0),(60,0),(70,0),(50,1),(60,1),(70,1),(80,1),(90,1)]:
        if not valid[fi, side]: continue
        t = hib_t[fi, side]; n = np.linalg.norm(t)
        if n < 1e-4: continue
        R_use = R_right if side == 0 else R_left
        dot = (hib_R[fi, side] @ R_use @ palm_normal_spider) @ (-t / n)
        print(f"  f{fi:3d} {'R' if side==0 else 'L'}: {dot:+.3f}")

    R_bc0 = FPOSE[0, :3, :3]
    R_cw = R_FPOSE2SIM @ np.linalg.inv(R_bc0)
    pn_w = R_cw @ CALIB['hand_cam_R'][0, 0] @ R_right @ palm_normal_spider
    print(f"  f0 rest (R): palm_normal = {pn_w.round(3)}")


if __name__ == "__main__":
    main()
