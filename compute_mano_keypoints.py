"""Offline: run MANO forward kinematics on HaMeR output to get 21 3D keypoints
per hand per frame (in MANO canonical frame, wrist at origin).

Layout: 16 joints from MANO FK + 5 fingertip vertices (thumb/index/middle/ring/pinky).
Full indices:
  0: wrist
  1-3: index MCP/PIP/DIP
  4-6: middle MCP/PIP/DIP
  7-9: pinky MCP/PIP/DIP
  10-12: ring MCP/PIP/DIP
  13-15: thumb CMC/MCP/IP
  16: thumb tip
  17: index tip
  18: middle tip
  19: ring tip
  20: pinky tip
"""
import os, json
import numpy as np
import torch
import smplx
from scipy.spatial.transform import Rotation as Rot

BASE = os.path.dirname(os.path.abspath(__file__))
MANO_JSON = os.path.join(BASE, 'output/pick_bottle_video/mano_results.json')
MANO_R_DIR = '/mnt/users/yjy/sim/video2robot-retarget/HaWoR/_DATA/data/mano'
MANO_L_DIR = '/mnt/users/yjy/sim/video2robot-retarget/HaWoR/_DATA/data_left/mano_left'
OUT = os.path.join(BASE, 'output/mano_keypoints.npz')


def rotmat_to_axang(R):
    return Rot.from_matrix(R).as_rotvec()


def main():
    with open(MANO_JSON) as f:
        rows = json.load(f)

    by_frame = {}
    for r in rows:
        by_frame.setdefault(r['img_name'], {})[('right' if r['is_right'] else 'left')] = r
    keys = sorted(by_frame.keys())
    N = len(keys)
    print(f"frames: {N}")

    mR = smplx.MANO(model_path=MANO_R_DIR, is_rhand=True,
                    use_pca=False, flat_hand_mean=True)
    mL = smplx.MANO(model_path=MANO_L_DIR, is_rhand=False,
                    use_pca=False, flat_hand_mean=True)

    # Standard MANO tip vertex indices (thumb, index, middle, ring, pinky)
    TIP_VIDX = [744, 320, 443, 554, 671]

    right = np.full((N, 21, 3), np.nan, dtype=np.float32)
    left  = np.full((N, 21, 3), np.nan, dtype=np.float32)
    vR = np.zeros(N, dtype=bool); vL = np.zeros(N, dtype=bool)

    for i, k in enumerate(keys):
        fr = by_frame[k]
        for side, layer, out_arr, vflag in [
                ('right', mR, right, vR), ('left', mL, left, vL)]:
            if side not in fr: continue
            hp = np.array(fr[side]['mano_hand_pose'])     # (15,3,3)
            go = np.array(fr[side]['mano_global_orient'])[0]  # (3,3)
            betas = np.array(fr[side]['mano_betas'])
            hp_axang = np.stack([rotmat_to_axang(hp[j]) for j in range(15)]).reshape(-1)
            # Keep global_orient ZERO so keypoints stay in wrist-local frame.
            with torch.no_grad():
                out = layer(
                    hand_pose=torch.tensor(hp_axang, dtype=torch.float32).unsqueeze(0),
                    betas=torch.tensor(betas, dtype=torch.float32).unsqueeze(0),
                    global_orient=torch.zeros(1, 3))
            joints = out.joints[0].numpy()            # (16,3)
            tips = out.vertices[0, TIP_VIDX].numpy()  # (5,3)
            pts = np.concatenate([joints, tips], axis=0)  # (21,3)
            pts = pts - pts[0]  # wrist at origin
            out_arr[i] = pts
            vflag[i] = True

    np.savez(OUT, right=right, left=left, valid_right=vR, valid_left=vL,
             frame_names=np.array(keys))
    print(f"right valid: {vR.sum()}  left valid: {vL.sum()}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
