"""Per-finger IK: given MANO 16 keypoints (wrist-local) and a Spider hand model,
solve Spider joint angles so finger body positions match MANO joints.

MANO keypoint indices (smplx MANO layer output):
    0: wrist
    1-3: index MCP, PIP, DIP
    4-6: middle MCP, PIP, DIP
    7-9: pinky MCP, PIP, DIP
    10-12: ring MCP, PIP, DIP
    13-15: thumb CMC, MCP, IP

Spider joint order (per right.xml body tree):
    [0]=index1y, [1]=index1z, [2]=index2, [3]=index3,
    [4]=middle1y, [5]=middle1z, [6]=middle2, [7]=middle3,
    [8]=pinky1y, [9]=pinky1z, [10]=pinky2, [11]=pinky3,
    [12]=ring1y, [13]=ring1z, [14]=ring2, [15]=ring3,
    [16]=thumb1x, [17]=thumb1y, [18]=thumb1z, [19]=thumb2y, [20]=thumb2z, [21]=thumb3
"""
import os
import numpy as np
import mujoco
from scipy.optimize import least_squares

# Finger chains: (qpos indices, mano keypoint indices, spider body-or-site names).
# Tip targets use the Spider fingertip SITE (e.g., "index_tip"); joint targets use bodies.
# Mano indices: 0 wrist, 1-3 index MCP/PIP/DIP, 4-6 middle, 7-9 pinky, 10-12 ring,
# 13-15 thumb CMC/MCP/IP, 16 thumb tip, 17 index tip, 18 middle tip, 19 ring tip, 20 pinky tip.
FINGER_SPEC = {
    'index':  {'qpos': [0, 1, 2, 3],    'mano_kp': [1, 2, 3, 17],
               'bodies': ['index1z', 'index2', 'index3'],  'tip_site': 'index_tip'},
    'middle': {'qpos': [4, 5, 6, 7],    'mano_kp': [4, 5, 6, 18],
               'bodies': ['middle1z', 'middle2', 'middle3'], 'tip_site': 'middle_tip'},
    'pinky':  {'qpos': [8, 9, 10, 11],  'mano_kp': [7, 8, 9, 20],
               'bodies': ['pinky1z', 'pinky2', 'pinky3'],  'tip_site': 'pinky_tip'},
    'ring':   {'qpos': [12, 13, 14, 15],'mano_kp': [10, 11, 12, 19],
               'bodies': ['ring1z', 'ring2', 'ring3'],    'tip_site': 'ring_tip'},
    'thumb':  {'qpos': [16, 17, 18, 19, 20, 21], 'mano_kp': [13, 14, 15, 16],
               'bodies': ['thumb1z', 'thumb2z', 'thumb3'], 'tip_site': 'thumb_tip'},
}


class FingerIK:
    """Per-hand finger IK solver. Uses a standalone hand MJCF to run FK cheaply."""

    def __init__(self, mjcf_path, prefix):
        os.chdir(os.path.dirname(mjcf_path))
        self.model = mujoco.MjModel.from_xml_path(os.path.basename(mjcf_path))
        self.data = mujoco.MjData(self.model)
        self.prefix = prefix  # 'right' or 'left'

        # Map: finger qpos index -> model.jnt_qposadr
        jnames_by_tree = [
            'index1y','index1z','index2','index3',
            'middle1y','middle1z','middle2','middle3',
            'pinky1y','pinky1z','pinky2','pinky3',
            'ring1y','ring1z','ring2','ring3',
            'thumb1x','thumb1y','thumb1z','thumb2y','thumb2z','thumb3',
        ]
        self.qadr = []
        self.jrange = []
        for nm in jnames_by_tree:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                                    f"{prefix}_j_{nm}")
            self.qadr.append(self.model.jnt_qposadr[jid])
            self.jrange.append(self.model.jnt_range[jid].copy())
        self.qadr = np.array(self.qadr)
        self.jrange = np.array(self.jrange)

        # Body IDs for the 3 joint-chain bodies + tip site id for each finger
        self.body_ids = {}
        self.tip_sid = {}
        for name, spec in FINGER_SPEC.items():
            self.body_ids[name] = [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}_{b}")
                for b in spec['bodies']
            ]
            self.tip_sid[name] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, f"{prefix}_{spec['tip_site']}")

        self.palm_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                          f"{prefix}_palm")

    def _fk_finger(self, qs, finger):
        idxs = FINGER_SPEC[finger]['qpos']
        for j, q in zip(idxs, qs):
            self.data.qpos[self.qadr[j]] = q
        mujoco.mj_kinematics(self.model, self.data)
        # Positions in world == palm frame since palm is free root at identity.
        bodies = np.array([self.data.xpos[bid] for bid in self.body_ids[finger]])
        tip = self.data.site_xpos[self.tip_sid[finger]]
        return np.vstack([bodies, tip])  # (4,3)

    def solve_finger(self, finger, targets_palm, init_q=None):
        spec = FINGER_SPEC[finger]
        n = len(spec['qpos'])
        lo = np.array([self.jrange[j, 0] for j in spec['qpos']])
        hi = np.array([self.jrange[j, 1] for j in spec['qpos']])
        if init_q is None:
            init_q = np.clip(np.full(n, 0.2), lo, hi)
        else:
            # Ensure nonzero so least_squares relative-step Jacobian isn't degenerate.
            init_q = np.where(np.abs(init_q) < 1e-3,
                              np.clip(0.2, lo, hi), init_q)
            init_q = np.clip(init_q, lo, hi)

        def resid(q):
            pos = self._fk_finger(q, finger)          # (4,3)
            e = (pos - targets_palm).reshape(-1)      # 12 residuals
            # Tiny regularizer to prefer smaller joint angles
            return np.concatenate([e, 1e-3 * (q - init_q)])

        res = least_squares(resid, init_q, bounds=(lo, hi),
                            diff_step=0.02, max_nfev=120,
                            xtol=1e-7, ftol=1e-7)
        return res.x, res.cost

    def solve_frame(self, mano_kp_palm, prev_q=None):
        """mano_kp_palm: (16,3) MANO keypoints in Spider palm frame (wrist at origin)."""
        q_full = np.zeros(22)
        if prev_q is None:
            prev_q = np.zeros(22)
        for name, spec in FINGER_SPEC.items():
            tgt = mano_kp_palm[spec['mano_kp']]       # (3,3)
            init = prev_q[spec['qpos']]
            qf, _ = self.solve_finger(name, tgt, init)
            q_full[spec['qpos']] = qf
        return q_full


def solve_all_frames(mano_kp, R_m2s, mjcf_path, prefix, valid):
    """mano_kp: (N,16,3) MANO keypoints (wrist-local).
       R_m2s: (3,3) MANO wrist canonical -> Spider palm.
       Returns (N,22) array of joint angles (rad).
    """
    ik = FingerIK(mjcf_path, prefix)
    N = len(mano_kp)
    out = np.zeros((N, 22))
    prev_q = np.zeros(22)
    # Transform MANO keypoints into Spider palm frame:
    # p_spider = R_m2s^T @ p_mano
    kp_spider = np.einsum('ij,nkj->nki', R_m2s.T, np.nan_to_num(mano_kp))
    for fi in range(N):
        if not valid[fi]:
            out[fi] = prev_q
            continue
        q = ik.solve_frame(kp_spider[fi], prev_q=prev_q)
        out[fi] = q
        prev_q = q
    return out
