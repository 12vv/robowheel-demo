"""
V1.7: calibrated trajectory + 6-DOF base follow (mocap).

Upgrade from v1.6:
  - Replace 3-slide base with mocap body -> full 6-DOF kinematic control.
  - Drive both position AND orientation from calibrated MANO trajectory.
  - Hand wrist pose per frame = T_bottle_world * T_hand_in_bottle.

Still kinematic replay (no RL) to validate trajectory alignment.
"""
import os, json
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import mujoco
import cv2
from scipy.spatial.transform import Rotation as Rot
from scipy.ndimage import uniform_filter1d

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPIDER_MANO   = os.path.join(BASE_DIR, 'spider/spider/assets/robots/mano')
OUT_DIR       = os.path.join(BASE_DIR, 'output/rl_v17_calibrated_6dof')
MANO_PATH     = os.path.join(BASE_DIR, 'output/pick_bottle_video/mano_results.json')
BOTTLE_VISUAL = os.path.abspath(os.path.join(BASE_DIR, 'output/bottle_recon/bottle_visual.obj'))
CALIB_PATH    = os.path.join(BASE_DIR, 'output/calibrated_trajectory.npz')
FPOSE_PATH    = os.path.join(BASE_DIR, 'output/bottle_6dof_poses.npz')

# Rotation: FoundationPose bottle frame (Y = long axis) -> sim frame (Z = up)
R_FPOSE2SIM = Rot.from_euler('x', 90, degrees=True).as_matrix()

# MANO (HaMeR camera-frame) wrist -> Spider wrist canonical. Continuous 3-DOF
# per-hand calibration via calibrate_wrist_offset.py (rest + multi-grip anchors).
_R_M2S_PATH = os.path.join(BASE_DIR, 'output/R_MANO_TO_SPIDER.npz')
if os.path.exists(_R_M2S_PATH):
    _d = np.load(_R_M2S_PATH)
    R_M2S_RIGHT = _d['right']
    R_M2S_LEFT  = _d['left']
else:
    R_M2S_RIGHT = R_M2S_LEFT = Rot.from_euler('z', 90, degrees=True).as_matrix()


def load_mano_fingers():
    with open(MANO_PATH) as f:
        results = json.load(f)
    frame_dict = {}
    for r in results:
        frame_dict.setdefault(r['img_name'], {})[('right' if r['is_right'] else 'left')] = r
    keys = sorted(frame_dict.keys())
    N = len(keys)

    # D1 mapping: decompose each MANO bone-local rotation matrix in the Euler
    # order matching the Spider joint chain of that bone.
    #
    # Spider body tree (per 4-finger): body_1y (axis Y) -> body_1z (axis Z) ->
    # body_2 (axis Z) -> body_3 (axis Z). Thumb: 1x -> 1y -> 1z -> 2y -> 2z -> 3.
    #
    # MANO bone local X axis points along the bone toward the tip; Spider body
    # local X points toward the base (meshes drawn at negative X). Flipping X
    # negates Ry and Rz angles about the other two axes, handled by SIGN_* below.
    SIGN_SPREAD =  1.0   # MCP abduction (joint *1y)
    SIGN_CURL   =  1.0   # MCP/PIP/DIP flexion (joint *1z/*2/*3): MANO +Z = palm-ward
    SIGN_THUMB_X = 1.0
    SIGN_THUMB_Y = 1.0
    SIGN_THUMB_Z = 1.0
    CURL_GAIN   = 1.7    # HaMeR underpredicts grip tightness; scale curl angles.
    THUMB_GAIN  = 1.4    # Thumb slightly less (its range is tighter).

    def decompose_yz(R):
        # R ≈ Ry(a) @ Rz(b)  (intrinsic Y-then-Z); twist about X ignored.
        return Rot.from_matrix(R).as_euler('YZX')[:2]

    def decompose_z(R):
        # PIP/DIP: single curl DOF about local Z.
        return Rot.from_matrix(R).as_euler('YZX')[1]

    def decompose_xyz(R):
        # Thumb CMC: full 3-DOF, Spider chain is x-then-y-then-z.
        return Rot.from_matrix(R).as_euler('XYZ')

    def set_mcp(joints, base, R):
        y, z = decompose_yz(R)
        joints[base]     = SIGN_SPREAD * y
        joints[base + 1] = SIGN_CURL * CURL_GAIN * z

    def mano_to_artimano(hp):
        joints = np.zeros(22)
        g = CURL_GAIN
        # Index (hp[0..2]) -> joints[0..3]
        set_mcp(joints, 0, hp[0])
        joints[2] = SIGN_CURL * g * decompose_z(hp[1])
        joints[3] = SIGN_CURL * g * decompose_z(hp[2])
        # Middle (hp[3..5]) -> joints[4..7]
        set_mcp(joints, 4, hp[3])
        joints[6] = SIGN_CURL * g * decompose_z(hp[4])
        joints[7] = SIGN_CURL * g * decompose_z(hp[5])
        # Pinky (hp[6..8]) -> joints[8..11]
        set_mcp(joints, 8, hp[6])
        joints[10] = SIGN_CURL * g * decompose_z(hp[7])
        joints[11] = SIGN_CURL * g * decompose_z(hp[8])
        # Ring (hp[9..11]) -> joints[12..15]
        set_mcp(joints, 12, hp[9])
        joints[14] = SIGN_CURL * g * decompose_z(hp[10])
        joints[15] = SIGN_CURL * g * decompose_z(hp[11])
        # Thumb CMC/MCP/IP
        tg = THUMB_GAIN
        t = decompose_xyz(hp[12])
        joints[16] = SIGN_THUMB_X * t[0]
        joints[17] = SIGN_THUMB_Y * t[1]
        joints[18] = SIGN_THUMB_Z * tg * t[2]
        ty, tz = decompose_yz(hp[13])
        joints[19] = SIGN_THUMB_Y * ty
        joints[20] = SIGN_THUMB_Z * tg * tz
        joints[21] = SIGN_THUMB_Z * tg * decompose_z(hp[14])
        return joints

    rf = np.zeros((N, 22)); lf = np.zeros((N, 22))
    for i, k in enumerate(keys):
        fr = frame_dict[k]
        if 'right' in fr: rf[i] = mano_to_artimano(np.array(fr['right']['mano_hand_pose']))
        if 'left'  in fr: lf[i] = mano_to_artimano(np.array(fr['left']['mano_hand_pose']))
    rf = uniform_filter1d(rf, size=5, axis=0)
    lf = uniform_filter1d(lf, size=5, axis=0)
    return N, rf, lf


def build_model():
    os.chdir(SPIDER_MANO)
    hr = mujoco.MjSpec.from_file('right.xml')
    hl = mujoco.MjSpec.from_file('left.xml')

    s = mujoco.MjSpec()
    s.option.gravity = [0, 0, -9.81]
    s.option.timestep = 0.002
    s.option.impratio = 10
    s.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    w = s.worldbody

    l = w.add_light(); l.pos=[0,-0.3,0.5]; l.dir=[0,0.2,-1]; l.diffuse=[1,1,1]; l.castshadow=True
    l2 = w.add_light(); l2.pos=[0.3,0.2,0.4]; l2.dir=[-0.2,-0.1,-0.5]; l2.diffuse=[0.4,0.4,0.4]

    f = w.add_geom(); f.type=mujoco.mjtGeom.mjGEOM_PLANE; f.size=[0.5,0.5,0.01]; f.rgba=[0.92,0.92,0.92,1]
    t = w.add_geom(); t.type=mujoco.mjtGeom.mjGEOM_BOX; t.size=[0.18,0.14,0.01]
    t.pos=[0,0,0.15]; t.rgba=[0.38,0.28,0.20,1]; t.friction=[1.0,0.005,0.0001]

    bottle_mesh = s.add_mesh(); bottle_mesh.name="bottle_mesh"; bottle_mesh.file=BOTTLE_VISUAL

    bottle = w.add_body(); bottle.name="bottle"; bottle.pos=[0,0,0.26]
    bj = bottle.add_freejoint(); bj.name="bottle_joint"
    bv = bottle.add_geom(); bv.type=mujoco.mjtGeom.mjGEOM_MESH; bv.meshname="bottle_mesh"
    bv.quat = Rot.from_euler('x', 90, degrees=True).as_quat(scalar_first=True).tolist()
    bv.rgba=[0.20,0.55,0.85,0.45]; bv.contype=0; bv.conaffinity=0; bv.group=1; bv.mass=0
    bg = bottle.add_geom(); bg.type=mujoco.mjtGeom.mjGEOM_CYLINDER
    bg.size=[0.030,0.075,0]; bg.rgba=[0.15,0.50,0.85,0.0]; bg.mass=0.20
    bg.friction=[2.0,0.01,0.001]; bg.contype=1; bg.conaffinity=1; bg.group=3

    # MANO spec's palm already has 6 DOF (3 slide + 3 hinge). Attach directly.
    for hand_spec, prefix in [(hr, "rh_"), (hl, "lh_")]:
        w.add_frame().attach_body(hand_spec.worldbody.first_body(), prefix, "")

    model = s.compile()
    return model, mujoco.MjData(model)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== V1.7 Calibrated + 6-DOF Mocap Base ===")

    N_mano, rf, lf = load_mano_fingers()
    calib = np.load(CALIB_PATH)
    hib_t = calib['hand_in_bottle_t'].copy()   # (N,2,3)
    hib_R = calib['hand_in_bottle_R'].copy()   # (N,2,3,3)
    valid = calib['valid']                     # (N,2)
    scale = float(calib['scale'])
    N = min(N_mano, len(hib_t))
    print(f"  frames: {N}   scale: {scale:.5f}")

    # Fill invalid frames: forward-fill from first valid; back-fill initial gap
    for side in range(2):
        first_valid = next((i for i in range(N) if valid[i, side]), None)
        if first_valid is None:
            hib_t[:, side] = 0.0
            hib_R[:, side] = np.eye(3)
            continue
        # back-fill
        for i in range(first_valid):
            hib_t[i, side] = hib_t[first_valid, side]
            hib_R[i, side] = hib_R[first_valid, side]
        # forward-fill
        last_t = hib_t[first_valid, side].copy()
        last_R = hib_R[first_valid, side].copy()
        for i in range(first_valid, N):
            if valid[i, side]:
                last_t = hib_t[i, side].copy()
                last_R = hib_R[i, side].copy()
            else:
                hib_t[i, side] = last_t
                hib_R[i, side] = last_R

    # FPose local frame -> sim frame (Y-up -> Z-up)
    hib_t = hib_t @ R_FPOSE2SIM.T
    hib_R = np.einsum('ij,nsjk->nsik', R_FPOSE2SIM, hib_R)

    # Discard intro-card MANO noise: find the first frame where BOTH hands are
    # valid (video intro f0-~f30 is a title card; HaMeR hallucinates a hand).
    # For frames before that, freeze pose to the first reliable grip frame.
    _gs = next((i for i in range(N) if valid[i, 0] and valid[i, 1]), 0)
    if _gs > 0:
        for side in range(2):
            hib_t[:_gs, side] = hib_t[_gs, side]
            hib_R[:_gs, side] = hib_R[_gs, side]
        rf[:_gs] = rf[_gs]
        lf[:_gs] = lf[_gs]
        print(f"  intro-skip: freeze hand pose to f{_gs} for frames [0..{_gs-1}]")

    # Smooth (HaMeR jitter)
    hib_t[:, 0] = uniform_filter1d(hib_t[:, 0], size=7, axis=0)
    hib_t[:, 1] = uniform_filter1d(hib_t[:, 1], size=7, axis=0)

    model, data = build_model()
    # 6 base joints per hand: pos_x/y/z (slide) + rot_x/y/z (hinge, euler XYZ)
    def jadr(name):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return model.jnt_qposadr[jid]
    base_r_qadr = [jadr(f"rh_right_pos_{a}") for a in "xyz"] + [jadr(f"rh_right_rot_{a}") for a in "xyz"]
    base_l_qadr = [jadr(f"lh_left_pos_{a}")  for a in "xyz"] + [jadr(f"lh_left_rot_{a}")  for a in "xyz"]
    bottle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'bottle')
    bottle_qadr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'bottle_joint')]

    # Load FoundationPose bottle trajectory (in camera frame) and compute cam->world anchor
    fpose = np.load(FPOSE_PATH)['poses'][:N]        # (N, 4, 4) mesh-in-camera
    R_bc = fpose[:, :3, :3]                          # mesh rotation in cam
    t_bc_scaled = fpose[:, :3, 3]                    # mesh translation in cam (metric)
    # Anchor: at frame 0, place mesh upright (Rx(+90)) at world [0, 0, 0.26]
    R_cw = R_FPOSE2SIM @ np.linalg.inv(R_bc[0])      # cam -> world rotation
    t_cw = np.array([0.0, 0.0, 0.26]) - R_cw @ t_bc_scaled[0]
    # Body frame = mesh frame * Rx(-90) (inverse of bv.quat). Body pose in world:
    R_bv_inv = Rot.from_euler('x', -90, degrees=True).as_matrix()
    body_t_world = np.einsum('ij,nj->ni', R_cw, t_bc_scaled) + t_cw   # (N,3)
    body_R_world = np.einsum('ij,njk,kl->nil', R_cw, R_bc, R_bv_inv)  # (N,3,3)

    # -- Override bottle trajectory (FPose drifts in orientation and freezes
    # position after ~f80; the real bottle stays upright and follows the hands).
    # Strategy: force upright orientation; drive position from hand-cam positions
    # (via R_cw,t_cw) during grip; hold at initial FPose position before grip.
    hand_cam_t = calib['hand_cam_t_metric'].copy()   # (N,2,3)  may contain NaN
    palm_world_raw = np.einsum('ij,nsj->nsi', R_cw, np.nan_to_num(hand_cam_t)) + t_cw

    grip_start = next((i for i in range(N) if valid[i, 0] and valid[i, 1]), N)
    initial_t = body_t_world[0].copy()

    override_t = np.tile(initial_t, (N, 1))  # default: stay at initial
    for fi in range(N):
        if fi < grip_start:
            continue
        # Bottle = palm - hib_t (reversing hand = bottle + hib_t), averaged over
        # valid hands; hib_t is already in the sim frame (R_FPOSE2SIM applied).
        contribs = []
        for side in range(2):
            if valid[fi, side]:
                contribs.append(palm_world_raw[fi, side] - hib_t[fi, side])
        if contribs:
            override_t[fi] = np.mean(contribs, axis=0)
        else:
            override_t[fi] = override_t[fi - 1]

    # Smooth the post-grip trajectory to kill HaMeR jitter and prevent jumps.
    if grip_start < N - 1:
        override_t[grip_start:] = uniform_filter1d(override_t[grip_start:], size=5, axis=0)

    body_t_world = override_t
    body_R_world = np.tile(np.eye(3), (N, 1, 1))   # force upright

    # Find finger joints (exclude the 6 base pos/rot joints)
    rh_jnts, lh_jnts = [], []
    for i in range(model.njnt):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or ""
        if "_pos_" in nm or "_rot_" in nm:
            continue
        if nm.startswith("rh_"): rh_jnts.append(i)
        elif nm.startswith("lh_"): lh_jnts.append(i)

    model.vis.global_.offwidth = 640
    model.vis.global_.offheight = 480
    renderer = mujoco.Renderer(model, height=480, width=640)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0, 0, 0.28]; cam.distance = 0.60; cam.azimuth = 145; cam.elevation = -25
    cam2 = mujoco.MjvCamera(); cam2.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam2.lookat[:] = [0, 0, 0.28]; cam2.distance = 0.70; cam2.azimuth = 45; cam2.elevation = -15

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    bottle_init_z = data.xpos[bottle_id][2]

    frames = []
    max_lift = 0.0

    for fi in range(N):
        # Kinematically drive the bottle body with FPose trajectory
        b_pos = body_t_world[fi]
        R_bw = body_R_world[fi]
        bq = Rot.from_matrix(R_bw).as_quat()  # xyzw
        data.qpos[bottle_qadr:bottle_qadr+3]   = b_pos
        data.qpos[bottle_qadr+3:bottle_qadr+7] = [bq[3], bq[0], bq[1], bq[2]]  # wxyz

        for side, qadrs in [(0, base_r_qadr), (1, base_l_qadr)]:
            target_pos = b_pos + R_bw @ hib_t[fi, side]
            R_m2s = R_M2S_RIGHT if side == 0 else R_M2S_LEFT
            target_R   = R_bw @ hib_R[fi, side] @ R_m2s
            euler = Rot.from_matrix(target_R).as_euler('xyz')
            for k in range(3): data.qpos[qadrs[k]]   = target_pos[k]
            for k in range(3): data.qpos[qadrs[3+k]] = euler[k]

        # Fingers: write qpos directly (pure kinematic - no dynamics)
        for j in range(min(22, len(rh_jnts))):
            jid = rh_jnts[j]
            jr = model.jnt_range[jid]
            qadr = model.jnt_qposadr[jid]
            data.qpos[qadr] = np.clip(rf[fi, j], jr[0], jr[1])
        for j in range(min(22, len(lh_jnts))):
            jid = lh_jnts[j]
            jr = model.jnt_range[jid]
            qadr = model.jnt_qposadr[jid]
            data.qpos[qadr] = np.clip(lf[fi, j], jr[0], jr[1])

        mujoco.mj_forward(model, data)

        if fi in (0, 60, 100):
            pr_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'rh_right_palm')
            print(f"    f{fi}: target={(b_pos + R_bw @ hib_t[fi,0]).round(3)}  palm_xpos={data.xpos[pr_id].round(3)}")

        bz = data.xpos[bottle_id][2]
        max_lift = max(max_lift, bz - bottle_init_z)

        renderer.update_scene(data, cam)
        img1 = renderer.render()
        renderer.update_scene(data, cam2)
        img2 = renderer.render()
        combo = np.hstack([img1, img2])
        bgr = cv2.cvtColor(combo, cv2.COLOR_RGB2BGR)
        cv2.putText(bgr, f"[{fi}/{N}] lift={(bz-bottle_init_z)*100:.1f}cm con={data.ncon}   (L: cam145  R: cam45)",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)
        frames.append(bgr)

    print(f"  Max lift: {max_lift*100:.1f}cm, frames: {len(frames)}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_mp4 = os.path.join(OUT_DIR, 'v17_6dof.mp4')
    w = cv2.VideoWriter(out_mp4, fourcc, 15, (1280, 480))
    for fr in frames: w.write(fr)
    w.release()
    for idx in [0, 20, 40, 60, 80, 100, 120, min(150, len(frames)-1)]:
        if idx < len(frames):
            cv2.imwrite(os.path.join(OUT_DIR, f'kf_{idx:04d}.jpg'), frames[idx])
    print(f"  -> {out_mp4}")


if __name__ == "__main__":
    main()
