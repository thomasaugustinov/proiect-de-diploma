#!/usr/bin/env python
import argparse
import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
import sys

# Make repo root importable
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from utils.common import coef_dict_to_vertices
from data.datasets     import LmdbDataset
from models.flame      import FLAME, FLAMEConfig

def batch_coef2verts(coef_dict, flame, batch_size=8):
    """
    Convert (shape, exp, pose) → [b, L, V, 3] but *ignore* global head rotation.
    """
    N = next(iter(coef_dict.values())).shape[0]
    for i in range(0, N, batch_size):
        chunk = {k: v[i:i+batch_size].contiguous() for k,v in coef_dict.items()}
        yield coef_dict_to_vertices(
            chunk,
            flame,
            rot_repr='aa',
            ignore_global_rot=True    # <-- strip out yaw/pitch/roll
        )

def load_coefs(npz_path: Path, stats_file: Path):
    stats = {k: torch.tensor(v) for k,v in dict(np.load(stats_file)).items()}
    data  = dict(np.load(npz_path))
    pred  = torch.from_numpy(data['pred']).float()  # [N, L, D]
    exp, pose = pred[..., :50], pred[..., 50:]
    exp  = exp  * stats['exp_std'][None,None,:]  + stats['exp_mean'][None,None,:]
    # pose = pose * stats['pose_std'][None,None,:] + stats['pose_mean'][None,None,:]
 
    # ——— support 4-dim (rot_x,rot_y,rot_z,jaw) *or* 6-dim (rot_x…jaw,tx,ty)
    D = pose.shape[-1]
    pm = stats['pose_mean']
    ps = stats['pose_std']
    if D != pm.shape[0]:
        # slice off the translation dims (tx,ty) from the stats
        pm = pm[:D]
        ps = ps[:D]
    pose = pose * ps[None,None,:] + pm[None,None,:]
    # placeholder for shape (will overwrite from GT)
    shape = torch.zeros(exp.shape[0], exp.shape[1], 100)
    # return {'shape': shape, 'exp': exp, 'pose': pose}
    
    # — if you only predicted 4 dims, pad (tx,ty)=0 so FLAME sees 6 dims
    if pose.shape[-1] == 4:
        z = torch.zeros(*pose.shape[:-1], 2, dtype=pose.dtype, device=pose.device)
        pose = torch.cat([pose, z], dim=-1)
    return {'shape': shape, 'exp': exp, 'pose': pose}

def load_gt_coefs(gt_root: Path, split_file: Path, stats_file: Path, fps: int, n_motions: int):
    ds = LmdbDataset(
        lmdb_dir=str(gt_root),
        split_file=str(split_file),
        coef_stats_file=None,    # <-- do NOT let LMDB also normalize
        coef_fps=fps,
        n_motions=n_motions,
        crop_strategy='center',
        rot_repr='aa'
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    shapes, exps, poses = [], [], []
    for _, coef_pair, _ in loader:
        cu = coef_pair[1]  # the “sample” window
        shapes.append(cu['shape'])
        exps.append(cu['exp'])
        poses.append(cu['pose'])
    shape = torch.cat(shapes, 0)
    exp   = torch.cat(exps,   0)
    pose  = torch.cat(poses,  0)
    stats = {k: torch.tensor(v) for k,v in dict(np.load(stats_file)).items()}
    shape = shape * stats['shape_std'][None,None,:]  + stats['shape_mean'][None,None,:]
    exp   =   exp * stats['exp_std'][None,None,:]    + stats['exp_mean'][None,None,:]
    pose  =  pose * stats['pose_std'][None,None,:]   + stats['pose_mean'][None,None,:]
    return {'shape': shape, 'exp': exp, 'pose': pose}

def compute_lve(vp, vg, lip_ix):
    d  = (vp - vg).norm(dim=-1)            # [b,L,V]
    ld = d[..., lip_ix]                    # lip verts
    pf = ld.max(-1).values.reshape(-1)     # flatten
    return pf.mean().item()*1000, pf.std().item()*1000

def compute_fdd(vp, vg, upper_ix):
    std_p = vp.std(dim=1)  # [b, V, 3]
    std_g = vg.std(dim=1)
    df    = (std_p - std_g).abs().mean(-1)  # [b, V]
    return df[:, upper_ix].mean().item() * 1e5

def compute_mod(vp, vg, mouth_ix):
    # per-pair mouth opening error over *all* top/bottom lips
    yp = vp[..., 1]; yg = vg[..., 1]         # y-coords
    pairs = mouth_ix.reshape(-1,2)
    diffs = []
    for top,bot in pairs:
        op = (yp[..., top] - yp[..., bot]).reshape(-1)
        og = (yg[..., top] - yg[..., bot]).reshape(-1)
        diffs.append((op - og).abs())
    all_diff = torch.cat(diffs, dim=0)
    return all_diff.mean().item()*1000, all_diff.std().item()*1000

def compute_diversity(coefs, dim_exp=50):
    A = coefs['exp'].reshape(-1,dim_exp).numpy()
    P = coefs['pose'][...,:3].reshape(-1,3).numpy()
    return float(np.std(A,axis=0).mean()*1e4), float(np.std(P,axis=0).mean()*1e4)

def main(args):
    flame = FLAME(FLAMEConfig).to('cpu')

    # 1) load & de-norm predictions
    pred = load_coefs(args.pred_coefs, args.stats_file)

    # 2) drop conditioning frames
    n_prev = args.n_prev_motions
    for k in pred:
        pred[k] = pred[k][:, n_prev:, :]

    # 3) how many to predict
    n_pred = next(iter(pred.values())).shape[1]

    # 4) load GT and drop same conditioning
    gt_full = load_gt_coefs(
        gt_root    = args.gt_root,
        split_file = args.gt_root.parent/'test.txt',
        stats_file = args.stats_file,
        fps        = args.fps,
        n_motions  = n_prev + n_pred
    )
    gt = {k: v[:, n_prev:, :] for k,v in gt_full.items()}

    # 5) inject real shape into preds
    pred['shape'] = gt['shape']

    # 6) load landmark index sets
    IDX_DIR  = ROOT/'tools'
    lip_ix   = np.load(IDX_DIR/'lip_vertex_indices.npy')
    upper_ix = np.load(IDX_DIR/'upper_face_indices.npy')
    mouth_ix = np.load(IDX_DIR/'mouth_opening_indices.npy')

    # 7) compute verts and per-frame metrics
    lve_vals, fdd_vals, mod_vals = [], [], []
    for vp, vg in zip(
        batch_coef2verts(pred, flame, batch_size=4),
        batch_coef2verts(gt,   flame, batch_size=4),
    ):
        l,m   = compute_lve(vp, vg, lip_ix);      lve_vals.append((l,m))
        f     = compute_fdd(vp, vg, upper_ix);    fdd_vals.append(f)
        mm,ms = compute_mod(vp, vg, mouth_ix);    mod_vals.append((mm,ms))

    # 8) head-beat accuracy on *jaw-opening* velocity
    #    jaw-opening axis-angle = channels [3:6] of pose
    pred_jaw = pred['pose'][..., 3:6].numpy()   # [B, L, 3]
    gt_jaw   =   gt['pose'][..., 3:6].numpy()
    ang_p    = np.linalg.norm(pred_jaw, axis=-1)  # [B, L]
    ang_g    = np.linalg.norm(gt_jaw,   axis=-1)
    vp       = np.abs(np.diff(ang_p, axis=1))     # [B, L-1]
    vg       = np.abs(np.diff(ang_g, axis=1))

    def simple_peaks(v):
        return np.nonzero((v[1:-1] > v[:-2]) & (v[1:-1] > v[2:]))[0] + 1

    f1s = []
    for seq_p, seq_g in zip(vp, vg):
        pk_p = simple_peaks(seq_p)
        pk_g = simple_peaks(seq_g)
        tp   = len(set(pk_p) & set(pk_g))
        prec = tp / (len(pk_p) + 1e-8)
        rec  = tp / (len(pk_g) + 1e-8)
        f1s.append(2 * prec * rec / (prec + rec + 1e-8))
    ba = float(np.mean(f1s))

    # 9) diversity
    de, dh = compute_diversity(pred)

    # 10) final averages
    lve_m = np.mean([x for x,_ in lve_vals])
    lve_s = np.mean([s for _,s in lve_vals])
    fdd_a = np.mean(fdd_vals)
    mod_m = np.mean([x for x,_ in mod_vals])
    mod_s = np.mean([s for _,s in mod_vals])

    out = {
      'LVE_mm':      [lve_m,    lve_s],
      'FDD_x1e5':    fdd_a,
      'MOD_mm':      [mod_m,    mod_s],
      'BA':          ba,
      'Div_exp_x1e4':de,
      'Div_hp_x1e4': dh
    }
    Path(args.out).parent.mkdir(exist_ok=True, parents=True)
    with open(args.out,'w') as f:
        json.dump(out, f, indent=2)
    print(f"=> wrote metrics to {args.out}")

if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--pred-coefs',    type=Path, required=True)
    p.add_argument('--gt-root',       type=Path, required=True)
    p.add_argument('--stats-file',    type=Path, required=True)
    p.add_argument('--out',           type=Path, required=True)
    p.add_argument('--fps',           type=int, default=25)
    p.add_argument('--n-prev-motions',type=int, default=10,
                   help="number of conditioning frames to drop")
    args = p.parse_args()
    main(args)
