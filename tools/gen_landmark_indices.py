# tools/gen_landmark_indices.py
import numpy as np
import torch
from scipy.spatial import cKDTree
from pathlib import Path

import sys
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

# import FLAME and its config namespace
from models.flame import FLAME, FLAMEConfig

def main():
    # 1) instantiate FLAME on CPU
    device = 'cpu'
    flame = FLAME(FLAMEConfig).to(device)

    # 2) load the landmark‐to‐face‐idx lookup (static + dynamic)
    emb = np.load(FLAMEConfig.flame_lmk_embedding_path, allow_pickle=True)[()]
    full_lmk_faces_idx = emb['full_lmk_faces_idx']   # (68,) face‐indices

    # 3) render one neutral FLAME mesh and its 3D landmarks
    #    shape,exp,pose all zeros
    with torch.no_grad():
        verts, _, lm3d = flame(
            torch.zeros(1, FLAMEConfig.n_shape),
            torch.zeros(1, FLAMEConfig.n_exp),
            torch.zeros(1, 6),  # axis‐angle head+jaw
            return_lm2d=False,
            return_lm3d=True
        )
    verts = verts[0].cpu().numpy()   # (V,3)
    lm3d  = lm3d[0].cpu().numpy()    # (68,3)

    # 4) build a KD‐tree on vertices, map each 3D‐landmark to nearest vertex
    tree = cKDTree(verts)
    lmk2vert = tree.query(lm3d)[1]    # (68,) vertex‐indices

    # 5) slice out the subsets
    #    * LIPS: landmarks 48–67
    lip_inds   = lmk2vert[48:68]
    #    * UPPER‐FACE: landmarks 17–47
    upper_inds = lmk2vert[17:48]
    #    * MOUTH OPENING: pairwise (48/49, 50/51, … , 66/67)
    mouth_inds = []
    for t,b in zip(range(48,68,2), range(49,68,2)):
        mouth_inds += [ int(lmk2vert[t]), int(lmk2vert[b]) ]

    # 6) save into tools/
    tools_dir = Path(__file__).parent
    np.save(tools_dir / 'lip_vertex_indices.npy',   lip_inds)
    np.save(tools_dir / 'upper_face_indices.npy',   upper_inds)
    np.save(tools_dir / 'mouth_opening_indices.npy', mouth_inds)

    # 6) save into tools/
    tools_dir = Path(__file__).parent
    np.save(tools_dir / 'lip_vertex_indices.npy',          lip_inds)
    np.save(tools_dir / 'upper_face_indices.npy',          upper_inds)
    np.save(tools_dir / 'mouth_opening_indices.npy',       mouth_inds)

    # 7) also save the two center‐of‐mouth vertices (landmarks 51 and 57)
    mouth_center_top = int(lmk2vert[51])
    mouth_center_bot = int(lmk2vert[57])
    np.save(tools_dir / 'mouth_center_top.npy',  np.array([mouth_center_top]))
    np.save(tools_dir / 'mouth_center_bot.npy',  np.array([mouth_center_bot]))

    print(f"Saved lip:{len(lip_inds)} upper:{len(upper_inds)} mouth_pairs:{len(mouth_inds)//2}")

if __name__ == '__main__':
    main()
