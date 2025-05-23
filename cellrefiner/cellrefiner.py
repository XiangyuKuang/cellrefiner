import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import scanpy as sc
import anndata
from sklearn.neighbors import kneighbors_graph
from scipy.stats import pearsonr
import squidpy as sq
from scipy.sparse import csr_matrix
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.special import kl_div
from sklearn.neighbors import NearestNeighbors
import tangram as tg
import seaborn as sns
from cellrefiner.util import cal_glvs,glvs,pre_cal1,sparsify,F_gc,V_xy,F_spot

class CellRefiner(object):
    """An object for transcriptomic data.

    :param adata_sc: anndata.AnnData object as input for scRNA-seq data
    :type adata_sc: class: anndata.AnnData
    :param adata_st: anndata.AnnData object as input for ST data i.e. Visium
    :type adata_st: class: anndata.AnnData
    :param db: ligand receptor

    """

    def __init__(self, adata_st=None, adata_sc=None, db=None):

        self.adata_st = adata_st
        self.adata_sc = adata_sc
        self.db = db

    def gen_w(self):
        """
        Generate affinity matrix
        """
        adata_sc = self.adata_sc
        
        if isinstance(adata_sc.X, csr_matrix):
            a = adata_sc.X.toarray()
        else:
            a = adata_sc.X

        db = self.db
        tl = np.zeros((a.shape[0], len(db['interaction_name'])))  # ligand expression matrix
        tr = np.zeros((a.shape[0], len(db['interaction_name'])))  # receptor expression matrix
        for i in range(len(db['interaction_name'])):
            int_name = db['interaction_name'][i].split('_')  # interaction
            lig = int_name[0]  # ligand
            rec = int_name[1:]  # receptor/s
            lig_ind = adata_sc.var_names == lig  # ligand indices as boolean array
            if sum(lig_ind) > 0:
                tl[:, i] = a[:, lig_ind].flatten()
                check = 0
                for j in range(len(rec)):  # see if all receptors are present
                    if sum(adata_sc.var_names == rec[j]) > 0:
                        check += 1

                if check == len(rec):
                    rec_ct = a[:, adata_sc.var_names == rec[0]]
                    for j in range(len(rec)):
                        rec_ct = np.minimum(rec_ct, a[:, adata_sc.var_names == rec[j]])

                    tr[:, i] = rec_ct.flatten()

        # calculate cell by cell affinity matrix
        W = np.add(np.matmul(tl, tr.T), np.matmul(tr, tl.T))  # T_L * T_R' + T_R * T_L'
        W = np.divide(W, np.amax(W))

        self.W = W

    def pp_cr(self, device="cuda:0",spatial_key = 'spatial', group="cell_class", Lx=5000, map=None, k = 5):
        self.k = k
        adata_sc = self.adata_sc
        adata_st = self.adata_st
        self.Lx = Lx
        sc.tl.rank_genes_groups(adata_sc, groupby=group, use_raw=False)
        markers_df = pd.DataFrame(adata_sc.uns["rank_genes_groups"]["names"]).iloc[0:100, :]
        markers = list(np.unique(markers_df.melt().value.values))
        tg.pp_adatas(adata_sc, adata_st, genes=markers) # produce adata.uns['overlap_genes'], required in belowing
        if map is None: # using tangram
            if device == "cpu":
                ad_map = tg.map_cells_to_space(adata_sc, adata_st,
                                            mode="cells",
                                            density_prior='rna_count_based',
                                            num_epochs=600,
                                            device="cpu",
                                            )
            else:
                ad_map = tg.map_cells_to_space(adata_sc, adata_st,
                                            mode="cells",
                                            density_prior='rna_count_based',
                                            num_epochs=600,
                                            device="cuda:0",
                                            # lambda_r=1
                                            )
            gmap = ad_map.X
        else:
            gmap = map # using input mapping
        
        x_coord = adata_st.obsm[spatial_key]
        #  estimate spot distance
        if Lx is None:
            nc = adata_st.shape[0]
            sq.gr.spatial_neighbors(adata_st,delaunay=True,spatial_key=spatial_key,coord_type='generic')
            distance_matrix = adata_st.obsp['spatial_distances']
            ds = np.zeros(nc)
            indices = distance_matrix.indices
            indptr = distance_matrix.indptr
            for cid in range(nc):
                j = indices[indptr[cid]:indptr[cid+1]]
                ds[cid] = np.mean(distance_matrix[cid,j])
            self.scale = 200/np.median(ds)*np.sqrt(k/5)
            x_coord = x_coord*self.scale
        else:
            self.scale = None
            scale = np.abs(np.max(x_coord[:, 0]) - np.min(x_coord[:, 0]))
            x_coord = x_coord / scale * self.Lx
        a = np.tile(x_coord[:, 0], (k, 1)).T.flatten()
        b = np.tile(x_coord[:, 1], (k, 1)).T.flatten()
        xs = np.concatenate(([a], [b]), axis=0).T
        xc = xs + np.random.normal(0, 20, size=xs.shape)
        neigh = NearestNeighbors(n_neighbors=5)
        neigh.fit(xc)
        x_id = neigh.kneighbors(xs)  # first entry is distance, second is indices
        x_id1 = []  # list of boolean arrays for neighboring spots
        for i in range(xs.shape[0]):
            x_id1.append(np.linalg.norm(xs - xs[i, :], axis=1) < 150)

        gmap1 = gmap.copy()
        mean_map = np.zeros(gmap1.shape[1])
        # create spot by cell index matrix for the top k cells
        cell5 = np.zeros((gmap.shape[1], k))
        for i in range(gmap1.shape[1]):
            cell5[i, :] = np.argpartition(gmap1[:, i], -k)[-k:]
            mean_map[i] = gmap1[cell5[i, :].astype(int), i].mean()
            gmap1[cell5[i, :].astype(int), :] = 0

        cell5m = cell5.flatten().astype(int)
        cell_codes = pd.Categorical(adata_sc.obs[group]).codes[cell5m]

        adata_sc1 = adata_sc[cell5m, adata_sc.uns['overlap_genes']]
        if 'highly_variable' in adata_sc1.var.columns: #check if highly variable genes exist
            adata_sc1 = adata_sc1[:, adata_sc1.var.highly_variable] 
        X_sc2m2 = adata_sc1.obsm['X_pca'].toarray()

        self.xc = xc  # cell coordinates
        self.xs = xs  # spot coordinates
        self.x_id1 = x_id1  # cell neighborhood assignments
        self.X_sc2m2 = X_sc2m2  # PCA matrix
        self.cell_codes = cell_codes  # cell type numbers
        self.cell5m = cell5m
        self.spots = cell5
        self.mean_map = mean_map#
        self.map = gmap

    def sim_cr(self, iterations=30, W=None, tissue_bound=0, dt=0.0028, m_val=125, rS=100):
        xc = self.xc
        xs = self.xs
        x_id1 = self.x_id1
        X_sc2m2 = self.X_sc2m2
        cell5m = self.cell5m

        if W is None:
            W = self.W
            W1 = W[cell5m, :]
            W1 = W1[:, cell5m]
            W1 = W1 / np.max(W1)
        else:
            W1 = W

        m_val = 125  # scaling param
        U0 = 0.1 / (2.85 / m_val)
        V0 = 1.1 / (2.85 / m_val)
        xi1 = 1.21 / (2.85 / m_val)
        xi2 = 1.9 / (2.85 / m_val)

        Sigma = np.array([[10000, 0], [0, 10000]])
        z_cutoff = 0.4  # level set cutoff for defining tissue boundary

        degree = np.diag(np.sum(W1, axis=1))
        L = degree - W1
        if np.linalg.det(L) > 0:
            q = pre_cal1(W1)
            H = sparsify(W1, q)
        else:
            H = np.zeros(np.shape(W1))

        pos_s = np.tile(xs, [iterations + 1, 1, 1])
        pos = np.tile(xc, [iterations + 1, 1, 1])  # coordinates, recorded each iterations
        F_gc_const = np.linspace(1, 0, iterations) ** 2
        z_val = z_cutoff * np.amax(cal_glvs(pos[0, :, :]))  # tissue boundary constraint
        for i in range(iterations):
            p = pos[i, :, :].copy()
            p += F_spot(pos[i, :, :], pos_s[i, :, :], rS*.7)  # add spot force to every cell
            for j in range(pos.shape[1]):
                for k in np.arange(0, xc.shape[0])[x_id1[j]]:
                    if j != k:
                        dv = V_xy(pos[i, k, :], pos[i, j, :], V0, U0, xi1, xi2)
                        p[j, :] += -dt * 7100 * dv
                        p[j, :] += 4 * F_gc_const[i] * F_gc(pos[i, k, :], pos[i, j, :], X_sc2m2[k, :],
                                                            X_sc2m2[j, :])  # gene force
                        p[j, :] += 4 * F_gc_const[i] * H[k, j]
            pos[i + 1, :, :] = p

            if tissue_bound == 1:
                z2 = np.empty(pos[i + 1, :, 0].shape)  # enforce tissue boundary
                for j in range(pos.shape[1]):
                    z2 += glvs(pos[i + 1, :, :], pos[0, j, :], Sigma)

                z_ind = z2 < z_val  # indices of cells outside tissues
                pos[i + 1, z_ind, :] = pos[i, z_ind, :] + 0.1 * (pos[i + 1, z_ind, :] - pos[i, z_ind, :])

        x_coord = self.adata_st.obsm['spatial']
        scale = np.abs(np.max(x_coord[:, 0]) - np.min(x_coord[:, 0]))
        if self.scale is None:
            self.pos = pos * scale / self.Lx
        else:
            self.pos = pos / self.scale
