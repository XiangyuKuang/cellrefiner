from typing import Optional, Union, List
import numpy as np
import pandas as pd
from anndata import AnnData
from .SEM import SEM
from scipy.sparse import csr_matrix

def contact_signal(df_ligrec: pd.DataFrame,
                   sem: Optional[SEM] = None,
                   adata: Optional[AnnData] = None,
                   contact_key: Optional[str] = 'contacts',
                   lr_delimiter: str = '-',
                   heteromeric_delimiter: str = '_'):
    '''
    Contact signal inference

    signal matrix is stored as .obsp['ligand-receptor']

    signal names are stored as .uns['contact_signal_info']

    rows are sender cells, columns are receiver cells
    '''

    if df_ligrec.shape[0] == 0:
        raise ValueError("empty ligand-receptor DB")
    if sem is None: # sem is not provided, using adata contact matrix
        contact_matrix = adata.obsp[contact_key]
    else: # sem is provided
        if adata is None: # adata is not provided, use sem.adata
            adata = sem.adata
        else: # adata is provided, use input adata
            if adata is not sem.adata: # check if same adata
                Warning('Provide adata is not an attribute of sem, sem.adata will be unchanged')
        if sem.contact_matrix is None:
            print('compute cell-cell contact')
            sem.compute_contact()
        else:
            contact_matrix = sem.contact_matrix
    df_ligrec = df_ligrec.copy()
    # get cell pair index
    nc = adata.shape[0]
    indices = contact_matrix.indices
    indptr = contact_matrix.indptr
    ci = []
    cj = []
    for i in range(nc):
        j = indices[indptr[i]:indptr[i+1]]
        ci.append(np.tile(i,len(j)))
        cj.append(j)
    ci = np.concatenate(ci)
    cj = np.concatenate(cj)

    # contact signal
    lr_keys = []
    I = np.ones(df_ligrec.shape[0],dtype=bool)
    # ligand-receptors pairs
    for i in range(df_ligrec.shape[0]):
        l = df_ligrec.iloc[i,0]
        r = df_ligrec.iloc[i,1]
        l_data = np.prod(adata[ci,l.split(heteromeric_delimiter)].X.toarray(),axis=1)
        r_data = np.prod(adata[cj,r.split(heteromeric_delimiter)].X.toarray(),axis=1)
        key = f'{l}{lr_delimiter}{r}'
        sig_mat = csr_matrix((l_data*r_data,indices.copy(),indptr.copy()), shape=(nc, nc)) # .copy() is necessary. eliminate_zeros() removes indices and indptr inplace
        sig_mat.eliminate_zeros()
        I[i] = sig_mat.nnz>0
        if I[i]:
            adata.obsp[key] = sig_mat
            lr_keys.append(key)
    df_ligrec = df_ligrec[I]

    # pathway and total
    pth_keys = df_ligrec.iloc[:,2].unique().tolist()
    for n,pth in enumerate(pth_keys):
        lr_idx = np.where(df_ligrec.iloc[:,2]==pth)[0]
        data = csr_matrix((nc,nc))
        for i in lr_idx:
            l = df_ligrec.iloc[i,0]
            r = df_ligrec.iloc[i,1]
            data += adata.obsp[f'{l}{lr_delimiter}{r}'].copy()
        adata.obsp[pth] = data.copy()
        if n == 0:
            total = data.copy()
        else:
            total += data.copy()
    adata.obsp['total'] = total
    adata.uns['contact_signal_info'] = {'lr_pair': lr_keys, 'pathway': pth_keys, 'total': ['total'], 'db': df_ligrec}

def cluster_communication(adata: AnnData,
                          cluster_key: str,
                          signal: str = 'total',
                          n_permutations: int = 100,
                          seed: int = 0):
    """
    Cluster-cluster communication

    add cluster communication to .uns
    """

    cluster_list = list(adata.obs[cluster_key].cat.categories)
    cluster_cell = adata.obs[cluster_key].to_numpy()
    sig_mat = adata.obsp[signal]
    rng = np.random.default_rng(seed)
    tmp_df, tmp_p_value = summarize_cluster(sig_mat,cluster_cell,cluster_list,rng,n_permutations=n_permutations)
    adata.uns[cluster_key+'-'+signal] = {'communication_matrix': tmp_df, 'communication_pvalue': tmp_p_value}

def signal_vector(adata: AnnData,
                  signal_type: Optional[Union[List,str]] = ['lr_pair','pathway','total'],
                  return_output=False):
    """
    Compute the sender signals and receiver signals for each cells

    add 'sender_signal' 'receiver_signal' to .obsm
    """

    signal_type = [signal_type] if type(signal_type) is str else signal_type
    signal_list = []
    for key in signal_type:
        signal_list+=adata.uns['contact_signal'][key]
    sdim = len(signal_list)
    signal_vec_s = np.zeros((adata.shape[0], sdim))
    signal_vec_r = np.zeros((adata.shape[0], sdim))
    for si,signal in enumerate(signal_list):
        signal_vec_s[:,si] = np.sum(adata.obsp[signal].toarray(),axis=1)# sender signal
        signal_vec_r[:,si] = np.sum(adata.obsp[signal].toarray(),axis=0)# receiver signal
    df_s = pd.DataFrame(index = adata.obs.index, columns=signal_list,data=signal_vec_s)
    df_r = pd.DataFrame(index = adata.obs.index, columns=signal_list,data=signal_vec_r)
    adata.obsm['sender_signal'] = df_s
    adata.obsm['receiver_signal'] = df_r
    print("add 'sender_signal' 'receiver_signal' to .obsm")
    if return_output:
        return df_s, df_r

def summarize_signal(adata: AnnData, cluster_key: str):
    df_r = pd.DataFrame(index = adata.obs.index, columns=adata.uns['contact_signal'])
    df_s = pd.DataFrame(index = adata.obs.index, columns=adata.uns['contact_signal'])
    for sig_key in adata.uns['contact_signal']:
        df_r[sig_key] = adata.obsp[sig_key].toarray().sum(axis=0)# receiver signal
        df_s[sig_key] = adata.obsp[sig_key].toarray().sum(axis=1)# sender signal
    selected_columns = [col for col in adata.uns['contact_signal'] if len(col.split('-')) >1]
    df_r_sel = df_r[selected_columns]
    df_s_sel = df_s[selected_columns]
    df_r_sel['cell_type'] = adata.obs[cluster_key]
    df_s_sel['cell_type'] = adata.obs[cluster_key]
    return df_r_sel.groupby('cell_type').mean(),df_s_sel.groupby('cell_type').mean()
    

def summarize_cluster(X, clusterid, clusternames, rng, n_permutations):
    # Input a sparse matrix of cell signaling and output a pandas dataframe
    # for cluster-cluster signaling
    n = len(clusternames)
    X_cluster = np.empty([n,n], float)
    p_cluster = np.zeros([n,n], float)
    for i in range(n):
        tmp_idx_i = np.where(clusterid==clusternames[i])[0]
        for j in range(n):
            tmp_idx_j = np.where(clusterid==clusternames[j])[0]
            X_cluster[i,j] = X[tmp_idx_i,:][:,tmp_idx_j].mean()
            
    for i in range(n_permutations):
        clusterid_perm = rng.permutation(clusterid)
        X_cluster_perm = np.empty([n,n], float)
        for j in range(n):
            tmp_idx_j = np.where(clusterid_perm==clusternames[j])[0]
            for k in range(n):
                tmp_idx_k = np.where(clusterid_perm==clusternames[k])[0]
                X_cluster_perm[j,k] = X[tmp_idx_j,:][:,tmp_idx_k].mean()
        p_cluster[X_cluster_perm >= X_cluster] += 1.0
    p_cluster = p_cluster / n_permutations
    df_cluster = pd.DataFrame(data=X_cluster, index=clusternames, columns=clusternames)
    df_p_value = pd.DataFrame(data=p_cluster, index=clusternames, columns=clusternames)
    return df_cluster, df_p_value