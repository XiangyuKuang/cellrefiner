#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
from SpaceFlow import SpaceFlow
import seaborn as sns
from cellrefiner.util import 

class cellrefiner(object):
    
    def __init__(self,adata_st=None,adata_sc=None,db=None)
    
        self.adata_st=adata_st
        self.adata_sc=adata_sc
        self.db=db
    
    def gen_W(self)
        """
        Generate affinity matrix
        """
        adata_sc=self.adata_sc
        a=a.X.toarray()
        db=self.db
        db['interaction_name']=db['interaction_name'].apply(str.lower)
        TL=np.zeros((a.shape[0],len(db['interaction_name']))) # ligand expression matrix
        TR=np.zeros((a.shape[0],len(db['interaction_name'])))# receptor expression matrix
        for i in range(len(db['interaction_name'])):
            int_name=db['interaction_name'][i].split('_') # interaction
            lig=int_name[0] # ligand
            rec=int_name[1:] # receptor/s
            lig_ind=adata_sc.var_names==lig # ligand indices as boolean array
            if sum(lig_ind)>0:
                TL[:,i]=a[:,lig_ind].flatten()
                check=0
                for j in range(len(rec)): # see if all receptors are present
                    if sum(adata_sc.var_names==rec[j])>0:
                        check+=1

                if check==len(rec):
                    rec_ct=a[:,adata_sc.var_names==rec[0]]
                    for j in range(len(rec)):
                        rec_ct=np.minimum(rec_ct,a[:,adata_sc.var_names==rec[j]])

                    TR[:,i]=rec_ct.flatten()
                    
        # calculate cell by cell affinity matrix
        W=np.add(np.matmul(TL,TR.T),np.matmul(TR,TL.T)) # T_L * T_R' + T_R * T_L'
        W=np.divide(W,np.amax(W))
        
        self.W=W    
    
    def pp_cr(self):
        adata_sc=self.adata_sc
        adata_st=self.adata_st
        sc.tl.rank_genes_groups(adata_sc, groupby="cell_subclass", use_raw=False)
        markers_df = pd.DataFrame(adata_sc.uns["rank_genes_groups"]["names"]).iloc[0:100, :]
        markers = list(np.unique(markers_df.melt().value.values))
        tg.pp_adatas(adata_sc, adata_st,genes=markers)
        ad_map = tg.map_cells_to_space(adata_sc, adata_st,
            mode="cells",
            density_prior='rna_count_based',
            num_epochs=500,
            device="cuda:0",
            )
        x_coord=adata_st.obsm['spatial']
        a=np.tile(x_coord[:,0],(5,1)).T.flatten()
        b=np.tile(x_coord[:,1],(5,1)).T.flatten()
        xs=np.concatenate(([a],[b]),axis=0).T
        xc=xs+np.random.normal(0,20,size=xs.shape)
        neigh = NearestNeighbors(n_neighbors=5)
        neigh.fit(xc)
        x_id=neigh.kneighbors(xs)# first entry is distance, second is indices
        x_id1=[] # list of boolean arrays for neighboring spots
        for i in range(xs.shape[0]):
            x_id1.append(np.linalg.norm(xs-xs[i,:],axis=1)<150) 

        gmap=ad_map.X

        # create spot by cell index matrix for the top 5 cells
        cell5=np.zeros((gmap.shape[1],5))
        gmap1=gmap
        for i in range(gmap1.shape[1]):
            cell5[i,:]= np.argpartition(gmap1[:,i], -5)[-5:]
            gmap1[cell5[i,:].astype(int),:]=0

        cell5m=cell5.flatten().astype(int)
        cell_codes=pd.Categorical(adata_sc.obs['cell_subclass']).codes[cell5m]

        adata_sc1=adata_sc[cell5m,adata_sc.uns['overlap_genes']]
        adata_sc1=adata_sc1[:,adata_sc1.var.highly_variable] # should also check if highly variable genes exist
        X_sc2m2=adata_sc1.obsm['X_pca'].toarray()
        
        self.xc=xc # cell coordinates
        self.xs=xs # spot coordinates
        self.x_id1=x_id1 # cell neighborhood assignments
        self.X_sc2m2=X_sc2m2 # PCA matrix
        self.cell_codes=cell_codes # cell type numbers
        self.cell5m=cell5m
        
    def sim_cr(self,iterations=30):
        xc=self.xc
        xs=self.xs
        x_id1=self.x_id1
        X_sc2m2=self.X_sc2m2
        W=self.W
        
        W1=W[cell5m,:]
        W1=W1[:,cell5m]
        W1=W1/np.max(W1)
        
        m_val=125 # use 75 else, scaling param
        U0=0.1/(2.85/m_val)
        V0=1.1/(2.85/m_val)
        xi1=1.21/(2.85/m_val)
        xi2=1.9/(2.85/m_val)
        dt=20
        rS=150 # desired spot radius
        
        q=pre_cal1(W1)
        H=sparsify(W1,q)
        
        x1=xc
        pos_s=np.tile(xs,[iterations+1,1,1])
        pos=np.tile(x1,[iterations+1,1,1]) # coordinates, recorded each iterations
        F_gc_const=np.linspace(1,0,iterations)**2
        for i in range(iterations):
            p=pos[i,:,:].copy()
            p+=F_spot(pos[i,:,:],pos_s[i,:,:],rS) # add spot force to every cell
            for j in range(pos.shape[1]):
                for k in np.arange(0,xc.shape[0])[x_id1[j]]:
                    if j!=k:
                        dv=V_xy(pos[i,k,:],pos[i,j,:],V0,U0,xi1,xi2)
                        p[j,:]+=-dt*dv
                        p[j,:]+=4*F_gc_const[i]*F_gc(pos[i,k,:],pos[i,j,:],X_sc2m2[k,:],X_sc2m2[j,:])# gene force
                        p[j,:]+=4*F_gc_const[i]*a1[k,j]
            pos[i+1,:,:]=p
        
            z2=np.empty(pos[i+1,:,0].shape) # enforce tissue boundary
            for j in range(pos.shape[1]):
                z2+=glvs(pos[i+1,:,:],pos[0,j,:],Sigma)
        
            z_ind=z2<z_val # indices of cells outside tissues
            pos[i+1,z_ind,:]=pos[i,z_ind,:]+0.1*(pos[i+1,z_ind,:]-pos[i,z_ind,:])
            
        self.pos=pos
    
    
    


# In[ ]:




