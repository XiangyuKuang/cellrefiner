from typing import Optional, Union, List, Dict
from math import pi, sqrt
from scipy.spatial import Delaunay, distance
from scipy.sparse import lil_matrix, coo_matrix, csr_matrix
from matplotlib import colormaps
from matplotlib.colors import to_rgb
import numpy as np
from numba import cuda
import pickle
import os
from .sem_util import AlphaShape
from sklearn.neighbors import NearestNeighbors
from anndata import AnnData
import warnings

class CellBase:
    """Base class for cell morphology representations"""
    
    adata: AnnData
    """Linked annotation data"""
    nc: int
    """cell number"""
    dim: int 
    """spatial dimension"""
    xc: np.ndarray
    """cell coordinates. shape=(nc,dim)"""
    ctype: np.ndarray
    """cell type code. shape=(nc)"""
    ctype_list: np.ndarray
    """cell type category. shape=(nct)"""
    color_list: np.ndarray
    """colors for cell type category. shape=(nct,3)"""
    ne: int
    """element number"""
    xe: np.ndarray
    """element coordinates. shape=(ne,dim)"""
    ecid: np.ndarray
    """id of cell to which each element belongs. shape: (ne)"""
    ceidn: np.ndarray
    """first elements id of each cell. shape=(nc+1)
    
    Elements of cell i can be retrieve by ceidn[i]:ceidn[i+1]. 
    
    Number of elements of cell i is ceidn[i+1]-ceidn[i]"""
    scale: float
    """normalization scaling factor"""
    deltax: np.ndarray
    """normalization scaling offset. shape=(1,dim)"""
    contact_matrix: csr_matrix
    """cell-cell contact matrix. shape=(nc,nc)"""
    spatial_distances: Dict[str, csr_matrix]
    """dictionary to store cell-cell distance matrix"""
    alphashape: List[AlphaShape]
    """list of alpha-shape object for each cell. len=nc"""
    alphashape_info: Dict[str, Union[bool, float, int, None]]
    """information of alpha-shape"""
    alpha_radius : np.ndarray
    """alpha radius of each cell"""
    ns_default: int
    """number of augment points for alpha-shape"""
    def __init__(self,
                 adata: Optional[AnnData] = None,
                 xc: Optional[np.ndarray] = None,
                 ctype: Optional[np.ndarray] = None,
                 cluster_key: str = 'leiden',
                 spatial_key: str = 'spatial',
                 color_list: Optional[np.ndarray] = None):
        """ Initialize common attributes"""
        
        self.adata = adata
        self.contact_matrix = None
        self.spatial_distances  = dict()
        self.alphashape_info = {'computed': False, 'alpha': None, 'ns': None, 'r': None}
        self.ns_default = 0
        self.scale = 1.
        self.deltax = np.zeros(2)
        
        # Initialize common attributes
        if self.adata is not None:
            self._init_from_adata(cluster_key,spatial_key)
        else:
            self._init_from_direct_inputs(xc,ctype,color_list)

        # Set default colors if not provided
        if self.color_list is None:
            self.set_color()

        # check 3d
        if self.dim > 2:
            warnings.warn('xc is 3d, use first two dim')
            self.dim = 2
            self.xc = self.xc[:,[0,1]]
        self.alpha_radius = np.zeros(self.nc)
        
    def _init_from_adata(self,cluster_key,spatial_key):
        """Initialize properties from AnnData object"""
        if spatial_key in self.adata.obsm:
            self.xc = self.adata.obsm[spatial_key].astype(np.float32)
            self.nc, self.dim = self.xc.shape
        else:
            raise KeyError(f"Spatial key '{spatial_key}' not found in adata.obsm")
        
        if cluster_key in self.adata.obs:
            self.ctype = self.adata.obs[cluster_key].cat.codes.to_numpy()
            self.ctype_list = self.adata.obs[cluster_key].cat.categories.to_numpy()
        else:
            raise KeyError(f"Cluster key '{cluster_key}' not found in adata.obs")
        
        if f'{cluster_key}_colors' in self.adata.uns:
            color_list = self.adata.uns[f'{cluster_key}_colors']
            if isinstance(color_list[0], str):
                color_list = np.array([to_rgb(x) for x in color_list])
            self.color_list = color_list

    def _init_from_direct_inputs(self,xc,ctype,color_list):
        """Initialize properties from direct inputs"""
        if xc is None:
            raise ValueError("Either adata or xc must be provided")
        else:
            self.xc = xc.astype(np.float32)
            self.nc, self.dim = xc.shape
        
        self.ctype = ctype
        if self.ctype is None:
            warnings.warn('ctype are not provided, set cell type to 0')
            self.ctype = np.zeros(self.nc, dtype=int)
        self.ctype_list = np.unique(self.ctype)

        self.color_list = color_list
        if self.color_list is not None:
            if isinstance(self.color_list[0], str):
                self.color_list = np.array([to_rgb(x) for x in self.color_list])

    def set_color(self, cmap_name: str = 'Set1') -> None:
        """
        Set a color map for `sem.ctype` (cell types)

        Parameters
        -------
        cmap_name: str
            Name of a matplotlib colormap (`matplotlib.colormaps`)
        """
        if cmap_name not in colormaps:
            cmap_name = 'Set1'
        cmap = colormaps[cmap_name]
        self.color_list = cmap(np.linspace(0, 1, len(self.ctype_list)))[:, :3]

    def compute_contact(self,
                        k: int = 8,
                        d_th: Optional[float] = None,
                        add_key: str = 'contacts') -> None:
        """
        Compute cell-cell contacts

        add .contact_matrix (csr_matrix)

        stored in adata.obsp[add_key]

        Parameters
        ------
        k : int, default=8
            Number of neighbors
        d_th : Optional[float], default=None
            Distance threshold
        add_key : str, default='contacts'
            Key for storing cell-cell contacts matrix to .obsp
        """
        if d_th is None:
            d_th = 2*self._get_e_radius() 
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(self.xe)
        distances, indices = nbrs.kneighbors(self.xe)
        row_indices = []
        col_indices = []
        data = []
        for ci in range(self.nc):
            neighbor = indices[self.ceidn[ci]:self.ceidn[ci+1], 1:].flatten()
            d_contact = distances[self.ceidn[ci]:self.ceidn[ci+1], 1:].flatten()
            contact_eid = np.unique(neighbor[d_contact<=d_th])
            contact_cid = self.ecid[contact_eid]
            for cj in contact_cid[contact_cid!=ci]:
                row_indices.append(ci)
                col_indices.append(cj)
                data.append(1)

        # make symmetrical
        contact_matrix = coo_matrix((data, (row_indices, col_indices)), shape=(self.nc, self.nc))
        self.contact_matrix = (contact_matrix+contact_matrix.T)/2 # contact_matrix is csr_matrix. tocoo() do not have .coords
        
        # if linked with adata, add contact_matrix to .obsp[add_key]
        if self.adata is not None:
            print(f"add .obsp['{add_key}'], .uns['{add_key}']")
            self.adata.obsp[add_key] = self.contact_matrix
            self.adata.uns[add_key] = {'k':k, 'd_th':d_th}

    def _get_e_radius() -> float: # replaced in sub-class
        """get default d_th for computing contact and alphashape"""
        return 1.0
    
    def compute_distance(self,
                         method: str,
                         k: int = 3,
                         return_distances: bool = False) -> Union[csr_matrix, None]:
        """
        Compute cell-cell distances in contact matrix

        Add .spatial_distances[method] (csr_matrix)

        Parameters
        ------
        method : str
            valid methods: 'knn', 'delaunay', 'contact'
        k : int, default: 3
            k for knn
        return_distances : bool, default: False
            Whether to return the distances matrix
        
        Return
        ------
        distance_matrix : csr_matrix
            Cell-cell distances if return_distances is True, None otherwise
        """

        xc = self.xc
        distance_matrix = lil_matrix((self.nc, self.nc))
        assert method in ('knn', 'delaunay', 'contact'), f"method must be 'knn' or 'delaunay', got {method}"
        if method == 'knn':
            nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(xc)
            distances,indices = nbrs.kneighbors(xc)
            for i in range(self.nc):
                for nj,j in enumerate(indices[i,1:]):
                    distance_matrix[i, j] = distances[i,nj+1]
                    distance_matrix[j, i] = distances[i,nj+1]
        
        if method == 'delaunay':
            tri = Delaunay(xc)
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i + 1, 3):
                        d = distance.euclidean(xc[simplex[i]],xc[simplex[j]])
                        distance_matrix[simplex[i], simplex[j]] = d
                        distance_matrix[simplex[j], simplex[i]] = d

        if method == 'contact':
            if self.contact_matrix is None:
                raise ValueError('contact is not computed')
            else:
                indices = self.contact_matrix.indices
                indptr = self.contact_matrix.indptr
                for i in range(self.nc):
                    for j in indices[indptr[i]:indptr[i+1]]:
                        d = distance.euclidean(xc[i],xc[j])
                        distance_matrix[i, j] = d
                        distance_matrix[j, i] = d
        if return_distances:
            return distance_matrix.tocsr()
        else:
            self.spatial_distances[method] = distance_matrix.tocsr()
    
    def compute_alphashape(self,
                           alpha: Optional[Union[np.ndarray,float]] = None,
                           ns: Optional[int] = None,
                           r: Optional[float] = None) -> None:
        """
        Compute alpha-shape for each cell
        
        Add .alphashape: List[AlphaShape]
        """
        if r is None:
            r = self._get_e_radius()/2
        if ns is None:
            ns = self.ns_default
        alpha_array = np.full(self.nc, alpha) if isinstance(alpha, float) else alpha
        # Check if recomputing alphashape are needed
        if not self.alphashape_info['computed'] or self.alphashape_info['ns']!=ns or self.alphashape_info['r']!=r:
            print(f"Computing alpha-shape with parameters: alpha={alpha}, ns={ns}, r={r}")
            xe = self.xe*self.scale+self.deltax
            self.alphashape = []
            for cid in range(self.nc):
                alpha_i = alpha_array[cid] if alpha_array is not None else None
                shp = AlphaShape(xe[self.ceidn[cid]:self.ceidn[cid+1]],alpha=alpha_i,ns=ns,r=r*self.scale)
                if alpha_i is None:
                    shp.update(2*shp.alpha_best)
                self.alpha_radius[cid] = shp.alpha
                self.alphashape.append(shp)
        elif self.alphashape_info['alpha']!=alpha:
            print(f"Updating alpha to {alpha}")
            for cid,shp in enumerate(self.alphashape):
                alpha_i = alpha_array[cid] if alpha_array is not None else None
                if alpha_i is None:
                    if shp.alpha_best is None:
                        shp.optimize_alpha()
                    alpha_i = 2*shp.alpha_best
                shp.update(alpha_i)
                self.alpha_radius[cid] = shp.alpha
        # else do nothing

        # update alphashape_info
        self.alphashape_info['computed'] = True
        self.alphashape_info.update({'alpha':alpha, 'ns':ns, 'r':r})

    def get_alpha(self) -> Union[np.ndarray, None]:
        """
        Get alpha radius of each cell

        Return
        ------
        alpha : Union[np.ndarray, None]
            Alpha radius of each cell if alphashapes have been computed, otherwise None
        """
        if not self.alphashape_info['computed']:
            warnings.warn('alphashape is not computed')
        return self.alpha_radius
    
    def get_area(self) -> Union[np.ndarray, None]:
        """
        Get cell areas

        Return
        ------
        area : Union[np.ndarray, None]
            Cell areas if alphashapes have been computed, otherwise None
        """
        if self.alphashape_info['computed']:
            area = np.zeros(self.nc)
            for i in range(self.nc):
                area[i] = self.alphashape[i].get_area()
            return area
        else:
            warnings.warn('alphashape is not computed')
            return None
    
    def get_elements(self, i: int) -> np.ndarray:
        '''
        Get elements of cell i

        Parameters
        ------
        i : int 
            Cell index
        
        Return 
        ------
        xe : np.ndarray 
            Element coordinates of cell i
        '''
        return self.xe[self.ceidn[i]:self.ceidn[i+1]]*self.scale+self.deltax
    
    def update_xc(self) -> None:
        """
        Update .xc (cell coordinates)
        """
        self.xc = np.array([np.mean(self.xe[self.ceidn[i]:self.ceidn[i+1]], axis=0) for i in range(self.nc)])

class SEM(CellBase):
    """Subcellular Element Method"""

    sim_name: str
    """simulation name"""
    t: int
    """time step""" 
    param: dict
    """simulation parameters"""

    def __init__(self, 
                 ne_per_cell: int,
                 re: float,
                 rd_ratio: float = 2.5,
                 adata: Optional[AnnData] = None,
                 cluster_key: str = 'leiden',
                 spatial_key: str = 'spatial',
                 embedding_key: str = 'X_pca',
                 xc: Optional[np.ndarray] = None,
                 ctype: Optional[np.ndarray] = None,
                 sim_name: str = 'untitled',# param: dict = {}
                 seed: int = 1):
        """
        Create a SEM object

        Parameters
        -------
        ne_per_cell : int, default: 20
            Number of elements per cell
        re : float
            Element radius
        rd_ratio : float
            Cell radius-distance ratio

            rd_ratio>2: cell radius < cell distance/2, tissue with gaps

            rd_ratio=2: cell radius = cell distance/2, no gaps (confluent tissue)

            rd_ratio<2: cell radius > cell distance/2, overcrowded
        adata : Anndata
            Anndata with .obsm[spatial_key] for cell coordinates, .obs[cluster_key] for cell types, .obsm[embedding_key] for low-dim embedding

            If not provided, xc and ctype are required
        xc : Optional[np.ndarray]
            cell coordinates. Ignored, if adata.obsm[spatial_key] is provided
        ctype : Optional[np.ndarray]
            cell types.  Ignored, if adata.obs[cluster_key] is provided
        cluster_key : str, default: 'leiden'
            Key for cell type in .obs
        spatial_key : str, default: 'spatial'
            Key for spatial coordinates in .obsm
        embedding_key : str, default: 'X_pca'
            Key for low-dim embedding in .obsm, used for computing gene similarity
        sim_name : str, default: 'untitled'
            Simulation name
        """

        super().__init__(adata, xc, ctype, cluster_key, spatial_key, None)
        # Initialize simulation-specific properties
        # element info
        self.ne = ne_per_cell * self.nc # total number of elements
        self.ecid = np.repeat(np.arange(self.nc, dtype=np.int32), ne_per_cell) # id of cell to which each element belongs
        self.ceidn = np.insert(np.cumsum([ne_per_cell]*self.nc), 0, 0)# first elements id of each cell. Elements of cell i can be retrieve by ceidn[i]:ceidn[i+1]. Number of elements of cell i is ceidn[i+1]-ceidn[i]
        self.xe = np.zeros((self.ne, self.dim), dtype=np.float32) # elements coordinates, n_element*dim
        # cell radius
        if self.nc > 2:
            # estimate cell radius by Delaunay
            distance_matrix = self.compute_distance('delaunay', return_distances=True)
            dc = np.zeros(self.nc)
            for cid in range(self.nc):
                _,j=distance_matrix[cid].nonzero()
                dc[cid] = np.mean(distance_matrix[cid,j]) if len(j)>0 else np.nan # some points might overlap with others
            rc = np.median(dc)/rd_ratio
            # rd_ratio>2: cell radius < cell distance/2, tissue with gaps
            # rd_ratio=2: cell radius = cell distance/2, no gaps (confluent tissue)
            # rd_ratio<2: cell radius > cell distance/2, overcrowded
        elif self.nc == 2:
            # only two cells
            rc = distance.euclidean(self.xc[0],self.xc[1])/2
        else:
            # only one cell
            rc = 1
        self.rc = rc
        rc_n = np.sqrt(ne_per_cell*(re/2)**2)# to-do: better estimation for steady state radius
        self.scale = rc/rc_n
        self.deltax = np.mean(self.xc, axis=0)
        self.xc = (self.xc-self.deltax)/self.scale ## scaling xc to xc_n/rc_n = xc/rc
        ## random number generator
        self.rng = np.random.default_rng(seed)
        self.rng_seed = seed
        ## deploy elements to the spherical region around each cell coordinates
        for cid in range(self.nc):
            # generate element in a spherical region following uniform distribution
            ne_i = self.ceidn[cid+1] - self.ceidn[cid]
            r = rc_n*np.sqrt(self.rng.uniform(0, 1, size=(ne_i, 1))) #np.sqrt() # cell_r = rc_n*self.ne_per_cell[i]/ne_per_cell
            phi = self.rng.uniform(-pi, pi, size=(ne_i, 1))
            xe = np.concatenate((r*np.cos(phi), r*np.sin(phi)), axis=1)
            xe = xe-np.mean(xe, axis=0)+self.xc[cid] # move initial element to cell center
            self.xe[self.ceidn[cid]:self.ceidn[cid+1]] = xe.astype(np.float32)

        # simulation info
        self.sim_name = sim_name
        self.t = 0
        self.param = dict()

        # adhesion based on gene simarity
        if self.adata is None:
            self.corr_matrix = np.ones((self.nc,self.nc), dtype=np.float32)
        else:
            X_em = self.adata.obsm[embedding_key]# default PCA matrix
            corr_matrix = np.corrcoef(X_em)
            c_min = 0.05
            corr_matrix[corr_matrix<c_min] = c_min
            self.corr_matrix = corr_matrix.astype(np.float32) # gene simarity matrix, (n_c*n_c)
            
        self.ns_default = 10 # alphashape default param
    
    def __repr__(self):
        return f'Simulation Name: {self.sim_name}\nt: {self.t}\nCell Number: {self.nc}\nElement Number: {self.xe.shape[0]}\nDim: {self.dim}\nParameters: {self.param}\nContact Matrix: {self.contact_matrix.__repr__()}'

    def _get_e_radius(self) -> float:
        """get SEM default d_th for computing contact and alphashape"""
        if len(self.param) > 0:
            d_th = self.param['rm_inter']
        else:
            warnings.warn('rm_inter is not provided, using d_th = 1')
            d_th = 1.0
        return d_th
    
    def sim_gpu(self, param: dict, T: int) -> None:
        """
        Implement SEM simulation

        Parameters
        ------
        param : dict
            Parameters
        T : int
            Time steps
        """
        self.param = param
        # get parameters
        rm_intra = param["rm_intra"]
        rm_inter = param["rm_inter"]
        dt = param["dt"]
        sigma = param["sigma"]
        gamma = param["gamma"]
        alpha_max,alpha_min = param["alpha"]
        
        cmax = self.corr_matrix.max()
        cmin = self.corr_matrix.min()
        if cmax==cmin:
            # corr_matrix is constant, set alpha to ones
            alpha = alpha_max*np.ones_like(self.corr_matrix)
        else:
            # scale corr_matrix to [alpha_min,alpha_max]
            alpha = (alpha_max-alpha_min)/(cmax-cmin)*self.corr_matrix+(alpha_min*cmax-alpha_max*cmin)/(cmax-cmin)
        sigmadt = sqrt(dt) * sigma

        # transfer array to gpu
        d_xe = cuda.to_device(self.xe)
        d_xe_F = cuda.to_device(self.xe)
        d_ecid = cuda.to_device(self.ecid)
        d_alpha = cuda.to_device(alpha)

        # gpu thread number
        tpb = 128
        bpg = 128
        # iteration
        cuda.synchronize()
        for t in range(T):
            x_randt = cuda.to_device((sigmadt*np.sqrt((T-t)/T) * self.rng.normal(0, 1, size=self.xe.shape)).astype(np.float32))#*self.cell_size[self.ecid,np.newaxis]
            dynamics2d_gpu2[bpg, tpb](d_xe, d_xe_F, d_ecid, d_alpha, gamma, x_randt, rm_intra, rm_inter, dt)
            cuda.synchronize()
            # var:t-1, var_F:t
            d_xe[:, :] = d_xe_F # update xe to t
            cuda.synchronize()
            # if self.t % vis_interval ==0:
            #     print(self.t)
            self.t += 1
        # close
        cuda.synchronize()
        self.xe = d_xe.copy_to_host()
        self.update_xc()
        self.alphashape_info['computed'] = False # marks alpha shapes need to be updated

    def save_sim(self) -> None:
        """
        Save simulation
        """
        filename = f'{self.sim_name}_{self.t}'
        if os.path.exists(filename+'.pkl'):
            warnings.warn(f"File '{filename}' already exists.")
            filename = filename + '_temp'
        filename = filename+'.pkl'

        with open(filename, 'wb') as f:
            data = {
                'xe': self.xe,
                'ceidn': self.ceidn,
                'ecid': self.ecid,
                'param': self.param,
                'scale': self.scale,
                'deltax': self.deltax
            }
            if self.alphashape_info['computed']:
                data['alpha_radius']= self.alpha_radius
            pickle.dump(data, f)
        print(f"saved as {filename}")

    def load_sim(self, sim_name: str , t: float, path: str = '.', rename: bool = True) -> None:
        '''
        Restore a simulation from `{path}/{sim_name}_{t}.pkl`

        Parameters
        ------
        sim_name : str
            Name of simulation
        t : float
            Time point
        path : str, default: '.'
            Path to simulation data
        rename : bool, default: True
            If True, rename the `sem` to `sim_name`
        '''
        filename = f'{path}/{sim_name}_{t}.pkl'
        with open(filename, 'rb') as f:
            print(f'load sim data from {filename}')
            data = pickle.load(f)
            self.xe = data['xe']
            self.ceidn = data['ceidn']
            self.ecid = data['ecid']
            self.scale = data['scale']
            self.deltax = data['deltax']
            if 'param' in data:
                print('.param loaded')
                self.param = data['param']
            if 'alpha_radius' in data:
                print('.alpha_radius loaded')
                self.compute_alphashape(alpha=data['alpha_radius'])
        self.t = t
        self.update_xc()
        if rename:
            print(f'Simulation renamed as {sim_name}')
            self.sim_name = sim_name

@cuda.jit
def dynamics2d_gpu2(xe, xe_F, ecid, alpha, gamma, x_randt, rm_intra, rm_inter, dt):
    """
    Simulation function
    """
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    ne = xe.shape[0]
    for i in range(start, ne, stride):
        cid = ecid[i]
        for j in range(ne):
            if j == i :
                continue
            deltax = xe[i, 0]-xe[j, 0]
            deltay = xe[i, 1]-xe[j, 1]
            if abs(deltax) < 30 and abs(deltay) < 30:
                r = sqrt(deltax**2 + deltay**2)
                if ecid[j] == cid:
                    dV = max(2*d_potential_LJ_gpu(r, rm_intra, 1.5)+ gamma*r, -10.0 )
                else:
                    dV = max(alpha[cid,ecid[j]]*d_potential_LJ_gpu(r, rm_inter, 1.5), -10.0)
                xe_F[i, 0] += -dt * dV * deltax
                xe_F[i, 1] += -dt * dV * deltay
        xe_F[i, 0] += x_randt[i, 0]
        xe_F[i, 1] += x_randt[i, 1]

@cuda.jit(device=True)
def d_potential_LJ_gpu(r, rm, epsilon):
    rs6 = (rm/r)**6
    return epsilon*r**-2*(rs6-rs6*rs6)

class cellshape_GT(CellBase):
    """Cell shape representation for experimental data visualization"""
    def __init__(self, 
                 xe: np.ndarray,
                 ecid: np.ndarray,
                 ceidn: np.ndarray,
                 xc: Optional[np.ndarray] = None,
                 ctype: Optional[np.ndarray] = None,
                 color_list: Optional[np.ndarray] = None,
                 adata: Optional[AnnData] = None,
                 spatial_key: str = 'spatial',
                 cluster_key: str = 'leiden'):
        self.nc = ceidn.shape[0]-1
        super().__init__(adata, xc, ctype, cluster_key, spatial_key, color_list)
        # Visualization-specific properties
        self.xe = xe
        self.ecid = ecid
        self.ceidn = ceidn
        self.dim = xe.shape[1]
        self.ne_per_cell: np.ndarray = ceidn[1:]-ceidn[:-1]
        if xc is None and adata is None:
            print('compute xc from xe')
            self.update_xc()
    
    def __repr__(self):
        return f'Cell Number: {self.nc}\nElement Number: {self.xe.shape[0]}\nDim: {self.dim}'