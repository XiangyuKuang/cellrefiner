from typing import Optional, Union
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection,PolyCollection
from matplotlib.axes import Axes

# to-do:
# compute norm
class AlphaShape():
    """Alpha-shape for element representation of cells.

    Attributes
    ----------
    x : np.ndarray
        Input points with shape (n_points, 2).
    alpha : Optional[float]
        Current alpha radius value.
    alpha_best : Optional[float]
        Optimized alpha radius value.
    simplices : np.ndarray
        Delaunay triangulation simplices.
    neighbors : np.ndarray
        Neighboring simplices indices.
    dmax : np.ndarray
        Maximum edge length of each simplex.
    edge : np.ndarray
        Unique edges in the triangulation.
    edge_counts : np.ndarray
        Occurrence count of each edge.
    tri2edge_I : np.ndarray
        Mapping from simplices to edges.
    alpha_range : list[float]
        Valid alpha range [min_edge, max_edge].
    simplices_adjacency_matrix : csr_matrix
        Adjacency matrix of simplices.
    is_boundary : np.ndarray
        Boolean array marking boundary edges.
    valid_simplices : np.ndarray
        Boolean array marking valid simplices.
    """

    x: np.ndarray
    """Point coordinates"""
    n_points: int
    """Number of points"""
    simplices: np.ndarray
    """Delaunay triangulation simplices, shape=(n_simplices,3), dtype=np.int32"""
    neighbors: np.ndarray
    """Neighboring simplices indices, shape=(n_simplices,3), dtype=np.int32"""
    simplices_adjacency_matrix: csr_matrix
    """Simplices adjacency matrix, shape=(n_simplices,n_simplices)"""
    dmax: np.ndarray
    """Maximum edge length of each simplex, shape=(n_simplices)"""
    edge : np.ndarray
    """Unique edges in the triangulation"""
    is_boundary : np.ndarray
    """Boolean array marking boundary edges"""
    valid_simplices : np.ndarray
    """Boolean array marking shape simplices"""
    alpha: float
    """Current alpha radius"""
    alpha_best: float
    """Optimized alpha radius"""
    
    def __init__(self,
                 x: np.ndarray, 
                 alpha: Optional[float] = None,
                 no_hole: bool = True,
                 ns: int=0,
                 r: float=0.0):
        """
        Initialize the alpha shape for a given set of points.
        
        Parameters
        ------
        x : np.ndarray
            2D coordinates of the point set.
        alpha : Optional[float]
            Initial alpha radius. If None, the alpha radius will be optimized to ensure the shape enclosing all of the points, having only one region.
        no_hole : bool,
            If True, the alpha radius optimization will include no holes in shape as the objective.
        ns : int default=0
            Number of 
        r : float default=0
        """
        def tri2edge(simplex):
            """map a triangle into its edges"""
            return [[simplex[0],simplex[1]],[simplex[0],simplex[2]],[simplex[1],simplex[2]]]
        
        if (ns > 0) and (r > 0):
            # add points around elements
            theta = np.linspace(0, 2*np.pi, ns+1)[:-1]
            xs = r*np.column_stack([np.cos(theta),np.sin(theta)])
            x = np.repeat(x, ns, axis=0) + np.tile(xs,(x.shape[0],1))
        self.x = x
        self.n_points = self.x.shape[0]
        tri = Delaunay(x) # perform Delaunay triangulation on the points
        self.simplices = tri.simplices # int32,
        self.neighbors = tri.neighbors # int32,
        self.dmax = np.array([max(pdist(x[simplex])) for simplex in self.simplices])
        # Get unique edges, their indices, and edge counts
        all_edge = np.concatenate([tri2edge(sorted(simplex)) for simplex in self.simplices]) # all edges in the Delaunay triangulation
        self.edge, I, self.edge_counts = np.unique(all_edge,axis=0,return_inverse=True,return_counts=True) # unique edges in the Delaunay triangulation, edges indices, edge counts
        self.tri2edge_I = I.reshape(-1,3) # map triangle to their edges
        
        # Range of alpha radius
        self.alpha_range = [np.min(self.dmax),np.max(self.dmax)]
        
        # Simplices adjacency matrix
        n_simplices = len(self.simplices)
        simplex_indices, neighbor_offsets = np.where(self.neighbors!= -1) # method2 310 µs ± 201 ns per loop
        neighbor_indices = self.neighbors[simplex_indices, neighbor_offsets]
        rows = np.concatenate([simplex_indices, neighbor_indices])
        cols = np.concatenate([neighbor_indices, simplex_indices])
        coo_adj = csr_matrix((np.ones_like(rows), (rows, cols)), shape=(n_simplices, n_simplices)).tocoo()
        coo_adj.sum_duplicates()
        self.simplices_adjacency_matrix = coo_adj.tocsr()
        
        # simplices_adjacency_matrix = np.zeros((n_simplices, n_simplices), dtype=int) # method0 2.65 ms ± 2.62 µs per loop 
        # for simplex_idx, neighbors in enumerate(self.neighbors):
        #     for neighbor_idx in neighbors:
        #         if neighbor_idx != -1:
        #             simplices_adjacency_matrix[simplex_idx, neighbor_idx] = 1
        #             simplices_adjacency_matrix[neighbor_idx, simplex_idx] = 1
        # self.simplices_adjacency_matrix = csr_matrix(simplices_adjacency_matrix) # stored as csr_matrix
        
        # adj = lil_matrix((n_simplices, n_simplices), dtype=int) # method1 16.7 ms ± 31.2 µs per loop
        # for i, neigh in enumerate(self.neighbors):
        #     valid_neigh = neigh[neigh != -1]
        #     adj[i, valid_neigh] = 1
        # self.simplices_adjacency_matrix = adj.tocsr() # stored as csr_matrix
        
        # If no alpha radius is provided, use optimized alpha radius
        self.alpha = alpha
        self.alpha_best = None
        if self.alpha is None:
            self.optimize_alpha(no_hole)
            self.alpha = self.alpha_best
        self.update(self.alpha) # update the alpha shape
        
    def __repr__(self):
        return f'alpha = {self.alpha}, simplices = {len(self.simplices)} simplices, x = {self.n_points} points'
    
    def update(self, alpha: float) -> None:
        """
        Update the alpha shape based on the given alpha radius. Calculate valid simplices and boundary edges.
        
        Parameters
        ------
        alpha : float
            Alpha radius used to determine valid simplices and boundaries.
        """
        self.alpha = alpha
        # Determine boundary edges, whose counts = 1
        edge_counts_update = self.edge_counts.copy()
        for i in self.tri2edge_I[self.dmax>alpha].flatten():
            edge_counts_update[i] -= 1
        self.is_boundary = edge_counts_update == 1
        # Mark simplices (maximum edge length< alpha radius) as valid
        self.valid_simplices = self.dmax<=alpha
        
    def get_shape_info(self):
        simplices_adjacency_matrix = self.simplices_adjacency_matrix.toarray()
        cover_points = np.unique(self.simplices[self.valid_simplices].flatten())
        n_components, labels = connected_components(csr_matrix(simplices_adjacency_matrix[self.valid_simplices,:][:,self.valid_simplices]))
        return cover_points, n_components, labels

    def optimize_alpha(self, no_hole: bool = True, return_alpha: bool = False) -> Union[float, None]:
        """
        Optimize the alpha radius using binary search to ensure that the shape contains all points.
        
        Optionally, ensures that the shape has no holes.

        Parameters
        -------
        no_hole : bool, default=True
            If True, the optimization ensures that no holes exist in the shape.
        return_alpha : bool, default=False
            If True, return the optimized alpha value.
        
        Return
        -------
        alpha_best : Union[float, None]
            Optimized alpha value if `return_alpha=True`, otherwise None.
        """
        # All points in one connected region
        def allpoints_1region(simplices,outer,adjacency_matrix,mask,n_points):
            I = np.where(mask)[0]
            n_components, _ = connected_components(adjacency_matrix[I,:][:,I], directed=False)
            return n_components == 1 and np.unique(simplices[mask].flatten()).shape[0] == n_points
        
        # All points in one connected region and no hole
        def allpoints_1region_0hole(simplices,outer,adjacency_matrix,mask,n_points):
            I = np.where(mask)[0]
            I1 = np.where(~mask)[0]
            n_components, _ = connected_components(adjacency_matrix[I,:][:,I], directed=False)
            _, labels = connected_components(adjacency_matrix[I1,:][:,I1], directed=False)
            return n_components == 1 and np.unique(simplices[mask].flatten()).shape[0] == n_points and np.setdiff1d(np.unique(labels),np.unique(labels[outer[~mask]])).shape[0]==0
        
        # Select the optimization objective
        opt_obj = allpoints_1region_0hole if no_hole else allpoints_1region
        
        #  Initialize the best alpha as the maximum possible alpha
        alpha_low, alpha_high = self.alpha_range[0], self.alpha_range[1]
        alpha_best = alpha_high

        tol=1e-3 # Tolerance for stopping the search
        max_iterations=10 # Maximum number of iterations for binary search
        # n_points = self.x.shape[0]
        outer = np.any(self.neighbors==-1,axis=1)# identify the outer simplices (those touching the boundary)
        # Binary search to find the optimal alpha value
        for _ in range(max_iterations):
            alpha_mid = (alpha_low + alpha_high) / 2# Midpoint alpha value
            mask = self.dmax <= alpha_mid # Mark simplices (maximum edge length< alpha_mid) as valid
            if opt_obj(self.simplices,outer,self.simplices_adjacency_matrix,mask,self.n_points):
                # Narrow the search to lower values
                alpha_best = alpha_mid
                alpha_high = alpha_mid
            else:
                # Narrow the search to higher values
                alpha_low = alpha_mid
            if alpha_high - alpha_low < tol: # stop searching
                break
        self.alpha_best = alpha_best
        if return_alpha:
            return alpha_best
    
    def close_hole(self):
        """
        Close holes in the alpha shape by marking invalid simplices corresponding to holes as valid.
        """
        # Identify outer simplices (those touching the boundary)
        outer = np.any(self.neighbors == -1, axis=1)
        # Compute valid simplices for current alpha shape
        mask = self.valid_simplices
        # Identify connected components among invalid simplices (corresponding to holes)
        _, labels = connected_components(csr_matrix(self.simplices_adjacency_matrix[~mask ,:][:,~mask ]))
        invalid_simplices_idx = np.where(~mask)[0]
        # Find the connected components corresponding to holes
        hole_labels = np.setdiff1d(np.unique(labels), np.unique(labels[outer[~mask]]))
        if hole_labels.size > 0:
            # Loop through the hole_labels and close the holes by marking those simplices as valid
            for hole_label in hole_labels:
                hole_simplices = np.where(labels == hole_label)[0]
                # Mark the simplices corresponding to the hole as valid
                self.valid_simplices[invalid_simplices_idx[hole_simplices]] = True
        # update boundary after closing holes
        edge_counts_update = self.edge_counts.copy()
        for i in self.tri2edge_I[~self.valid_simplices].flatten():
            edge_counts_update[i] -= 1
        self.is_boundary = edge_counts_update == 1

    def get_boundary_vertices(self) -> list:
        boundary_edges = self.edge[self.is_boundary]
        boundary_points = [boundary_edges[0][0], boundary_edges[0][1]]
        boundary_edges = np.delete(boundary_edges, 0, axis=0)
        while len(boundary_edges) > 1:
            last_point = boundary_points[-1]
            match_idx = np.where(boundary_edges == last_point )[0]
            if len(match_idx) == 0:
                break  # 如果没有找到下一个相接的点，说明没有闭合
            match_edge = boundary_edges[match_idx[0]]
            boundary_edges = np.delete(boundary_edges, match_idx[0], axis=0)
            # 将匹配到的边的另一个点加入序列
            if match_edge[0] == last_point:
                boundary_points.append(match_edge[1])
            else:
                boundary_points.append(match_edge[0])
        return boundary_points
    
    def get_boundary(self, close = True) -> np.ndarray:
        boundary_points = self.get_boundary_vertices()
        if close:
            boundary_points = np.append(boundary_points,boundary_points[0])
        return self.x[boundary_points]
    
    def get_simplices(self) -> np.ndarray:
        return self.simplices[self.valid_simplices]
    
    def get_area(self) -> np.floating:
        area = 0.
        for spx in self.get_simplices():
            area+=np.cross(self.x[spx[1]]-self.x[spx[0]],self.x[spx[2]]-self.x[spx[0]])
        return area/2

    def plot_shape(self, alpha: Optional[float] = None, fill:bool = True, ax: Optional[Axes] = None, **kwargs):
        """
        Plot the alpha shape. Optionally update the alpha value before plotting.
        
        Parameters
        ----------
        alpha : float, optional
            The alpha value to update the shape with before plotting.
        fill : bool
            If True, the shape will be filled with color. Default is True.
        ax : sAxes, optional
            The axis to plot on. If None, a new axis will be created.
        **kwargs : dict, optional
            Additional styling arguments for the boundary and face (boundarywidth, boundarycolor, facecolor, facealpha).
        """
        if alpha is not None:
            print(f'update alpha to {alpha}')
            self.update(alpha)
        if ax is None:
            fig,ax=plt.subplots()
        line_kwargs = dict()
        poly_kwargs = dict(facecolors='gray',edgecolors='gray')
        for k, v in kwargs.items():
            if k == 'boundarywidth':
                line_kwargs['linewidths'] = v
            elif k == 'boundarycolor':
                line_kwargs['colors'] = v
            elif k == 'facecolor':
                poly_kwargs['facecolors'] = v
                poly_kwargs['edgecolors'] = v
            elif k == 'facealpha':
                poly_kwargs['alpha'] = v
        boundary_x = [self.get_boundary()]
        lc = LineCollection(boundary_x,**line_kwargs)
        ax.add_collection(lc)
        if fill:
            poly = PolyCollection(boundary_x, **poly_kwargs)
            ax.add_collection(poly)
        ax.autoscale()
        return ax