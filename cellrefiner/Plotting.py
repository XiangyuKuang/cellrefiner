from typing import Optional, Union, Tuple
from anndata import AnnData
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import Normalize, to_rgb
from matplotlib.patches import Patch
from matplotlib.transforms import offset_copy
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import ConvexHull
from scipy.sparse import csr_matrix, isspmatrix
from math import pi
from .SEM import SEM

def contour_plot(sem:SEM, ax=None, iso=0.5, ns = 50,fill=False, save = False, linewidths=0.1, cid_list= None):
    # contour_plot(sem,ax=ax,linewidths=0.5,iso=0.25,fill=True,cid_list=clist) #4s
    scale = sem.scale
    delta = sem.deltax
    if sem.dim == 2:
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        if cid_list is None:
            cid_list = range(sem.nc)

        for cid in cid_list:
            eid = np.arange(sem.ceidn[cid],sem.ceidn[cid+1])
            # a_eid = eid[self.etype[eid]==1]
            xl = np.min(sem.xe[eid], axis=0)-3.0
            xr = np.max(sem.xe[eid], axis=0)+3.0
            x, y = np.meshgrid(np.linspace(
                xl[0], xr[0], ns), np.linspace(xl[1], xr[1], ns))
            s = np.zeros((ns, ns))
            for e in sem.xe[eid]:
                s += np.exp(-np.power((x-e[0])**2+(y-e[1])**2, 5))
            cc = sem.color_list[sem.ctype[cid]]
            # calculate contour but do not show
            fig_temp, ax_temp = plt.subplots()
            cont = ax_temp.contour(x*scale+delta[0], y*scale+delta[1], s, levels=[iso], colors=['k'],linewidths=linewidths)
            plt.close(fig_temp)
            # find longest contour
            cont_list = cont.collections[0].get_paths()
            cont_length = np.zeros(len(cont_list))
            for i,path in enumerate(cont_list):
                cont_length[i] = path.vertices.shape[0]
            path = cont_list[np.argmax(cont_length)]
            if fill:
                ax.fill(path.vertices[:,0],path.vertices[:,1],facecolor=cc)
                ax.plot(path.vertices[:,0],path.vertices[:,1],'k',linewidth = linewidths)
                # ax.contourf(x*scale+delta[0], y*scale+delta[1], s, levels=[iso,np.max(s)+1], colors=[cc],linewidths=linewidths)
            else:
                ax.plot(path.vertices[:,0],path.vertices[:,1],color=cc,linewidth = linewidths)

        # ax.legend(loc = 'center right',ncol=2,bbox_to_anchor=(1.65,0.5),framealpha=0,markerscale=3)
        # self.apply_vis_config(ax)
        # if box is not None:
        #     ax.set_xlim([box[0],box[2]])
        #     ax.set_ylim([box[1],box[3]])
        ax.set_aspect('equal', adjustable='box')
        ax.invert_yaxis()
    if save:
        fig.savefig(f"{sem.sim_name}/{sem.sim_name}_{sem.t}.pdf", dpi=300)
        plt.close(fig)
    return ax

def convex_plot(sem:SEM, ax=None, ns=10, r=0.5, fill=True, save=False, linewidths=1, cid_list=None):
    # convex_plot(sem,ax=ax,linewidths=0.5,r=0.5,fill=True,cid_list=clist) #1.5s
    scale = sem.scale
    delta = sem.deltax
    theta = 2*pi*np.linspace(0,1,ns+1)-pi
    xs = r*scale*np.vstack([np.cos(theta[:-1]),np.sin(theta[:-1])]).T
    xe = sem.xe*scale+delta
    if sem.dim == 2:
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        if cid_list is None:
            cid_list = range(sem.nc)
        
        for cid in cid_list:
            cc = sem.color_list[sem.ctype[cid]]
            if ns > 0:
                xes = np.repeat(xe[sem.ceidn[cid]:sem.ceidn[cid+1]], ns, axis=0) + np.tile(xs,(sem.ceidn[cid+1]-sem.ceidn[cid],1))
            else:
                xes = xe[sem.ceidn[cid]:sem.ceidn[cid+1]]
            hull = ConvexHull(xes)
            boundary = np.array([[xes[simplex[0]], xes[simplex[1]]] for simplex in hull.simplices])
            if fill:
                line_segments = LineCollection(boundary, colors='k', linewidths=linewidths)
                ax.fill(xes[hull.vertices, 0], xes[hull.vertices, 1], cc, alpha=1)
            else:
                line_segments = LineCollection(boundary, colors=cc, linewidths=linewidths)
            ax.add_collection(line_segments)
        ax.set_aspect('equal', adjustable='box')
        ax.invert_yaxis()
    if save:
        fig.savefig(f"{sem.sim_name}/{sem.sim_name}_{sem.t}.pdf", dpi=300)
        plt.close(fig)
    return ax

def alphashape_plot(sem: SEM, 
                    vis_key: Optional[str] = None,
                    arr: Optional[Union[np.ndarray, pd.Series]] = None, 
                    summary: str = 'sender',
                    compute_alphashape: bool = False, 
                    cid_list: Optional[np.ndarray] = None, 
                    cmap_name: str = 'Reds',
                    vmax: Optional[float] = None,
                    vmin: Optional[float] = None,
                    boundary_width: float = 1, 
                    boundary_color: Optional[Union[str, Tuple]] = None, 
                    boundary_alpha: float = 1, 
                    face_alpha: float = 1,
                    show_axis: bool = True,
                    enable_annotation: bool= False,
                    enable_legend: bool = False,
                    enable_colorbar: bool = False,
                    ax: Optional[Axes] = None, 
                    save_name: Optional[str] = None, 
                    **kwargs) -> Axes:
    """
    Plot cell shape using alpha shape, visualize cell data by colors.

    Parameters
    ----------
    sem : SEM
        Subcellular element method object
    vis_key : str, optional
        Key to retrieve visualization data from `sem.adata`
    arr : np.ndarray or pd.Series, optional
        Data for visualization. shape = (nc,) or (len(cid_list),)
        
        Ignored if `vis_key` is provided .
        
        `sem.ctype` will be visualized if `arr` and `vis_key` both are not provided.
    compute_alphashape : bool, default=False
        Compute alphashape if True
    cid_list : ndarray, optional
        Array of index for cells to be visualized. Default: all cells
    cmap_name : str, default='Reds'
        Valid matplotlib colormap name to visualize data
    vmax : float, optional
        Colormap upper bound. Default: 95th percentile for positive data
    vmin : float, optional
        Colormap lower bound. Default: data min
    boundary_width : float, default=1
        Cell boundary line width
    boundary_color : str or tuple, optional
        Cell boundary line color, Default: matches face color
    boundary_alpha : float, default=1
        Cell boundary line opacity, 0 (fully transparent), 1 (fully opaque)
    face_alpha : float, default=1
        Cell shape face opacity, 0 (fully transparent), 1 (fully opaque)
    show_axis : bool, default=True
        Show axis
    enable_annotation : bool, default=False
        Annotate cells with index at centroids
    enable_legend : bool, default=False
        Show categorical legend (only for category data)
    enable_colorbar : bool, default=False
        Show colorbar (only for continuous data)
    ax : Axes, optional
        Target matplotlib axes object. Creates new figure if None.
    save_name : str, optional
        Output path for figure saving (e.g., 'figure.pdf')
    **kwargs
        keyword arguments passed to `sem.compute_alphashape()`
        
    Returns
    ----------
    ax : Axes
    """

    if compute_alphashape or not sem.alphashape_info['computed'] or kwargs:
        sem.compute_alphashape(**kwargs)

    fig, ax = _get_axes(ax)
    cid_list, _ = _get_cid_list(sem, cid_list)
    arr = _get_arr(sem, vis_key, arr, summary)
    
    if arr is None:
        # vis sem.ctype
        if vis_key is None:
            # use cell type color in sem
            cat_code = sem.ctype[cid_list]
            cat_list = sem.ctype_list
            color_list = sem.color_list
            colors = color_list[cat_code]
            facecolors = np.insert(colors,3,face_alpha,axis=1)
            edgecolors = np.insert(colors,3,boundary_alpha,axis=1)
            enable_colorbar = False
        else:
            raise KeyError(f"vis_key '{vis_key}' not found in genes or adata.obs")
    else:
        # vis arr
        if arr.dtype.name == 'category':
            # obtain category and color from arr
            cat_code, cat_list, color_list = _get_cat_arr_color(sem,arr,cid_list,vis_key,cmap_name)
            colors = color_list[cat_code]
            facecolors = np.insert(colors,3,face_alpha,axis=1)
            edgecolors = np.insert(colors,3,boundary_alpha,axis=1)
            enable_colorbar = False
        else:
            # set color based on arr
            if len(arr) == sem.nc:
                arr = arr[cid_list]
            elif len(arr)!=len(cid_list):
                raise ValueError('len(arr)!=len(cid_list)')
            
            cmap = colormaps[cmap_name]
            # vmax = np.percentile(arr,95) if vmax is None else vmax
            vmax = arr.max() if vmax is None else vmax
            vmin = arr.min() if vmin is None else vmin
            norm = Normalize(vmin=vmin, vmax=vmax, clip=False)
            facecolors = cmap(norm(arr))
            edgecolors = cmap(norm(arr))
            facecolors[:,3] = face_alpha
            edgecolors[:,3] = boundary_alpha
            # enable_colorbar = True
            if enable_legend:
                print('visualize data, cannot use legend')
            enable_legend = False

    # draw cell shape
    all_boundaries = []
    fc = []
    bc = []
    for i, cid in enumerate(cid_list):
        all_boundaries.append(sem.alphashape[cid].get_boundary())
        fc.append(facecolors[i])
        bc.append(facecolors[i])
    
    if boundary_color is not None:
        bc = boundary_color
    polyc = PolyCollection(all_boundaries,
                           facecolors=fc,
                           edgecolors=bc,
                           linewidths = boundary_width)
    ax.add_collection(polyc)
    _set_axes(ax, show_axis)
    
    if enable_colorbar:
        # draw colorbar
        _add_colorbar(fig, ax, cmap, norm)
    elif enable_legend:
        # draw legend
        legend_patches = []
        for i in np.unique(cat_code):
            legend_patches.append(Patch(color=color_list[i],label=cat_list[i]))
        transform = offset_copy(ax.transAxes, x=5, y=0, units='points',fig=fig) 
        ax.legend(handles=legend_patches,
                  loc='center left',
                  bbox_to_anchor=(1, 0.5),
                  bbox_transform=transform,
                  frameon=False)
    
    if enable_annotation:
        spatial_coor = sem.xc*sem.scale+sem.deltax
        for i in cid_list:
            ax.annotate(f'{i}',spatial_coor[i],ha='center',va='center',fontweight='bold')

    if save_name is not None:
        fig.savefig(save_name, dpi=500, bbox_inches='tight', transparent=True)
    return ax

def element_plot(sem: SEM,
                 vis_key: Optional[str] = None,
                 arr: Optional[Union[np.ndarray, pd.Series]] = None,
                 summary: str = 'sender',
                 cid_list: Optional[np.ndarray] = None, 
                 cmap_name: str ='Reds', 
                 spot_size: float = 1,
                 scaling: bool = True, 
                 show_axis: bool = True, 
                 enable_colorbar: bool = True, 
                 enable_legend: bool = True,
                 ax: Optional[Axes] = None,
                 save_name: Optional[str] = None,) -> Axes:
    """
    Plotting cell elements

    Parameters
    ----------
    sem : SEM
        Subcellular element method object
    vis_key : str, optional
        Key to retrieve visualization data from `sem.adata`.
    arr : np.ndarray or pd.Series, optional
        Data for visualization. Accepts both cell-level (nc,) and element-level (ne,)
    summary : str, default='sender'
        'sender' represents sender signal, retrieves data from adata.obsm['sender_signal'][vis_key]

        'receiver' retrieves receiver signal data from adata.obsm['receiver_signal'][vis_key]

        'gene' retrieves gene expression data from adata
    cid_list : ndarray, optional
        Array of index for cells to be visualized. Default: all cells
    cmap_name : str, default='Reds'
        Valid matplotlib colormap name to visualize data
    spot_size : float, default=1
        Markersize for `matplotlib.pyplot.scatter`
    scaling : bool, default=True
        Scale coordinates back to original data(`xc`) if True, otherwise visualize directly.
    show_axis : bool, default=True
        Show axis.
    enable_legend : bool, default=False
        Show categorical legend (only for category data).
    enable_colorbar : bool, default=False
        Show colorbar (only for continuous data).
    ax : Axes, optional
        Target matplotlib axes object. Creates new figure if None
    save_name : str, optional
        Output path for figure saving (e.g., 'figure.pdf')
    
    Returns
    ----------
    ax : Axes
    """

    fig, ax = _get_axes(ax)
    cid_list, xe = _get_cid_list(sem, cid_list, scaling)
    arr = _get_arr(sem, vis_key, arr, summary)

    ec = None
    if arr is None:
        # vis sem.ctype
        if vis_key is None:
            # use cell type color in sem
            cat_code = sem.ctype[cid_list]
            cat_list = sem.ctype_list
            color_list = sem.color_list
        else:
            raise KeyError(f"vis_key '{vis_key}' not found in genes or adata.obs")
    else:
        # vis arr
        if arr.dtype.name == 'category':
            # obtain category and color from arr
            cat_code, cat_list, color_list = _get_cat_arr_color(sem,arr,cid_list,vis_key,cmap_name)
        else:
            cmap = colormaps[cmap_name]
            # color norm
            if arr.min()>=0:
                norm = Normalize(vmin=arr.min(), vmax=np.percentile(arr,95), clip=False)
            else:
                a = np.percentile(np.abs(arr),95)
                norm = Normalize(vmin=-a, vmax=a, clip=False)
            # set color
            if arr.shape[0] == sem.nc:
                # cell color
                cc = cmap(norm(arr))
                # cell color -> element color
                ec = np.zeros((sem.ne,cc.shape[1]))
                for cid in range(sem.nc):
                    ne_i = sem.ceidn[cid+1]-sem.ceidn[cid]
                    ec[sem.ceidn[cid]:sem.ceidn[cid+1],:] = np.tile(cc[cid],(ne_i,1))
            else:
                ec = cmap(norm(arr)) # element color
    # plot
    if ec is None:
        # cell color
        ecid = []
        for n,cid in enumerate(cid_list):
            ecid.append(n*np.ones(sem.ceidn[cid+1]-sem.ceidn[cid]))
        ecid = np.concatenate(ecid).astype(int)
        element_cat = cat_code[ecid]
        for i in np.unique(cat_code):
            vis = element_cat == i
            ax.scatter(xe[vis, 0], xe[vis, 1],
                       c = color_list[i][np.newaxis],
                       label=cat_list[i],
                       s=spot_size)
        if enable_legend:
            # draw legend
            transform = offset_copy(ax.transAxes, x=5, y=0, units='points',fig=fig) 
            ax.legend(loc='center left',
                      bbox_to_anchor=(1, 0.5),
                      bbox_transform=transform,
                      frameon=False,
                      markerscale=5/spot_size)
    else:
        # element color
        for cid in cid_list:
            ax.scatter(xe[sem.ceidn[cid]:sem.ceidn[cid+1], 0], xe[sem.ceidn[cid]:sem.ceidn[cid+1], 1], 
                       c = ec[sem.ceidn[cid]:sem.ceidn[cid+1]],
                       s=spot_size)
        if enable_colorbar:
            # draw colorbar
            _add_colorbar(fig, ax, cmap, norm)
    _set_axes(ax, show_axis)
    
    if save_name is not None:
        fig.savefig(save_name, dpi=500, bbox_inches='tight', transparent=True)
    return ax

def vis_contact_signal(sem: Optional[SEM] = None,
                       adata: Optional[AnnData] = None,
                       sig_mat: Optional[Union[csr_matrix,np.ndarray]] = None,
                       signal: Optional[str] = None,
                       cid_list: Optional[np.ndarray] = None,
                       scaling: bool = True,
                       line_width: float = 1,
                       line_color: Union[str, tuple] = 'k',
                       line_alpha: float = 1,
                       ax: Optional[Axes] = None):
    """
    Visualize contact signals or relationships between cells

    Parameters
    ----------
    sem : SEM
        A subcellular element method object.
    sig_mat : csr_matrix or ndarray, optional
        Signal matrix to visualize. If `signal` is provided, this parameter will be ignored, 
        and the signal matrix will be retrieved from `sem.adata.obsp[signal]`.
        
        If `sig_mat` and `signal` both are None, the contact matrix `sem.contact_matrix` will be visualized.
    signal : str, optional
        Key for signal matrix in `sem.adata.obsp`. If given, `sig_mat` will be ignored.
    cid_list : ndarray, optional
        Array of index for cells to be visualized. Default: all cells
    scaling : bool, default=True
        Scale coordinates back to original data if True, otherwise visualize directly
    line_width : float, default=1
        Cell-cell contacts line width
    line_color : str or tuple, default='k'
        Cell-cell contacts line color
    line_alpha : float, default=1
        Cell-cell contacts line opacity, 0 (fully transparent), 1 (fully opaque)
    ax : Axes, optional
        Target matplotlib axes object. Creates new figure if None

    Returns
    ----------
    ax : Axes
    """
    fig, ax = _get_axes(ax)
    cid_list, _ = _get_cid_list(sem, cid_list)
    if sem is None:
        assert(adata is not None)
        nc = adata.shape[0]
        if signal:
            sig_mat = adata.obsp[signal]
        spatial_coor = adata.obsm['spatial']
    else:
        nc = sem.nc
        if signal:
            sig_mat = sem.adata.obsp[signal]
        elif sig_mat is None:
            if sem.contact_matrix is None:
                print('compute cell-cell contact')
                sem.compute_contact()
            sig_mat = sem.contact_matrix

        if scaling:
            spatial_coor = sem.xc*sem.scale+sem.deltax
        else:
            spatial_coor = sem.xc

    seg = []
    if isinstance(line_width, (list, tuple)):
        data = sig_mat.data
        linewidths = np.abs(data)/data.max()*line_width[1]
    else:
        linewidths = line_width
    if isinstance(line_alpha, (list, tuple)):
        data = sig_mat.data
        linealphas = np.abs(data)/data.max()*line_alpha[1]
    else:
        linealphas = line_alpha
    if isspmatrix(sig_mat):
        indices = sig_mat.indices
        indptr = sig_mat.indptr
        for i in cid_list:
            for j in indices[indptr[i]:indptr[i+1]]:
                if j in cid_list:
                    seg.append([spatial_coor[i],spatial_coor[j]])

    else:
        for j in cid_list:
            sender_i = np.where(sig_mat[:,j]>0)[0]
            for i in sender_i:
                if i in cid_list:
                    seg.append([spatial_coor[i],spatial_coor[j]])
    lc = LineCollection(seg, linewidths=linewidths, colors=line_color, alpha=linealphas)
    ax.add_collection(lc)
    return ax

def _get_axes(ax: Optional[Axes] = None) -> Tuple[plt.Figure, Axes]:
    """create or get axes"""
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    return fig, ax

def _get_cid_list(sem: SEM, cid_list: Optional[np.ndarray], scaling=True):
    if scaling:
        xe = sem.xe*sem.scale+sem.deltax
    else:
        xe = sem.xe

    if cid_list is None:
        cid_list = np.arange(sem.nc)
    else:
        xe_vis = []
        for cid in cid_list:
            xe_vis.append(xe[sem.ceidn[cid]:sem.ceidn[cid+1]])
        xe = np.vstack(xe_vis)
    return cid_list, xe

def _add_colorbar(fig, ax, cmap, norm):
    """add colorbar"""
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.4) #to do fix colorbar width 
    cb = plt.colorbar(sm,cax=cax)

def _set_axes(ax, show_axis):
    """aspect equal, axis off, invert yaxis"""
    ax.set_aspect('equal', adjustable='box')
    ax.autoscale(tight=True)
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    if not show_axis:
        ax.set_axis_off()

def _get_arr(sem,vis_key,arr,summary):
    if (sem.adata is not None) & (vis_key is not None):
        if summary == 'gene' and vis_key in sem.adata.var_names:
            arr = sem.adata[:,vis_key].X.toarray()[:,0] # retrieve gene expression
        elif summary == 'sender' and 'sender_signal' in sem.adata.obsm and vis_key in sem.adata.obsm['sender_signal']:
            arr = sem.adata.obsm['sender_signal'][vis_key].to_numpy() # retrieve sender signal
        elif summary == 'receiver' and 'receiver_signal' in sem.adata.obsm and vis_key in sem.adata.obsm['receiver_signal']:
            arr = sem.adata.obsm['receiver_signal'][vis_key].to_numpy() # retrieve receiver signal
        elif vis_key in sem.adata.obs:
            arr = sem.adata.obs[vis_key] # retrieve adata.obs
    return arr

def _get_cat_arr_color(sem,arr,cid_list,vis_key,cmap_name):
    cat_code = arr.cat.codes[cid_list]
    cat_list = arr.cat.categories
    if (vis_key+'_colors') in sem.adata.uns:
        # use cluster color in the adata
        color_list = sem.adata.uns[vis_key+'_colors']
        if type(color_list[0]) is str:
            color_list = np.array([to_rgb(x) for x in color_list])
    else:
        cmap = colormaps[cmap_name]
        color_list = cmap( np.linspace( 0,1,len(cat_list) ) )[:,:3]
    return cat_code, cat_list, color_list

'''
def find_cell(adata,box,spatial_key = 'spatial'):
    x = adata.obsm[spatial_key]
    return np.where( x[:,0]>=box[0] & x[:,0]<box[0]+box[2] & x[:,1]>=box[1] & x[:,1]<box[1]+box[3] )[0]
'''

'''
def contour_plot(sem, ax=None, iso=0.5, ns = 50,fill=False, save = False, linewidths=0.1, cid_list= None):
    # contour_plot(sem,ax=ax,linewidths=0.5,iso=0.25,fill=True,cid_list=clist) #4s
    scale = sem.scale
    delta = sem.deltax
    if sem.dim == 2:
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        if cid_list is None:
            cid_list = range(sem.nc)

        for cid in cid_list:
            # eid = np.arange(sem.ceidn[cid]-sem.ne_per_cell,sem.ceidn[cid])
            eid = np.arange(sem.ceidn[cid],sem.ceidn[cid+1])
            # a_eid = eid[self.etype[eid]==1]
            xl = np.min(sem.xe[eid], axis=0)-3.0
            xr = np.max(sem.xe[eid], axis=0)+3.0
            x, y = np.meshgrid(np.linspace(
                xl[0], xr[0], ns), np.linspace(xl[1], xr[1], ns))
            s = np.zeros((ns, ns))
            for e in sem.xe[eid]:
                s += np.exp(-np.power((x-e[0])**2+(y-e[1])**2, 5))
            cc = sem.color_list[sem.ctype[cid]]
            if fill:
                ax.contour(x*scale+delta[0], y*scale+delta[1], s, levels=[iso], colors=['k'],linewidths=linewidths)
                ax.contourf(x*scale+delta[0], y*scale+delta[1], s, levels=[iso,np.max(s)+1], colors=[cc],linewidths=linewidths)
            else:
                ax.contour(x*scale+delta[0], y*scale+delta[1], s, levels=[iso], colors=[cc],linewidths=linewidths)

        # ax.legend(loc = 'center right',ncol=2,bbox_to_anchor=(1.65,0.5),framealpha=0,markerscale=3)
        # self.apply_vis_config(ax)
        # if box is not None:
        #     ax.set_xlim([box[0],box[2]])
        #     ax.set_ylim([box[1],box[3]])
        # ax.set_aspect('equal', adjustable='box')
        ax.invert_yaxis()
    if save:
        fig.savefig(f"{sem.sim_name}/{sem.sim_name}_{sem.t}.pdf", dpi=300)
        plt.close(fig)
    return ax
'''

'''def element_plot(sem:SEM, ax=None,spot_size=1,save_name=None,scaling=True,enable_legend=False):
    # element_plot(sem,ax=ax)
    if scaling:
        xe = sem.xe*sem.scale+sem.deltax
    else:
        xe = sem.xe
    
    if sem.dim == 2:
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        
        element_ctype = sem.ctype[sem.ecid]
        for i in range(len(sem.ctype_list)):
            cc = sem.color_list[i]
            vis = element_ctype==i
            ax.plot(xe[vis, 0], xe[vis, 1], '.',color = cc, markersize=spot_size,label=sem.ctype_list[i])
        ax.set_aspect('equal', adjustable='box')
        ax.invert_yaxis()
        if enable_legend:
            ax.legend(loc = 'center right',bbox_to_anchor=(1.1,0.5),markerscale=10/spot_size)
    
    if save_name is not None:
        fig.savefig(save_name, dpi=500, bbox_inches='tight', transparent=True)
    return ax
'''