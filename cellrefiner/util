import numpy as np
from scipy.stats import pearsonr

def cal_glvs(pos):
        # input should be a nx2 matrix of coordinates
        # This is for visualiztion of level set and finding max val only, hence construction of the grid
        x_offset=0.1*(np.amax(pos[:,0])-np.amin(pos[:,0])) #distance from tissue edge to compute points
        y_offset=0.1*(np.amax(pos[:,1])-np.amin(pos[:,1]))
        x,y=np.meshgrid(np.linspace(np.amin(pos[:,0])-x_offset,np.amax(pos[:,0])+x_offset,100),np.linspace(np.amin(pos[:,1])-y_offset,np.amax(pos[:,1])+y_offset,100))
        Sigma = np.array([[ 10000 , 0], [0,  10000]])
        pts = np.empty(x.shape + (2,))
        pts[:, :, 0] = x
        pts[:, :, 1] = y
        z=np.zeros((pts.shape[0],pts.shape[1]))
        for i in range(pos.shape[0]):
            z+=z+glvs(pts,pos[i,:],Sigma)

        return z
    
def glvs(pos, mu, Sigma):
        """Return the multivariate Gaussian distribution on array pos."""

        n = mu.shape[0]
        Sigma_det = np.linalg.det(Sigma)
        Sigma_inv = np.linalg.inv(Sigma)
        N = np.sqrt((2*np.pi)**n * Sigma_det)
        # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
        # way across all the input variables.
        fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

        return np.exp(-fac / 2) / N
    
def pre_cal1(N):
        degree=np.diag(np.sum(N,axis=1))
        L=degree-N
        L_inv=np.linalg.pinv(L,rcond=np.finfo(float).tiny)
        n=len(N)
        q=np.multiply(N,np.diag(L_inv)+np.reshape(np.diag(L_inv.T),(-1,1))-L_inv.T-L_inv)
        return q
    
def sparsify(W1,q):
        n=len(W1)
        P=np.minimum(1,np.log(n)*q)
        H=np.ones((n,n))*np.finfo(float).tiny
        rand=np.random.rand(n,n)
        idx=np.where(rand<P)
        H[idx]=W1[idx]/P[idx]
        kept=len(H[H>np.finfo(float).tiny])/n**2
        percent=.40
        time=0
        while kept<percent:
            if time>300:
                break
            rand=np.random.rand(n,n)
            idx=np.where(rand<P)
            H[idx]=W1[idx]/P[idx]
            kept=len(H[H>np.finfo(float).tiny])/n**2
            time+=1
            print('kept is:',kept)

        H=H/np.amax(H)
        return H
    
def F_gc(xi,xj,gi,gj): # modified gene force for just attraction
    c=pearsonr(gi,gj)[0]
    if np.logical_and(np.linalg.norm(xi-xj)>0,c>0):
        f=c*(xi-xj)/np.linalg.norm(xi-xj)
    else:
        f=0
    return f

def V_xy(xi,xj,V0,U0,xi1,xi2):
        # assume xi,xj are vectors
    r2=(xj[0]-xi[0])**2+(xj[1]-xi[1])**2
    r=np.sqrt(r2)
    dVdr=-2*r*V0/xi1**2*np.exp(-r2/xi1**2)+2*r/xi2**2*U0*np.exp(-r2/xi2**2)
    drdx=(xj[0]-xi[0])*r**(-0.5)
    drdy=(xj[1]-xi[1])*r**(-0.5)
    dVdx=dVdr*drdx
    dVdy=dVdr*drdy
    return np.array([dVdx,dVdy])
    
def F_spot(xi,si,rS):
    var1=si-xi
    f=var1/np.reshape(np.linalg.norm(var1,axis=1),(len(np.linalg.norm(var1,axis=1)),1))
    fs=np.minimum((np.linalg.norm(var1,axis=1)-rS)**2,30)
    f[np.linalg.norm(var1,axis=1)<rS]=0
    return f*np.reshape(fs,(len(fs),1))
