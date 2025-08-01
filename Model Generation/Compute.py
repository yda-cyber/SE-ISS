from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from scipy import linalg
import numpy as np
import numba as nb
import warnings

@nb.njit(cache=True,fastmath=True)
def computeAngle(p1,p2,p3):
        
    v1 = p1-p2
    v2 = p2-p3
    
    a1 = np.arccos(np.dot(v1,v2)/np.sqrt(np.sum(v1**2))/np.sqrt(np.sum(v2**2)))
    
    return a1/np.pi*180

@nb.njit(cache=True)
def computeDistance(p1,p2):
    
    v1 = p1-p2
    return np.sqrt(np.sum(v1**2))


@nb.njit(cache=True,fastmath=True)
def computeDihedral(p0,p1,p2,p3):

    b0 = -1.0*(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    
    
    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= np.sqrt(np.sum(b1**2))
    b1 = b1.astype(np.float32)
    b0 = b0.astype(np.float32)
    b2 = b2.astype(np.float32)
    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - np.dot(b0, b1)*b1
    w = b2 - np.dot(b2, b1)*b1
    
    #v = v.astype(np.float32)
    #w = w.astype(np.float32)
    
    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    
    return np.degrees(np.arctan2(y, x))

def computeCoordinateMissingICSolve(p1,p2,p3,d1,a,d2):
    """
    

    Parameters
    ----------
    p1,p2,p3 : coordinate, 1d array

    d1 : p3-p4 distance, in coordinate, float
    a : angle between p2,p3,p4, in degree
    d2 : dihedrak p1,p2,p3,p4, in degree

    Returns
    -------
    r(p4): coordinate, 1d array.

    """
    
    def func(i):
        return [
            computeDihedralNoNumba(p1,p2,p3,i)-d2,
            computeAngleNoNumba(p2,p3,i)-a,
            100*(computeDistanceNoNumba(p3, i)-d1)
            ]
    
    warnings.filterwarnings('ignore', 'The iteration is not making good progress')
    r=fsolve(func, p3+np.random.random(3),maxfev=1000)
    
    #print('error = ' , np.sum(r-computeCoordinateMissingICVec(p1,p2,p3,d1,a,d2)))
    #plt.scatter(d2, np.sum(r-computeCoordinateMissingICVector(p1,p2,p3,d1,a,d2)))
    return r
    

#@nb.njit(cache=True,fastmath=True)
def computeCoordinateMissingIC(A,B,C,d1,a,d2,verbose=0):
    #A,B,C = p1,p2,p3
    
    d2 = -d2/180*np.pi
    a = (180-a)/180*np.pi
    
    # Set C as the center of the spherical coordinate
    # Set B-C axis as the z-axis
    z = (B-C)/np.linalg.norm(B-C)
    
    # Build Normal Vector
    u1 = A-C
    out = np.dot(u1,z)
    x = u1 - out*z
    #if out < 0:
    #    x = -x
    x /= np.linalg.norm(x)
    
    # Create y axis
    y = np.cross(z,x)
    #y /= np.linalg.norm(y)
    #print(np.linalg.norm(x),np.linalg.norm(y),np.linalg.norm(z))
    
    #print(x,y,z)
    #print(np.cross(y,z), np.cross(z,x), np.cross(x,y))
    
    # In new spherical axis, calculate the coordinate of p4
    xN = d1 * np.sin(a) * np.cos(d2)
    yN = d1 * np.sin(a) * np.sin(d2)
    zN = d1 * np.cos(a)
    
    #print(xN,yN,zN)
    #print(xN*x + yN*y + zN*z + C)
    
    i= xN*x + yN*y + zN*z + C
    
    #print(d2, a, d1)
    if verbose:
        print(computeDihedralNoNumba(A,B,C,i),
                180-computeAngleNoNumba(B,C,i),
                (computeDistanceNoNumba(C, i)))

    return i

