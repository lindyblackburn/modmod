# modmod: modular modeling for interferometry

# outstanding questions:
#  - use amplitude or total flux?
#  - compatible with pymc3 distributions?

# future 3.0 compatibility
from __future__ import division
from __future__ import print_function
str = type('')
from builtins import object

import numpy as np
from scipy.special import j0, j1

# model of sky (xy or uv)
class model(object):

    # initialize model as copy of an existing model, maintain coord
    def __init__(self, other=None):
        # defaults
        self.flux = lambda: 1.0
        self.com = lambda: np.array((0., 0.))
        if other is not None:
            self.eval = other.eval
            self.flux = other.flux
            self.com = other.com
            self.var = other.var
            self.pp = other.pp
        self.parent = other # do we need to record keep to preserve model tree?

    # operators on model, will return a new transformed model
    def __repr__(self):
        return self.pp()

    # return a copy of self: model(self)
    def copy(self):
        return model(self)

    # shift model by dx and dy in image domain [radians on sky]
    # note that we never anticipate shifting in uv coords, or mulitplying by complex exp in xy coords
    def shift(self, dx, dy=0):
        transformed = model(self)
        def eval(r, s, coord='xy'):
            if coord == 'uv':
                return np.exp(-2.*np.pi*1j*(dx*r + dy*s)) * self.eval(r, s, coord)
            else: # xy
                return self.eval(r-dx, s-dy, coord)
        transformed.eval = eval
        transformed.com = lambda: self.com() + np.array((dx, dy)) # center of mass in x,y
        transformed.pp = lambda: "%s(x-%s, y-%s)" % (self.pp(), str(dx), str(dy))
        return transformed

    # stretch model by factors hx and hy in image domain maintain peak flux
    def scale(self, hx, hy=None, norm=False):
        if hy is None:
            hy = hx
        if norm:
            transformed = self.scale(hx, hy).divide(hx*hy)
            return transformed
        transformed = model(self)
        def eval(r, s, coord='xy'):
            if coord == 'uv':
                return hx*hy * self.eval(hx*r, hy*s, coord)
            else: # xy
                return self.eval(r/hx, s/hy, coord) # truediv? theano will truediv
        transformed.eval = eval
        transformed.flux = lambda: self.flux() * hx * hy
        transformed.var = lambda: np.array((hx**2, hy**2)) * self.var()
        transformed.pp = lambda: "%s(x/%s, y/%s)" % (self.pp(), str(hx), str(hy))
        return transformed

    # rotate model by theta [radians]
    def rotate(self, theta, deg=False):
        transformed = model(self)
        if deg:
            theta = theta * np.pi/180. 
        (cth, sth) = (np.cos(theta), np.sin(theta)) # will return theano output on theano argument
        transformed.eval = lambda r,s,coord='xy': self.eval(cth*r + sth*s, -sth*r + cth*s, coord) # negative rotate coords
        def com():
            (x, y) = self.com()
            return np.array(((cth*x - sth*y), (sth*x + cth*y)))
        def var(): # guess
            (x, y) = self.var()
            return np.array(((cth*x - sth*y), (sth*x + cth*y)))
        transformed.com = com
        transformed.var = var
        transformed.pp = lambda: "R[%s, %s deg]" % (self.pp(), str(theta*180./np.pi))
        return transformed

    # multiply model total flux by constant factor (support for model x model?)
    def multiply(self, factor):
        transformed = model(self)
        transformed.eval = lambda r,s,coord='xy': factor * self.eval(r, s, coord)
        transformed.flux = lambda: factor * self.flux()
        transformed.pp = lambda: "(%s x %s)" % (str(factor), self.pp())
        return transformed

    # divide model total flux by constant factor (support for 1./model?)
    def divide(self, factor):
        transformed = model(self)
        transformed.eval = lambda r,s,coord='xy': self.eval(r, s, coord) / factor
        transformed.flux = lambda: self.flux() / factor
        transformed.pp = lambda: "(%s / %s)" % (self.pp(), str(factor))
        return transformed

    # add an additional model to model
    # calculating moments here is getting inefficient -- better with some memoization scheme
    def add(self, other):
        transformed = model(self)
        transformed.eval = lambda r,s,coord='xy': self.eval(r, s, coord) + other.eval(r, s, coord)
        transformed.com = lambda: (self.flux()*self.com() + other.flux()*other.com()) / (self.flux() + other.flux())
        def var():
            (f1, f2) = (self.flux(), other.flux())
            (c1, c2) = (self.com(), other.com())
            com = (f1*c1 + f2*c2) / (f1+f2) # new center-of-mass
            (d1, d2) = (c1-com, c2-com) # difference vector to new center-of-mass
            return (f1*(self.var() + d1**2) + f2*(other.var() + d2**2)) / (f1+f2)
        transformed.flux = lambda: self.flux() + other.flux()
        transformed.var = var
        transformed.pp = lambda: "(%s + %s)" % (self.pp(), other.pp())
        return transformed

    # subtract a model from model
    def sub(self, other):
        transformed = model(self)
        transformed.eval = lambda r,s,coord='xy': self.eval(r, s, coord) - other.eval(r, s, coord)
        transformed.com = lambda: (self.flux()*self.com() - other.flux()*other.com()) / (self.flux() + other.flux())
        transformed.var = lambda: self.var() + other.var() # too complicated to do right..
        transformed.flux = lambda: self.flux() - other.flux()
        transformed.pp = lambda: "(%s - %s)" % (self.pp(), other.pp())
        return transformed

    # convolve model with additional model
    def convolve(self, other):
        transformed = model(self)
        def eval(r, s, coord='xy'):
            if coord == 'uv':
                return self.eval(r, s, coord) * other.eval(r, s, coord)
            else: # xy
                m1 = self.eval(r, s, coord)
                m2 = other.eval(r, s, coord)
                # try loop doesn't work here.. sometimes fftconvolve perfectly happy with T object
                # note that for this to work, coord must be a uniform grid
                if all((isinstance(m1, (int, long, float, complex, np.ndarray)) for obj in (m1, m2))):
                    from scipy.signal import fftconvolve
                    dv = (r[0,1]-r[0,0]) * (s[1,0]-s[0,0]) # must be from e.g. meshgrid
                    ret = fftconvolve(m1, m2, mode='same') * dv
                    print("numpy path")
                    return ret
                else:
                    print("theano path")
                    import theano.tensor as T
                    dv = (r[0,1]-r[0,0]) * (s[1,0]-s[0,0]) # must be from e.g. meshgrid
                    ret = fftconvolve(m1, m2, mode='same') * dv
                    m1pad = T.shape_padleft(m1, 2)
                    m2pad = T.shape_padleft(m2, 2)
                    ret = T.nnet.conv2d(m1pad, m2pad, border_mode='half', filter_flip=False)[0,0] / dv
                    return ret
        transformed.eval = eval
        transformed.com = lambda: self.com() + other.com()
        transformed.flux = lambda: self.flux() * other.flux()
        transformed.var = lambda: self.var() + other.var()
        transformed.pp = lambda: "(%s o %s)" % (self.pp(), other.pp())
        return transformed

    # center image to (0, 0): shortcut to shift by -com()
    def center(self):
        transformed = model(self)
        transformed.com = lambda: np.array((0., 0.))
        def eval(xx, yy, coord='xy'):
            com = self.com()
            return self.shift(-com[0], -com[1]).eval(xx, yy, coord)
        transformed.eval = eval
        transformed.pp = lambda: "Center[%s]" % (self.pp())
        return transformed

    # blur by Gaussian kernel, shortcut to convolve(Gauss.scale(hx, hy).rotate(theta))
    def blur(self, hx, hy=None, theta=None, deg=False):
        if hy is None:
            hy = hx
        kern = Gauss.scale(hx, hy, norm=True)
        if theta is None:
            mod = self.convolve(kern)
        else:
            mod = self.convolve(kern.rotate(theta, deg))
        return mod

    # for xy use image size as FOV
    # for uv use size of smallest feature for FOV
    def show(self, coord='xy', fov=None, n=256, cmap='afmhot'):
        import matplotlib.pyplot as plt
        if fov is None:
            fov = 3. * np.sqrt(np.max(self.var()))
        if coord == 'uv':
            r = np.linspace(-2./fov, 2./fov, n, endpoint=False)
        else:
            r = np.linspace(-fov/2., fov/2., n, endpoint=False)
        dr = r[1] - r[0]
        (rr, ss) = np.meshgrid(r, r)
        a = np.abs(self.eval(rr, ss, coord))
        plt.imshow(a, origin='lower', vmin=0, extent=[-(fov+dr)/2., (fov-dr)/2., -(fov+dr)/2., (fov-dr)/2.], cmap=cmap)
        plt.xlabel(coord[0])
        plt.ylabel(coord[1], rotation=0.)

    # overloaded binary operators for some transforms

    __add__ = add
    __sub__ = sub
    __mul__ = multiply
    __rmul__ = multiply
    __div__ = divide

# model primitives

def err(msg):
    raise(Exception(msg))

# point source (delta function) at 0, 0 with total flux = 1
Point = model()
Point.pp = lambda: "Point"
# how best to do this and preserve shape independent of data type? u/u? what is norm for xy coords?
Point.eval = lambda r,s,coord='xy': 1.+0j if coord=='uv' else 1. * ((r==0.) & (s==0.))
Point.var = lambda: np.array((0., 0.))

# sigma_xy=1, sigma_uv=1/2pi circular gaussian at 0, 0 with total flux = 1
Gauss = model()
Gauss.pp = lambda: "Gauss" # sigma_uv = 1/(2pi*sigma_xy)
Gauss.eval = lambda r,s,coord='xy': (np.exp(-2.*np.pi**2*(r**2 + s**2)) * 1.+0j) if coord=='uv' \
                          else (np.exp(-0.5*(r**2 + s**2)) * (2.*np.pi)**(-1))  # np.sqrt okay theano
Gauss.var = lambda: np.array((1., 1.))

# r=1 circle with total flux = 1 (uv only)
Circle = model()
Circle.pp = lambda: "Circle" # unit circle of unit flux (delta function at r=1 / 2pi)
Circle.eval = lambda r,s,coord='xy': j0(2*np.pi*(r**2 + s**2)) if coord=='uv' \
                          else (np.sqrt(r**2 + s**2) == 1.) / (2*np.pi)
Circle.var = lambda: np.array((0.5, 0.5))

# r=1 disk at 0, 0 with total flux = 1
Disk = model()
Disk.pp = lambda: "Disk"
Disk.eval = lambda r,s,coord='xy': np.nan_to_num(j1(2*np.pi*(np.sqrt(r**2 + s**2)))/(np.sqrt(r**2 + s**2)))/np.pi + 1.*(r**2 + s**2 == 0.) if coord=='uv' \
                          else (np.sqrt(r**2 + s**2) < 1) / np.pi # np.sqrt okay theano
Disk.var = lambda: np.array((1./3., 1./3.))

# Crescent model with radius r1<r2, total flux=1
# default is to center at middle of outer disk
def Crescent(r1=0.75, r2=1.0):
    mod = (Disk.scale(r2) - Disk.scale(r1).shift(r2-r1,0)).divide((r2**2-r1**2))
    mod.pp = lambda: "Crescent(%s,%s)" % (str(r1), str(r2))
    return mod

# Ring model with radius r1<r2, total flux=1
def Ring(r1=0.75, r2=1.0):
    mod = (Disk.scale(r2) - Disk.scale(r1)).divide((r2**2 - r1**2))
    mod.pp = lambda: "Ring(%s,%s)" % (str(r1), str(r2))
    return mod

