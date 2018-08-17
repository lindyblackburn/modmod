# modmod: modular modeling for interferometry
# delayed evaluation allows building of complex model graph from primitives
# 2017, 2018 L. Blackburn

# outstanding questions:
#  - use amplitude or total flux for normalization?
#  - compatible with pymc3 distributions?
#  - edward? pymc4?
#  - sympy compatibility

# future 3.0 compatibility
# from __future__ import division
# from __future__ import print_function
# str = type('')
# from builtins import object

import numpy as np
import ztypes as zt
from scipy.special import j0, j1

from matplotlib import colors as col
from matplotlib.colors import LinearSegmentedColormap
anglemap = col.LinearSegmentedColormap.from_list(
   'anglemap', [col.BASE_COLORS[c] for c in "kbwrk"], N=256, gamma=1.0)

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

    # shift model by dx and dy in image domain [e.g. radians on sky]
    # note that we never anticipate shifting in uv coords, or mulitplying by complex exp in xy coords
    # expand: if true, use ztypes to split complex number into real, imag
    def shift(self, dx, dy=0, expand=None):
        transformed = model(self)
        def eval(r, s, coord='xy'):
            if coord == 'uv':
                other = self.eval(r, s, coord)
                phasor = -2.* np.pi * (dx*r + dy*s)
                # if hasattr(other, 'zeros_like') or hasattr(phasor, 'zeros_like'): # theano tensor
                if expand or type(other) is zt.Complex:
                    real = np.cos(phasor)
                    imag = np.sin(phasor)
                    return zt.Complex(real=real, imag=imag) * self.eval(r, s, coord)
                else: # numpy use complex data type
                    return np.exp(1j*phasor) * self.eval(r, s, coord)
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
        def eval(r, s, coord):
            # print self.eval(r, s, coord)
            # print other.eval(r, s, coord)
            return self.eval(r, s, coord) + other.eval(r, s, coord)
        transformed.eval = eval
        # transformed.eval = lambda r,s,coord='xy': self.eval(r, s, coord) + other.eval(r, s, coord)
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
                    print("numpy path (convolve)")
                    return ret
                else:
                    print("theano path (convolve)")
                    import theano.tensor as T
                    dv = (r[0,1]-r[0,0]) * (s[1,0]-s[0,0]) # must be from e.g. meshgrid
                    # ret = fftconvolve(m1, m2, mode='same') * dv
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

    def show(self, n=256, colorbar='horizontal', fov=None, zoom=(3, 3), cmap='afmhot', pmap=anglemap):
        import matplotlib.pyplot as plt
        if fov is None:
            fov = np.sqrt(self.var())  # set FOV to 1 sigma
        if not hasattr(zoom, '__getitem__'):
            zoom = (zoom, zoom) # set x and y zoom to be the same
        fovxy = zoom[0] * max(fov)
        fovuv = zoom[1] / (2. * np.pi * min(fov))
        x = np.linspace(-fovxy, fovxy, n, endpoint=False) # endpoint=False includes a zero point
        u = np.linspace(-fovuv, fovuv, n, endpoint=False) # n=2**m allows faster convolve
        dx = x[1]-x[0]
        du = u[1]-u[0]
        (xx, yy) = np.meshgrid(x, x)
        (uu, vv) = np.meshgrid(u, u)
        vxy = self.eval(xx, yy, 'xy')
        vuv = self.eval(uu, vv, 'uv')
        plt.subplot(1, 3, 1)
        plt.imshow(vxy, origin='lower', vmin=min(np.min(vxy), 0),
            extent=[-fovxy-dx/2., fovxy-dx/2., -fovxy-dx/2., fovxy-dx/2.], cmap=cmap)
        plt.xlabel('x')
        plt.ylabel('y', rotation=0.)
        if colorbar != 'none': plt.colorbar(orientation=colorbar)
        plt.subplot(1, 3, 2)
        plt.imshow(np.abs(vuv), origin='lower', vmin=0,
            extent=[-fovuv-du/2., fovuv-du/2., -fovuv-du/2., fovuv-du/2.], cmap=cmap)
        plt.xlabel('u')
        plt.ylabel('v', rotation=0.)
        if colorbar != 'none': plt.colorbar(orientation=colorbar)
        plt.subplot(1, 3, 3)
        plt.imshow(180.*np.angle(vuv)/np.pi, origin='lower', vmin=-180, vmax=180,
            extent=[-fovuv-du/2., fovuv-du/2., -fovuv-du/2., fovuv-du/2.], cmap=pmap)
        plt.xlabel('u')
        plt.ylabel('v', rotation=0.)
        if colorbar != 'none': plt.colorbar(orientation=colorbar)

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
Point.eval = lambda r,s,coord='xy': 1. if coord=='uv' else 1. * ((r==0.) & (s==0.))
Point.var = lambda: np.array((0., 0.))

# sigma_xy=1, sigma_uv=1/2pi circular gaussian at 0, 0 with total flux = 1
Gauss = model()
Gauss.pp = lambda: "Gauss" # sigma_uv = 1/(2pi*sigma_xy)
Gauss.eval = lambda r,s,coord='xy': (np.exp(-2.*np.pi**2*(r**2 + s**2))) if coord=='uv' \
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

# Crescent model with radius r1<r2, total flux=1, e.g. simple case of Kamruddin & Dexter 2013
# default is to center at middle of outer disk, recenter with Crescent.center()
# -1 < asymmetry < 1 places inner disk boundary with respect to outer disk along x
def Crescent(r1=0.75, r2=1.0, asymmetry=1.0):
    mod = (Disk.scale(r2) - Disk.scale(r1).shift(asymmetry*(r2-r1),0)).divide((r2**2-r1**2))
    mod.pp = lambda: "Crescent(%s,%s,%s)" % (str(r1), str(r2), asymmetry)
    return mod

# Ring model with radius r1<r2, total flux=1
def Ring(r1=0.75, r2=1.0):
    mod = (Disk.scale(r2) - Disk.scale(r1)).divide((r2**2 - r1**2))
    mod.pp = lambda: "Ring(%s,%s)" % (str(r1), str(r2))
    return mod

