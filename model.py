# modmod: modular modeling for interferometry
# delayed evaluation allows building of complex model graph from primitives
# 2017, 2018, 2019 L. Blackburn

# outstanding questions:
#  - use amplitude or total flux for normalization?
#  - compatible with pymc3 distributions?
#  - edward? pymc4? tensorflow?
#  - sympy compatibility

# future 3.0 compatibility
# from __future__ import division
# from __future__ import print_function
# str = type('')
# from builtins import object

import numpy as np
# python relative imports have changed ?
# from . import ztypes as zt
import ztypes as zt
from scipy.special import j0, j1

from matplotlib import colors as col
from matplotlib.colors import LinearSegmentedColormap
anglemap = col.LinearSegmentedColormap.from_list(
   'anglemap', [col.BASE_COLORS[c] for c in "kbwrk"], N=256, gamma=1.0)

# set to True to default convert complex to ztypes.Complex
expand=False

# model of sky (xy or uv)
class model(object):

    # initialize model as copy of an existing model, maintain coord
    def __init__(self, other=None):
        # defaults
        self.flux = lambda: 1.0
        self.com = lambda: np.array((0., 0.))
        self.expand = expand       # toggle for ztypes
        if other is not None:
            self.expand = other.expand
            self.im   = other.im   # visuator function xy space
            self.vis  = other.vis  # visuator function uv space
            self.flux = other.flux # total flux
            self.com  = other.com  # center-of-mass
            self.var  = other.var  # covariance matrix
            self.duv  = other.duv  # gradient in u,v
            self.pp   = other.pp   # pretty-print operation
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
    def shift(self, dx, dy=0., expand=expand):
        transformed = model(self)
        def vis(u, v):
            other = self.vis(u, v)
            phasor = -2.* np.pi * (dx*u + dy*v)
            # if hasattr(other, 'zeros_like') or hasattr(phasor, 'zeros_like'): # theano tensor
            if expand or type(other) is zt.Complex:
                real = np.cos(phasor)
                imag = np.sin(phasor)
                return zt.Complex(real=real, imag=imag) * other
            else: # numpy use complex data type
                return np.exp(1j*phasor) * other
        def im(x, y):
                return self.im(x-dx, y-dy)
        transformed.vis = vis
        transformed.im = im
        def duv(u, v):
            other = self.vis(u, v)
            dother = self.duv(u, v)
            phasor = -2. * np.pi * (dx*u + dy*v)
            if expand or type(other) is zt.Complex:
                real = np.cos(phasor)
                imag = np.sin(phasor)
                # (d/du) exp(2pi au+bv) f(u,v) = 2pia exp(2pi au+bv) f(uv) + exp(2pi au+bv) df(uv)
                du = zt.Complex(real=real, imag=imag) * (zt.Complex(real=0, imag=-2*np.pi*dx) * other + dother[0])
                dv = zt.Complex(real=real, imag=imag) * (zt.Complex(real=0, imag=-2*np.pi*dy) * other + dother[1])
                return np.array((du, dv))
            else: # numpy use complex data type
                return np.exp(1j*phasor) * (-2j*np.pi*(np.array((dx, dy))) * other + dother)
        transformed.duv = duv
        transformed.com = lambda: self.com() + np.array((dx, (0 if dy is None else dy))) # center of mass in x,y
        transformed.pp = lambda: "%s(x-%s, y-%s)" % (self.pp(), str(dx), '0' if dy is None else str(dy))
        return transformed

    # stretch model by factors hx and hy in image domain
    # maintain peak flux (norm=False) or total flux (norm=True)
    def scale(self, hx, hy=None, norm=False):
        if hy is None:
            hy = hx
        factor = hx*hy
        if norm:
            transformed = self.scale(hx, hy, norm=False).divide(factor)
            transformed.flux = self.flux # avoid unnecessary computation
            return transformed
        transformed = model(self)
        transformed.vis = lambda u,v: factor * self.vis(hx*u, hy*v)
        transformed.im = lambda x,y: self.im(x/hx, y/hy) # truediv? theano will truediv
        transformed.duv = lambda u,v: factor * self.duv(hx*u, hy*v) * np.array((hx, hy))
        transformed.flux = lambda: self.flux() * factor
        transformed.var = lambda: np.array(((hx**2, factor), (factor, hy**2))) * self.var()
        transformed.pp = lambda: "%s(x/%s, y/%s)" % (self.pp(), str(hx), str(hy))
        return transformed

    # rotate model by theta [radians]
    def rotate(self, theta, deg=False):
        transformed = model(self)
        if deg:
            theta = theta * np.pi/180. 
        (cth, sth) = (np.cos(theta), np.sin(theta)) # will return theano output on theano argument
        R = np.array(((cth, -sth), (sth, cth))) # rotation matrix
        transformed.vis = lambda u,v: self.vis(cth*u + sth*v, -sth*u + cth*v) # negative rotate coords
        transformed.im  = lambda x,y: self.im(cth*x + sth*y, -sth*x + cth*y) # negative rotate coords
        def duv(u, v):
            dother = self.duv(cth*u + sth*v, -sth*u + cth*v) # derivative at original coordinate
            return np.dot(R, dother) # rotate old derivative vector
        def com():
            return np.matmul(R, self.com())
        def var(): # guess
            return np.matmul(R.T, np.matmul(R, self.var()))
        transformed.com = com
        transformed.var = var
        transformed.duv = duv
        transformed.pp = lambda: "R[%s, %s deg]" % (self.pp(), str(theta*180./np.pi))
        return transformed

    # multiply model total flux by constant factor (support for model x model?)
    def multiply(self, factor):
        transformed = model(self)
        transformed.vis = lambda u,v: factor * self.vis(u, v)
        transformed.duv = lambda u,v: factor * self.duv(u, v)
        transformed.im = lambda x,y: factor * self.im(x, y)
        transformed.flux = lambda: factor * self.flux()
        transformed.pp = lambda: "(%s x %s)" % (str(factor), self.pp())
        return transformed

    # divide model total flux by constant factor (support for 1./model?)
    def divide(self, factor):
        transformed = model(self)
        transformed.vis = lambda u,v: self.vis(u, v) / factor
        transformed.duv = lambda u,v: self.duv(u, v) / factor
        transformed.im = lambda x,y: self.im(x, y) / factor
        transformed.flux = lambda: self.flux() / factor
        transformed.pp = lambda: "(%s / %s)" % (self.pp(), str(factor))
        return transformed

    # add an additional model to model
    # calculating moments here is getting inefficient -- better with some memoization scheme
    def add(self, other):
        transformed = model(self)
        transformed.vis = lambda u,v: self.vis(u, v) + other.vis(u, v)
        transformed.duv = lambda u,v: self.duv(u, v) + other.duv(u, v)
        transformed.im = lambda x,y: self.im(x, y) + other.im(x, y)
        transformed.flux = lambda: self.flux() + other.flux()
        transformed.com = lambda: (self.flux()*self.com() + other.flux()*other.com()) / (self.flux() + other.flux())
        def var():
            (f1, f2) = (self.flux(), other.flux())
            (c1, c2) = (self.com(), other.com())
            (v1, v2) = (self.var(), other.var())
            tot = f1 + f2
            (a1, a2) = (f1/tot, f2/tot)
            com = a1*c1 + a2*c2 # new center-of-mass
            (d1, d2) = (c1-com, c2-com) # difference vector to new center-of-mass
            return a1*(v1 + np.outer(d1, d1)) + a2*(v2 + np.outer(d2, d2))
        transformed.var = var
        transformed.pp = lambda: "(%s + %s)" % (self.pp(), other.pp())
        return transformed

    # subtract a model from model
    def sub(self, other):
        transformed = model(self)
        transformed.vis = lambda u,v: self.vis(u, v) - other.vis(u, v)
        transformed.duv = lambda u,v: self.duv(u, v) - other.duv(u, v)
        transformed.im = lambda x,y: self.im(x, y) - other.im(x, y)
        transformed.flux = lambda: self.flux() - other.flux()
        transformed.com = lambda: (self.flux()*self.com() - other.flux()*other.com()) / (self.flux() - other.flux())
        def var():
            (f1, f2) = (self.flux(), other.flux())
            (c1, c2) = (self.com(), other.com())
            (v1, v2) = (self.var(), other.var())
            tot = f1 - f2
            (a1, a2) = (f1/tot, f2/tot)
            com = a1*c1 - a2*c2 # new center-of-mass
            (d1, d2) = (c1-com, c2-com) # difference vector to new center-of-mass
            return a1*(v1 + np.outer(d1, d1)) - a2*(v2 + np.outer(d2, d2))
        transformed.var = var
        transformed.pp = lambda: "(%s - %s)" % (self.pp(), other.pp())
        return transformed

    # convolve model with additional model
    def convolve(self, other):
        transformed = model(self)
        transformed.vis = lambda u,v: self.vis(u, v) * other.vis(u, v)
        transformed.duv = lambda u,v: self.vis(u, v) * other.duv(u, v) + other.vis(u, v) * self.duv(u, v)
        def im(x, y):
            m1 = self.im(x, y)
            m2 = other.im(x, y)
            # try loop doesn't work here.. sometimes fftconvolve perfectly happy with T object
            # note that for this to work, coord must be a uniform grid
            if all((isinstance(m1, (int, float, complex, np.ndarray)) for obj in (m1, m2))):
                from scipy.signal import fftconvolve
                dv = (x[0,1]-x[0,0]) * (y[1,0]-y[0,0]) # must be from e.g. meshgrid
                ret = fftconvolve(m1, m2, mode='same') * dv
                return ret
            else:
                print("theano path (convolve)")
                import theano.tensor as T
                dv = (x[0,1]-x[0,0]) * (y[1,0]-y[0,0]) # must be from e.g. meshgrid
                # ret = fftconvolve(m1, m2, mode='same') * dv
                m1pad = T.shape_padleft(m1, 2)
                m2pad = T.shape_padleft(m2, 2)
                ret = T.nnet.conv2d(m1pad, m2pad, border_mode='half', filter_flip=False)[0,0] / dv
                return ret
        transformed.im = im
        transformed.com = lambda: self.com() + other.com()
        transformed.flux = lambda: self.flux() * other.flux()
        transformed.var = lambda: self.var() + other.var()
        transformed.pp = lambda: "(%s o %s)" % (self.pp(), other.pp())
        return transformed

    # center image to (0, 0): shortcut to shift by -com()
    # not so efficient for any real work..
    def center(self):
        transformed = model(self)
        transformed.com = lambda: np.array((0., 0.))
        def vis(u, v):
            com = self.com()
            return self.shift(-com[0], -com[1]).vis(u, v)
        def im(x, y):
            com = self.com()
            return self.shift(-com[0], -com[1]).im(x, y)
        def duv(u, v):
            com = self.com()
            return self.shift(-com[0], -com[1]).duv(u, v)
        transformed.vis = vis
        transformed.im = im
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

    # normalize to total flux of 1
    def norm(self):
        transformed = model(self)
        transformed.flux = lambda: 1.0
        transformed.vis = lambda u,v: self.vis(u,v) / self.flux()
        transformed.duv = lambda u,v: self.duv(u,v) / self.flux()
        transformed.im = lambda x,y: self.im(x,y) / self.flux()
        transformed.pp = lambda: "Norm[%s]" % (self.pp())
        return transformed

    def show(self, n=256, colorbar='horizontal', fov=None, zoom=(3, 3), cmap='afmhot', pmap=anglemap):
        import matplotlib.pyplot as plt
        if fov is None:
            fov = np.sqrt(np.diagonal(self.var()))  # set FOV to 1 sigma
        fov[fov==0] = 1. # replace any zero fov (e.g. point source)
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
        vxy = self.im(xx, yy)
        vuv = self.vis(uu, vv)
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
        plt.setp(plt.gcf(), figwidth=12, figheight=4.5)
        plt.tight_layout()

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
# Point.vis = lambda u,v: 1.
# Point.duv = lambda u,v: 0.
Point.vis = lambda u,v: np.ones_like(u)
Point.duv = lambda u,v: np.zeroes_like(u)
Point.im = lambda x,y: 1. * ((x==0.) & (y==0.))
Point.var = lambda: np.array(((0., 0.), (0., 0.)))
Point.duv = None

# sigma_xy=1, sigma_uv=1/2pi circular gaussian at 0, 0 with total flux = 1
Gauss = model()
Gauss.pp = lambda: "Gauss" # sigma_uv = 1/(2pi*sigma_xy)
Gauss.vis = lambda u,v: np.exp(-2.*np.pi**2*(u**2 + v**2))
Gauss.duv = lambda u,v: (-2.*np.pi**2) * np.array((2*u, 2*v)) * np.exp(-2.*np.pi**2*(u**2 + v**2))
Gauss.im = lambda x,y: (np.exp(-0.5*(x**2 + y**2)) * (2.*np.pi)**(-1))
Gauss.var = lambda: np.array(((1., 0.), (0., 1.)))
Gauss.duv = None

# r=1 circle with total flux = 1 (uv only)
Circle = model()
Circle.pp = lambda: "Circle" # unit circle of unit flux (delta function at r=1 / 2pi)
Circle.vis = lambda u,v: j0(2*np.pi*(u**2 + v**2))
Circle.im = lambda x,y: (np.sqrt(x**2 + y**2) == 1.) / (2*np.pi)
Circle.var = lambda: np.array(((0.5, 0.), (0., 0.5)))
Circle.duv = None

# r=1 disk at 0, 0 with total flux = 1
Disk = model()
Disk.pp = lambda: "Disk"
def vis(u, v):
    r = np.sqrt(u**2 + v**2)
    return np.nan_to_num(j1(2*np.pi*r)/r)/np.pi + 1.*(r == 0.)
Disk.vis = lambda u,v: vis(u, v)
Disk.im = lambda x,y: (np.sqrt(x**2 + y**2) < 1) / np.pi # np.sqrt okay theano
Disk.var = lambda: np.array(((1./3., 0.), (0., 1./3.)))
Disk.duv = None

# Crescent model with radius r1<r2, total flux=1, e.g. simple case of Kamruddin & Dexter 2013
# default is to center at middle of outer disk, recenter with Crescent.center()
# -1 < asymmetry < 1 places inner disk boundary with respect to outer disk along x
# contrast: 1 = zero flux inside, 0 = fully-filled disk
def Crescent(r1=0.75, r2=1.0, asymmetry=1.0, contrast=1, expand=expand):
    mod = (Disk.scale(r2) - contrast*Disk.scale(r1).shift(asymmetry*(r2-r1),0,expand=expand)).divide((r2**2-contrast*r1**2))
    mod.pp = lambda: "Crescent(%s,%s,%s)" % (str(r1), str(r2), asymmetry)
    return mod

# Ring model with radius r1<r2, total flux=1
def Ring(r1=0.75, r2=1.0):
    mod = (Disk.scale(r2) - Disk.scale(r1)).divide((r2**2 - r1**2))
    mod.pp = lambda: "Ring(%s,%s)" % (str(r1), str(r2))
    return mod

