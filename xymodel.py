# modmod: modular modeling for interferometry

# outstanding questions:
#  - use theano or manual recursion for visib evaluation?
#    - manual may be compatible with basic variables (delayed evaluation) and theano (after evaluation)
#  - use amplitude or total flux?
#  - compatible with pymc3 distributions?
#  - how to initialize?
#  - how many subclasses and for what? (mostly bookkeeping)
#  - support for image model?

import numpy as np
from scipy.signal import fftconvolve

# uv model of sky
class xymodel():

    # initialize model as copy of an existing model
    def __init__(self, model=None):
        self.parent = model # do we need to record keep to preserve model tree?

    # operators on model, will return a new transformed model

    # shift assuming image domain (for test)
    def shift(self, dx, dy):
        transformed = xymodel(self)
        transformed.eval = lambda x,y: self.eval(x-dx, y-dy)
        transformed.__repr__ = lambda: "%s(x-%s, y-%s)" % (repr(self), repr(dx), repr(dy))
        return transformed

    # rotate model by theta [radians]
    def rotate(self, theta):
        transformed = xymodel(self)
        (cth, sth) = (np.cos(theta), np.sin(theta)) # this will call T.cos and T.sin, theano compatible
        transformed.eval = lambda x,y: self.eval(cth*x + sth*y, -sth*x + cth*y) # negative rotate coords
        transformed.__repr__ = lambda: "R[%s, %s rad]" % (repr(self), repr(theta))
        return transformed

    # stretch model by hx and hy in image domain, maintain total flux
    def scale(self, hx, hy):
        transformed = xymodel(self)
        transformed.eval = lambda x,y: (hx*hy)**(-1) * self.eval(x/hx, y/hy) # truediv? theano will truediv
        transformed.__repr__ = lambda: "%s(%sx, %sy)" % (repr(self), repr(hx), repr(hy))
        return transformed

    # scale model total flux by factor
    def multiply(self, factor):
        transformed = xymodel(self)
        transformed.eval = lambda x,y: factor * self.eval(x, y)
        transformed.__repr__ = lambda: "(%s x %s)" % (repr(factor), repr(self))
        return transformed

    # add an additional model to model
    def add(self, model):
        transformed = xymodel(self)
        transformed.eval = lambda x,y: self.eval(x, y) + model.eval(x, y)
        transformed.__repr__ = lambda: "(%s + %s)" % (repr(self), repr(model))
        return transformed

    # add an additional model to model
    def sub(self, model):
        transformed = xymodel(self)
        transformed.eval = lambda x,y: self.eval(x, y) - model.eval(x, y)
        transformed.__repr__ = lambda: "(%s - %s)" % (repr(self), repr(model))
        return transformed

    # convolve model with additional model
    def convolve(self, model):
        transformed = xymodel(self)
        def eval(x, y):
            m1 = self.eval(x, y)
            m2 = model.eval(x, y)
            # try loop doesn't work here.. sometimes fftconvolve perfectly happy with T object
            if isinstance(m1, np.ndarray) and isinstance(m2, np.ndarray):
                ret = fftconvolve(m1, m2, mode='same')
                print "numpy path"
                return ret
            else:
                print "theano path"
                import theano.tensor as T
                m1pad = T.shape_padleft(m1, 2)
                m2pad = T.shape_padleft(m2, 2)
                ret = T.nnet.conv2d(m1pad, m2pad, border_mode='half', filter_flip=False)[0,0]
                return ret
        transformed.eval = eval
        transformed.__repr__ = lambda: "(%s o %s)" % (repr(self), repr(model))
        return transformed

    # overloaded binary operators for some transforms

    __add__ = add
    __sub__ = sub
    __mul__ = multiply
    __rmul__ = multiply

    # visibility function ? (manual recursion)
    def eval(self, x, y):
        raise(Exception("eval called on empty model"))

# model primitives

# point source (delta function) at 0, 0 with total flux = 1, what is normalization?
Point = xymodel()
Point.__repr__ = lambda: "Point"
Point.eval = lambda x,y: 1. * ((x==0.) & (y==0.)) # how best to do this and preserve shape independent of data type? u/u?

# sigma=1 circular gaussian at 0, 0 with total flux = 1
Gauss = xymodel()
Gauss.__repr__ = lambda: "Gauss"
Gauss.eval = lambda x,y: (2.*np.pi)**(-1) * np.exp(-0.5*(x**2 + y**2)) # make theano compatible?

# r=1 disk at 0, 0 with total flux = 1
Disk = xymodel()
Disk.__repr__ = lambda: "Disk"
Disk.eval = lambda x,y: (2.*np.pi)**(-1) * (np.sqrt(x**2 + y**2) < 1) # make theano compatible?

