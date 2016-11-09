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

# uv model of sky
class uvmodel():

    # initialize model as copy of an existing model
    def __init__(self, model=None):
        self.parent = model # do we need to record keep to preserve model tree?

    # operators on model, will return a new transformed model

    # shift model by dx and dy in image domain [radians on sky]
    def shift(self, dx, dy):
        transformed = uvmodel(self)
        transformed.eval = lambda u,v: np.exp(-2.*np.pi*1j*(dx*u + dy*v)) * self.eval(u, v)
        transformed.__repr__ = lambda: "%s(x-%s, y-%s)" % (repr(self), repr(dx), repr(dy))
        return transformed

    # shift assuming image domain (for test)
    def imshift(self, dx, dy):
        transformed = uvmodel(self)
        transformed.eval = lambda u,v: self.eval(u-dx, v-dy)
        transformed.__repr__ = lambda: "%s(x-%s, y-%s)" % (repr(self), repr(dx), repr(dy))
        return transformed

    # rotate model by theta [radians]
    def rotate(self, theta):
        transformed = uvmodel(self)
        (cth, sth) = (np.cos(theta), np.sin(theta))
        transformed.eval = lambda u,v: self.eval(cth*u + sth*v, -sth*u + cth*v) # negative rotate coords
        transformed.__repr__ = lambda: "R[%s, %s rad]" % (repr(self), repr(theta))
        return transformed

    # stretch model by hx and hy in image domain, maintain total flux
    def scale(self, hx, hy):
        transformed = uvmodel(self)
        transformed.eval = lambda u,v: self.eval(hx*u, hy*v)
        transformed.__repr__ = lambda: "%s(x/%s, y/%s)" % (repr(self), repr(hx), repr(hy))
        return transformed

    # scale model total flux by factor
    def multiply(self, factor):
        transformed = uvmodel(self)
        transformed.eval = lambda u,v: factor * self.eval(u, v)
        transformed.__repr__ = lambda: "(%s x %s)" % (repr(factor), repr(self))
        return transformed

    # add an additional model to model
    def add(self, model):
        transformed = uvmodel(self)
        transformed.eval = lambda u,v: self.eval(u, v) + model.eval(u, v)
        transformed.__repr__ = lambda: "(%s + %s)" % (repr(self), repr(model))
        return transformed

    # add an additional model to model
    def sub(self, model):
        transformed = uvmodel(self)
        transformed.eval = lambda u,v: self.eval(u, v) - model.eval(u, v)
        transformed.__repr__ = lambda: "(%s - %s)" % (repr(self), repr(model))
        return transformed

    # convolve model with additional model
    def convolve(self, model):
        transformed = uvmodel(self)
        transformed.eval = lambda u,v: self.eval(u, v) * model.eval(u, v)
        transformed.__repr__ = lambda: "(%s o %s)" % (repr(self), repr(model))
        return transformed

    # overloaded binary operators for some transforms

    __add__ = add
    __sub__ = sub
    __mul__ = multiply
    __rmul__ = multiply

    # visibility function ? (manual recursion)
    def eval(self, u, v):
        raise(Exception("eval called on empty model"))

# model primitives

# point source (delta function) at 0, 0 with total flux = 1
Point = uvmodel()
Point.__repr__ = lambda: "Point"
Point.eval = lambda u,v: 1.+ 0j # how best to do this and preserve shape independent of data type? u/u?

# sigma=1 circular gaussian at 0, 0 with total flux = 1
Gauss = uvmodel()
Gauss.__repr__ = lambda: "Gauss"
Gauss.eval = lambda u,v: np.exp(-0.5*(u**2 + v**2)) + 0j # make theano compatible?

# r=1 disk at 0, 0 with total flux = 1
Disk = uvmodel()
Disk.__repr__ = lambda: "Disk"
Disk.eval = lambda u,v: 1. * (np.sqrt(u**2 + v**2) < 1) + 0j # make theano compatible?

