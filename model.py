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

# model of sky (xy or uv)
class model(object):

    # initialize model as copy of an existing model, maintain coord
    def __init__(self, other=None):
        self.parent = other # do we need to record keep to preserve model tree?

    # operators on model, will return a new transformed model
    def __repr__(self):
        return self.pp()

    # shift model by dx and dy in image domain [radians on sky]
    # note that we never anticipate shifting in uv coords, or mulitplying by complex exp in xy coords
    def shift(self, dx, dy):
        transformed = model(self)
        def eval(r, s, coord):
            if coord == 'uv':
                return np.exp(-2.*np.pi*1j*(dx*r + dy*s)) * self.eval(r, s, coord)
            else: # xy
                return self.eval(r-dx, s-dy, coord)
        transformed.eval = eval
        transformed.pp = lambda: "%s(x-%s, y-%s)" % (self.pp(), repr(dx), repr(dy))
        return transformed

    # stretch model by factors hx and hy in image domain, maintain total flux
    def scale(self, hx, hy):
        transformed = model(self)
        def eval(r, s, coord):
            if coord == 'uv':
                return self.eval(hx*r, hy*s, coord)
            else: # xy
                return (hx*hy)**(-1) * self.eval(r/hx, s/hy, coord) # truediv? theano will truediv
        transformed.eval = eval
        transformed.pp = lambda: "%s(x/%s, y/%s)" % (self.pp(), repr(hx), repr(hy))
        return transformed

    # rotate model by theta [radians]
    def rotate(self, theta):
        transformed = model(self)
        (cth, sth) = (np.cos(theta), np.sin(theta)) # will return theano output on theano argument
        transformed.eval = lambda r,s,coord: self.eval(cth*r + sth*s, -sth*r + cth*s, coord) # negative rotate coords
        transformed.pp = lambda: "R[%s, %s rad]" % (self.pp(), repr(theta))
        return transformed

    # multiply model total flux by constant factor (support for model x model?)
    def multiply(self, factor):
        transformed = model(self)
        transformed.eval = lambda r,s,coord: factor * self.eval(r, s, coord)
        transformed.pp = lambda: "(%s x %s)" % (repr(factor), self.pp())
        return transformed

    # add an additional model to model
    def add(self, other):
        transformed = model(self)
        transformed.eval = lambda r,s,coord: self.eval(r, s, coord) + other.eval(r, s, coord)
        transformed.pp = lambda: "(%s + %s)" % (self.pp(), other.pp())
        return transformed

    # add an additional model to model
    def sub(self, other):
        transformed = model(self)
        transformed.eval = lambda r,s,coord: self.eval(r, s, coord) - other.eval(r, s, coord)
        transformed.pp = lambda: "(%s - %s)" % (self.pp(), other.pp())
        return transformed

    # convolve model with additional model
    def convolve(self, other):
        transformed = model(self)
        def eval(r, s, coord):
            if coord == 'uv':
                return self.eval(r, s, coord) * other.eval(r, s, coord)
            else: # xy
                m1 = self.eval(r, s, coord)
                m2 = other.eval(r, s, coord)
                # try loop doesn't work here.. sometimes fftconvolve perfectly happy with T object
                if isinstance(m1, np.ndarray) and isinstance(m2, np.ndarray):
                    from scipy.signal import fftconvolve
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
        transformed.pp = lambda: "(%s o %s)" % (self.pp(), other.pp())
        return transformed

    # overloaded binary operators for some transforms

    __add__ = add
    __sub__ = sub
    __mul__ = multiply
    __rmul__ = multiply

# model primitives

# point source (delta function) at 0, 0 with total flux = 1
Point = model()
Point.pp = lambda: "Point"
# how best to do this and preserve shape independent of data type? u/u? what is norm for xy coords?
Point.eval = lambda r,s,coord: 1.+0j if coord=='uv' else 1. * ((r==0.) & (s==0.))

# sigma=1 circular gaussian at 0, 0 with total flux = 1
Gauss = model()
Gauss.pp = lambda: "Gauss"
Gauss.eval = lambda r,s,coord: np.exp(-0.5*(r**2 + s**2)) * (1.+0j if coord=='uv' else (2.*np.pi)**(-1))  # np.sqrt okay theano

# r=1 disk at 0, 0 with total flux = 1
Disk = model()
Disk.pp = lambda: "Disk"
Disk.eval = lambda r,s,coord: None if coord=='uv' else 1. * (np.sqrt(r**2 + s**2) < 1) # np.sqrt okay theano

