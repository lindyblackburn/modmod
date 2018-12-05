"""
Complex types

2018.06.05 - Lindy Blackburn, basic construction
2018.12.03 - Lindy Blackburn, additional tune-up
"""

import numpy as np

# Complex type has two attributes: real, imag
# each is supposed to be "real-valued" and represent the re,im components
# what is the goal here ...
# if it is desired to use ztypes, then user should inject a ztypes element early on
# ztypes should greedy promote objects to ztypes when needed
# the only use case for ztypes right now is to support transparent fake "complex" types for theano vars
# theano vars shouldn't necessarily use ztypes, only if autograd is desired
# so it should not be automatically triggered, it should be manually triggered
# note we want to think about post-theano work as well
class Complex(object):
    """
    Complex pair a[...,0] + 1j*a[...,1]
    """

    # create from complex numbers or real, imag, no copy!
    def __init__(self, real=None, imag=None):
        if type(real) == type(self): # already a Complex type, do nothing
            if imag is not None:
                raise Exception('cannot set imag value with Complex type input')
            (self.real, self.imag) = (real.real, real.imag)
        elif imag is not None:  # explicitly send real, imag - iterators and lists will be left as is
            (self.real, self.imag) = (real, imag)
        # examples: 1j, [1j, 2j], np.array(1j)
        # not examples: theano.tensor.cscalar() 
        # due to conversion to array, there are some differences between e.g. array(1) and 1 in output
        elif np.iscomplexobj(real):    # already a complex numpy array or list of complex numbers
            if imag is not None:
                raise Exception('cannot set imag value with complex object input')
            real = np.array(real) # need to convert to to array if taking in a list
            (self.real, self.imag) = (real.real, real.imag) # split real, imag and set separately
        # also will be converted to array
        elif hasattr(real, '__len__'): # some other kind of tuple, list, array, but not iterator..
            (self.real, self.imag) = (np.array(real), np.zeros(len(real)))
        else:                          # single item, iterator, or theano real object
            (self.real, self.imag) = (real, 0.)

    # index into data self[key], propagate key to real and imag separately
    def __getitem__(self, key):
        return Complex(real=self.real[key], imag=self.imag[key])

    # currently returns (real, imag) tuples for each element
    def __iter__(self):
        return ((real, imag) for (real, imag) in zip(self.real, self.imag))

    def __len__(self):
        return len(self.real)

    def conj(a):
        return Complex(real=a.real, imag=-a.imag)

    def norm(a):
        return a.real**2 + a.imag**2

    def abs(a):
        return np.sqrt(a.norm())

    # complex to tuple
    # expand_dims will create extra (..,2) dimension instead of flat array
    def c2t(self, z):
        return np.expand_dims(z, -1).view(dtype=np.float)

    # return complex numpy array
    @property
    def z(self):
        return self.real + 1j*self.imag

    # don't rely on numpy wrapping because might use non-complex type
    def __add__(self, b):
        other = Complex(b)
        return Complex(real=self.real + other.real, imag=self.imag + other.imag)

    # don't rely on numpy wrapping because might use non-complex type
    def __radd__(self, b):
        other = Complex(b)
        return Complex(real=other.real + self.real, imag=other.imag + self.imag)

    def __sub__(self, b):
        other = Complex(b)
        return Complex(real=self.real - other.real, imag=other.imag - self.imag)

    def __rsub__(self, b):
        other = Complex(b)
        return Complex(real=other.real - self.real, imag=other.imag - self.imag)

    # complex multiply (a[...,0] + 1j*a[...,0]) * (b[...,0] + 1j*b[...,1])
    def multiply(a, b):
        return Complex(real=a.real*b.real - a.imag*b.imag, imag=a.real*b.imag + a.imag*b.real)

    def __mul__(self, b):
        print 'mul'
        return self.multiply(Complex(b))

    def __rmul__(self, b):
        print 'rmul'
        return self.multiply(Complex(b))

    def __truediv__(self, b):
        other = Complex(b)
        numer = self * other.conj()
        denom = other.norm()
        return Complex(real=numer.real / denom, imag=numer.imag / denom)

    def __rtruediv__(self, b):
        other = Complex(b)
        numer = other * self.conj()
        denom = self.norm()
        return Complex(real=numer.real / denom, imag=numer.imag / denom)

    def __div__(self, b):
        return self.__truediv__(b)

    def __rdiv__(self, b):
        return self.__rtruediv__(b)

    __array_priority__ = 10000 # use __rmul__ instead of numpy __mul__ (default is 0, np.array is 10)

