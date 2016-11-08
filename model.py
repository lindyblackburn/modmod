# modmod: modular modeling for interferometry

# outstanding questions:
#  - use theano or manual recursion for visib evaluation?
#    - manual may be compatible with basic variables (delayed evaluation) and theano (after evaluation)
#  - use amplitude or total flux?
#  - compatible with pymc3 distributions?
#  - how to initialize?
#  - how many subclasses and for what? (mostly bookkeeping)
#  - support for image model?

# uv model of sky
class uvmodel():

    # model amplitude factor
    amplitude = 1.


    # operators on model, will return a new transformed model

    # shift model by dx and dy in image domain [radians on sky]
    def shift(self, dx, dy):
        return None

    # rotate model by theta [radians]
    def rotate(self, theta):
        return None

    # stretch model by hx and hy in image domain, maintain total flux
    def scale(self, hx, hy):
        return None

    # scale model total flux by factor
    def multiply(self, factor):
        return None

    # add an additional model to model
    def add(self, model):
        return None

    # convolve model with additional model
    def convolve(self, model):
        return None

    # overloaded binary operators for some transforms


    # visibility variable
    v = None

    # total flux variable
    total_flux = None

    # visibility function ? (manual recursion)
    def veval(self, u, v):
        return None


    # hierarchy of transforms ? (record keeping)


