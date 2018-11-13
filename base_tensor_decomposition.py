

class BaseTensorDecomposition:
    def __init__(self):
        pass

    def decompose(self, tensor):
        pass
    
    def compose_from_factors(self, factors):
        pass

    def compute_loss(self, factors):
        pass


class Base_CP(BaseTensorDecomposition):
    def __init__(self, rank, init_scheme):
        pass

    def init_factors(self, tensor, rank, init_scheme):
        pass

    def _random_init(self, tensor, rank):
        pass

    def _svd_init(self, tensor, rank):
        pass

    def compute_loss(self, tensor, rank):
        pass

    def compose_from_factors(self, factors):
        pass

class CP_als(Base_CP):
    def __init__(self, rank, init_scheme, max_it, tol):
        pass

    def als_cycle(self, factors, tensor):
        pass

    def als_it(self, factors, tensor):
        pass
