from functools import wraps


def ensure_monotonicity(decomposer, function, loss, tol=0):
    function = getattr(decomposer, function)
    @wraps(function)
    def new_func(*args, **kwargs):
        old_loss = getattr(decomposer, loss)
        return_val = function(*args, **kwargs)
        assert getattr(decomposer, loss) <= old_loss + tol
        return return_val
    return new_func

