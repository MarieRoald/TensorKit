from functools import wraps


def ensure_monotonicity(decomposer, function, loss, skips=0, rtol=1e-10, atol=1e-10):
    function = getattr(decomposer, function)
    @wraps(function)
    def new_func(*args, **kwargs):
        nonlocal skips
        
        old_loss = getattr(decomposer, loss)
        return_val = function(*args, **kwargs)
        if skips <= 0:
            assert getattr(decomposer, loss) - old_loss <= rtol*abs(old_loss) + atol
        else:
            skips -= 1
        return return_val
    return new_func
