def quadratic_form_trace(A, X):
    """Compute :math:`tr(X^T A X)`
    """
    return sum(X_i.T@A@X_i for X_i in X.T)
