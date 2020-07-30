def nan_to_zero_hook(grad):
    grad[grad != grad] = 0 # NaN != NaN
    return grad
