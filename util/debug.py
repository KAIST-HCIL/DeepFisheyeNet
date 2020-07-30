def viz_grad_sum(name, abs = True):
    def hook(grad):
        
        if abs:
            print(name, "(abs sum):", grad.abs().sum())
        else:
            print(name, "(sum):", grad.sum())

    return hook

def viz_grad_mean(name, abs = True):
    def hook(grad):
        if abs:
            print(name, "(abs mean):", grad.abs().mean(), grad.shape)
        else:
            print(name, "(mean):", grad.mean())
    return hook

def viz_grad(name):
    def hook(grad):
        print(name, grad)
    return hook
