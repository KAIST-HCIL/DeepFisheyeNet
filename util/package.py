import importlib

def find_class_using_name(module_name, name, postfix):
    """
        Imports class with name.
        The name should have certain pattern with under the module.
    """
    filename = '{}.{}_{}'.format(module_name, name, postfix)
    module_lib = importlib.import_module(filename)
    target_name = (name + postfix).replace('_', '')

    target_cls = None
    for name, cls in module_lib.__dict__.items():
        if name.lower() == target_name.lower():
            target_cls = cls

    if target_cls is None:
        NotImplementedError('Class {} is not implemented in module {}'.format(name, module_name))

    return target_cls
