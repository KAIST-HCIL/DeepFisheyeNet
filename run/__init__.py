import inspect
from util.package import find_class_using_name
from .base.base_run import BaseRun

def find_run_using_name(run_name):
    subpackage_name = find_subpackage(run_name)
    package_name = 'run.{}'.format(subpackage_name)
    run_cls = find_class_using_name(package_name, run_name, 'run')
    if inspect.isclass(run_cls) and issubclass(run_cls, BaseRun):
        return run_cls

    raise Exception("{} is not correctely implemented as BaseRun class".format(run_name))

def find_subpackage(run_name):
    subpackage_name = run_name.split('_')[:-1]
    subpackage_name = '_'.join(subpackage_name)

    return subpackage_name
