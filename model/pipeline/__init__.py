import inspect
from util.package import find_class_using_name
from .base_pipeline import BasePipeline

def find_pipeline_using_name(pipeline_name):
    if not pipeline_name:
        raise Exception("'pipeline_name' is empty")
    pipeline_cls = find_class_using_name('model.pipeline', pipeline_name, 'pipeline')
    if inspect.isclass(pipeline_cls) and issubclass(pipeline_cls, BasePipeline):
        return pipeline_cls

    raise Exception("{} is not correctely implemented as PipelineModel class".format(pipeline_name))
