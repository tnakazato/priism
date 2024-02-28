import inspect

import casatasks


def run_casa_task(taskname, **kwargs):
    task = getattr(casatasks, taskname)
    task_signature = inspect.signature(task)
    clean_args = dict(
        (k, v) for k, v in kwargs.items() if k in task_signature.parameters
    )
    return task(**clean_args)
