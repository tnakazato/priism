from __future__ import absolute_import


def register(func):
    # casa-specific shutdown handler
    import casa_shutdown
    casa_shutdown.add_shutdown_hook(func)
