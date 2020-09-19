# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import importlib
import os



GS_SCHEDULER_REGISTRY = {}


def build_gs_scheduler(args):
    return GS_SCHEDULER_REGISTRY[args.gs_scheduler](args)


def register_gs_scheduler(name):
    """Decorator to register a new LR scheduler."""

    def register_gs_scheduler_cls(cls):
        if name in GS_SCHEDULER_REGISTRY:
            raise ValueError('Cannot register duplicate GS scheduler ({})'.format(name))

        GS_SCHEDULER_REGISTRY[name] = cls
        return cls

    return register_gs_scheduler_cls


# automatically import any Python files in the optim/lr_scheduler/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('fairseq.optim.gumbel_softmax_scheduler.' + module)
