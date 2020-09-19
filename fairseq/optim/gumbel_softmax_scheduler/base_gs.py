from . import  register_gs_scheduler
import math
@register_gs_scheduler('base_gs')
class BaseGsSchedule(object):

    def __init__(self, args,):

        self.tau_max = args.gumbel_softmax_max
        self.tau_r = args.gumbel_softmax_tau_r
        self.tau_min = args.gumbel_softmax_min
        self.update_freq = args.gumbel_softmax_update_freq
        self.step_update(0)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--gumbel-softmax-max', type=float, default=10., )
        parser.add_argument('--gumbel-softmax-min', type=float, default=1, )
        parser.add_argument('--gumbel-softmax-tau-r', type=float, default=1e-4)
        parser.add_argument('--gumbel-softmax-update-freq', type=int, default=5000)


    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        update_cliped = math.floor(num_updates / self.update_freq) * self.update_freq
        self.tau = max(self.tau_max * math.exp(- self.tau_r * update_cliped), self.tau_min)
        return self.tau

    def get_gs_tau(self):
        return self.tau