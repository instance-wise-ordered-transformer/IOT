# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch
from fairseq import utils
from . import FairseqCriterion, register_criterion


@register_criterion('label_smoothed_cross_entropy')
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.kl = args.kl
        self.encoder_max_order = args.encoder_max_order
        self.decoder_max_order = args.decoder_max_order
        self.diversity = args.diversity
        self.decoder_orders = args.decoder_orders
        assert len(self.decoder_orders) == self.decoder_max_order
        print('| decoder orders ', self.decoder_orders)


    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--kl', default=0., type=float)
        parser.add_argument('--diversity', default=0.1, type=float)
        parser.add_argument('--encoder-max-order', default=1, type=int)
        parser.add_argument('--decoder-max-order', default=2, type=int)
        parser.add_argument('--decoder-orders', default=[0, 1], nargs='+', type=int)
        parser.add_argument('--gs-clamp', action='store_true')
        # fmt: on

    def forward(self, model, sample, reduce=True, gs_tau=0.5, gs_hard=False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        encoder_orders = list(range(0, self.encoder_max_order))
        decoder_orders = self.decoder_orders
        l_index = []
        l_net_output = []
        for i in range(len(decoder_orders)):
            e_index = min(i, self.encoder_max_order - 1)
            d_index = min(i, self.decoder_max_order - 1)
            model.set_perm_order(encoder_orders[e_index], decoder_orders[d_index])
            l_index.append(decoder_orders[d_index])
            l_net_output.append(model(**sample['net_input']))

        loss, nll_loss, kl_loss, old_loss, diversity_loss = self.compute_loss(model, l_net_output, sample,
                                                                              reduce=reduce, l_index=l_index,
                                                                              gs_tau=gs_tau, gs_hard=gs_hard)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'kl_loss': utils.item(kl_loss) if reduce else kl_loss,
            'old_loss': utils.item(old_loss) if reduce else old_loss,
            'diversity_loss': utils.item(diversity_loss) if reduce else diversity_loss,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,

        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, l_net_output, sample, reduce=True, l_index=None, gs_tau=0.5, gs_hard=False):
        l_lprobs = []
        l_orders = []
        l_softmaxouts = []
        l_softmaxouts_clamp = []
        for net_output in l_net_output:
            lprobs, orders, softmaxout, softmaxout_clamp = model.get_normalized_probs(net_output, log_probs=True, gs_tau=gs_tau, gs_hard=gs_hard)
            l_lprobs.append(lprobs)
            l_orders.append(orders)
            l_softmaxouts.append(softmaxout)
            l_softmaxouts_clamp.append(softmaxout_clamp)
        ts = lprobs.size(1)
        orders = torch.mean(torch.stack(l_orders, dim=0), dim=0)
        softmaxout = torch.mean(torch.stack(l_softmaxouts, dim=0), dim=0)
        softmaxout_clamp = torch.mean(torch.stack(l_softmaxouts_clamp, dim=0), dim=0)
        l_orders = [orders[:, i].view(-1, 1).expand(-1, ts).contiguous().view(-1, 1) for i in range(len(l_index))]
        kl = torch.log(softmaxout_clamp)
        kl = -torch.mean(kl, dim=-1) - math.log(self.decoder_max_order)
        l_lprobs = [lprobs.view(-1, lprobs.size(-1)) for lprobs in l_lprobs]
        target = model.get_targets(sample, net_output).view(-1, 1)
        non_pad_mask = target.ne(self.padding_idx)
        l_nll_loss_nopad = [-lprobs.gather(dim=-1, index=target) for lprobs in l_lprobs]
        nll_loss = torch.sum(torch.stack(l_orders, dim=0) *
                             torch.stack(l_nll_loss_nopad, dim=0),
                             dim=0)[non_pad_mask]
        smooth_loss = torch.sum(torch.stack([-lprobs.sum(dim=-1, keepdim=True) for lprobs in l_lprobs], dim=0),
                                dim=0)[non_pad_mask]
        diversity_loss = torch.mean(softmaxout, dim = 0)
        diversity_loss = torch.log(diversity_loss)
        diversity_loss = (-torch.mean(diversity_loss, dim=-1) - math.log(self.decoder_max_order))
        kl_loss = - torch.mean(kl)
        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
            kl_loss = kl_loss * sample['ntokens']
            diversity_loss = diversity_loss * sample['ntokens']
        eps_i = self.eps / lprobs.size(-1)
        old_loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss
        loss = old_loss + self.kl * kl_loss + self.diversity * diversity_loss
        return loss, nll_loss, kl_loss, old_loss, diversity_loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'old_loss': sum(log.get('old_loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
            'kl_loss': sum(log.get('kl_loss', 0) for log in logging_outputs) / ntokens ,
            'diversity_loss': sum(log.get('diversity_loss', 0) for log in logging_outputs ) / ntokens,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
