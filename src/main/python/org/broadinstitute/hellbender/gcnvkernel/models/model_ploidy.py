import argparse
import inspect
import logging

import numpy as np
import pymc3 as pm
import theano as th
import theano.tensor as tt
import pymc3.distributions.dist_math as pm_dist_math
from pymc3 import Normal, Deterministic, DensityDist, Dirichlet, Bound, Uniform, NegativeBinomial, Poisson, Gamma, Exponential
from typing import List, Dict, Set, Tuple
import matplotlib.pyplot as plt

from . import commons
from .fancy_model import GeneralizedContinuousModel
from .. import config, types
from ..structs.interval import Interval
from ..structs.metadata import IntervalListMetadata, SampleMetadataCollection
from ..tasks.inference_task_base import HybridInferenceParameters

_logger = logging.getLogger(__name__)
np.set_printoptions(threshold=np.inf)


class PloidyModelConfig:
    """Germline ploidy-model hyperparameters."""
    def __init__(self,
                 ploidy_state_priors_map: Dict[List[str], Dict[List[int], float]] = None,
                 ploidy_concentration_scale: float = 0.1,
                 depth_upper_bound: float = 1000.0,
                 error_rate_upper_bound: float = 0.1,
                 contig_bias_lower_bound: float = 0.1,
                 contig_bias_upper_bound: float = 2.0,
                 contig_bias_scale: float = 10.0,
                 mosaicism_bias_lower_bound: float = -0.5,
                 mosaicism_bias_upper_bound: float = 0.5,
                 mosaicism_bias_scale: float = 0.001,
                 psi_scale: float = 0.001):
        """Initializer.

        Args:
            ploidy_state_priors_map: Map of the ploidy-state priors. This is a defaultdict(OrderedDict).  The keys
                                     of the defaultdict are the contig tuples.  The keys of the OrderedDict
                                     are the ploidy states, and the values of the OrderedDict are the normalized
                                     prior probabilities.
            ploidy_concentration_scale: Scale factor for the concentration parameters of the per-contig-set
                                        Dirichlet prior on ploidy states
            depth_upper_bound: Upper bound of the uniform prior on the per-sample depth
            error_rate_upper_bound: Upper bound of the uniform prior on the error rate
            contig_bias_lower_bound: Lower bound of the Gamma prior on the per-contig bias
            contig_bias_upper_bound: Upper bound of the Gamma prior on the per-contig bias
            contig_bias_scale: Scale factor for the Gamma prior on the per-contig bias
            mosaicism_bias_lower_bound: Lower bound of the Gaussian prior on the per-sample-and-contig mosaicism bias
            mosaicism_bias_upper_bound: Upper bound of the Gaussian prior on the per-sample-and-contig mosaicism bias
            mosaicism_bias_scale: Standard deviation of the Gaussian prior on the per-sample-and-contig "
                                  mosaicism bias
            psi_scale: Inverse mean of the exponential prior on the per-sample unexplained variance
        """
        assert ploidy_state_priors_map is not None
        self.ploidy_state_priors_map = ploidy_state_priors_map
        self.ploidy_concentration_scale = ploidy_concentration_scale
        self.depth_upper_bound = depth_upper_bound
        self.error_rate_upper_bound = error_rate_upper_bound
        self.contig_bias_lower_bound = contig_bias_lower_bound
        self.contig_bias_upper_bound = contig_bias_upper_bound
        self.contig_bias_scale = contig_bias_scale
        self.mosaicism_bias_lower_bound = mosaicism_bias_lower_bound
        self.mosaicism_bias_upper_bound = mosaicism_bias_upper_bound
        self.mosaicism_bias_scale = mosaicism_bias_scale
        self.psi_scale = psi_scale

    @staticmethod
    def expose_args(args: argparse.ArgumentParser, hide: Set[str] = None):
        """Exposes arguments of `__init__` to a given instance of `ArgumentParser`.

        Args:
            args: an instance of `ArgumentParser`
            hide: a set of arguments not to expose

        Returns:
            None
        """
        group = args.add_argument_group(title="Ploidy-model parameters")
        if hide is None:
            hide = set()

        initializer_params = inspect.signature(PloidyModelConfig.__init__).parameters
        valid_args = {"--" + arg for arg in initializer_params.keys()}
        for hidden_arg in hide:
            assert hidden_arg in valid_args, \
                "Initializer argument to be hidden {0} is not a valid initializer arguments; possible " \
                "choices are: {1}".format(hidden_arg, valid_args)

        def process_and_maybe_add(arg, **kwargs):
            full_arg = "--" + arg
            if full_arg in hide:
                return
            kwargs['default'] = initializer_params[arg].default
            group.add_argument(full_arg, **kwargs)

        process_and_maybe_add("ploidy_concentration_scale",
                              type=float,
                              help="Scale factor for the concentration parameters of the per-contig-set "
                                   "Dirichlet prior on ploidy states",
                              default=initializer_params['ploidy_concentration_scale'].default)

        process_and_maybe_add("depth_upper_bound",
                              type=float,
                              help="Upper bound of the uniform prior on the per-sample depth",
                              default=initializer_params['depth_upper_bound'].default)

        process_and_maybe_add("error_rate_upper_bound",
                              type=float,
                              help="Upper bound of the uniform prior on the error rate",
                              default=initializer_params['error_rate_upper_bound'].default)

        process_and_maybe_add("contig_bias_lower_bound",
                              type=float,
                              help="Lower bound of the Gamma prior on the per-contig bias",
                              default=initializer_params['contig_bias_lower_bound'].default)

        process_and_maybe_add("contig_bias_upper_bound",
                              type=float,
                              help="Upper bound of the Gamma prior on the per-contig bias",
                              default=initializer_params['contig_bias_upper_bound'].default)

        process_and_maybe_add("contig_bias_scale",
                              type=float,
                              help="Scale factor for the Gamma prior on the per-contig bias",
                              default=initializer_params['contig_bias_scale'].default)

        process_and_maybe_add("mosaicism_bias_lower_bound",
                              type=float,
                              help="Lower bound of the Gaussian prior on the per-sample-and-contig mosaicism bias",
                              default=initializer_params['mosaicism_bias_lower_bound'].default)

        process_and_maybe_add("mosaicism_bias_upper_bound",
                              type=float,
                              help="Upper bound of the Gaussian prior on the per-sample-and-contig mosaicism bias",
                              default=initializer_params['mosaicism_bias_upper_bound'].default)

        process_and_maybe_add("mosaicism_bias_scale",
                              type=float,
                              help="Standard deviation of the Gaussian prior on the per-sample-and-contig "
                                   "mosaicism bias",
                              default=initializer_params['mosaicism_bias_scale'].default)

        process_and_maybe_add("psi_scale",
                              type=float,
                              help="Inverse mean of the exponential prior on the per-sample unexplained variance",
                              default=initializer_params['psi_scale'].default)

    @staticmethod
    def from_args_dict(args_dict: Dict):
        """Initialize an instance of `PloidyModelConfig` from a dictionary of arguments.

        Args:
            args_dict: a dictionary of arguments; the keys must match argument names in
                `PloidyModelConfig.__init__`

        Returns:
            an instance of `PloidyModelConfig`
        """
        relevant_keys = set(inspect.getfullargspec(PloidyModelConfig.__init__).args)
        relevant_kwargs = {k: v for k, v in args_dict.items() if k in relevant_keys}
        return PloidyModelConfig(**relevant_kwargs)


class PloidyWorkspace:
    epsilon: float = 1e-10

    """Workspace for storing data structures that are shared between continuous and discrete sectors
    of the germline contig ploidy model."""
    def __init__(self,
                 ploidy_config: PloidyModelConfig,
                 interval_list_metadata: IntervalListMetadata,
                 sample_names: List[str],
                 sample_metadata_collection: SampleMetadataCollection):
        self.ploidy_config = ploidy_config
        self.interval_list_metadata = interval_list_metadata
        self.sample_names = sample_names
        self.sample_metadata_collection = sample_metadata_collection

        assert sample_metadata_collection.all_samples_have_coverage_metadata(sample_names), \
            "Some samples do not have coverage metadata"

        # define useful quantities and shared tensors
        self.eps = self.epsilon

        self.num_samples: int = len(sample_names)
        self.num_contigs = interval_list_metadata.num_contigs
        self.num_counts = sample_metadata_collection.get_sample_coverage_metadata(sample_names[0]).max_count + 1

        # in the below, s = sample index, i = contig-tuple index, j = contig index,
        # k = ploidy-state index, l = ploidy index (equal to ploidy), m = count index

        # process the ploidy-state priors map
        self.contig_tuples: List[Tuple[str]] = list(self.ploidy_config.ploidy_state_priors_map.keys())
        self.num_contig_tuples = len(self.contig_tuples)
        self.num_contigs_i = [len(contig_tuple) for contig_tuple in self.contig_tuples]
        self.ploidy_states_i_k: List[List[Tuple[int]]] = \
            [list(self.ploidy_config.ploidy_state_priors_map[contig_tuple].keys())
             for contig_tuple in self.contig_tuples]
        self.ploidy_state_priors_i_k: List[np.ndarray] = \
            [np.array(list(self.ploidy_config.ploidy_state_priors_map[contig_tuple].values()))
             for contig_tuple in self.contig_tuples]
        self.log_ploidy_state_priors_i_k: List[np.ndarray] = \
            [np.log(ploidy_state_priors_k + self.eps) for ploidy_state_priors_k in self.ploidy_state_priors_i_k]
        self.num_ploidy_states_i = np.array([len(ploidy_states_k) for ploidy_states_k in self.ploidy_states_i_k])
        self.num_ploidies = np.max([np.max(ploidy_state_k)
                                    for ploidy_states_k in self.ploidy_states_i_k
                                    for ploidy_state_k in ploidy_states_k ]) + 1
        self.is_ploidy_in_ploidy_state_i_jkl = [np.zeros((self.num_contigs_i[i], self.num_ploidy_states_i[i], self.num_ploidies))
                                                for i in range(self.num_contig_tuples)]

        self.contigs: List[str] = []
        self.contig_to_ij_map = {}
        self.ploidy_i_jk: List[np.ndarray] = []
        for i, contig_tuple in enumerate(self.contig_tuples):
            self.ploidy_i_jk.append(np.transpose([ploidy_state for ploidy_state in self.ploidy_states_i_k[i]]))
            for j, contig in enumerate(contig_tuple):
                self.contigs.append(contig)
                self.contig_to_ij_map[contig] = (i, j)
                for k in range(self.num_ploidy_states_i[i]):
                    ploidy = self.ploidy_i_jk[i][j, k]
                    self.is_ploidy_in_ploidy_state_i_jkl[i][j, k, ploidy] = 1

        assert set(self.contigs) == interval_list_metadata.contig_set, \
            "The set of contigs present in the coverage files must match exactly " \
            "the set of contigs present in the ploidy-state-priors file."

        # count-distribution data
        self.hist_i_sjm_full = [np.zeros((self.num_samples, self.num_contigs_i[i], self.num_counts), dtype=types.med_uint)
                                for i in range(self.num_contig_tuples)]
        for s, sample_name in enumerate(self.sample_names):
            sample_metadata = sample_metadata_collection.get_sample_coverage_metadata(sample_name)
            for contig, ij in self.contig_to_ij_map.items():
                i, j = ij
                self.hist_i_sjm_full[i][s, j] = sample_metadata.contig_hist_m[contig]
        self.counts_m = np.arange(self.num_counts, dtype=types.med_uint)

        # mask for count bins
        # mask_percentile = 40
        # hist_sjm_masked = np.ma.masked_where(self.hist_sjm_full == 0, self.hist_sjm_full)
        # hist_sjm_masked = np.ma.filled(hist_sjm_masked.astype(float), np.nan)
        # hist_cutoff_sj = np.nanpercentile(hist_sjm_masked, mask_percentile, axis=-1)
        # self.hist_sjm_mask = self.hist_sjm_full > hist_cutoff_sj[:, :, np.newaxis]
        self.hist_i_sjm_mask = self._construct_mask(self.hist_i_sjm_full)

        for s in range(self.num_samples):
            fig, ax = plt.subplots()
            t_j = np.zeros(self.num_contigs)
            for i, contig_tuple in enumerate(self.contig_tuples):
                for j, contig in enumerate(contig_tuple):
                    counts_m_masked = self.counts_m[self.hist_i_sjm_mask[i][s, j]]
                    t_j[j] = np.mean(np.repeat(counts_m_masked[1:], self.hist_i_sjm_full[i][s, j, counts_m_masked[1:]]))
                    plt.semilogy(self.hist_i_sjm_full[i][s, j] / np.sum(self.hist_i_sjm_full[i][s, j]), color='k', lw=0.5, alpha=0.1)
                    plt.semilogy(counts_m_masked, self.hist_i_sjm_full[i][s, j, counts_m_masked] / np.sum(self.hist_i_sjm_full[i][s, j]),
                                 c='b' if i < self.num_contig_tuples - 1 else 'r', lw=1, alpha=0.25)
                    # plt.semilogy(self.hist_sjm_full[s, j], color='k', lw=0.5, alpha=0.1)
                    # plt.semilogy(counts_m_masked, self.hist_sjm_full[s, j, counts_m_masked],
                    #              c='b' if j < self.num_contigs - 2 else 'r', lw=1, alpha=0.25)
                    ax.set_xlim([0, self.num_counts])
                    ax.set_ylim([1E-5, 1E-1])
                    # ax.set_ylim([1, 2 * np.max(self.hist_sjm_full)])

            print(s, t_j / np.mean(t_j))
            ax.set_xlabel('count', size=14)
            # ax.set_ylabel('number of intervals', size=14)
            fig.tight_layout(pad=0.1)
            fig.savefig('/home/slee/working/gatk/test_files/plots/sample_{0}.png'.format(s))

        average_ploidy = 2. # TODO
        self.d_s_testval = 100 #np.median(np.sum(self.hist_sjm_full * np.arange(self.num_counts), axis=-1) / np.sum(self.hist_sjm_full, axis=-1), axis=-1) / average_ploidy

        self.hist_i_sjm : List[types.TensorSharedVariable] = \
            [th.shared(hist_sjm_full, name='hist_%d_sjm' % i, borrow=config.borrow_numpy)
             for i, hist_sjm_full in enumerate(self.hist_i_sjm_full)]

        # ploidy log posteriors (initial value is immaterial)
        self.log_q_ploidy_i_sjl: List[types.TensorSharedVariable] = \
            [th.shared(np.zeros((self.num_samples, self.num_contigs_i[i], self.num_ploidies)),
                       name='log_q_ploidy_%d_sjl' % i, borrow=config.borrow_numpy)
             for i in range(self.num_contig_tuples)]

        # ploidy log emission (initial value is immaterial)
        self.log_ploidy_emission_i_sjl: List[types.TensorSharedVariable] = \
            [th.shared(np.zeros((self.num_samples, self.num_contigs_i[i], self.num_ploidies)),
                       name='log_ploidy_emission_%d_sjl' % i, borrow=config.borrow_numpy)
             for i in range(self.num_contig_tuples)]

    @staticmethod
    def _get_contig_set_from_interval_list(interval_list: List[Interval]) -> Set[str]:
        return {interval.contig for interval in interval_list}

    @staticmethod
    def _construct_mask(hist_i_sjm):
        # count_states = np.arange(0, hist_sjm.shape[2])
        # mode_sj = np.argmax(hist_sjm * (count_states >= 5), axis=2)
        # mask_sjm = np.full(np.shape(hist_sjm), False)
        # for s in range(np.shape(hist_sjm)[0]):
        #     for j in range(np.shape(hist_sjm)[1]):
        #         min_sj = np.argmin(hist_sjm[s, j, :mode_sj[s, j] + 1])
        #         if mode_sj[s, j] <= 10:
        #             mode_sj[s, j] = 0
        #             cutoff = 0.
        #         else:
        #             cutoff = 0.05
        #         for m in range(mode_sj[s, j], np.shape(hist_sjm)[2]):
        #             if hist_sjm[s, j, m] >= cutoff * hist_sjm[s, j, mode_sj[s, j]]:
        #                 if hist_sjm[s, j, m] > 0:
        #                     mask_sjm[s, j, m] = True
        #             else:
        #                 break
        #         for m in range(mode_sj[s, j], min_sj, -1):
        #             if hist_sjm[s, j, m] >= cutoff * hist_sjm[s, j, mode_sj[s, j]]:
        #                 if hist_sjm[s, j, m] > 0:
        #                     mask_sjm[s, j, m] = True
        #             else:
        #                 break
        mask_i_sjm = []
        for hist_sjm in hist_i_sjm:
            mask_sjm = np.full(np.shape(hist_sjm), True)
            mask_sjm[:, :, 0] = False
            mask_i_sjm.append(mask_sjm)

        return mask_i_sjm


class PloidyModel(GeneralizedContinuousModel):
    """Declaration of the germline contig ploidy model (continuous variables only; posterior of discrete
    variables are assumed to be known)."""

    def __init__(self,
                 ploidy_config: PloidyModelConfig,
                 ploidy_workspace: PloidyWorkspace):
        super().__init__()
        self.ploidy_config = ploidy_config
        self.ploidy_workspace = ploidy_workspace

        # shorthands
        ploidy_concentration_scale = ploidy_config.ploidy_concentration_scale
        depth_upper_bound = ploidy_config.depth_upper_bound
        error_rate_upper_bound = ploidy_config.error_rate_upper_bound
        contig_bias_lower_bound = ploidy_config.contig_bias_lower_bound
        contig_bias_upper_bound = ploidy_config.contig_bias_upper_bound
        contig_bias_scale = ploidy_config.contig_bias_scale
        mosaicism_bias_lower_bound = ploidy_config.mosaicism_bias_lower_bound
        mosaicism_bias_upper_bound = ploidy_config.mosaicism_bias_upper_bound
        mosaicism_bias_scale = ploidy_config.mosaicism_bias_scale
        psi_scale = ploidy_config.psi_scale
        num_contig_tuples = ploidy_workspace.num_contig_tuples
        num_samples = ploidy_workspace.num_samples
        num_contigs_i = ploidy_workspace.num_contigs_i
        counts_m = ploidy_workspace.counts_m
        hist_i_sjm = ploidy_workspace.hist_i_sjm
        hist_i_sjm_mask = ploidy_workspace.hist_i_sjm_mask
        ploidy_state_priors_i_k = ploidy_workspace.ploidy_state_priors_i_k
        log_ploidy_state_priors_i_k = ploidy_workspace.log_ploidy_state_priors_i_k
        ploidy_i_jk = ploidy_workspace.ploidy_i_jk
        is_ploidy_in_ploidy_state_i_jkl = ploidy_workspace.is_ploidy_in_ploidy_state_i_jkl
        d_s_testval = ploidy_workspace.d_s_testval
        eps = ploidy_workspace.eps

        register_as_global = self.register_as_global
        register_as_sample_specific = self.register_as_sample_specific

        d_s = Uniform('d_s',
                      upper=depth_upper_bound,
                      shape=num_samples,
                      testval=d_s_testval)
        register_as_sample_specific(d_s, sample_axis=0)

        def bound(logp, *conditions, **kwargs):
            broadcast_conditions = kwargs.get('broadcast_conditions', True)
            if broadcast_conditions:
                alltrue = pm_dist_math.alltrue_elemwise
            else:
                alltrue = pm_dist_math.alltrue_scalar
            return tt.switch(alltrue(conditions), logp, 0)

        def negative_binomial_logp(mu, alpha, value, mask=True):
            return bound(pm_dist_math.factln(value + alpha - 1) - pm_dist_math.factln(alpha - 1)
                         + pm_dist_math.logpow(mu / (mu + alpha), value)
                         + pm_dist_math.logpow(alpha / (mu + alpha), alpha),
                         mu > 0, value > 0, alpha > 0, mask)   # mask out value = 0

        for i in range(num_contig_tuples):
            b_j = Bound(Gamma,
                        lower=contig_bias_lower_bound,
                        upper=contig_bias_upper_bound)('b_%d_j' % i,
                                                       alpha=contig_bias_scale,
                                                       beta=contig_bias_scale,
                                                       shape=num_contigs_i[i])
            register_as_global(b_j)
            b_j_norm = Deterministic('b_%d_j_norm' % i, var=b_j / tt.mean(b_j))

            f_sj = Bound(Normal,
                         lower=mosaicism_bias_lower_bound,
                         upper=mosaicism_bias_upper_bound)('f_%d_js' % i,
                                                           sd=mosaicism_bias_scale,
                                                           shape=(num_samples, num_contigs_i[i]))
            register_as_sample_specific(f_sj, sample_axis=0)

            if len(ploidy_state_priors_i_k[i]) > 1:
                pi_sk = Dirichlet('pi_%d_sk' % i,
                                  a=ploidy_concentration_scale * ploidy_state_priors_i_k[i],
                                  shape=(num_samples, len(ploidy_state_priors_i_k[i])),
                                  transform=pm.distributions.transforms.t_stick_breaking(eps),
                                  testval=ploidy_state_priors_i_k[i][np.newaxis, :])
                register_as_sample_specific(pi_sk, sample_axis=0)
            else:
                pi_sk = Deterministic('pi_%d_sk' % i, var=tt.ones((num_samples, 1)))

            error_rate_sj = Uniform('error_rate_%d_sj' % i,
                                    lower=0.,
                                    upper=error_rate_upper_bound,
                                    shape=(num_samples, num_contigs_i[i]))
            register_as_sample_specific(error_rate_sj, sample_axis=0)

            mu_sjk = pm.Deterministic('mu_%d_sjk' % i,
                                      var=d_s.dimshuffle(0, 'x', 'x') * b_j_norm.dimshuffle('x', 0, 'x') * \
                                          (tt.maximum(ploidy_i_jk[i][np.newaxis, :, :] + f_sj.dimshuffle(0, 1, 'x') * (ploidy_i_jk[i][np.newaxis, :, :] > 0),
                                                      error_rate_sj[:, :, np.newaxis])))

            psi_sj = Exponential(name='psi_%d_sj' % i,
                                 lam=1.0 / psi_scale,
                                 shape=(num_samples, num_contigs_i[i]))
            register_as_sample_specific(psi_sj, sample_axis=0)
            alpha_sj = pm.Deterministic('alpha_%d_sj' % i, tt.inv((tt.exp(psi_sj) - 1.0 + eps)))

            logp_sjkm = negative_binomial_logp(mu=mu_sjk.dimshuffle(0, 1, 2, 'x') + eps,
                                               alpha=alpha_sj.dimshuffle(0, 1, 'x', 'x'),
                                               value=counts_m[np.newaxis, np.newaxis, np.newaxis, :],
                                               mask=hist_i_sjm_mask[i][:, :, np.newaxis, :])

            # def negative_binomial_logp(mu, alpha, value, mask=True):
            #     return bound(pm_dist_math.factln(value + alpha - 1) - pm_dist_math.factln(alpha - 1) - pm_dist_math.factln(value)
            #                               + pm_dist_math.logpow(mu / (mu + alpha), value)
            #                               + pm_dist_math.logpow(alpha / (mu + alpha), alpha),
            #                               mu > 0, value >= 0, alpha > 0, mask)
            #
            # def poisson_logp(mu, value, mask=True):
            #     log_prob = bound(pm_dist_math.logpow(mu, value) - mu, mu >= 0, value >= 0, mask)
            #     # Return zero when mu and value are both zero
            #     return tt.switch(tt.eq(mu, 0) * tt.eq(value, 0), 0, log_prob)
            #
            # num_occurrences_sj = np.sum(self.ploidy_workspace.hist_sjm_full * hist_sjm_mask, axis=-1)
            # p_j_skm = [tt.exp(negative_binomial_logp(mu=mu_j_sk[j].dimshuffle(0, 1, 'x') + eps,
            #                                          alpha=alpha_js[j].dimshuffle(0, 'x', 'x'),
            #                                          value=counts_m[np.newaxis, np.newaxis, :],
            #                                          mask=hist_sjm_mask[:, j, np.newaxis, :]))
            #            for j in range(num_contigs)]
            # [pm.Deterministic('hist_%d_skm' % j,
            #                   var=num_occurrences_sj[:, j, np.newaxis, np.newaxis] * p_j_skm[j] + eps)
            #  for j in range(num_contigs)]

            def _logp_hist(_hist_sjm):
                # logp_hist_j_skm = [poisson_logp(mu=num_occurrences_sj[:, j, np.newaxis, np.newaxis] * p_j_skm[j] + eps,
                #                                 value=_hist_sjm[:, j, :].dimshuffle(0, 'x', 1),
                #                                 mask=hist_sjm_mask[:, j, np.newaxis, :])
                #                    for j in range(num_contigs)]
                # return tt.stack([tt.sum(log_ploidy_state_priors_i_k[i][np.newaxis, :, np.newaxis] + \
                #                         tt.sum(hist_sjm_mask[:, contig_to_ij_map[contig], np.newaxis, :] * \
                #                                pm.logsumexp(tt.log(pi_i_sk[i][:, :, np.newaxis] + eps) + logp_hist_j_skm[contig_to_ij_map[contig]], axis=1)))
                #         for i, contig_tuple in enumerate(contig_tuples) for contig in contig_tuple])
                return tt.sum(log_ploidy_state_priors_i_k[i][np.newaxis, :, np.newaxis] + \
                              tt.sum(_hist_sjm[:, :, np.newaxis, :] * \
                                     pm.logsumexp(tt.log(pi_sk[:, np.newaxis, :, np.newaxis] + eps) + logp_sjkm, axis=1)))

            DensityDist(name='hist_%d_sjm' % i, logp=_logp_hist, observed=hist_i_sjm[i])

            pm.Deterministic(name='log_ploidy_emission_%d_sjl' % i,
                             var=tt.log(tt.dot(pi_sk, is_ploidy_in_ploidy_state_i_jkl[i]) + eps))


class PloidyEmissionBasicSampler:
    """Draws posterior samples from the ploidy log emission probability for a given variational
    approximation to the ploidy model posterior."""
    def __init__(self, ploidy_model: PloidyModel, samples_per_round: int):
        self.ploidy_model = ploidy_model
        self.ploidy_workspace = ploidy_model.ploidy_workspace
        self.samples_per_round = samples_per_round
        self._simultaneous_log_ploidy_emission_sampler = None

    def update_approximation(self, approx: pm.approximations.MeanField):
        """Generates a new compiled sampler based on a given approximation.
        Args:
            approx: an instance of PyMC3 mean-field approximation

        Returns:
            None
        """
        self._simultaneous_log_ploidy_emission_sampler = \
            self._get_compiled_simultaneous_log_ploidy_emission_sampler(approx)

    def is_sampler_initialized(self):
        return self._simultaneous_log_ploidy_emission_sampler is not None

    def draw(self) -> np.ndarray:
        return self._simultaneous_log_ploidy_emission_sampler()

    @th.configparser.change_flags(compute_test_value="off")
    def _get_compiled_simultaneous_log_ploidy_emission_sampler(self, approx: pm.approximations.MeanField):
        """For a given variational approximation, returns a compiled theano function that draws posterior samples
        from the log ploidy emission."""
        log_ploidy_emission_i_sjl = [commons.stochastic_node_mean_symbolic(
            approx, self.ploidy_model['log_ploidy_emission_%d_sjl' % i], size=self.samples_per_round)
            for i in range(self.ploidy_workspace.num_contig_tuples)]
        return th.function(inputs=[], outputs=log_ploidy_emission_i_sjl)


class PloidyBasicCaller:
    """Bayesian update of germline contig ploidy log posteriors."""
    def __init__(self,
                 inference_params: HybridInferenceParameters,
                 ploidy_workspace: PloidyWorkspace):
        self.ploidy_workspace = ploidy_workspace
        self.inference_params = inference_params
        self._update_log_q_ploidy_sjl_theano_func = self._get_update_log_q_ploidy_sjl_theano_func()

    @th.configparser.change_flags(compute_test_value="off")
    def _get_update_log_q_ploidy_sjl_theano_func(self) -> th.compile.function_module.Function:
        # new_log_q_ploidy_sjl = self.ploidy_workspace.log_ploidy_emission_i_sjl - pm.logsumexp(self.ploidy_workspace.log_ploidy_emission_i_sjl, axis=2)
        # old_log_q_ploidy_sjl = self.ploidy_workspace.log_q_ploidy_i_sjl
        # admixed_new_log_q_ploidy_sjl = commons.safe_logaddexp(
        #     new_log_q_ploidy_sjl + np.log(self.inference_params.caller_external_admixing_rate),
        #     old_log_q_ploidy_sjl + np.log(1.0 - self.inference_params.caller_external_admixing_rate))
        # update_norm_sj = commons.get_hellinger_distance(admixed_new_log_q_ploidy_sjl, old_log_q_ploidy_sjl)
        # return th.function(inputs=[],
        #                    outputs=[update_norm_sj],
        #                    updates=[(self.ploidy_workspace.log_q_ploidy_i_sjl, admixed_new_log_q_ploidy_sjl)])
        return th.function(inputs=[],
                           outputs=[],
                           updates=[(log_q_ploidy_sjl, log_q_ploidy_sjl)
                                    for log_q_ploidy_sjl in self.ploidy_workspace.log_q_ploidy_i_sjl])

    def call(self) -> np.ndarray:
        return self._update_log_q_ploidy_sjl_theano_func()
