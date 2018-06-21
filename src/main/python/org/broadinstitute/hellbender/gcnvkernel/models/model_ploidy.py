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
                 contig_bias_lower_bound: float = 0.8,
                 contig_bias_upper_bound: float = 1.2,
                 contig_bias_scale: float = 100.0,
                 mosaicism_bias_lower_bound: float = -0.5,
                 mosaicism_bias_upper_bound: float = 0.5,
                 mosaicism_bias_scale: float = 0.01):
        """Initializer.

        Args:
            ploidy_state_priors_map: Map of the ploidy-state priors. This is a defaultdict(OrderedDict).  The keys
                                     of the defaultdict are the contig tuples.  The keys of the OrderedDict
                                     are the ploidy states, and the values of the OrderedDict are the normalized
                                     prior probabilities.
            ploidy_concentration_scale: Scaling factor for the concentration parameters of the per-contig-set
                                        Dirichlet prior on ploidy states
            depth_upper_bound: Upper bound of the uniform prior on the per-sample depth
            error_rate_upper_bound: Upper bound of the uniform prior on the error rate
            contig_bias_lower_bound: Lower bound of the Gamma prior on the per-contig bias
            contig_bias_upper_bound: Upper bound of the Gamma prior on the per-contig bias
            contig_bias_scale: Scaling factor for the Gamma prior on the per-contig bias
            mosaicism_bias_lower_bound: Lower bound of the Gaussian prior on the per-sample-and-contig mosaicism bias
            mosaicism_bias_upper_bound: Upper bound of the Gaussian prior on the per-sample-and-contig mosaicism bias
            mosaicism_bias_scale: Standard deviation of the Gaussian prior on the per-sample-and-contig "
                                  mosaicism bias
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
                              help="Scaling factor for the concentration parameters of the per-contig-set "
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
                              help="Scaling factor for the Gamma prior on the per-contig bias",
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
        self.ploidy_states_i_k: List[List[Tuple[int]]] = \
            [list(self.ploidy_config.ploidy_state_priors_map[contig_tuple].keys())
             for contig_tuple in self.contig_tuples]
        self.ploidy_state_priors_i_k: List[np.ndarray] = \
            [np.array(list(self.ploidy_config.ploidy_state_priors_map[contig_tuple].values()))
             for contig_tuple in self.contig_tuples]
        self.ploidy_j_k: List[np.ndarray] = []
        self.contigs: List[str] = []
        for i, contig_tuple in enumerate(self.contig_tuples):
            for j, contig in enumerate(contig_tuple):
                self.contigs.append(contig)
                self.ploidy_j_k.append(np.array([ploidy_state[j]
                                                 for ploidy_state in self.ploidy_states_i_k[i]]))

        assert set(self.contigs) == interval_list_metadata.contig_set, \
            "The set of contigs present in the coverage files must match exactly " \
            "the set of contigs present in the ploidy-state-priors file."

        self.contig_to_index_map = {contig: index for index, contig in enumerate(self.contigs)}
        self.num_ploidy_states_j = np.array([len(ploidy_k) for ploidy_k in self.ploidy_j_k])
        self.num_ploidies = np.max([np.max(ploidy_k) for ploidy_k in self.ploidy_j_k]) + 1
        self.is_contig_in_contig_tuple_ij = np.zeros((self.num_contig_tuples, self.num_contigs),
                                                     dtype=types.small_uint)
        self.is_ploidy_in_ploidy_state_j_kl = [np.zeros((self.num_ploidy_states_j[j], self.num_ploidies))
                                               for j in range(self.num_contigs)]
        self.ploidy_priors_jl = 1E-10 * np.ones((self.num_contigs, self.num_ploidies),
                                                dtype=types.floatX)

        for j in range(self.num_contigs):
            for k in range(self.num_ploidy_states_j[j]):
                ploidy = self.ploidy_j_k[j][k]
                self.is_ploidy_in_ploidy_state_j_kl[j][k, ploidy] = 1

        for i, contig_tuple in enumerate(self.contig_tuples):
            for contig in contig_tuple:
                j = self.contig_to_index_map[contig]
                self.is_contig_in_contig_tuple_ij[i, j] = 1
                self.ploidy_priors_jl[j] = np.sum(self.ploidy_state_priors_i_k[i][:, np.newaxis] * self.is_ploidy_in_ploidy_state_j_kl[j], axis=0)

        # count-distribution data
        self.hist_sjm_full = np.zeros((self.num_samples, self.num_contigs, self.num_counts), dtype=types.med_uint)
        for si, sample_name in enumerate(self.sample_names):
            sample_metadata = sample_metadata_collection.get_sample_coverage_metadata(sample_name)
            self.hist_sjm_full[si] = sample_metadata.hist_jm[:, :]


        # # mask for count bins
        # self.mask_sjm, self.counts_m = self._construct_mask(self.hist_sjm_full)

        self.counts_m = np.arange(self.num_counts, dtype=types.med_uint)

        average_ploidy = 2. # TODO
        self.d_s_testval = np.median(np.sum(self.hist_sjm_full * np.arange(self.num_counts), axis=-1) / np.sum(self.hist_sjm_full, axis=-1), axis=-1) / average_ploidy

        # self.hist_sjm : types.TensorSharedVariable = \
        #     th.shared(self.hist_sjm_full[:, :, self.counts_m], name='hist_sjm', borrow=config.borrow_numpy)

        self.hist_sjm : types.TensorSharedVariable = \
            th.shared(self.hist_sjm_full, name='hist_sjm', borrow=config.borrow_numpy)

        # self.num_occurrences_sj = np.sum(self.hist_sjm_full[:, :, self.counts_m], axis=2)

        # ploidy log posteriors (initialize to priors)
        self.log_q_ploidy_sjl: types.TensorSharedVariable = \
            th.shared(np.tile(np.log(self.ploidy_priors_jl), (self.num_samples, 1, 1)),
                      name='log_q_ploidy_sjl', borrow=config.borrow_numpy)

        # ploidy log emission (initial value is immaterial)
        self.log_ploidy_emission_sjl: types.TensorSharedVariable = \
            th.shared(np.zeros((self.num_samples, self.num_contigs, self.num_ploidies), dtype=types.floatX),
                      name='log_ploidy_emission_sjl', borrow=config.borrow_numpy)


    @staticmethod
    def _get_contig_set_from_interval_list(interval_list: List[Interval]) -> Set[str]:
        return {interval.contig for interval in interval_list}

    # @staticmethod
    # def _construct_mask(hist_sjm):
    #     count_states = np.arange(0, hist_sjm.shape[2])
    #     mode_sj = np.argmax(hist_sjm * (count_states >= 5), axis=2)
    #     mask_sjm = np.full(np.shape(hist_sjm), False)
    #     for s in range(np.shape(hist_sjm)[0]):
    #         for j in range(np.shape(hist_sjm)[1]):
    #             min_sj = np.argmin(hist_sjm[s, j, :mode_sj[s, j] + 1])
    #             if mode_sj[s, j] <= 10:
    #                 mode_sj[s, j] = 0
    #                 cutoff = 0.
    #             else:
    #                 cutoff = 0.05
    #             for m in range(mode_sj[s, j], np.shape(hist_sjm)[2]):
    #                 if hist_sjm[s, j, m] >= cutoff * hist_sjm[s, j, mode_sj[s, j]]:
    #                     if hist_sjm[s, j, m] > 0:
    #                         mask_sjm[s, j, m] = True
    #                 else:
    #                     break
    #             for m in range(mode_sj[s, j], min_sj, -1):
    #                 if hist_sjm[s, j, m] >= cutoff * hist_sjm[s, j, mode_sj[s, j]]:
    #                     if hist_sjm[s, j, m] > 0:
    #                         mask_sjm[s, j, m] = True
    #                 else:
    #                     break
    #         mask_sjm[:, :, 0] = False
    #
    #     counts_m = []
    #     for m in range(hist_sjm.shape[2]):
    #         if np.any(mask_sjm[:, :, m]):
    #             counts_m.append(m)
    #
    #     return mask_sjm[:, :, counts_m], np.array(counts_m, dtype=types.small_uint)


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
        # mosaicism_bias_lower_bound = ploidy_config.mosaicism_bias_lower_bound
        # mosaicism_bias_upper_bound = ploidy_config.mosaicism_bias_upper_bound
        # mosaicism_bias_scale = ploidy_config.mosaicism_bias_scale
        contig_tuples = ploidy_workspace.contig_tuples
        num_samples = ploidy_workspace.num_samples
        num_contigs = ploidy_workspace.num_contigs
        counts_m = ploidy_workspace.counts_m
        contig_to_index_map = ploidy_workspace.contig_to_index_map
        # mask_sjm = ploidy_workspace.mask_sjm
        hist_sjm = ploidy_workspace.hist_sjm
        ploidy_state_priors_i_k = ploidy_workspace.ploidy_state_priors_i_k
        ploidy_j_k = ploidy_workspace.ploidy_j_k
        is_ploidy_in_ploidy_state_j_kl = ploidy_workspace.is_ploidy_in_ploidy_state_j_kl
        d_s_testval = ploidy_workspace.d_s_testval
        eps = ploidy_workspace.eps

        register_as_global = self.register_as_global
        register_as_sample_specific = self.register_as_sample_specific

        d_s = Uniform('d_s',
                      upper=depth_upper_bound,
                      shape=num_samples,
                      testval=d_s_testval)
        register_as_sample_specific(d_s, sample_axis=0)

        b_j = Bound(Gamma,
                    lower=contig_bias_lower_bound,
                    upper=contig_bias_upper_bound)('b_j',
                                                   alpha=contig_bias_scale,
                                                   beta=contig_bias_scale,
                                                   shape=num_contigs)
        register_as_global(b_j)
        b_j_norm = Deterministic('b_j_norm', var=b_j / tt.mean(b_j))

        # f_js = Bound(Normal,
        #              lower=mosaicism_bias_lower_bound,
        #              upper=mosaicism_bias_upper_bound)('f_js',
        #                                                sd=mosaicism_bias_scale,
        #                                                shape=(num_contigs, num_samples))
        # register_as_sample_specific(f_js, sample_axis=1)

        pi_i_sk = []
        for i, contig_tuple in enumerate(contig_tuples):
            if len(ploidy_state_priors_i_k[i]) > 1:
                pi_i_sk.append(Dirichlet('pi_%d_sk' % i,
                                         a=ploidy_concentration_scale * ploidy_state_priors_i_k[i],
                                         shape=(num_samples, len(ploidy_state_priors_i_k[i])),
                                         transform=pm.distributions.transforms.t_stick_breaking(eps),
                                         testval=ploidy_state_priors_i_k[i]))
                register_as_sample_specific(pi_i_sk[i], sample_axis=0)
            else:
                pi_i_sk.append(Deterministic('pi_%d_sk' % i, var=tt.ones((num_samples, 1))))

        error_rate_j = Uniform('error_rate_j',
                               lower=0.,
                               upper=error_rate_upper_bound,
                               shape=num_contigs)
        register_as_global(error_rate_j)

        mu_j_sk = [d_s.dimshuffle(0, 'x') * b_j_norm[j] * \
                   # (tt.maximum(ploidy_j_k[j][np.newaxis, :] + f_js[j].dimshuffle(0, 'x') * (ploidy_j_k[j][np.newaxis, :] > 0),
                   #             e_js[j].dimshuffle(0, 'x')))
                   tt.maximum(ploidy_j_k[j][np.newaxis, :], error_rate_j[j])
                   for j in range(num_contigs)]

        psi_js = Exponential(name='psi_js',
                             lam=100.0, #1.0 / ploidy_config.psi_scale,
                             shape=(num_contigs, num_samples))
        register_as_sample_specific(psi_js, sample_axis=1)
        alpha_js = tt.inv((tt.exp(psi_js) - 1.0 + eps))

        def bound(logp, *conditions, **kwargs):
            broadcast_conditions = kwargs.get('broadcast_conditions', True)
            if broadcast_conditions:
                alltrue = pm_dist_math.alltrue_elemwise
            else:
                alltrue = pm_dist_math.alltrue_scalar
            return tt.switch(alltrue(conditions), logp, 0)

        def negative_binomial_logp(mu, alpha, value):
            return bound(pm_dist_math.binomln(value + alpha - 1, value)
                                      + pm_dist_math.logpow(mu / (mu + alpha), value)
                                      + pm_dist_math.logpow(alpha / (mu + alpha), alpha),
                                      mu > 0, value >= 10, alpha > 0)


        logp_j_skm = [negative_binomial_logp(mu=mu_j_sk[j].dimshuffle(0, 1, 'x') + eps,
                                             alpha=alpha_js[j].dimshuffle(0, 'x', 'x'),
                                             value=counts_m[np.newaxis, np.newaxis, :])
                      for j in range(num_contigs)]

        # p_j_skm = [tt.exp(NegativeBinomial.dist(mu=mu_j_sk[j].dimshuffle(0, 1, 'x') + eps,
        #                                         alpha=alpha_js[j].dimshuffle(0, 'x', 'x'))
        #                   .logp(th.shared(np.array(counts_m, dtype=types.small_uint), borrow=config.borrow_numpy).dimshuffle('x', 'x', 0)))
        #            for j in range(num_contigs)]
        # hist_norm_sj = pm.Uniform('hist_norm_sj',
        #                           shape=(num_samples, num_contigs))
        # register_as_sample_specific(hist_norm_sj, sample_axis=0)
        # hist_mu_j_skm = [pm.Deterministic('hist_mu_%d_skm' % j,
        #                                   self.ploidy_workspace.num_occurrences_tot_sj[:, j, np.newaxis, np.newaxis] * hist_norm_sj[:, j, np.newaxis, np.newaxis] * p_j_skm[j] + eps)
        #                  for j in range(num_contigs)
        # hist_mu_j_skm = [pm.Deterministic('hist_mu_%d_skm' % j,
        #                                   self.ploidy_workspace.num_occurrences_sj[:, j, np.newaxis, np.newaxis] * p_j_skm[j] + eps)
        #                  for j in range(num_contigs)]

        def _logp_hist(_hist_sjm):
            # logp_hist_j_skm = [pm.Poisson.dist(mu=hist_mu_j_skm[j]).logp(_hist_sjm[:, j, :].dimshuffle(0, 'x', 1))
            #                    for j in range(num_contigs)]
            # return [tt.sum(tt.log(ploidy_state_priors_i_k[i][np.newaxis, :, np.newaxis] + eps) + \
            #                tt.sum(mask_sjm[:, contig_to_index_map[contig], np.newaxis, :] * \
            #                       pm.logsumexp(tt.log(pi_i_sk[i][:, :, np.newaxis] + eps) + logp_hist_j_skm[contig_to_index_map[contig]], axis=1)))
            #         for i, contig_tuple in enumerate(contig_tuples) for contig in contig_tuple]
            return tt.stack([tt.sum(tt.log(ploidy_state_priors_i_k[i][np.newaxis, :, np.newaxis] + eps) + \
                                    # tt.sum(mask_sjm[:, contig_to_index_map[contig], np.newaxis, :] * _hist_sjm[:, contig_to_index_map[contig], np.newaxis, :] * \
                                    tt.sum(_hist_sjm[:, contig_to_index_map[contig], np.newaxis, :] * \
                                           pm.logsumexp(tt.log(pi_i_sk[i][:, :, np.newaxis] + eps) + logp_j_skm[contig_to_index_map[contig]], axis=1)))
                             for i, contig_tuple in enumerate(contig_tuples) for contig in contig_tuple])

        DensityDist(name='hist_sjm', logp=_logp_hist, observed=hist_sjm)

        # logp_j_sk = [tt.log(ploidy_state_priors_i_k[i][np.newaxis, :] + eps) + \
        #              tt.sum(mask_sjm[:, contig_to_index_map[contig], np.newaxis, :] * hist_sjm[:, contig_to_index_map[contig], np.newaxis, :] * \
        #                     (tt.log(pi_i_sk[i][:, :, np.newaxis] + eps) + logp_j_skm[contig_to_index_map[contig]]), axis=-1)
        #              for i, contig_tuple in enumerate(contig_tuples) for contig in contig_tuple]
        #
        # pm.Deterministic(name='log_ploidy_emission_sjl',
        #                  var=tt.stack([
        #                      pm.logsumexp(logp_j_sk[j][:, :, np.newaxis] + tt.log(is_ploidy_in_ploidy_state_j_kl[j][np.newaxis, :, :] + 1E-100), axis=1)
        #                      for j in range(num_contigs)]).dimshuffle(1, 0, 3))

        pm.Deterministic(name='log_ploidy_emission_sjl',
                         var=tt.stack([
                             tt.log(tt.dot(pi_i_sk[i], is_ploidy_in_ploidy_state_j_kl[contig_to_index_map[contig]]) + eps)
                             for i, contig_tuple in enumerate(contig_tuples) for contig in contig_tuple]).dimshuffle(1, 0, 2))


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
        out = self._simultaneous_log_ploidy_emission_sampler()
        log_ploidy_emission_sjl = out[0]
        d_s = out[1]
        b_j_norm = out[2]
        pi_i_sk = out[3:]
        q_ploidy_sjl = np.exp(self.ploidy_workspace.log_q_ploidy_sjl.get_value(borrow=True))
        for s, q_ploidy_jl in enumerate(q_ploidy_sjl):
            print('sample_{0}:'.format(s), np.argmax(q_ploidy_jl, axis=1))
        # print("log_q_ploidy_sjl")
        # print(self.ploidy_workspace.log_q_ploidy_sjl.get_value(borrow=True))
        print("pi_i_sk")
        print(pi_i_sk)
        print("d_s")
        print(d_s)
        print("b_j_norm")
        print(b_j_norm)

        counts_m = self.ploidy_workspace.counts_m
        hist_sjm_full = self.ploidy_workspace.hist_sjm_full
        hist_sjm = self.ploidy_workspace.hist_sjm_full[:, :, counts_m]
        for s in range(self.ploidy_workspace.num_samples):
            fig, ax = plt.subplots()
            for i, contig_tuple in enumerate(self.ploidy_workspace.contig_tuples):
                for contig in contig_tuple:
                    # k = np.argmax(pi_i_sk[i][s])
                    j = self.ploidy_workspace.contig_to_index_map[contig]
                    plt.semilogy(hist_sjm_full[s, j], color='b', lw=0.5)
                    plt.semilogy(counts_m, hist_sjm[s, j], color='r', lw=3)
                    # plt.semilogy(counts_m, hist_mu_j_skm[j][s, k], color='g', lw=3)
                    y_max = 2 * np.max(hist_sjm_full[s, j])
                    ax.set_xlim([0, hist_sjm_full.shape[2]])
                    ax.set_ylim([1, y_max])
            ax.set_xlabel('count', size=14)
            ax.set_ylabel('number of intervals', size=14)
            fig.tight_layout(pad=0.1)
            fig.savefig('/home/slee/working/gatk/test_files/sample_{0}.png'.format(s))

        return log_ploidy_emission_sjl
        # return self._simultaneous_log_ploidy_emission_sampler()

    @th.configparser.change_flags(compute_test_value="off")
    def _get_compiled_simultaneous_log_ploidy_emission_sampler(self, approx: pm.approximations.MeanField):
        """For a given variational approximation, returns a compiled theano function that draws posterior samples
        from the log ploidy emission."""
        pi_i_sk = [commons.stochastic_node_mean_symbolic(
            approx, self.ploidy_model['pi_%d_sk' % i], size=self.samples_per_round)
            for i in range(self.ploidy_workspace.num_contig_tuples)]
        log_ploidy_emission_sjl = commons.stochastic_node_mean_symbolic(
            approx, self.ploidy_model['log_ploidy_emission_sjl'], size=self.samples_per_round)
        d_s = commons.stochastic_node_mean_symbolic(
            approx, self.ploidy_model['d_s'], size=self.samples_per_round)
        b_j_norm = commons.stochastic_node_mean_symbolic(
            approx, self.ploidy_model['b_j_norm'], size=self.samples_per_round)
        # hist_mu_j_skm = [commons.stochastic_node_mean_symbolic(
        #     approx, self.ploidy_model['hist_mu_%d_skm' % j], size=self.samples_per_round)
        #     for j in range(self.ploidy_workspace.num_contigs)]
        return th.function(inputs=[], outputs=[log_ploidy_emission_sjl, d_s, b_j_norm] + pi_i_sk)
        # return th.function(inputs=[], outputs=[log_ploidy_emission_sjl, d_s, psi_js, b_j_norm] + pi_i_sk + hist_mu_j_skm)
        # return th.function(inputs=[], outputs=log_ploidy_emission_sjl)


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
        # new_log_q_ploidy_sjl = self.ploidy_workspace.log_p_ploidy_jl.dimshuffle('x', 0, 1) + self.ploidy_workspace.log_ploidy_emission_sjl
        # new_log_q_ploidy_sjl -= pm.logsumexp(new_log_q_ploidy_sjl, axis=2)
        new_log_q_ploidy_sjl = self.ploidy_workspace.log_ploidy_emission_sjl - pm.logsumexp(self.ploidy_workspace.log_ploidy_emission_sjl, axis=2)
        old_log_q_ploidy_sjl = self.ploidy_workspace.log_q_ploidy_sjl
        admixed_new_log_q_ploidy_sjl = commons.safe_logaddexp(
            new_log_q_ploidy_sjl + np.log(self.inference_params.caller_external_admixing_rate),
            old_log_q_ploidy_sjl + np.log(1.0 - self.inference_params.caller_external_admixing_rate))
        update_norm_sj = commons.get_hellinger_distance(admixed_new_log_q_ploidy_sjl, old_log_q_ploidy_sjl)
        return th.function(inputs=[],
                           outputs=[update_norm_sj],
                           updates=[(self.ploidy_workspace.log_q_ploidy_sjl, admixed_new_log_q_ploidy_sjl)])

    def call(self) -> np.ndarray:
        return self._update_log_q_ploidy_sjl_theano_func()
