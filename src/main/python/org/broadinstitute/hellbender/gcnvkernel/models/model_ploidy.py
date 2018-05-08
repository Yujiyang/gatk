import argparse
import inspect
import logging

import numpy as np
import pymc3 as pm
import theano as th
import theano.tensor as tt
from pymc3 import Normal, Deterministic, DensityDist, Dirichlet, Bound, Uniform, NegativeBinomial, Poisson, Gamma, Exponential
from typing import List, Dict, Set, Tuple

from . import commons
from .fancy_model import GeneralizedContinuousModel
from .. import config, types
from ..structs.interval import Interval
from ..structs.metadata import IntervalListMetadata, SampleMetadataCollection
from ..tasks.inference_task_base import HybridInferenceParameters

import matplotlib.pyplot as plt

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
                 mosaicism_bias_lower_bound: float = -0.9,
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
                assert contig not in self.contigs, "Contig tuples must be disjoint."
                self.contigs.append(contig)
                self.ploidy_j_k.append(np.array([ploidy_state[j]
                                                 for ploidy_state in self.ploidy_states_i_k[i]]))

        assert all([contig in self.contigs for contig in interval_list_metadata.contig_set]), \
            "Some contigs do not have ploidy priors"

        self.contig_to_index_map = {contig: index for index, contig in enumerate(self.contigs)}

        self.num_ploidy_states_j = np.array([len(ploidy_k) for ploidy_k in self.ploidy_j_k])
        self.max_num_ploidy_states = np.max(self.num_ploidy_states_j)
        self.num_ploidies = np.max([np.max(ploidy_k) for ploidy_k in self.ploidy_j_k]) + 1

        self.is_contig_in_contig_tuple_ij = np.zeros((self.num_contig_tuples, self.num_contigs),
                                                     dtype=types.small_uint)
        self.is_ploidy_in_ploidy_state_jkl = np.zeros((self.num_contigs, self.max_num_ploidy_states, self.num_ploidies),
                                                      dtype=types.small_uint)
        self.is_ploidy_in_ploidy_state_j_kl = [np.zeros((self.num_ploidy_states_j[j], self.num_ploidies))
                                               for j in range(self.num_contigs)]
        self.ploidy_jk = np.zeros((self.num_contigs, self.max_num_ploidy_states),
                                  dtype=types.small_uint)
        self.ploidy_state_priors_ik = 1E-10 * np.ones((self.num_contig_tuples, self.max_num_ploidy_states),
                                                      dtype=types.floatX)
        self.ploidy_priors_jl = 1E-10 * np.ones((self.num_contigs, self.num_ploidies),
                                                dtype=types.floatX)

        for j in range(self.num_contigs):
            for k in range(self.num_ploidy_states_j[j]):
                ploidy = self.ploidy_j_k[j][k]
                self.ploidy_jk[j, k] = ploidy
                self.is_ploidy_in_ploidy_state_jkl[j, k, ploidy] = 1
                self.is_ploidy_in_ploidy_state_j_kl[j][k, ploidy] = 1

        for i, contig_tuple in enumerate(self.contig_tuples):
            unpadded_priors = np.array(list(self.ploidy_config.ploidy_state_priors_map[contig_tuple].values()))
            self.ploidy_state_priors_ik[i, :len(unpadded_priors)] = unpadded_priors
            for contig in contig_tuple:
                j = self.contig_to_index_map[contig]
                self.is_contig_in_contig_tuple_ij[i, j] = 1
                self.ploidy_priors_jl[j] = np.sum(self.ploidy_state_priors_ik[i, :, np.newaxis] * self.is_ploidy_in_ploidy_state_jkl[j, :, :], axis=0)

        # count-distribution data
        hist_sjm = np.zeros((self.num_samples, self.num_contigs, self.num_counts), dtype=types.med_uint)
        for si, sample_name in enumerate(self.sample_names):
            sample_metadata = sample_metadata_collection.get_sample_coverage_metadata(sample_name)
            hist_sjm[si, :] = sample_metadata.hist_jm[:]

        # mask for count bins
        mask_sjm, self.counts_m = self._construct_mask(hist_sjm)
        self.mask_sjm = mask_sjm[:, :, self.counts_m]

        average_ploidy = 2. # TODO
        self.d_s_testval = np.median(np.sum(hist_sjm * np.arange(hist_sjm.shape[2]), axis=-1) / np.sum(hist_sjm, axis=-1), axis=-1) / average_ploidy

        self.hist_sjm : types.TensorSharedVariable = \
            th.shared(hist_sjm[:, :, self.counts_m], name='hist_sjm', borrow=config.borrow_numpy)

        for s in range(self.num_samples):
            fig, ax = plt.subplots()
            for i, contig_tuple in enumerate(self.contig_tuples):
                for contig in contig_tuple:
                    j = self.contig_to_index_map[contig]
                    hist_m_masked = hist_sjm[s, j][mask_sjm[s, j]]
                    plt.semilogy(hist_sjm[s, j], color='b', lw=0.5)
                    plt.semilogy(np.arange(hist_sjm.shape[2])[mask_sjm[s, j]], hist_m_masked, color='r', lw=3)
                    y_max = 2 * np.max(hist_sjm[s, j])
                    ax.set_xlim([0, hist_sjm.shape[2]])
                    ax.set_ylim([1, y_max])
            ax.set_xlabel('count', size=14)
            ax.set_ylabel('number of intervals', size=14)
            fig.tight_layout(pad=0.1)
            fig.savefig('/home/slee/working/gatk/test_files/sample_{0}.png'.format(s))

        # print('ploidy_states_i_k')
        # print(self.ploidy_states_i_k)
        # print('ploidy_j_k')
        # print(self.ploidy_j_k)
        # print('ploidy_state_priors_ik')
        # print(self.ploidy_state_priors_ik)
        # print('is_contig_in_contig_tuple_ij')
        # print(self.is_contig_in_contig_tuple_ij)
        # print('is_ploidy_in_ploidy_state_jkl')
        # print(self.is_ploidy_in_ploidy_state_jkl)
        # print('ploidy_jk')
        # print(self.ploidy_jk)
        # print('ploidy_state_priors_ik')
        # print(self.ploidy_state_priors_ik)
        # print('ploidy_priors_jl')
        # print(self.ploidy_priors_jl)
        print('counts_m')
        print(self.counts_m)
        print('hist_sjm')
        print(hist_sjm[:, :, self.counts_m])

        # ploidy log priors
        self.log_p_ploidy_jl: types.TensorSharedVariable = \
            th.shared(np.log(self.ploidy_priors_jl),
                      name='log_p_ploidy_sjl', borrow=config.borrow_numpy)

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

    @staticmethod
    def _construct_mask(hist_sjm):
        count_states = np.arange(0, hist_sjm.shape[2])
        mode_sj = np.argmax(hist_sjm * (count_states >= 5), axis=2)
        mask_sjm = np.full(np.shape(hist_sjm), False)
        for s in range(np.shape(hist_sjm)[0]):
            for j in range(np.shape(hist_sjm)[1]):
                min_sj = np.argmin(hist_sjm[s, j, :mode_sj[s, j] + 1])
                if mode_sj[s, j] <= 10:
                    mode_sj[s, j] = 0
                    cutoff = 0.
                else:
                    cutoff = 0.05
                for m in range(mode_sj[s, j], np.shape(hist_sjm)[2]):
                    if hist_sjm[s, j, m] >= cutoff * hist_sjm[s, j, mode_sj[s, j]]:
                        if hist_sjm[s, j, m] > 0:
                            mask_sjm[s, j, m] = True
                    else:
                        break
                for m in range(mode_sj[s, j], min_sj, -1):
                    if hist_sjm[s, j, m] >= cutoff * hist_sjm[s, j, mode_sj[s, j]]:
                        if hist_sjm[s, j, m] > 0:
                            mask_sjm[s, j, m] = True
                    else:
                        break
            mask_sjm[:, :, 0] = False

        counts_m = []
        for m in range(hist_sjm.shape[2]):
            if np.any(mask_sjm[:, :, m]):
                counts_m.append(m)

        return mask_sjm, counts_m


class PloidyModel(GeneralizedContinuousModel):
    """Declaration of the germline contig ploidy model (continuous variables only; posterior of discrete
    variables are assumed to be known)."""

    epsilon: float = 1e-10

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
        contig_tuples = ploidy_workspace.contig_tuples
        num_contig_tuples = ploidy_workspace.num_contig_tuples
        num_samples = ploidy_workspace.num_samples
        num_contigs = ploidy_workspace.num_contigs
        counts_m = ploidy_workspace.counts_m
        contig_to_index_map = ploidy_workspace.contig_to_index_map
        mask_sjm = ploidy_workspace.mask_sjm
        hist_sjm = ploidy_workspace.hist_sjm
        ploidy_state_priors_ik = ploidy_workspace.ploidy_state_priors_ik
        ploidy_state_priors_i_k = ploidy_workspace.ploidy_state_priors_i_k
        ploidy_jk = ploidy_workspace.ploidy_jk
        ploidy_j_k = ploidy_workspace.ploidy_j_k
        max_num_ploidy_states = self.ploidy_workspace.max_num_ploidy_states
        is_contig_in_contig_tuple_ij = self.ploidy_workspace.is_contig_in_contig_tuple_ij
        is_ploidy_in_ploidy_state_jkl = self.ploidy_workspace.is_ploidy_in_ploidy_state_jkl
        is_ploidy_in_ploidy_state_j_kl = self.ploidy_workspace.is_ploidy_in_ploidy_state_j_kl
        num_ploidies = self.ploidy_workspace.num_ploidies
        d_s_testval = self.ploidy_workspace.d_s_testval
        eps = self.epsilon

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

        e_js = Uniform('e_js',
                       lower=0.,
                       upper=error_rate_upper_bound,
                       shape=(num_contigs, num_samples))
        register_as_sample_specific(e_js, sample_axis=1)

        mu_j_sk = [d_s.dimshuffle(0, 'x') * b_j_norm[j] * \
                   # (tt.maximum(ploidy_j_k[j][np.newaxis, :] + f_js[j].dimshuffle(0, 'x') * (ploidy_j_k[j][np.newaxis, :] > 0),
                   #             e_js[j].dimshuffle(0, 'x')))
                   tt.maximum(ploidy_j_k[j][np.newaxis, :], e_js[j].dimshuffle(0, 'x'))
                   for j in range(num_contigs)]

        psi_js = Exponential(name='psi_js',
                             lam=10.0, #1.0 / ploidy_config.psi_scale,
                             shape=(num_contigs, num_samples))
        register_as_sample_specific(psi_js, sample_axis=1)
        alpha_js = tt.inv((tt.exp(psi_js) - 1.0))

        p_j_skm = [tt.exp(NegativeBinomial.dist(mu=mu_j_sk[j].dimshuffle(0, 1, 'x') + eps,
                                                alpha=alpha_js[j].dimshuffle(0, 'x', 'x'))
                          .logp(th.shared(np.array(counts_m, dtype=types.small_uint), borrow=config.borrow_numpy).dimshuffle('x', 'x', 0)))
                   for j in range(num_contigs)]

        # p_j_skm = [tt.exp(Poisson.dist(mu=mu_j_sk[j].dimshuffle(0, 1, 'x') + eps)
        #                   .logp(th.shared(np.array(counts_m, dtype=types.small_uint), borrow=config.borrow_numpy).dimshuffle('x', 'x', 0)))
        #            for j in range(num_contigs)]

        def _logp_hist_j_skm(_hist_sjm):
            num_occurrences_tot_sj = tt.sum(_hist_sjm * mask_sjm, axis=2)
            logp_hist_j_skm = [pm.Poisson.dist(mu=num_occurrences_tot_sj[:, j].dimshuffle(0, 'x', 'x') * p_j_skm[j] + eps) \
                                   .logp(_hist_sjm[:, j, :].dimshuffle(0, 'x', 1))
                               for j in range(num_contigs)]
            return [mask_sjm[:, contig_to_index_map[contig], np.newaxis, :] * \
                   (tt.log(ploidy_state_priors_i_k[i][np.newaxis, :, np.newaxis] + eps) + tt.log(pi_i_sk[i][:, :, np.newaxis] + eps) + logp_hist_j_skm[contig_to_index_map[contig]])
                    for i, contig_tuple in enumerate(contig_tuples) for contig in contig_tuple]
            # return [mask_sjm[:, contig_to_index_map[contig], np.newaxis, :] * _hist_sjm[:, contig_to_index_map[contig], np.newaxis, :] * \
            #         (tt.log(ploidy_state_priors_i_k[i][np.newaxis, :, np.newaxis] + eps) + tt.log(pi_i_sk[i][:, :, np.newaxis] + eps) + tt.log(p_j_skm[contig_to_index_map[contig]] + eps))
            #         for i, contig_tuple in enumerate(contig_tuples) for contig in contig_tuple]

        DensityDist(name='hist_sjm',
                    logp=lambda _hist_sjm: tt.sum([pm.logsumexp(logp_hist_skm, axis=1)
                                                   for logp_hist_skm in _logp_hist_j_skm(_hist_sjm)]),
                    observed=hist_sjm)

        # logp_hist_j_sk = [tt.sum(logp_hist_skm, axis=-1)
        #                   for logp_hist_skm in _logp_hist_j_skm(hist_sjm)]
        # logp_jsl = tt.as_tensor_variable([tt.sum(logp_hist_j_sk[j][:, :, np.newaxis] * is_ploidy_in_ploidy_state_j_kl[j][np.newaxis, :, :], axis=1)
        #                                   for j in range(num_contigs)])

        logp_jsl = tt.as_tensor_variable([tt.log(pi_i_sk[i] + eps) * is_contig_in_contig_tuple_ij[i, j]
                          for i, contig_tuple in enumerate(contig_tuples) for contig in contig_tuple])
        Deterministic(name='logp_sjl', var=logp_jsl.dimshuffle(1, 0, 2))


class PloidyEmissionBasicSampler:
    """Draws posterior samples from the ploidy log emission probability for a given variational
    approximation to the ploidy model posterior."""
    def __init__(self, ploidy_model: PloidyModel, samples_per_round: int):
        self.ploidy_model = ploidy_model
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
        psi_js = out[2]
        pi_i_sk = out[3:]
        print(pi_i_sk)
        print(d_s)
        print(1. / (np.exp(psi_js) - 1))
        print(np.exp(log_ploidy_emission_sjl))
        return log_ploidy_emission_sjl
        # return self._simultaneous_log_ploidy_emission_sampler()

    @th.configparser.change_flags(compute_test_value="off")
    def _get_compiled_simultaneous_log_ploidy_emission_sampler(self, approx: pm.approximations.MeanField):
        """For a given variational approximation, returns a compiled theano function that draws posterior samples
        from the log ploidy emission."""
        log_ploidy_emission_sjl = commons.stochastic_node_mean_symbolic(
            approx, self.ploidy_model['logp_sjl'], size=self.samples_per_round)
        pi_i_sk = [commons.stochastic_node_mean_symbolic(
            approx, self.ploidy_model['pi_%d_sk' % i], size=self.samples_per_round)
            for i in range(self.ploidy_model.ploidy_workspace.num_contig_tuples)]
        d_s = commons.stochastic_node_mean_symbolic(
            approx, self.ploidy_model['d_s'], size=self.samples_per_round)
        psi_js = commons.stochastic_node_mean_symbolic(
            approx, self.ploidy_model['psi_js'], size=self.samples_per_round)
        return th.function(inputs=[], outputs=[log_ploidy_emission_sjl, d_s, psi_js] + pi_i_sk)


class PloidyBasicCaller:
    """Bayesian update of germline contig ploidy log posteriors."""
    def __init__(self,
                 inference_params: HybridInferenceParameters,
                 ploidy_workspace: PloidyWorkspace):
        self.ploidy_workspace = ploidy_workspace
        self.inference_params = inference_params
        self._update_log_q_ploidy_sjl_theano_func = self._update_log_q_ploidy_sjl_theano_func()

    @th.configparser.change_flags(compute_test_value="off")
    def _update_log_q_ploidy_sjl_theano_func(self) -> th.compile.function_module.Function:
        # new_log_q_ploidy_sjl = self.ploidy_workspace.log_p_ploidy_jl.dimshuffle('x', 0, 1) + self.ploidy_workspace.log_ploidy_emission_sjl
        new_log_q_ploidy_sjl = self.ploidy_workspace.log_ploidy_emission_sjl
        new_log_q_ploidy_sjl -= pm.logsumexp(new_log_q_ploidy_sjl, axis=2)
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
