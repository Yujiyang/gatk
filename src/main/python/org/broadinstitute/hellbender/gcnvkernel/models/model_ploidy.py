import argparse
import inspect
import logging
from typing import List, Dict, Set, Tuple

import numpy as np
import pymc3 as pm
import theano as th
import theano.tensor as tt
from pymc3 import Normal, Deterministic, DensityDist, Dirichlet, Bound, Uniform, NegativeBinomial, Poisson, Gamma
from typing import List, Dict, Set, Tuple

from . import commons
from .fancy_model import GeneralizedContinuousModel
from .. import config, types
from ..structs.interval import Interval
from ..structs.metadata import IntervalListMetadata, SampleMetadataCollection
from ..tasks.inference_task_base import HybridInferenceParameters

import matplotlib.pyplot as plt

_logger = logging.getLogger(__name__)


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

        self.is_ploidy_in_ploidy_state_jkl = np.zeros((self.num_contigs, self.max_num_ploidy_states, self.num_ploidies),
                                                      dtype=types.small_uint)
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

        for i, contig_tuple in enumerate(self.contig_tuples):
            unpadded_priors = np.array(list(self.ploidy_config.ploidy_state_priors_map[contig_tuple].values()))
            self.ploidy_state_priors_ik[i, :len(unpadded_priors)] = unpadded_priors
            for contig in contig_tuple:
                j = self.contig_to_index_map[contig]
                self.ploidy_priors_jl[j] = np.sum(self.ploidy_state_priors_ik[i, :, np.newaxis] * self.is_ploidy_in_ploidy_state_jkl[j, :, :], axis=0)

        # count-distribution data
        hist_sjm = np.zeros((self.num_samples, self.num_contigs, self.num_counts), dtype=types.med_uint)
        for si, sample_name in enumerate(self.sample_names):
            sample_metadata = sample_metadata_collection.get_sample_coverage_metadata(sample_name)
            hist_sjm[si, :] = sample_metadata.hist_jm[:]

        # mask for count bins
        mask_sjm, self.counts_m = self._construct_mask(hist_sjm)
        self.mask_sjm = mask_sjm[:, :, self.counts_m]

        self.hist_sjm = hist_sjm[:, :, self.counts_m]

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
            fig.savefig('/home/BROAD.MIT.EDU/slee/working/gatk/test_files/sample_{0}.png'.format(s))

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
                min_sj = np.argmin(hist_sjm[s, j, :mode_sj[s, j]])
                if mode_sj[s, j] <= 10:
                    mode_sj[s, j] = 0
                    cutoff = 0.
                else:
                    cutoff = 0.05
                for m in range(mode_sj[s, j], np.shape(hist_sjm)[2]):
                    if hist_sjm[s, j, m] >= cutoff * hist_sjm[s, j, mode_sj[s, j]]:
                        if hist_sjm[s, j, m] >= 0:
                            mask_sjm[s, j, m] = True
                    else:
                        break
                for m in range(mode_sj[s, j], min_sj, -1):
                    if hist_sjm[s, j, m] >= cutoff * hist_sjm[s, j, mode_sj[s, j]]:
                        if hist_sjm[s, j, m] >= 0:
                            mask_sjm[s, j, m] = True
                    else:
                        break
            mask_sjm[:, :, 0] = False

        counts_m = []
        for m in range(hist_sjm.shape[2]):
            if np.any(hist_sjm[:, :, m]):
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
        ploidy_jk = ploidy_workspace.ploidy_jk
        max_num_ploidy_states = self.ploidy_workspace.max_num_ploidy_states
        is_ploidy_in_ploidy_state_jkl = self.ploidy_workspace.is_ploidy_in_ploidy_state_jkl
        eps = self.epsilon

        register_as_global = self.register_as_global
        register_as_sample_specific = self.register_as_sample_specific

        d_s = Uniform('d_s',
                      upper=depth_upper_bound,
                      shape=num_samples)
        register_as_sample_specific(d_s, sample_axis=0)

        b_j = Bound(Gamma,
                    lower=contig_bias_lower_bound,
                    upper=contig_bias_upper_bound)('b_j',
                                                   alpha=contig_bias_scale,
                                                   beta=contig_bias_scale,
                                                   shape=num_contigs)
        register_as_global(b_j)
        b_j_norm = Deterministic('b_j_norm', var=b_j / tt.mean(b_j))

        # f_sj = Bound(Normal,
        #              lower=mosaicism_bias_lower_bound,
        #              upper=mosaicism_bias_upper_bound)('f_sj',
        #                                                sd=mosaicism_bias_scale,
        #                                                shape=(num_samples, num_contigs))
        # register_as_sample_specific(f_sj, sample_axis=0)

        pi_sik = Dirichlet('pi_sik',
                           a=ploidy_concentration_scale * ploidy_state_priors_ik,
                           shape=(num_samples, num_contig_tuples, max_num_ploidy_states),
                           transform=pm.distributions.transforms.t_stick_breaking(eps))
        register_as_sample_specific(pi_sik, sample_axis=0)

        e_sj = Uniform('e_sj',
                       lower=0.,
                       upper=error_rate_upper_bound,
                       shape=(num_samples, num_contigs))
        register_as_sample_specific(e_sj, sample_axis=0)

        mu_sjk = d_s.dimshuffle(0, 'x', 'x') * b_j_norm.dimshuffle('x', 0, 'x') * \
                 tt.maximum(ploidy_jk[np.newaxis, :, :], e_sj.dimshuffle(0, 1, 'x'))
                   # (tt.maximum(ploidy_jk[np.newaxis, :, :] + f_sj.dimshuffle(0, 1, 'x') * (ploidy_jk[np.newaxis, :, :] > 0),
        alpha_sj = Uniform('alpha_sj',
                           upper=10000.,
                           shape=(num_samples, num_contigs))
        register_as_sample_specific(alpha_sj, sample_axis=0)

        logp_sjkm = NegativeBinomial.dist(mu=mu_sjk.dimshuffle(0, 1, 2, 'x') + eps,
                                          alpha=alpha_sj.dimshuffle(0, 1, 'x', 'x'))\
                        .logp(th.shared(np.array(counts_m, dtype=types.small_uint), borrow=config.borrow_numpy).dimshuffle('x', 'x', 'x', 0))

        def _logp_hist_sjm(_hist_sjm):
            num_occurrences_tot_sj = tt.sum(_hist_sjm * mask_sjm, axis=2)
            logp_hist_sjkm = Poisson.dist(mu=num_occurrences_tot_sj.dimshuffle(0, 1, 'x', 'x') * \
                                         tt.exp(logp_sjkm) + eps) \
                .logp(_hist_sjm.dimshuffle(0, 1, 'x', 2))
            return tt.sum(
                [pm.math.logsumexp(
                    mask_sjm[:, contig_to_index_map[contig], np.newaxis, :] * (tt.log(pi_sik[:, i, :, np.newaxis] + eps) + logp_hist_sjkm[:, contig_to_index_map[contig], :, :]),
                    axis=1)     # logsumexp over k
                    for i, contig_tuple in enumerate(contig_tuples) for contig in contig_tuple])

        DensityDist(name='hist_sjm', logp=_logp_hist_sjm, observed=hist_sjm)

        # for ploidy log emission sampling
        logp_sjl = pm.math.logsumexp(tt.sum(
            logp_sjkm.dimshuffle(0, 1, 2, 'x', 3) * is_ploidy_in_ploidy_state_jkl[np.newaxis, :, :, :, np.newaxis],
            axis=-1),               # sum over m
            axis=2)[:, :, 0, :]     # logsumexp over k
        Deterministic(name='logp_sjl', var=logp_sjl)


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
        return self._simultaneous_log_ploidy_emission_sampler()

    @th.configparser.change_flags(compute_test_value="off")
    def _get_compiled_simultaneous_log_ploidy_emission_sampler(self, approx: pm.approximations.MeanField):
        """For a given variational approximation, returns a compiled theano function that draws posterior samples
        from the log ploidy emission."""
        log_ploidy_emission_sjl = commons.stochastic_node_mean_symbolic(
            approx, self.ploidy_model['logp_sjl'], size=self.samples_per_round)
        return th.function(inputs=[], outputs=log_ploidy_emission_sjl)


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
        new_log_q_ploidy_sjl = self.ploidy_workspace.log_p_ploidy_jl.dimshuffle('x', 0, 1) + self.ploidy_workspace.log_ploidy_emission_sjl
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
