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
        self.ploidy_states_ik, self.ploidy_state_priors_ik, self.ploidy_jk, self.contig_tuples, self.contigs = \
            self._process_ploidy_state_priors_map(ploidy_state_priors_map)

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
    def _process_ploidy_state_priors_map(ploidy_state_priors_map: Dict[List[str], Dict[List[int], float]]) \
            -> Tuple[List[List[Tuple[int]]], List[np.ndarray], List[np.ndarray], List[Tuple[str]], List[str]]:
        contig_tuples: List[Tuple[str]] = list(ploidy_state_priors_map.keys())

        # i = contig-tuple index, j = contig index, k = ploidy-state index
        ploidy_states_ik: List[List[Tuple[int]]] = [list(ploidy_state_priors_map[contig_tuple].keys())
                                                    for contig_tuple in contig_tuples]
        ploidy_state_priors_ik: List[np.ndarray] = [np.fromiter(ploidy_state_priors_map[contig_tuple].values(),
                                                                 dtype=float)
                                                     for contig_tuple in contig_tuples]

        ploidy_jk: List[np.ndarray] = []
        contigs: List[str] = []
        for i, contig_tuple in enumerate(contig_tuples):
            for contig_index, contig in enumerate(contig_tuple):
                assert contig not in contigs, "Contig tuples must be disjoint."
                contigs.append(contig)
                ploidy_jk.append(np.array([ploidy_state[contig_index] for ploidy_state in ploidy_states_ik[i]]))

        return ploidy_states_ik, ploidy_state_priors_ik, ploidy_jk, contig_tuples, contigs

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
        self.interval_list_metadata = interval_list_metadata
        self.sample_metadata_collection = sample_metadata_collection
        self.ploidy_config = ploidy_config
        self.num_contigs = interval_list_metadata.num_contigs
        self.sample_names = sample_names
        self.num_samples: int = len(sample_names)
        self.max_count = sample_metadata_collection.get_sample_coverage_metadata(sample_names[0]).max_count
        self.contig_to_index_map = {contig: index for index, contig in enumerate(ploidy_config.contigs)}

        assert all([contig in ploidy_config.contigs for contig in interval_list_metadata.contig_set]), \
            "Some contigs do not have ploidy priors"
        assert sample_metadata_collection.all_samples_have_coverage_metadata(sample_names), \
            "Some samples do not have coverage metadata"

        # create shared theano tensors
        # s = sample index, i = contig-tuple index, j = contig index, k = ploidy-state index, m = count index

        # count-distribution data
        hist_sjm = np.zeros((self.num_samples, self.num_contigs, self.max_count + 1), dtype=types.med_uint)
        for si, sample_name in enumerate(self.sample_names):
            sample_metadata = sample_metadata_collection.get_sample_coverage_metadata(sample_name)
            hist_sjm[si, :] = sample_metadata.hist_jm[:]
        self.hist_sjm: types.TensorSharedVariable = th.shared(hist_sjm, name='hist_sjm', borrow=config.borrow_numpy)

        self.mask_sjm = self._construct_mask(hist_sjm)

        num_ploidy_states_j = np.array([len(ploidy_k) for ploidy_k in ploidy_config.ploidy_jk])

        # ploidy log posteriors (initial value is immaterial
        self.log_q_ploidy_j_sk: List[types.TensorSharedVariable] = \
            [th.shared(np.zeros((self.num_samples, num_ploidy_states_j[j]), dtype=types.floatX),
                       name='log_q_ploidy_%s_sk' % contig, borrow=config.borrow_numpy)
             for j, contig in enumerate(self.ploidy_config.contigs)]

        # ploidy log emission (initial value is immaterial)
        self.log_ploidy_emission_j_sk: List[types.TensorSharedVariable] = \
            [th.shared(np.zeros((self.num_samples, num_ploidy_states_j[j]), dtype=types.floatX),
                       name='log_ploidy_emission_%s_sk' % contig, borrow=config.borrow_numpy)
             for j, contig in enumerate(self.ploidy_config.contigs)]

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
        return mask_sjm


class PloidyModel(GeneralizedContinuousModel):
    """Declaration of the germline contig ploidy model (continuous variables only; posterior of discrete
    variables are assumed to be known)."""

    epsilon: float = 1e-10

    def __init__(self,
                 ploidy_config: PloidyModelConfig,
                 ploidy_workspace: PloidyWorkspace):
        super().__init__()

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
        contig_tuples = ploidy_config.contig_tuples
        num_samples = ploidy_workspace.num_samples
        num_contigs = ploidy_workspace.num_contigs
        max_count = ploidy_workspace.max_count
        contig_to_index_map = ploidy_workspace.contig_to_index_map
        mask_sjm = ploidy_workspace.mask_sjm
        hist_sjm = ploidy_workspace.hist_sjm
        ploidy_states_ik = ploidy_config.ploidy_states_ik
        ploidy_state_priors_ik = ploidy_config.ploidy_state_priors_ik
        ploidy_jk = ploidy_config.ploidy_jk
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
        register_as_global(b_j_norm)

        f_js = Bound(Normal,
                     lower=mosaicism_bias_lower_bound,
                     upper=mosaicism_bias_upper_bound)('f_js',
                                                       sd=mosaicism_bias_scale,
                                                       shape=(num_contigs, num_samples))
        register_as_sample_specific(f_js, sample_axis=1)

        pi_i_sk = [Dirichlet('pi_%s_sk' % str(contig_tuple),
                             a=ploidy_concentration_scale * ploidy_state_priors_ik[i],
                             shape=(num_samples, len(ploidy_states_ik[i])),
                             transform=pm.distributions.transforms.t_stick_breaking(eps))
                   if len(contig_tuple) > 1
                   else pm.Deterministic('pi_%s_sk' % str(contig_tuple), var=tt.ones((num_samples, 1)))
                   for i, contig_tuple in enumerate(contig_tuples)]
        for i in range(len(contig_tuples)):
            register_as_sample_specific(pi_i_sk[i], sample_axis=0)

        e_js = Uniform('e_js',
                       lower=0.,
                       upper=error_rate_upper_bound,
                       shape=(num_contigs, num_samples))
        register_as_sample_specific(e_js, sample_axis=1)

        mu_j_sk = [d_s.dimshuffle(0, 'x') * b_j_norm[j] * \
                   (tt.maximum(ploidy_jk[j][np.newaxis, :] + f_js[j].dimshuffle(0, 'x') * (ploidy_jk[j][np.newaxis, :] > 0),
                               e_js[j].dimshuffle(0, 'x')))
                   for j in range(num_contigs)]
        alpha_js = Uniform('alpha_js',
                           upper=10000.,
                           shape=(num_contigs, num_samples))
        register_as_sample_specific(alpha_js, sample_axis=1)

        logp_j_skm = [NegativeBinomial.dist(mu=mu_j_sk[j].dimshuffle(0, 1, 'x') + eps,
                                            alpha=alpha_js[j].dimshuffle(0, 'x', 'x'))
                          .logp(tt.arange(max_count + 1).dimshuffle('x', 'x', 0))
                      for j in range(num_contigs)]

        def _logp_sjm(_hist_sjm):
            num_occurrences_tot_sj = tt.sum(_hist_sjm * mask_sjm, axis=2)
            logp_hist_j_skm = [Poisson.dist(mu=num_occurrences_tot_sj[:, j].dimshuffle(0, 'x', 'x') * tt.exp(logp_j_skm[j]) + eps) \
                                   .logp(_hist_sjm[:, j, :].dimshuffle(0, 'x', 1))
                               for j in range(num_contigs)]
            return tt.sum(
                [pm.math.logsumexp(
                    mask_sjm[:, contig_to_index_map[contig], np.newaxis, :] * (tt.log(pi_i_sk[i][:, :, np.newaxis] + eps) + logp_hist_j_skm[contig_to_index_map[contig]]),
                    axis=1)
                    for i, contig_tuple in enumerate(contig_tuples) for contig in contig_tuple])

        DensityDist(name='hist_sjm', logp=_logp_sjm, observed=hist_sjm)

        # for log ploidy emission sampling
        for j, contig in enumerate(ploidy_config.contigs):
            Deterministic(name='logp_%s_skm' % contig, var=logp_j_skm[j])


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
        log_ploidy_emission_sjk = commons.stochastic_node_mean_symbolic(
            approx, self.ploidy_model['logp_sjk'], size=self.samples_per_round)
        return th.function(inputs=[], outputs=log_ploidy_emission_sjk)


class PloidyBasicCaller:
    """Bayesian update of germline contig ploidy log posteriors."""
    def __init__(self,
                 inference_params: HybridInferenceParameters,
                 ploidy_workspace: PloidyWorkspace):
        self.ploidy_workspace = ploidy_workspace
        self.inference_params = inference_params
        self._update_log_q_ploidy_sjk_theano_func = self._get_update_log_q_ploidy_sjk_theano_func()

    @th.configparser.change_flags(compute_test_value="off")
    def _get_update_log_q_ploidy_sjk_theano_func(self) -> th.compile.function_module.Function:
        new_log_q_ploidy_sjk = self.ploidy_workspace.log_ploidy_emission_sjk
        new_log_q_ploidy_sjk -= pm.logsumexp(new_log_q_ploidy_sjk, axis=2)
        old_log_q_ploidy_sjk = self.ploidy_workspace.log_q_ploidy_sjk
        admixed_new_log_q_ploidy_sjk = commons.safe_logaddexp(
            new_log_q_ploidy_sjk + np.log(self.inference_params.caller_external_admixing_rate),
            old_log_q_ploidy_sjk + np.log(1.0 - self.inference_params.caller_external_admixing_rate))
        update_norm_sj = commons.get_hellinger_distance(admixed_new_log_q_ploidy_sjk, old_log_q_ploidy_sjk)
        return th.function(inputs=[],
                           outputs=[update_norm_sj],
                           updates=[(self.ploidy_workspace.log_q_ploidy_sjk, admixed_new_log_q_ploidy_sjk)])

    def call(self) -> np.ndarray:
        return self._update_log_q_ploidy_sjk_theano_func()
