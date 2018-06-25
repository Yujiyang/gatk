import logging
import numpy as np
import pymc3 as pm
from typing import Callable
import matplotlib.pyplot as plt
from scipy.stats import nbinom

from .inference_task_base import Sampler, Caller, CallerUpdateSummary, \
    HybridInferenceTask, HybridInferenceParameters
from .. import config, types
from ..models.model_ploidy import PloidyModelConfig, PloidyModel, \
    PloidyWorkspace, PloidyEmissionBasicSampler, PloidyBasicCaller

_logger = logging.getLogger(__name__)


class PloidyCaller(Caller):
    """This class is a wrapper around `PloidyBasicCaller` to be used in a `HybridInferenceTask`."""
    def __init__(self,
                 hybrid_inference_params: HybridInferenceParameters,
                 ploidy_workspace: PloidyWorkspace):
        self.hybrid_inference_params = hybrid_inference_params
        self.ploidy_basic_caller = PloidyBasicCaller(hybrid_inference_params, ploidy_workspace)
        self.ploidy_workspace = ploidy_workspace

    def snapshot(self):
        """Snapshot is not necessary since there is no internal consistency loop."""
        pass

    def finalize(self):
        """Finalizing is not necessary since there is no internal consistency loop."""
        pass

    def call(self) -> 'PloidyCallerUpdateSummary':
        update_norm_sj = self.ploidy_basic_caller.call()
        return PloidyCallerUpdateSummary(
            update_norm_sj, self.hybrid_inference_params.caller_summary_statistics_reducer)

    def update_auxiliary_vars(self):
        pass


class PloidyCallerUpdateSummary(CallerUpdateSummary):
    def __init__(self,
                 update_norm_sj: np.ndarray,
                 reducer: Callable[[np.ndarray], float]):
        self.scalar_update = reducer(update_norm_sj)

    def __repr__(self):
        return "ploidy update size: {0:2.6}".format(self.scalar_update)

    def reduce_to_scalar(self) -> float:
        return self.scalar_update


class PloidyEmissionSampler(Sampler):
    """This class is a wrapper around `PloidyEmissionBasicSampler` to be used in a `HybridInferenceTask`."""
    def __init__(self,
                 hybrid_inference_params: HybridInferenceParameters,
                 ploidy_model: PloidyModel,
                 ploidy_workspace: PloidyWorkspace):
        super().__init__(hybrid_inference_params)
        self.ploidy_workspace = ploidy_workspace
        self.ploidy_emission_basic_sampler = PloidyEmissionBasicSampler(
            ploidy_model, self.hybrid_inference_params.log_emission_samples_per_round)

    def update_approximation(self, approx: pm.approximations.MeanField):
        self.ploidy_emission_basic_sampler.update_approximation(approx)

    def draw(self) -> np.ndarray:
        return self.ploidy_emission_basic_sampler.draw()

    def reset(self):
        self.ploidy_workspace.log_ploidy_emission_sjl.set_value(
            np.zeros((self.ploidy_workspace.num_samples,
                      self.ploidy_workspace.num_contigs,
                      self.ploidy_workspace.num_ploidies),
                     dtype=types.floatX), borrow=config.borrow_numpy)

    def increment(self, update):
        self.ploidy_workspace.log_ploidy_emission_sjl.set_value(
            self.ploidy_workspace.log_ploidy_emission_sjl.get_value(borrow=True) + update)

    def get_latest_log_emission_posterior_mean_estimator(self) -> np.ndarray:
        return self.ploidy_workspace.log_ploidy_emission_sjl.get_value(borrow=True)


class CohortPloidyInferenceTask(HybridInferenceTask):
    """Cohort germline contig ploidy determination task."""
    def __init__(self,
                 hybrid_inference_params: HybridInferenceParameters,
                 ploidy_config: PloidyModelConfig,
                 ploidy_workspace: PloidyWorkspace):
        _logger.info("Instantiating the germline contig ploidy determination model...")
        self.ploidy_model = PloidyModel(ploidy_config, ploidy_workspace)

        _logger.info("Instantiating the ploidy emission sampler...")
        ploidy_emission_sampler = PloidyEmissionSampler(hybrid_inference_params, self.ploidy_model, ploidy_workspace)

        _logger.info("Instantiating the ploidy caller...")
        ploidy_caller = PloidyCaller(hybrid_inference_params, ploidy_workspace)

        elbo_normalization_factor = ploidy_workspace.num_samples * ploidy_workspace.num_contigs
        super().__init__(hybrid_inference_params, self.ploidy_model, ploidy_emission_sampler, ploidy_caller,
                         elbo_normalization_factor=elbo_normalization_factor,
                         advi_task_name="denoising",
                         calling_task_name="ploidy calling")

        self.ploidy_config = ploidy_config
        self.ploidy_workspace = ploidy_workspace

    def disengage(self):
        print("Sampling...")
        num_samples = 100
        trace = self.continuous_model_approx.sample(num_samples)
        pi_i_sk = [np.mean(trace['pi_%d_sk' % i], axis=0)
                   for i in range(self.ploidy_workspace.num_contig_tuples)]
        d_s = np.mean(trace['d_s'], axis=0)
        b_j_norm = np.mean(trace['b_j_norm'], axis=0)
        mu_j_sk = [np.mean(trace['mu_%d_sk' % j], axis=0)
                   for j in range(self.ploidy_workspace.num_contigs)]
        alpha_js = np.mean(trace['alpha_js'], axis=0)
        # hist_j_skm = [np.mean(trace['hist_%d_skm' % j], axis=0)
        #               for j in range(self.ploidy_workspace.num_contigs)]
        print("Sampling done.")

        print("d_s")
        print(d_s)
        print("b_j_norm")
        print(b_j_norm)
        q_ploidy_sjl = np.exp(self.ploidy_workspace.log_q_ploidy_sjl.get_value(borrow=True))
        for s, q_ploidy_jl in enumerate(q_ploidy_sjl):
            print('sample_{0}:'.format(s), np.argmax(q_ploidy_jl, axis=1))
        for s in range(self.ploidy_workspace.num_samples):
            fig, ax = plt.subplots()
            t_j = np.zeros(self.ploidy_workspace.num_contigs)
            for i, contig_tuple in enumerate(self.ploidy_workspace.contig_tuples):
                for contig in contig_tuple:
                    j = self.ploidy_workspace.contig_to_index_map[contig]
                    counts_m_masked = self.ploidy_workspace.counts_m[self.ploidy_workspace.hist_sjm_mask[s, j]]
                    t_j[j] = np.mean(np.repeat(counts_m_masked[1:], self.ploidy_workspace.hist_sjm_full[s, j, counts_m_masked[1:]]))
                    k = np.argmax(pi_i_sk[i][s])
                    print(mu_j_sk[j][s, k], alpha_js[j, s])
                    mu = mu_j_sk[j][s, k]
                    alpha = alpha_js[j, s]
                    pdf = nbinom.pmf(k=counts_m_masked, n=alpha, p=alpha / (mu + alpha))
                    plt.semilogy(self.ploidy_workspace.hist_sjm_full[s, j] / np.sum(self.ploidy_workspace.hist_sjm_full[s, j]), color='k', lw=0.5, alpha=0.1)
                    plt.semilogy(counts_m_masked, self.ploidy_workspace.hist_sjm_full[s, j, counts_m_masked] / np.sum(self.ploidy_workspace.hist_sjm_full[s, j]),
                                 c='b' if j < self.ploidy_workspace.num_contigs - 2 else 'r', lw=1, alpha=0.25)
                    plt.semilogy(counts_m_masked, pdf, color='g')
                    # plt.semilogy(self.ploidy_workspace.hist_sjm_full[s, j], color='k', lw=0.5, alpha=0.1)
                    # plt.semilogy(counts_m_masked, self.ploidy_workspace.hist_sjm_full[s, j, counts_m_masked],
                    #              c='b' if j < self.ploidy_workspace.num_contigs - 2 else 'r', lw=1, alpha=0.25)
                    # plt.semilogy(counts_m_masked, hist_j_skm[j][s, k][counts_m_masked], color='g')
                    ax.set_xlim([0, self.ploidy_workspace.num_counts])
                    ax.set_ylim([1E-5, 1E-1])
            print(s, t_j / np.mean(t_j))
            ax.set_xlabel('count', size=14)
            # ax.set_ylabel('number of intervals', size=14)
            fig.tight_layout(pad=0.1)
            fig.savefig('/home/slee/working/gatk/test_files/plots/sample_{0}.png'.format(s))
