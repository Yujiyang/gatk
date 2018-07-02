import os

# set theano flags
os.environ["THEANO_FLAGS"] = "device=cpu,floatX=float64,optimizer=fast_run,compute_test_value=ignore,openmp=true,exception_verbosity='high'"

import argparse
import gcnvkernel

parser = argparse.ArgumentParser(description="gCNV contig ploidy and read-depth determination tool",
                                 formatter_class=gcnvkernel.cli_commons.GCNVHelpFormatter)

# logging args
gcnvkernel.cli_commons.add_logging_args_to_argparse(parser)

# add tool-specific args
group = parser.add_argument_group(title="Required arguments")

group.add_argument("--contig_count_distribution_collection_files",
                   type=str,
                   required=True,
                   nargs='+',  # one or more
                   default=argparse.SUPPRESS,
                   help="List of per-contig count-distribution files for all samples (in .tsv format; must include sample name header)")

group.add_argument("--input_model_path",
                   type=str,
                   required=True,
                   default=argparse.SUPPRESS,
                   help="Path to ploidy-model parameters")

group.add_argument("--output_calls_path",
                   type=str,
                   required=True,
                   default=argparse.SUPPRESS,
                   help="Output path to write posteriors")

# optional arguments
# Note: we are hiding parameters that are either set by the model or are irrelevant to the case calling task
gcnvkernel.PloidyModelConfig.expose_args(
    parser,
    hide={
        "--ploidy_concentration_scale",
        "--depth_upper_bound",
        "--error_rate_upper_bound",
        "--contig_bias_lower_bound",
        "--contig_bias_upper_bound",
        "--contig_bias_scale"
    })

# override some inference parameters
gcnvkernel.HybridInferenceParameters.expose_args(
    parser,
    override_default={
        "--learning_rate": 0.1,
        "--adamax_beta2": 0.999,
        "--log_emission_samples_per_round": 1000,
        "--log_emission_sampling_rounds": 50,
        "--log_emission_sampling_median_rel_error": 1e-3,
        "--max_advi_iter_first_epoch": 5000,
        "--max_advi_iter_subsequent_epochs": 1000,
        "--convergence_snr_averaging_window": 5000,
        "--convergence_snr_countdown_window": 100,
        "--num_thermal_advi_iters": 10000,
        "--max_calling_iters": 1,
        "--caller_update_convergence_threshold": 1e-3
    })

if __name__ == "__main__":

    # parse arguments
    args = parser.parse_args()
    gcnvkernel.cli_commons.set_logging_config_from_args(args)

    # check gcnvkernel version in the input model path
    gcnvkernel.io_commons.check_gcnvkernel_version_from_path(args.input_model_path)

    # read ploidy-state prior map from the model
    ploidy_state_priors_table = os.path.join(args.input_model_path,
                                            gcnvkernel.io_consts.default_ploidy_state_prior_tsv_filename)
    assert os.path.exists(ploidy_state_priors_table) and os.path.isfile(ploidy_state_priors_table), \
        "The provided ploidy model is corrupted: it does not include ploidy-state priors .tsv file"
    # read ploidy-state prior map from file
    ploidy_state_priors_map = gcnvkernel.io_ploidy.get_ploidy_state_priors_map_from_tsv_file(
        ploidy_state_priors_table)

    # load interval list from the model
    interval_list_file = os.path.join(args.input_model_path, gcnvkernel.io_consts.default_interval_list_filename)
    assert os.path.exists(interval_list_file) and os.path.isfile(interval_list_file), \
        "The provided ploidy model is corrupted: it does not include interval list .tsv file"
    interval_list = gcnvkernel.io_intervals_and_counts.load_interval_list_tsv_file(interval_list_file)

    # load sample coverage metadata
    sample_metadata_collection: gcnvkernel.SampleMetadataCollection = gcnvkernel.SampleMetadataCollection()
    sample_names = gcnvkernel.io_metadata.read_sample_coverage_metadata(
        sample_metadata_collection, args.contig_count_distribution_collection_files)

    # generate intervals metadata
    intervals_metadata: gcnvkernel.IntervalListMetadata = gcnvkernel.IntervalListMetadata(interval_list)

    # inject ploidy-state priors map to the dictionary of parsed args
    args_dict = args.__dict__
    args_dict['ploidy_state_priors_map'] = ploidy_state_priors_map

    # setup the case ploidy inference task
    ploidy_config = gcnvkernel.PloidyModelConfig.from_args_dict(args_dict)
    ploidy_inference_params = gcnvkernel.HybridInferenceParameters.from_args_dict(args_dict)
    ploidy_workspace = gcnvkernel.PloidyWorkspace(ploidy_config, intervals_metadata, sample_names,
                                                  sample_metadata_collection)
    ploidy_task = gcnvkernel.CasePloidyInferenceTask(
        ploidy_inference_params, ploidy_config, ploidy_workspace, args.input_model_path)

    # go!
    ploidy_task.engage()
    ploidy_task.disengage()

    # sample sample-specific posteriors
    gcnvkernel.io_ploidy.SamplePloidyWriter(
        ploidy_config, ploidy_workspace, ploidy_task.continuous_model,
        ploidy_task.continuous_model_approx, args.output_calls_path)()
