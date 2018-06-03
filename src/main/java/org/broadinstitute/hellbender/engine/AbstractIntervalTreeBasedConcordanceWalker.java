package org.broadinstitute.hellbender.engine;

import htsjdk.samtools.SAMSequenceDictionary;
import htsjdk.tribble.Feature;
import htsjdk.variant.variantcontext.VariantContext;
import htsjdk.variant.variantcontext.VariantContextComparator;
import htsjdk.variant.vcf.VCFHeader;
import org.apache.commons.collections4.Predicate;
import org.broadinstitute.barclay.argparser.Argument;
import org.broadinstitute.hellbender.cmdline.StandardArgumentDefinitions;
import org.broadinstitute.hellbender.exceptions.GATKException;
import org.broadinstitute.hellbender.tools.spark.sv.utils.SVInterval;
import org.broadinstitute.hellbender.tools.spark.sv.utils.SVIntervalTree;
import org.broadinstitute.hellbender.tools.walkers.validation.ConcordanceState;
import org.broadinstitute.hellbender.utils.SimpleInterval;
import org.broadinstitute.hellbender.utils.gcs.BucketUtils;
import org.broadinstitute.hellbender.utils.io.IOUtils;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

public abstract class AbstractIntervalTreeBasedConcordanceWalker extends GATKTool {

    public static final String TRUTH_VARIANTS_LONG_NAME = "truth";

    public static final String CONFIDENCE_REGION_LONG_NAME = "confidence";
    public static final String CONFIDENCE_REGION_SHORT_NAME = "C";

    // The distance in bases to look ahead and cache when querying feature sources.
    public static final int CACHE_LOOKAHEAD = 100_000;

    @Argument(shortName = TRUTH_VARIANTS_LONG_NAME, fullName = TRUTH_VARIANTS_LONG_NAME,
            doc = "A VCF containing truth variants", optional = false)
    public String truthVariantsFile;

    @Argument(doc = "TO BE IMPLEMENTED",
            fullName= CONFIDENCE_REGION_LONG_NAME,
            shortName = CONFIDENCE_REGION_SHORT_NAME,
            optional = true)
    protected String highConfidenceRegion;

    @Argument(doc = "path to a file holding mask on region to be excluded", fullName = "ref-region-mask-file", optional = true)
    protected String refRegionMaskFilePath;

    @Argument(fullName = StandardArgumentDefinitions.VARIANT_LONG_NAME, shortName = StandardArgumentDefinitions.VARIANT_SHORT_NAME,
            doc = "One or more VCF files containing call sets to be evaluated", common = false, optional = false)
    public List<String> evalVariantsFileList = new ArrayList<>(2);

    //////
    private FeatureDataSource<VariantContext> truthVariants;
    protected SVIntervalTree<List<TruthVariant>> truthTree;
    protected SVIntervalTree<String> maskedOut;
    private MultiVariantDataSource allEvalVariants;
    protected SAMSequenceDictionary refSeqDict;

    private VariantContextComparator variantContextComparator;

    /**
     * For testing two objects of type T, using information stored in object of type R.
     * See example in {@link #getConcordanceTester()} ()}
     */
    @FunctionalInterface
    public interface TriPredicate<T, R> {

        Boolean test(T t1, T t2, R r);

        default TriPredicate<T, R> and(final TriPredicate<? super T, ? super R> other) {
            Objects.requireNonNull(other);
            return (T t1, T t2, R r) -> test(t1, t2, r) && other.test(t1, t2, r);
        }

        default TriPredicate<T, R> negate() {
            return (T t1, T t2, R r) -> !test(t1, t2, r);
        }

        default TriPredicate<T, R> or(final TriPredicate<? super T, ? super R> other) {
            Objects.requireNonNull(other);
            return (T t1, T t2, R r) -> test(t1, t2, r) || other.test(t1, t2, r);
        }
    }

    protected TriPredicate<VariantContext, ReferenceContext> variantConcordanceTester;

    // =================================================================================================================

    @Override
    public final SAMSequenceDictionary getBestAvailableSequenceDictionary() {
        return refSeqDict;
    }

    public final VCFHeader getTruthHeader() { return getHeader(truthVariants); }

    public final VCFHeader getEvalHeader() {
        return allEvalVariants.getHeader();
    }

    private static VCFHeader getHeader(final FeatureDataSource<VariantContext> source) {
        final Object header = source.getHeader();
        if ( ! (header instanceof VCFHeader) ) {
            throw new GATKException("Header for " + source.getName() + " is not in VCF header format");
        }
        return (VCFHeader)header;
    }

    // INITIALIZATION, TRAVERSAL, AND TERMINATION ======================================================================

    @Override
    protected final void onStartup() {
        super.onStartup();

        IOUtils.assertFileIsReadable(Paths.get(truthVariantsFile));
        evalVariantsFileList.forEach(s ->  IOUtils.assertFileIsReadable(Paths.get(s)));

        features = new FeatureManager(this, FeatureDataSource.DEFAULT_QUERY_LOOKAHEAD_BASES,
                cloudPrefetchBuffer, cloudIndexPrefetchBuffer, referenceArguments.getReferencePath());

        truthVariants = new FeatureDataSource<>(new FeatureInput<>(truthVariantsFile, "truth"), CACHE_LOOKAHEAD, VariantContext.class);
        refSeqDict = truthVariants.getSequenceDictionary();

        // sorry, I cannot find the BED file reader.....
        maskedOut = new SVIntervalTree<>();
        if (refRegionMaskFilePath != null) {
            try ( final BufferedReader rdr =
                          new BufferedReader(new InputStreamReader(BucketUtils.openFile(refRegionMaskFilePath))) ) {
                String line;
                while ( (line = rdr.readLine()) != null ) {
                    String[] bedLine = line.split("\t");
                    maskedOut.put(new SVInterval(refSeqDict.getSequenceIndex(bedLine[0]),
                            Integer.valueOf(bedLine[1]), Integer.valueOf(bedLine[2])), "");
                }
            } catch (final IOException ioe ) {
                throw new GATKException("Unable to read intervals from " + refRegionMaskFilePath, ioe);
            }
        }

        truthTree = buildIntervalTreeFromTruth();

        initializeEvalVariantsList();

        if ( hasIntervals() ) {
            truthVariants.setIntervalsForTraversal(intervalsForTraversal);
            allEvalVariants.setIntervalsForTraversal(intervalsForTraversal);
        }
        variantContextComparator = getVariantContextComparator(refSeqDict);
        variantConcordanceTester = getConcordanceTester();
    }

    private void initializeEvalVariantsList() {
        final List<FeatureInput<VariantContext>> evalVariantsFeatureInputs = new ArrayList<>(2);
        evalVariantsFileList.forEach(
                f -> {
                    final FeatureInput<VariantContext> featureInput = new FeatureInput<>(f);
                    if (evalVariantsFeatureInputs.contains(featureInput)) {
                        throw new IllegalArgumentException("Feature inputs must be unique: " + featureInput.toString());
                    }
                    evalVariantsFeatureInputs.add(featureInput);
                    features.addToFeatureSources(0, featureInput, VariantContext.class, cloudPrefetchBuffer, cloudIndexPrefetchBuffer,
                            referenceArguments.getReferencePath());
                }
        );

        allEvalVariants = new MultiVariantDataSource(evalVariantsFeatureInputs, CACHE_LOOKAHEAD, cloudPrefetchBuffer, cloudIndexPrefetchBuffer,
                referenceArguments.getReferencePath());
    }

    /**
     * Evaluate each variant, both from the evaluation set and the truth set.
     *
     * One can override {@link #onTraversalStart()} for additional initialization
     * One can override {@link #onTraversalSuccess()} for additional post-traversal tasks
     */
    @Override
    public final void traverse() {
        int cnt = 0;
        final Predicate<VariantContext> evalPredicate = makeEvalVariantFilter();
        for (final VariantContext evalVariant : allEvalVariants) {
            if ( ! evalPredicate.evaluate(evalVariant)) {++cnt;continue;}
            final SimpleInterval variantInterval = new SimpleInterval(evalVariant);
            apply(evalVariant, new ReadsContext(reads, variantInterval), new ReferenceContext(reference, variantInterval));
        }
        logger.info(cnt + " eval variants skipped evaluation");
    }

    @Override
    protected final void onShutdown() {
        super.onShutdown();
        if( truthVariants != null ) {
            truthVariants.close();
        }
        if( allEvalVariants != null) {
            allEvalVariants.close();
        }
    }

    // CORE CONCEPT OF THIS WALKER =====================================================================================

    public interface ClassifiedVariant extends Feature {
        ConcordanceState getConcordanceState();
    }
    public interface TruthVariant extends ClassifiedVariant {
        VariantContext getTruth();
    }
    public interface EvalVariant extends ClassifiedVariant {
        VariantContext getEval();
    }

    public final class FalseNegative implements TruthVariant {
        private final VariantContext truth;
        private float supportingScore = 0f;

        public FalseNegative(final VariantContext truth) {
            this.truth = truth;
        }

        @Override
        public VariantContext getTruth() {
            return truth;
        }
        @Override
        public String getContig() {
            return truth.getContig();
        }
        @Override
        public int getStart() {
            return truth.getStart();
        }
        @Override
        public int getEnd() {
            return truth.getEnd();
        }

        @Override
        public ConcordanceState getConcordanceState() {
            return ConcordanceState.FALSE_NEGATIVE;
        }

        public float getSupportingScore() {
            return supportingScore;
        }

        public void setSupportingScore(final float supportingScore) {
            this.supportingScore = supportingScore;
        }
    }

    public static abstract class EvalOnlyVariant implements EvalVariant {
        private final VariantContext eval;

        EvalOnlyVariant(final VariantContext eval) {
            this.eval = eval;
        }

        @Override
        public VariantContext getEval() {
            return eval;
        }
        @Override
        public String getContig() {
            return eval.getContig();
        }
        @Override
        public int getStart() {
            return eval.getStart();
        }
        @Override
        public int getEnd() {
            return eval.getEnd();
        }
    }

    public static final class FalsePositive extends EvalOnlyVariant {
        private ReasonForFalsePositive reasonForFalsePositive;

        public FalsePositive(final VariantContext eval, final ReasonForFalsePositive reasonForFalsePositive) {
            super(eval);
            this.reasonForFalsePositive = reasonForFalsePositive;
        }

        public ConcordanceState getConcordanceState() {
            return ConcordanceState.FALSE_POSITIVE;
        }

        public ReasonForFalsePositive getReasonForFalsePositive() {
            return reasonForFalsePositive;
        }

        public enum ReasonForFalsePositive {
            NO_OVERLAPPING_TRUTH("NO_OVP"),
            OVERLAPPING_TRUTH_WITH_DIFF_TYPE_OR_LOW_SUPPORT("WR_TYPE_OR_LOW_SUPP");

            private final String abbreviation;

            ReasonForFalsePositive(final String abbreviation) {
                this.abbreviation = abbreviation;
            }

            public String getAbbreviation() { return abbreviation; }
        }
    }

    public final class FilteredTrueNegative extends EvalOnlyVariant {
        public FilteredTrueNegative(final VariantContext eval) {
            super(eval);
        }

        public ConcordanceState getConcordanceState() {
            return ConcordanceState.FILTERED_TRUE_NEGATIVE;
        }
    }

    public static final class SupportingTruth {

        public final VariantContext truthVar;
        public final float maxSupportScore;

        public SupportingTruth(final VariantContext truthVar, final float maxSupportScore) {
            this.truthVar = truthVar;
            this.maxSupportScore = maxSupportScore;
        }
    }
    public static abstract class BiVariant implements TruthVariant, EvalVariant {
        private final VariantContext eval;
        private final SupportingTruth supportingTruth;

        BiVariant(final VariantContext eval, final SupportingTruth supportingTruth) {
            this.eval = eval;
            this.supportingTruth = supportingTruth;
        }

        @Override
        public final VariantContext getTruth() {
            return supportingTruth.truthVar;
        }

        public final float getSupportScore() {
            return supportingTruth.maxSupportScore;
        }

        @Override
        public final VariantContext getEval() {
            return eval;
        }
        @Override
        public final String getContig() {
            return eval.getContig();
        }
        @Override
        public final int getStart() {
            return eval.getStart();
        }
        @Override
        public final int getEnd() {
            return eval.getEnd();
        }
    }

    public static final class TruePositive extends BiVariant {

        public TruePositive(final VariantContext eval, final SupportingTruth supportingTruth) {
            super(eval, supportingTruth);
        }

        public final ConcordanceState getConcordanceState() {
            return ConcordanceState.TRUE_POSITIVE;
        }
    }

    public static final class FilteredFalseNegative extends BiVariant {

        public FilteredFalseNegative(final VariantContext eval, final SupportingTruth supportingTruth) {
            super(eval, supportingTruth);
        }

        public final ConcordanceState getConcordanceState() {
            return ConcordanceState.FILTERED_FALSE_NEGATIVE;
        }
    }

    // OVERRIDE HIGHLY DESIRABLE =======================================================================================

    // classify {@code eval}
    protected abstract void apply(final VariantContext eval, final ReadsContext readsContext, final ReferenceContext refContext);

    /**
     * Pad provided eval call accordingly for overlapping with the truth interval tree.
     */
    protected abstract SVInterval getPaddedSvInterval(final VariantContext eval);

    /**
     * Override this to customize the degree of agreement required to call a true positive.
     * Sometimes, for example, we may want just a single alt allele to agree
     * and sometime we may require all alt alleles to match.
     */
    protected abstract TriPredicate<VariantContext, ReferenceContext> getConcordanceTester();

    /**
     * Computes a score of concordance between the eval and truth.
     * This is particularly useful for variants that spans a large interval, i.e. CNV and SV calls.
     */
    protected abstract float computeConcordanceScore(final VariantContext eval, final VariantContext truth, final ReferenceContext context);

    /**
     * Tools could override how padding are added,
     * if any padding is desired.
     */
    protected SVIntervalTree<List<TruthVariant>> buildIntervalTreeFromTruth() {
        final SVIntervalTree<List<TruthVariant>> truthTree = new SVIntervalTree<>();
        final Predicate<VariantContext> variantContextPredicate = makeTruthVariantFilter();
        truthVariants.forEach(truth -> {
            if (!variantContextPredicate.evaluate(truth)) return;
            final SVInterval svInterval = new SVInterval(refSeqDict.getSequenceIndex(truth.getContig()),
                   truth.getStart() - 1, truth.getEnd());
            SVIntervalTree.Entry<List<TruthVariant>> listEntry = truthTree.find(svInterval);
            // pre-classify every truth call as FN, that is, without matching eval call, later could be modified accordingly by apply(...)
            final List<TruthVariant> value;
            if (listEntry == null) {
                value = Arrays.asList(new FalseNegative(truth));
            } else {
                value = new ArrayList<>(listEntry.getValue()); // listEntry is immutable
                value.add(new FalseNegative(truth));
            }
            truthTree.put(svInterval, value);
        });
        return truthTree;
    }

    /**
     * For comparing variants that don't overlap
     */
    protected VariantContextComparator getVariantContextComparator(final SAMSequenceDictionary dict) {
        return new VariantContextComparator(dict);
    }

    protected Predicate<VariantContext> makeTruthVariantFilter() {
        return vc -> !vc.isFiltered();
    }

    /**
     * Override to filter out variants that should not be considered.
     * An example would be variants that are below the size threshold of 50 bp for SV call set evaluation.
     */
    protected Predicate<VariantContext> makeEvalVariantFilter() { return vc -> true; }
}
