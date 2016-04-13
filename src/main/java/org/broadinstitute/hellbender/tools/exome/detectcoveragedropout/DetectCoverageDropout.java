package org.broadinstitute.hellbender.tools.exome.detectcoveragedropout;

import org.broadinstitute.hellbender.cmdline.Argument;
import org.broadinstitute.hellbender.cmdline.CommandLineProgram;
import org.broadinstitute.hellbender.cmdline.CommandLineProgramProperties;
import org.broadinstitute.hellbender.cmdline.programgroups.CopyNumberProgramGroup;
import org.broadinstitute.hellbender.exceptions.UserException;
import org.broadinstitute.hellbender.tools.exome.*;

import java.io.*;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Command line tool to detect whether a sample has the "rainfall" problem.
 *
 * @author lichtens &lt;lichtens@broadinstitute.org&gt;
 */
@CommandLineProgramProperties(
        summary = "Detects whether there is an artificial dropout (or increase) in target coverage relative to what was " +
                " expected in the PoN.  This is a good measure of whether the PoN matches the case samples.  In " +
                "previous presentations and discussions with the Clinical Sequencing Platform, this was referred to" +
                " as the 'rainfall issue'.",
        oneLineSummary = "Detect whether a sample has coverage dropouts using segmentation (CNV) results.",
        programGroup = CopyNumberProgramGroup.class
)
public final class DetectCoverageDropout extends CommandLineProgram {
    protected static final String SEGFILE_SHORT_NAME = "S";
    protected static final String SEGFILE_LONG_NAME = "segments";

    protected static final String TARGET_FILE_SHORT_NAME = "T";
    protected static final String TARGET_FILE_LONG_NAME = "targets";

    protected static final String OUTPUT_FILE_SHORT_NAME = "o";
    protected static final String OUTPUT_FILE_LONG_NAME = "outputFile";

    protected static final String SEG_THRESHOLD_SHORT_NAME = "st";
    protected static final String SEG_THRESHOLD_LONG_NAME = "segThreshold";

    protected static final String MIN_WEIGHT_SHORT_NAME = "m";
    protected static final String MIN_WEIGHT_LONG_NAME = "minWeight";

    protected static final String MIN_DIST_SHORT_NAME = "mmd";
    protected static final String MIN_DIST_LONG_NAME = "minMeanDist";

    protected static final String WARNING_FILE = "WARNING_PON_MAY_NOT_MATCH_CASE_SAMPLE.";

    @Argument(
            doc = "Target file with copy ratio estimates.  Preferably, the *.tn.*.tsv file as generated by GATK CNV.",
            shortName = TARGET_FILE_SHORT_NAME,
            fullName = TARGET_FILE_LONG_NAME,
            optional = false
    )
    protected File targetsFile;

    @Argument(
            doc = "Segment file corresponding to target file.",
            shortName = SEGFILE_SHORT_NAME,
            fullName = SEGFILE_LONG_NAME,
            optional = false
    )
    protected File segmentsFile;

    @Argument(
            doc = "Output filename for result of the detector run.  This file is fairly simple and just lists " +
                    "the decision and the model that was fitted.",
            fullName = OUTPUT_FILE_LONG_NAME,
            shortName = OUTPUT_FILE_SHORT_NAME,
            optional = false
    )
    protected File outputFile;

    @Argument(
            doc = "(Advanced) This tool tags each segment as good or bad (bad being a possible dropout " +
                    "seen in the segment).  This parameter dictates what proportion (post-filtering) must " +
                    "be good in order to avoid being tagged a dropout.",
            fullName = SEG_THRESHOLD_LONG_NAME,
            shortName = SEG_THRESHOLD_SHORT_NAME,
            optional = true
    )
    protected double segmentThreshold = 0.75;

    @Argument(
            doc = "(Advanced) The minimum weight in the two component gaussian mixture model to be considered" +
                    " an actual two component GMM",
            fullName = MIN_WEIGHT_LONG_NAME,
            shortName = MIN_WEIGHT_SHORT_NAME,
            optional = true
    )
    protected double minWeight = 0.02;

    @Argument(
            doc = "(Advanced) The minimum distance in the two component gaussian mixture model means to be considered" +
                    " an actual two component GMM",
            fullName = MIN_DIST_LONG_NAME,
            shortName = MIN_DIST_SHORT_NAME,
            optional = true
    )
    protected double thresholdDistancePerSegment = 0.1;

    @Override
    protected Object doWork(){


        final ReadCountCollection counts;
        try {
           counts = ReadCountCollectionUtils.parse(targetsFile);
        } catch (final IOException e) {
            throw new UserException.CouldNotReadInputFile(targetsFile.getPath(), e);
        }

        final TargetCollection<ReadCountRecord.SingleSampleRecord> targetList = new HashedListTargetCollection<>(
                counts.records().stream().map(ReadCountRecord::asSingleSampleRecord).collect(Collectors.toList()));
        final List<ModeledSegment> segments = SegmentUtils.readModeledSegmentsFromSegmentFile(segmentsFile);

        logger.info("Input files loaded.  Targets: " + targetsFile + " and segments: " + segmentsFile);

        final CoverageDropoutDetector c = new CoverageDropoutDetector();
        final CoverageDropoutResult coverageDropoutResult = c.determineCoverageDropoutDetected(segments, targetList, .003, thresholdDistancePerSegment, segmentThreshold, minWeight);
        CoverageDropoutResult.writeCoverageDropoutResults(Collections.singletonList(coverageDropoutResult), outputFile);
        if(coverageDropoutResult.isCoverageDropout()){
            /**We write a file to let users know they may have the "rainfall" issue without causing the job to fail**/
            writeWarning(new File(outputFile.getAbsoluteFile().getParent()+"/" + WARNING_FILE+outputFile.getName()));
        }
        return 0;
    }

    private void writeWarning(final File outFile) {
        String warning = "The case sample and PoN combination are possibly a bad match. Please examine your " +
                "results and if they look poor (see example: http://gatkforums.broadinstitute.org/discussion/6076/what-causes-this-rainfall-effect" +
                ", you may need to create a user account), create a PoN " +
                "that includes normals sequenced similarly to this case sample";
        try (Writer writer = new BufferedWriter(new OutputStreamWriter(
                new FileOutputStream(outFile), "utf-8"))) {
            writer.write(warning);
        } catch (final IOException ioe) {
            throw new UserException.CouldNotCreateOutputFile(outFile, ioe.getMessage());
        }
    }
}
