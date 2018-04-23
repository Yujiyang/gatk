package org.broadinstitute.hellbender.tools.copynumber.formats.records;

import org.broadinstitute.hellbender.tools.copynumber.DetermineGermlineContigPloidy;
import org.broadinstitute.hellbender.tools.copynumber.formats.collections.SimpleCountCollection;
import org.broadinstitute.hellbender.tools.copynumber.formats.metadata.SampleMetadata;
import org.broadinstitute.hellbender.tools.copynumber.formats.metadata.SimpleSampleMetadata;
import org.broadinstitute.hellbender.utils.Utils;
import org.broadinstitute.hellbender.utils.param.ParamUtils;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Represents coverage distributions for each contig in an ordered set associated with a named sample.
 * Should only be used to write temporary files in {@link DetermineGermlineContigPloidy}.
 *
 * @author Samuel Lee &lt;slee@broadinstitute.org&gt;
 */
public final class ContigToCoverageDistributionMap {
    private final SampleMetadata metadata;
    private final int maximumCount;
    private final LinkedHashMap<String, Map<Integer, Integer>> contigToCoverageDistributionMap;

    public ContigToCoverageDistributionMap(final SimpleCountCollection readCounts,
                                           final int maximumCount) {
        Utils.nonNull(readCounts);
        ParamUtils.isPositiveOrZero(maximumCount, "Maximum count must be non-negative.");
        this.metadata = new SimpleSampleMetadata(readCounts.getMetadata().getSampleName());
        this.maximumCount = maximumCount;
        this.contigToCoverageDistributionMap = readCounts.getRecords().stream()
                .filter(c -> c.getCount() <= maximumCount)
                .collect(Collectors.groupingBy(
                        SimpleCount::getContig,
                        LinkedHashMap::new,
                        Collectors.groupingBy(
                                SimpleCount::getCount,
                                LinkedHashMap::new,
                                Collectors.summingInt(c -> 1))));
    }

    public SampleMetadata getMetadata() {
        return metadata;
    }

    public int getMaximumCount() {
        return maximumCount;
    }

    public Map<Integer, Integer> getCoverageDistribution(final String contig) {
        return contigToCoverageDistributionMap.get(contig);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }

        final ContigToCoverageDistributionMap that = (ContigToCoverageDistributionMap) o;

        return maximumCount == that.maximumCount &&
                metadata.equals(that.metadata) &&
                contigToCoverageDistributionMap.equals(that.contigToCoverageDistributionMap);
    }

    @Override
    public int hashCode() {
        int result = metadata.hashCode();
        result = 31 * result + maximumCount;
        result = 31 * result + contigToCoverageDistributionMap.hashCode();
        return result;
    }
}
