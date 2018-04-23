package org.broadinstitute.hellbender.tools.copynumber.formats.collections;

import org.apache.commons.collections4.ListUtils;
import org.broadinstitute.hellbender.exceptions.GATKException;
import org.broadinstitute.hellbender.tools.copynumber.DetermineGermlineContigPloidy;
import org.broadinstitute.hellbender.tools.copynumber.formats.metadata.LocatableMetadata;
import org.broadinstitute.hellbender.tools.copynumber.formats.metadata.Metadata;
import org.broadinstitute.hellbender.tools.copynumber.formats.records.ContigToCoverageDistributionMap;
import org.broadinstitute.hellbender.utils.tsv.TableColumnCollection;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * Represents a sequence dictionary and coverage distributions for each contig in an ordered set associated with a cohort of named samples.
 * Should only be used to write temporary files in {@link DetermineGermlineContigPloidy}.
 *
 * @author Samuel Lee &lt;slee@broadinstitute.org&gt;
 */
public final class CoveragePerContigCollection extends AbstractRecordCollection<LocatableMetadata, ContigToCoverageDistributionMap> {
    private static final String SAMPLE_NAME_TABLE_COLUMN = "SAMPLE_NAME";

    public CoveragePerContigCollection(final LocatableMetadata metadata,
                                       final List<ContigToCoverageDistributionMap> contigToCoverageDistributionMaps,
                                       final List<String> contigs) {
        super(
                metadata,
                contigToCoverageDistributionMaps,
                new TableColumnCollection(ListUtils.union(Collections.singletonList(SAMPLE_NAME_TABLE_COLUMN), contigs)),
                dataLine -> new ContigToCoverageDistributionMap(
                        dataLine.get(SAMPLE_NAME_TABLE_COLUMN),
                        contigs.stream().collect(Collectors.toMap(
                                Function.identity(),
                                dataLine::getInt,
                                (u, v) -> {
                                    throw new GATKException.ShouldNeverReachHereException("Cannot have duplicate contigs.");
                                },   //contigs should already be distinct
                                LinkedHashMap::new))),
                (contigToCoverageDistributionMap, dataLine) -> {
                    dataLine.append(contigToCoverageDistributionMap.getSampleName());
                    contigs.stream().map(contigToCoverageDistributionMap::getCoverage).forEach(dataLine::append);
                });
    }

    @Override
    protected Metadata.Type getMetadataType() {
        return Metadata.Type.LOCATABLE;
    }
}
