/*---------------------------------------------------------------------------*\
  ITHACA-FV
  Common sampled-submesh utility for DEIM and hyper-reduced projections.
\*---------------------------------------------------------------------------*/

#ifndef SampledMesh_H
#define SampledMesh_H

#include "fvCFD.H"
#include "fvMeshSubset.H"
#include "syncTools.H"

namespace Foam
{

class SampledMesh
{
private:

    const fvMesh& mesh_;
    labelList sampledCells_;
    label layers_;

    bitSet selectedCells_;
    autoPtr<fvMeshSubset> subset_;
    labelList sampledSubCells_;

    bool readDictionaries_;

    void build();

public:

    // Construct a reusable sampled mesh from local cell labels owned by
    // the current rank. In parallel all ranks must call this constructor
    // collectively, even if sampledCells is empty on some ranks.
    SampledMesh
    (
        const fvMesh& mesh,
        const labelList& sampledCells,
        const label layers,
        const bool readDictionaries = true
    );

    // No copy: fvMeshSubset owns a mesh and the cached fields should always
    // refer to the same object.
    SampledMesh(const SampledMesh&) = delete;
    void operator=(const SampledMesh&) = delete;

    const fvMesh& mesh() const
    {
        return mesh_;
    }

    const labelList& sampledCells() const
    {
        return sampledCells_;
    }

    const labelList& sampledSubCells() const
    {
        return sampledSubCells_;
    }

    label layers() const
    {
        return layers_;
    }

    const bitSet& selectedCells() const
    {
        return selectedCells_;
    }

    fvMeshSubset& subset()
    {
        return subset_();
    }

    const fvMeshSubset& subset() const
    {
        return subset_();
    }

    fvMesh& subMesh()
    {
        return subset_->subMesh();
    }

    const fvMesh& subMesh() const
    {
        return subset_->subMesh();
    }

    label localSubMeshSize() const
    {
        return subset_->subMesh().nCells();
    }

    label globalSubMeshSize() const;

    // Write one diagnostic scalar mask on the original mesh:
    //  - 0.0 outside the sampled submesh
    //  - 0.5 on halo/submesh cells
    //  - 1.0 on the true sampled/collocation cells
    void writeMask
    (
        const word& fieldName = "hrMask"
    ) const;


    // ---------------------------------------------------------------------
    // Static utilities.
    //
    // They are deliberately exposed so legacy users such as DEIM can reuse
    // exactly the same stencil/mapping logic without changing their public
    // ownership of fvMeshSubset.
    // ---------------------------------------------------------------------

    // Build the sampled-cell seed bitSet.
    static bitSet seedSelection
    (
        const fvMesh& mesh,
        const labelList& sampledCells
    );

    // Expand an existing selection by face-neighbour layers. Coupled
    // boundaries (processor/cyclic/etc.) are traversed using syncTools.
    // This routine is collective in parallel.
    static bitSet expandSelection
    (
        const fvMesh& mesh,
        const bitSet& initialSelection,
        const label layers
    );

    // Convenience: seed + expand.
    static bitSet cellSelection
    (
        const fvMesh& mesh,
        const labelList& sampledCells,
        const label layers
    );

    // Return the selected local cell labels, useful for the legacy DEIM
    // totalMagicPoints/uniqueMagicPoints bookkeeping.
    static labelList cellLabels
    (
        const fvMesh& mesh,
        const labelList& sampledCells,
        const label layers
    );

    // Configure an externally owned fvMeshSubset from a selection. This is
    // the bridge used by DEIM, which already stores autoPtr<fvMeshSubset>.
    static void setSubset
    (
        fvMeshSubset& subset,
        const fvMesh& mesh,
        const bitSet& selectedCells,
        const bool readDictionaries = true
    );

    static void setSubset
    (
        fvMeshSubset& subset,
        const fvMesh& mesh,
        const labelList& selectedCells,
        const bool readDictionaries = true
    );


    // Select the cells with the largest residual indicator.
    //
    // fraction is applied to the local mesh. For the current tutorial this
    // is intended for serial use. If exclusionLayers > 0, already selected
    // cells and their face-neighbour layers are excluded from subsequent
    // picks to avoid strongly clustered samples.
    static labelList residualBasedSamples
    (
        const scalarField& indicator,
        const fvMesh& mesh,
        const scalar fraction,
        const label exclusionLayers = 0
    );

    // Add cells adjacent to all physical, non-empty boundaries.
    // Coupled processor/cyclic patches and empty patches are excluded.
    static labelList addPhysicalBoundaryCells
    (
        const fvMesh& mesh,
        const labelList& sampledCells
    );

    // Original-local cell labels -> subset-local cell labels.
    static labelList mapToSubmesh
    (
        const labelList& sampledCells,
        const fvMeshSubset& subset,
        const bool fatalIfMissing = true
    );
};

} // End namespace Foam

#endif
