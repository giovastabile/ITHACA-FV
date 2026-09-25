/*---------------------------------------------------------------------------*\
  ITHACA-FV
  Common sampled-submesh utility for DEIM and hyper-reduced projections.
\*---------------------------------------------------------------------------*/

#include "SampledMesh.H"
#include "PstreamReduceOps.H"

namespace Foam
{

SampledMesh::SampledMesh
(
    const fvMesh& mesh,
    const labelList& sampledCells,
    const label layers,
    const bool readDictionaries
)
:
    mesh_(mesh),
    sampledCells_(sampledCells),
    layers_(layers),
    selectedCells_(mesh.nCells()),
    subset_(new fvMeshSubset(mesh)),
    sampledSubCells_(sampledCells.size(), -1),
    readDictionaries_(readDictionaries)
{
    build();
}


void SampledMesh::build()
{
    selectedCells_ =
        cellSelection
        (
            mesh_,
            sampledCells_,
            layers_
        );

    setSubset
    (
        subset_(),
        mesh_,
        selectedCells_,
        readDictionaries_
    );

    sampledSubCells_ =
        mapToSubmesh
        (
            sampledCells_,
            subset_(),
            true
        );
}


label SampledMesh::globalSubMeshSize() const
{
    label n = subset_->subMesh().nCells();
    reduce(n, sumOp<label>());
    return n;
}


bitSet SampledMesh::seedSelection
(
    const fvMesh& mesh,
    const labelList& sampledCells
)
{
    bitSet selected(mesh.nCells());

    forAll(sampledCells, sampleI)
    {
        const label celli = sampledCells[sampleI];

        if (celli < 0 || celli >= mesh.nCells())
        {
            FatalErrorInFunction
                << "Invalid sampled cell " << celli
                << " on processor " << Pstream::myProcNo()
                << ". Local mesh contains " << mesh.nCells()
                << " cells."
                << exit(FatalError);
        }

        selected.set(celli);
    }

    return selected;
}


bitSet SampledMesh::expandSelection
(
    const fvMesh& mesh,
    const bitSet& initialSelection,
    const label layers
)
{
    if (layers < 0)
    {
        FatalErrorInFunction
            << "Number of stencil layers must be non-negative, got "
            << layers
            << exit(FatalError);
    }

    if (initialSelection.size() != mesh.nCells())
    {
        FatalErrorInFunction
            << "Selection size (" << initialSelection.size()
            << ") differs from local mesh size (" << mesh.nCells() << ")"
            << exit(FatalError);
    }

    bitSet selected(initialSelection);

    const labelUList& owner = mesh.faceOwner();
    const labelUList& neighbour = mesh.faceNeighbour();
    const polyBoundaryMesh& patches = mesh.boundaryMesh();

    for (label layer = 0; layer < layers; ++layer)
    {
        bitSet expanded(selected);

        // Internal face-neighbours.
        for (label facei = 0; facei < mesh.nInternalFaces(); ++facei)
        {
            const label own = owner[facei];
            const label nei = neighbour[facei];

            if (selected.test(own))
            {
                expanded.set(nei);
            }

            if (selected.test(nei))
            {
                expanded.set(own);
            }
        }

        // Coupled face-neighbours (processor, cyclic, ...).
        //
        // Store the selected state of the local face-cell, then swap it
        // across coupled patches. After the swap, boundarySelected contains
        // the state of the cell on the opposite side of the coupled face.
        labelList boundarySelected(mesh.nBoundaryFaces(), 0);

        forAll(patches, patchI)
        {
            const polyPatch& pp = patches[patchI];

            if (!pp.coupled())
            {
                continue;
            }

            const labelUList& faceCells = pp.faceCells();

            forAll(faceCells, patchFaceI)
            {
                const label facei = pp.start() + patchFaceI;
                const label bFaceI = facei - mesh.nInternalFaces();

                boundarySelected[bFaceI] =
                    selected.test(faceCells[patchFaceI]) ? 1 : 0;
            }
        }

        syncTools::swapBoundaryFaceList(mesh, boundarySelected);

        forAll(patches, patchI)
        {
            const polyPatch& pp = patches[patchI];

            if (!pp.coupled())
            {
                continue;
            }

            const labelUList& faceCells = pp.faceCells();

            forAll(faceCells, patchFaceI)
            {
                const label facei = pp.start() + patchFaceI;
                const label bFaceI = facei - mesh.nInternalFaces();

                if (boundarySelected[bFaceI])
                {
                    expanded.set(faceCells[patchFaceI]);
                }
            }
        }

        selected = expanded;
    }

    return selected;
}


bitSet SampledMesh::cellSelection
(
    const fvMesh& mesh,
    const labelList& sampledCells,
    const label layers
)
{
    return
        expandSelection
        (
            mesh,
            seedSelection(mesh, sampledCells),
            layers
        );
}


labelList SampledMesh::cellLabels
(
    const fvMesh& mesh,
    const labelList& sampledCells,
    const label layers
)
{
    const bitSet selected =
        cellSelection
        (
            mesh,
            sampledCells,
            layers
        );

    DynamicList<label> labels(selected.count());

    for (label celli = 0; celli < mesh.nCells(); ++celli)
    {
        if (selected.test(celli))
        {
            labels.append(celli);
        }
    }

    labels.shrink();

    return labelList(labels);
}


void SampledMesh::setSubset
(
    fvMeshSubset& subset,
    const fvMesh& mesh,
    const bitSet& selectedCells,
    const bool readDictionaries
)
{
#if OPENFOAM >= 1812
    // syncPar=true is essential when the selected region crosses coupled
    // processor boundaries.
    subset.setCellSubset
    (
        selectedCells,
        -1,
        true
    );
#else
    labelHashSet selectedSet;

    for (label celli = 0; celli < mesh.nCells(); ++celli)
    {
        if (selectedCells.test(celli))
        {
            selectedSet.insert(celli);
        }
    }

    subset.setLargeCellSubset(selectedSet);
#endif

    if (readDictionaries)
    {
        fvMesh& subMesh = subset.subMesh();

        subMesh.fvSchemes::readOpt() =
            mesh.fvSchemes::readOpt();

        subMesh.fvSolution::readOpt() =
            mesh.fvSolution::readOpt();

        subMesh.fvSchemes::read();
        subMesh.fvSolution::read();
    }
}


void SampledMesh::setSubset
(
    fvMeshSubset& subset,
    const fvMesh& mesh,
    const labelList& selectedCells,
    const bool readDictionaries
)
{
    bitSet selected(mesh.nCells());

    forAll(selectedCells, i)
    {
        const label celli = selectedCells[i];

        if (celli < 0 || celli >= mesh.nCells())
        {
            FatalErrorInFunction
                << "Invalid selected cell " << celli
                << " on processor " << Pstream::myProcNo()
                << exit(FatalError);
        }

        selected.set(celli);
    }

    setSubset
    (
        subset,
        mesh,
        selected,
        readDictionaries
    );
}


labelList SampledMesh::mapToSubmesh
(
    const labelList& sampledCells,
    const fvMeshSubset& subset,
    const bool fatalIfMissing
)
{
    const labelList& cellMap = subset.cellMap();

    labelList sampledSubCells
    (
        sampledCells.size(),
        -1
    );

    // fullToSub avoids O(Nsample * Nsub) repeated scans.
    label maxFullCell = -1;

    forAll(cellMap, subCellI)
    {
        maxFullCell = max(maxFullCell, cellMap[subCellI]);
    }

    labelList fullToSub(maxFullCell + 1, -1);

    forAll(cellMap, subCellI)
    {
        const label fullCell = cellMap[subCellI];

        if (fullCell >= 0)
        {
            fullToSub[fullCell] = subCellI;
        }
    }

    forAll(sampledCells, sampleI)
    {
        const label fullCell = sampledCells[sampleI];

        if (fullCell >= 0 && fullCell < fullToSub.size())
        {
            sampledSubCells[sampleI] = fullToSub[fullCell];
        }

        if
        (
            fatalIfMissing
         && sampledSubCells[sampleI] < 0
        )
        {
            FatalErrorInFunction
                << "Sampled cell " << fullCell
                << " is not present in fvMeshSubset on processor "
                << Pstream::myProcNo()
                << exit(FatalError);
        }
    }

    return sampledSubCells;
}

} // End namespace Foam
