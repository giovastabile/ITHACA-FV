/*---------------------------------------------------------------------------*\
  ITHACA-FV
  Common sampled-submesh utility for DEIM and hyper-reduced projections.
\*---------------------------------------------------------------------------*/

#include "SampledMesh.H"
#include "PstreamReduceOps.H"
#include <algorithm>

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


void SampledMesh::writeMask
(
    const word& fieldName
) const
{
    volScalarField hrMask
    (
        IOobject
        (
            fieldName,
            mesh_.time().timeName(),
            mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh_,
        dimensionedScalar("zero", dimless, 0.0)
    );

    // First mark the entire retained submesh (sampled cells + halo)
    // as 0.5.
    for (label celli = 0; celli < mesh_.nCells(); ++celli)
    {
        if (selectedCells_.test(celli))
        {
            hrMask[celli] = 0.5;
        }
    }

    // Then overwrite the true sampled/collocation cells with 1.0.
    forAll(sampledCells_, sampleI)
    {
        const label celli = sampledCells_[sampleI];

        if (celli >= 0 && celli < mesh_.nCells())
        {
            hrMask[celli] = 1.0;
        }
    }

    hrMask.write();

    label nSamples = sampledCells_.size();
    label nSubmesh = selectedCells_.count();

    reduce(nSamples, sumOp<label>());
    reduce(nSubmesh, sumOp<label>());

    Info<< "SampledMesh: wrote diagnostic mask '" << fieldName << "'" << nl
        << "    0.0 = outside sampled submesh" << nl
        << "    0.5 = halo/submesh cell" << nl
        << "    1.0 = true sampled cell" << nl
        << "    global sampled cells = " << nSamples << nl
        << "    global submesh cells = " << nSubmesh << endl;
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



labelList SampledMesh::residualBasedSamples
(
    const scalarField& indicator,
    const fvMesh& mesh,
    const scalar fraction,
    const label exclusionLayers
)
{
    if (indicator.size() != mesh.nCells())
    {
        FatalErrorInFunction
            << "Indicator size (" << indicator.size()
            << ") differs from mesh size (" << mesh.nCells() << ")"
            << exit(FatalError);
    }

    if (fraction <= 0.0 || fraction > 1.0)
    {
        FatalErrorInFunction
            << "Sampling fraction must be in (0,1], got "
            << fraction
            << exit(FatalError);
    }

    if (exclusionLayers < 0)
    {
        FatalErrorInFunction
            << "exclusionLayers must be non-negative, got "
            << exclusionLayers
            << exit(FatalError);
    }

    if (Pstream::parRun() && exclusionLayers > 0)
    {
        FatalErrorInFunction
            << "The residual-based greedy spacing selector currently "
            << "supports exclusionLayers > 0 only in serial. "
            << "Use exclusionLayers=0 in parallel."
            << exit(FatalError);
    }

    List<label> order(mesh.nCells());

    forAll(order, i)
    {
        order[i] = i;
    }

    std::sort
    (
        order.begin(),
        order.end(),
        [&indicator](const label a, const label b)
        {
            return indicator[a] > indicator[b];
        }
    );

const label target =
    std::max<label>
    (
        1,
        std::min<label>
        (
            mesh.nCells(),
            label(fraction*mesh.nCells() + 0.5)
        )
    );

    DynamicList<label> selected(target);
    bitSet blocked(mesh.nCells());

    forAll(order, rankI)
    {
        if (selected.size() >= target)
        {
            break;
        }

        const label celli = order[rankI];

        if (blocked.test(celli))
        {
            continue;
        }

        selected.append(celli);

        if (exclusionLayers == 0)
        {
            blocked.set(celli);
        }
        else
        {
            labelList seed(1);
            seed[0] = celli;

            const bitSet exclusion =
                cellSelection(mesh, seed, exclusionLayers);

            blocked |= exclusion;
        }
    }

    selected.shrink();

    Info<< "Residual-based sampling selected "
        << selected.size() << " cells from "
        << mesh.nCells() << " local cells"
        << " (requested fraction = " << fraction
        << ", exclusionLayers = " << exclusionLayers << ")"
        << endl;

    return labelList(selected);
}


labelList SampledMesh::addPhysicalBoundaryCells
(
    const fvMesh& mesh,
    const labelList& sampledCells
)
{
    labelHashSet selected(2*sampledCells.size() + 128);

    forAll(sampledCells, i)
    {
        selected.insert(sampledCells[i]);
    }

    const polyBoundaryMesh& patches = mesh.boundaryMesh();

    forAll(patches, patchI)
    {
        const polyPatch& pp = patches[patchI];

        // Processor/cyclic/etc. are not physical boundaries.
        if (pp.coupled())
        {
            continue;
        }

        // Do not sample the front/back of a 2-D case.
        if (pp.type() == "empty")
        {
            continue;
        }

        const labelUList& faceCells = pp.faceCells();

        forAll(faceCells, faceI)
        {
            selected.insert(faceCells[faceI]);
        }
    }

    labelList result(selected.toc());
    Foam::sort(result);

    Info<< "After adding physical-boundary cells: "
        << result.size() << " local sampled cells" << endl;

    return result;
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
