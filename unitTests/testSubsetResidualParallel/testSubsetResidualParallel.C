#include "fvCFD.H"
#include "fvMeshSubset.H"
#include "syncTools.H"
#include "PstreamReduceOps.H"

#include <iomanip>

using namespace Foam;

class projectableScalarMatrix
:
    public fvScalarMatrix
{
public:
    projectableScalarMatrix(const fvScalarMatrix& A)
    :
        fvScalarMatrix(A)
    {}

    void prepare()
    {
        addBoundaryDiag(diag(), 0);
    }

    void prepareSource(scalarField& source)
    {
        addBoundarySource(source, false);
    }
};

label ownedNearestCell
(
    const fvMesh& mesh,
    const point& target,
    label& ownerProc
)
{
    label localCell = -1;
    scalar localBest = GREAT;

    forAll(mesh.C(), celli)
    {
        const scalar d = magSqr(mesh.C()[celli] - target);

        if (d < localBest)
        {
            localBest = d;
            localCell = celli;
        }
    }

    scalar globalBest = localBest;
    reduce(globalBest, minOp<scalar>());

    const scalar scale = max(scalar(1.0), mag(globalBest));

    const bool candidate =
        localCell >= 0
     && mag(localBest - globalBest) <= 100*SMALL*scale;

    ownerProc = candidate ? Pstream::myProcNo() : Pstream::nProcs();
    reduce(ownerProc, minOp<label>());

    return Pstream::myProcNo() == ownerProc ? localCell : -1;
}

bitSet expandCellSelectionParallel
(
    const fvMesh& mesh,
    const bitSet& initialSelection,
    const label nLayers
)
{
    bitSet selected(initialSelection);

    const labelUList& owner = mesh.faceOwner();
    const labelUList& neighbour = mesh.faceNeighbour();

    for (label layer = 0; layer < nLayers; ++layer)
    {
        bitSet expanded(selected);

        for (label facei = 0; facei < mesh.nInternalFaces(); ++facei)
        {
            const label own = owner[facei];
            const label nei = neighbour[facei];

            if (selected.test(own)) expanded.set(nei);
            if (selected.test(nei)) expanded.set(own);
        }

        labelList boundarySelected(mesh.nBoundaryFaces(), 0);
        const polyBoundaryMesh& patches = mesh.boundaryMesh();

        forAll(patches, patchI)
        {
            const polyPatch& pp = patches[patchI];
            if (!pp.coupled()) continue;

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
            if (!pp.coupled()) continue;

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

scalarField matrixResidual
(
    fvScalarMatrix& A,
    const volScalarField& field
)
{
    projectableScalarMatrix Ap(A);
    Ap.prepare();

    const scalarField& x = field.primitiveField();
    scalarField Ax(x.size(), 0.0);

    const lduInterfaceFieldPtrsList interfaces
    (
        field.boundaryField().scalarInterfaces()
    );

    Ap.Amul
    (
        Ax,
        x,
        Ap.boundaryCoeffs(),
        interfaces,
        0
    );

    scalarField rhs(A.source());
    Ap.prepareSource(rhs);

    scalarField residual(Ax.size(), 0.0);

    forAll(residual, celli)
    {
        residual[celli] = Ax[celli] - rhs[celli];
    }

    return residual;
}

int main(int argc, char *argv[])
{
    #include "setRootCase.H"

    Time runTime(Time::controlDictName, args);

    fvMesh mesh
    (
        IOobject
        (
            fvMesh::defaultRegion,
            runTime.timeName(),
            runTime,
            IOobject::MUST_READ
        )
    );

    if (Pstream::master())
    {
        std::cout
            << std::setprecision(16)
            << "\n========================================\n"
            << " PARALLEL fvMeshSubset residual test\n"
            << " MPI ranks = " << Pstream::nProcs()
            << "\n========================================\n"
            << std::endl;
    }

    scalar xmin = GREAT, xmax = -GREAT;
    scalar ymin = GREAT, ymax = -GREAT;
    scalar zmin = GREAT, zmax = -GREAT;

    forAll(mesh.C(), celli)
    {
        const vector& C = mesh.C()[celli];

        xmin = min(xmin, C.x()); xmax = max(xmax, C.x());
        ymin = min(ymin, C.y()); ymax = max(ymax, C.y());
        zmin = min(zmin, C.z()); zmax = max(zmax, C.z());
    }

    reduce(xmin, minOp<scalar>()); reduce(xmax, maxOp<scalar>());
    reduce(ymin, minOp<scalar>()); reduce(ymax, maxOp<scalar>());
    reduce(zmin, minOp<scalar>()); reduce(zmax, maxOp<scalar>());

    const scalar Lx = max(xmax - xmin, SMALL);
    const scalar Ly = max(ymax - ymin, SMALL);
    const scalar Lz = max(zmax - zmin, SMALL);
    const scalar pi = constant::mathematical::pi;

    volScalarField T
    (
        IOobject
        (
            "T",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        ),
        mesh
    );

    forAll(T, celli)
    {
        const vector& C = mesh.C()[celli];

        const scalar x = (C.x() - xmin)/Lx;
        const scalar y = (C.y() - ymin)/Ly;
        const scalar z = (C.z() - zmin)/Lz;

        T[celli] =
              Foam::sin(1.31*pi*x)
            + 0.37*Foam::cos(0.83*pi*y)
            + 0.21*Foam::sin(1.17*pi*z)
            + 0.11*x*y;
    }

    T.correctBoundaryConditions();

    List<point> targets(4);

    targets[0] = point(xmin + 0.31*Lx, ymin + 0.41*Ly, zmin + 0.44*Lz);
    targets[1] = point(xmin + 0.48*Lx, ymin + 0.37*Ly, zmin + 0.52*Lz);
    targets[2] = point(xmin + 0.52*Lx, ymin + 0.63*Ly, zmin + 0.47*Lz);
    targets[3] = point(xmin + 0.69*Lx, ymin + 0.58*Ly, zmin + 0.61*Lz);

    labelList targetLocalCell(targets.size(), -1);
    labelList targetOwner(targets.size(), -1);
    bitSet initialSelection(mesh.nCells());

    forAll(targets, targetI)
    {
        label ownerProc = -1;
        const label localCell = ownedNearestCell(mesh, targets[targetI], ownerProc);

        targetOwner[targetI] = ownerProc;

        if (localCell >= 0)
        {
            targetLocalCell[targetI] = localCell;
            initialSelection.set(localCell);
        }
    }

    if (Pstream::master())
    {
        Info << "Sample ownership:" << nl;
        forAll(targets, targetI)
        {
            Info << "  sample " << targetI
                 << " -> processor " << targetOwner[targetI] << nl;
        }
        Info << endl;
    }

    fvScalarMatrix fullA(-fvm::laplacian(T));

    forAll(fullA.source(), celli)
    {
        const vector& C = mesh.C()[celli];

        const scalar x = (C.x() - xmin)/Lx;
        const scalar y = (C.y() - ymin)/Ly;
        const scalar z = (C.z() - zmin)/Lz;

        fullA.source()[celli] =
              1.0
            + 0.40*x
            + 0.20*y
            + 0.10*z;
    }

    scalarField rFull = matrixResidual(fullA, T);

    const label minLayers = 0;
    const label maxLayers = 4;
    const scalar tolerance = 1e-10;

    label firstPassingLayer = -1;

    for (label layers = minLayers; layers <= maxLayers; ++layers)
    {
        bitSet selectedCells =
            expandCellSelectionParallel(mesh, initialSelection, layers);

        label globalSelected = selectedCells.count();
        reduce(globalSelected, sumOp<label>());

        fvMeshSubset subset(mesh);

        subset.setCellSubset
        (
            selectedCells,
            -1,
            true
        );

        fvMesh& subMesh = subset.subMesh();
        const labelList& cellMap = subset.cellMap();

        labelList sampleSubCell(targets.size(), -1);

        forAll(targets, targetI)
        {
            const label oldLocalCell = targetLocalCell[targetI];
            if (oldLocalCell < 0) continue;

            forAll(cellMap, subCellI)
            {
                if (cellMap[subCellI] == oldLocalCell)
                {
                    sampleSubCell[targetI] = subCellI;
                    break;
                }
            }

            if (sampleSubCell[targetI] < 0)
            {
                FatalErrorInFunction
                    << "Sample " << targetI
                    << " on processor " << Pstream::myProcNo()
                    << " was not found in the submesh."
                    << exit(FatalError);
            }
        }

        tmp<volScalarField> tTSub = subset.interpolate(T);
        volScalarField TSub(tTSub);

        fvScalarMatrix subA(-fvm::laplacian(TSub));

        forAll(subA.source(), subCellI)
        {
            const vector& C = subMesh.C()[subCellI];

            const scalar x = (C.x() - xmin)/Lx;
            const scalar y = (C.y() - ymin)/Ly;
            const scalar z = (C.z() - zmin)/Lz;

            subA.source()[subCellI] =
                  1.0
                + 0.40*x
                + 0.20*y
                + 0.10*z;
        }

        scalarField rSub = matrixResidual(subA, TSub);

        scalar maxAbsDiff = 0.0;
        scalar diff2 = 0.0;
        scalar full2 = 0.0;
        label nSamples = 0;

        forAll(targets, targetI)
        {
            const label fullCell = targetLocalCell[targetI];
            const label subCell = sampleSubCell[targetI];

            if (fullCell < 0 || subCell < 0) continue;

            const scalar fullValue = rFull[fullCell];
            const scalar subValue = rSub[subCell];
            const scalar diff = subValue - fullValue;

            maxAbsDiff = max(maxAbsDiff, mag(diff));
            diff2 += sqr(diff);
            full2 += sqr(fullValue);
            ++nSamples;

            Pout
                << "layers=" << layers
                << " sample=" << targetI
                << " fullCell=" << fullCell
                << " subCell=" << subCell
                << " rFull=" << fullValue
                << " rSub=" << subValue
                << " diff=" << diff
                << endl;
        }

        reduce(maxAbsDiff, maxOp<scalar>());
        reduce(diff2, sumOp<scalar>());
        reduce(full2, sumOp<scalar>());
        reduce(nSamples, sumOp<label>());

        const scalar relL2 = Foam::sqrt(diff2/max(full2, SMALL));

        const bool passed =
            nSamples == targets.size()
         && maxAbsDiff < tolerance;

        if (passed && firstPassingLayer < 0)
        {
            firstPassingLayer = layers;
        }

        if (Pstream::master())
        {
            Info
                << nl
                << "----------------------------------------" << nl
                << " layers                    : " << layers << nl
                << " global selected cells     : " << globalSelected << nl
                << " samples compared          : " << nSamples << nl
                << " max |rSub-rFull|          : " << maxAbsDiff << nl
                << " relative sampled L2 diff  : " << relL2 << nl
                << " result                    : "
                << (passed ? "PASS" : "FAIL") << nl
                << "----------------------------------------" << nl
                << endl;
        }
    }

    if (Pstream::master())
    {
        Info
            << nl
            << "========================================" << nl
            << " SUMMARY" << nl
            << "========================================" << nl
            << "MPI ranks                 : " << Pstream::nProcs() << nl
            << "Tolerance                 : " << tolerance << nl
            << "First passing halo depth  : " << firstPassingLayer << nl
            << "========================================" << nl
            << endl;
    }

    return firstPassingLayer >= 0 ? 0 : 1;
}
