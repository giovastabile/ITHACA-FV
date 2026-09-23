#include "fvCFD.H"
#include "fvMeshSubset.H"
#include "ITHACAutilities.H"
#include "Foam2Eigen.H"

#include <Eigen/Sparse>
#include <iomanip>

using namespace Foam;

label nearestCell(const fvMesh& mesh, const point& target)
{
    label bestCell = -1;
    scalar bestDist = GREAT;

    forAll(mesh.C(), celli)
    {
        const scalar d = magSqr(mesh.C()[celli] - target);

        if (d < bestDist)
        {
            bestDist = d;
            bestCell = celli;
        }
    }

    return bestCell;
}

labelList globalToLocalCells
(
    const labelList& sampledCells,
    const fvMeshSubset& subset
)
{
    const labelList& cellMap = subset.cellMap();
    labelList local(sampledCells.size(), -1);

    forAll(sampledCells, sampleI)
    {
        forAll(cellMap, subCellI)
        {
            if (cellMap[subCellI] == sampledCells[sampleI])
            {
                local[sampleI] = subCellI;
                break;
            }
        }

        if (local[sampleI] < 0)
        {
            FatalErrorInFunction
                << "Sampled cell " << sampledCells[sampleI]
                << " was not found in the submesh."
                << exit(FatalError);
        }
    }

    return local;
}

labelList buildStencilCells
(
    fvMesh& mesh,
    const labelList& sampledCells,
    const label layers
)
{
    List<labelList> neighbourhoods(sampledCells.size());

    forAll(sampledCells, i)
    {
        neighbourhoods[i] =
            ITHACAutilities::getIndices
            (
                mesh,
                sampledCells[i],
                layers
            );
    }

    labelList stencilCells =
        ITHACAutilities::combineList(neighbourhoods);

    labelHashSet uniqueCells;

    forAll(stencilCells, i)
    {
        uniqueCells.insert(stencilCells[i]);
    }

    forAll(sampledCells, i)
    {
        uniqueCells.insert(sampledCells[i]);
    }

    return uniqueCells.toc();
}

int main(int argc, char *argv[])
{
    #include "setRootCase.H"

    Time runTime
    (
        Time::controlDictName,
        args
    );

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

    if (Pstream::parRun())
    {
        FatalErrorInFunction
            << "This first fvMeshSubset residual test is intentionally serial."
            << nl
            << "Run without -parallel."
            << exit(FatalError);
    }

    std::cout << std::setprecision(16);

    scalar xmin = GREAT;
    scalar xmax = -GREAT;
    scalar ymin = GREAT;
    scalar ymax = -GREAT;
    scalar zmin = GREAT;
    scalar zmax = -GREAT;

    forAll(mesh.C(), celli)
    {
        const vector& C = mesh.C()[celli];

        xmin = min(xmin, C.x());
        xmax = max(xmax, C.x());
        ymin = min(ymin, C.y());
        ymax = max(ymax, C.y());
        zmin = min(zmin, C.z());
        zmax = max(zmax, C.z());
    }

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

    labelList sampledCells(3);

    sampledCells[0] = nearestCell
    (
        mesh,
        point
        (
            xmin + 0.35*Lx,
            ymin + 0.45*Ly,
            zmin + 0.50*Lz
        )
    );

    sampledCells[1] = nearestCell
    (
        mesh,
        point
        (
            xmin + 0.50*Lx,
            ymin + 0.55*Ly,
            zmin + 0.40*Lz
        )
    );

    sampledCells[2] = nearestCell
    (
        mesh,
        point
        (
            xmin + 0.65*Lx,
            ymin + 0.40*Ly,
            zmin + 0.60*Lz
        )
    );

    Info
        << nl
        << "========================================" << nl
        << " fvMeshSubset residual test (SERIAL)" << nl
        << "========================================" << nl
        << "Full mesh cells : " << mesh.nCells() << nl
        << "Sampled cells   : " << sampledCells << nl
        << endl;

    fvScalarMatrix fullA
    (
        -fvm::laplacian(T)
    );

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

    Eigen::SparseMatrix<double> AFull;
    Eigen::VectorXd bFull;

    Foam2Eigen::fvMatrix2Eigen
    (
        fullA,
        AFull,
        bFull
    );

    Eigen::VectorXd xFull =
        Foam2Eigen::field2Eigen(T);

    Eigen::VectorXd rFull =
        AFull*xFull - bFull;

    const label minLayers = 0;
    const label maxLayers = 4;
    const scalar tolerance = 1e-11;

    scalar bestMaxAbsDiff = GREAT;
    label firstExactLayer = -1;

    for
    (
        label layers = minLayers;
        layers <= maxLayers;
        ++layers
    )
    {
        labelList stencilCells =
            buildStencilCells
            (
                mesh,
                sampledCells,
                layers
            );

        labelHashSet cellSet;

        forAll(stencilCells, i)
        {
            cellSet.insert(stencilCells[i]);
        }

        fvMeshSubset subset(mesh);
        subset.setCellSubset(cellSet);

        fvMesh& subMesh = subset.subMesh();

        labelList localSampledCells =
            globalToLocalCells
            (
                sampledCells,
                subset
            );

        tmp<volScalarField> tTSub =
            subset.interpolate(T);

        volScalarField TSub(tTSub);

        fvScalarMatrix subA
        (
            -fvm::laplacian(TSub)
        );

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

        Eigen::SparseMatrix<double> ASub;
        Eigen::VectorXd bSub;

        Foam2Eigen::fvMatrix2Eigen
        (
            subA,
            ASub,
            bSub
        );

        Eigen::VectorXd xSub =
            Foam2Eigen::field2Eigen(TSub);

        Eigen::VectorXd rSub =
            ASub*xSub - bSub;

        scalar maxAbsDiff = 0.0;
        scalar l2Diff2 = 0.0;
        scalar l2Full2 = 0.0;

        Info
            << nl
            << "----------------------------------------" << nl
            << " layers        = " << layers << nl
            << " stencil cells = " << stencilCells.size() << nl
            << " submesh cells = " << subMesh.nCells() << nl
            << "----------------------------------------" << nl;

        forAll(sampledCells, sampleI)
        {
            const label globalCell = sampledCells[sampleI];
            const label localCell = localSampledCells[sampleI];

            const scalar fullValue = rFull(globalCell);
            const scalar subValue = rSub(localCell);
            const scalar diff = subValue - fullValue;

            maxAbsDiff = max(maxAbsDiff, mag(diff));
            l2Diff2 += sqr(diff);
            l2Full2 += sqr(fullValue);

            Info
                << "sample " << sampleI
                << "  global=" << globalCell
                << "  local=" << localCell
                << "  rFull=" << fullValue
                << "  rSub=" << subValue
                << "  diff=" << diff
                << nl;
        }

        const scalar relL2 =
            Foam::sqrt
            (
                l2Diff2
               /max(l2Full2, SMALL)
            );

        const bool passed =
            maxAbsDiff < tolerance;

        Info
            << nl
            << "max |rSub-rFull| = " << maxAbsDiff << nl
            << "relative sampled L2 difference = " << relL2 << nl
            << "result = " << (passed ? "PASS" : "FAIL") << nl
            << endl;

        bestMaxAbsDiff = min(bestMaxAbsDiff, maxAbsDiff);

        if (passed && firstExactLayer < 0)
        {
            firstExactLayer = layers;
        }
    }

    Info
        << nl
        << "========================================" << nl
        << " SUMMARY" << nl
        << "========================================" << nl
        << "Tolerance                 : " << tolerance << nl
        << "Best max absolute diff    : " << bestMaxAbsDiff << nl
        << "First layer count passing : " << firstExactLayer << nl
        << "========================================" << nl
        << endl;

    if (firstExactLayer < 0)
    {
        Info
            << "No tested stencil depth reproduced the full residual "
            << "within tolerance." << nl;

        return 1;
    }

    return 0;
}
