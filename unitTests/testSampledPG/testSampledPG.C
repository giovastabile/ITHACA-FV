#include "fvCFD.H"
#include "fvMeshSubset.H"
#include "syncTools.H"
#include "PstreamReduceOps.H"
#include <Eigen/Eigen>
#include <iomanip>

using namespace Foam;

class projectableScalarMatrix : public fvScalarMatrix
{
public:
    projectableScalarMatrix(const fvScalarMatrix& A) : fvScalarMatrix(A) {}
    void prepare() { addBoundaryDiag(diag(), 0); }
    void prepareSource(scalarField& source) { addBoundarySource(source, false); }
};

class projectableVectorMatrix : public fvVectorMatrix
{
public:
    projectableVectorMatrix(const fvVectorMatrix& A) : fvVectorMatrix(A) {}
    void prepareComponent(const scalarField& originalDiag, const direction cmpt)
    {
        diag() = originalDiag;
        addBoundaryDiag(diag(), cmpt);
    }
    void prepareSource(vectorField& source) { addBoundarySource(source, false); }
};

label ownedNearestCell(const fvMesh& mesh, const point& target, label& ownerProc)
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
                boundarySelected[bFaceI] = selected.test(faceCells[patchFaceI]) ? 1 : 0;
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
                if (boundarySelected[bFaceI]) expanded.set(faceCells[patchFaceI]);
            }
        }

        selected = expanded;
    }

    return selected;
}

labelList buildSampleSubCellMap
(
    const labelList& sampleLocalCell,
    const fvMeshSubset& subset
)
{
    const labelList& cellMap = subset.cellMap();
    labelList sampleSubCell(sampleLocalCell.size(), -1);

    forAll(sampleLocalCell, sampleI)
    {
        const label oldLocalCell = sampleLocalCell[sampleI];
        if (oldLocalCell < 0) continue;

        forAll(cellMap, subCellI)
        {
            if (cellMap[subCellI] == oldLocalCell)
            {
                sampleSubCell[sampleI] = subCellI;
                break;
            }
        }

        if (sampleSubCell[sampleI] < 0)
        {
            FatalErrorInFunction
                << "Sample " << sampleI << " not found on processor "
                << Pstream::myProcNo() << exit(FatalError);
        }
    }

    return sampleSubCell;
}

scalarField applyScalarMatrix(fvScalarMatrix& A, const volScalarField& phi)
{
    projectableScalarMatrix Ap(A);
    Ap.prepare();

    const scalarField& x = phi.primitiveField();
    scalarField Ax(x.size(), 0.0);
    const lduInterfaceFieldPtrsList interfaces(phi.boundaryField().scalarInterfaces());

    Ap.Amul(Ax, x, Ap.boundaryCoeffs(), interfaces, 0);
    return Ax;
}

scalarField scalarRhs(fvScalarMatrix& A)
{
    projectableScalarMatrix Ap(A);
    scalarField rhs(A.source());
    Ap.prepareSource(rhs);
    return rhs;
}

vectorField applyVectorMatrix(fvVectorMatrix& A, const volVectorField& phi)
{
    projectableVectorMatrix Ap(A);
    const scalarField originalDiag(Ap.diag());
    const lduInterfaceFieldPtrsList interfaces(phi.boundaryField().scalarInterfaces());

    vectorField Aphi(phi.size(), vector::zero);
    const vectorField& mode = phi.primitiveField();

    for (direction cmpt = 0; cmpt < vector::nComponents; ++cmpt)
    {
        Ap.prepareComponent(originalDiag, cmpt);
        scalarField x(mode.component(cmpt));
        scalarField Ax(x.size(), 0.0);
        FieldField<Field, scalar> bouCoeffsCmpt(Ap.boundaryCoeffs().component(cmpt));

        Ap.Amul(Ax, x, bouCoeffsCmpt, interfaces, cmpt);

        forAll(Ax, celli)
        {
            Aphi[celli][cmpt] = Ax[celli];
        }
    }

    return Aphi;
}

vectorField vectorRhs(fvVectorMatrix& A)
{
    projectableVectorMatrix Ap(A);
    vectorField rhs(A.source());
    Ap.prepareSource(rhs);
    return rhs;
}

Eigen::VectorXd sampleScalarField
(
    const scalarField& values,
    const labelList& sampleLocalCell
)
{
    Eigen::VectorXd sampled = Eigen::VectorXd::Zero(sampleLocalCell.size());

    forAll(sampleLocalCell, sampleI)
    {
        const label celli = sampleLocalCell[sampleI];
        scalar value = celli >= 0 ? values[celli] : 0.0;
        reduce(value, sumOp<scalar>());
        sampled(sampleI) = value;
    }

    return sampled;
}

Eigen::VectorXd sampleVectorField
(
    const vectorField& values,
    const labelList& sampleLocalCell
)
{
    Eigen::VectorXd sampled =
        Eigen::VectorXd::Zero(vector::nComponents*sampleLocalCell.size());

    forAll(sampleLocalCell, sampleI)
    {
        const label celli = sampleLocalCell[sampleI];

        for (direction cmpt = 0; cmpt < vector::nComponents; ++cmpt)
        {
            scalar value = celli >= 0 ? values[celli][cmpt] : 0.0;
            reduce(value, sumOp<scalar>());
            sampled(vector::nComponents*sampleI + cmpt) = value;
        }
    }

    return sampled;
}

template<class DerivedA, class DerivedB>
scalar relativeDifference
(
    const Eigen::MatrixBase<DerivedA>& A,
    const Eigen::MatrixBase<DerivedB>& B
)
{
    return (A - B).norm()/max(B.norm(), scalar(SMALL));
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
        std::cout << std::setprecision(16)
                  << "\n========================================\n"
                  << " SAMPLED PG / SUBMESH TEST\n"
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

    List<point> targets(4);
    targets[0] = point(xmin + 0.31*Lx, ymin + 0.41*Ly, zmin + 0.44*Lz);
    targets[1] = point(xmin + 0.48*Lx, ymin + 0.37*Ly, zmin + 0.52*Lz);
    targets[2] = point(xmin + 0.52*Lx, ymin + 0.63*Ly, zmin + 0.47*Lz);
    targets[3] = point(xmin + 0.69*Lx, ymin + 0.58*Ly, zmin + 0.61*Lz);

    labelList sampleLocalCell(targets.size(), -1);
    bitSet initialSelection(mesh.nCells());

    forAll(targets, sampleI)
    {
        label ownerProc = -1;
        const label localCell = ownedNearestCell(mesh, targets[sampleI], ownerProc);
        sampleLocalCell[sampleI] = localCell;
        if (localCell >= 0) initialSelection.set(localCell);
    }

    const label layers = 1;
    bitSet selectedCells = expandCellSelectionParallel(mesh, initialSelection, layers);

    label globalSelected = selectedCells.count();
    reduce(globalSelected, sumOp<label>());

    fvMeshSubset subset(mesh);
    subset.setCellSubset(selectedCells, -1, true);

    fvMesh& subMesh = subset.subMesh();

    labelList sampleSubCell =
        buildSampleSubCellMap(sampleLocalCell, subset);

    if (Pstream::master())
    {
        Info << "Samples        : " << targets.size() << nl
             << "Stencil layers : " << layers << nl
             << "Selected cells : " << globalSelected << nl << endl;
    }

    volScalarField T
    (
        IOobject
        (
            "T", runTime.timeName(), mesh,
            IOobject::MUST_READ, IOobject::NO_WRITE
        ),
        mesh
    );

    volVectorField U
    (
        IOobject
        (
            "U", runTime.timeName(), mesh,
            IOobject::MUST_READ, IOobject::NO_WRITE
        ),
        mesh
    );

    // ---------------- SCALAR MODES ----------------
    const label nScalarModes = 3;
    PtrList<volScalarField> scalarModes(nScalarModes);

    for (label modeI = 0; modeI < nScalarModes; ++modeI)
    {
        scalarModes.set
        (
            modeI,
            new volScalarField
            (
                IOobject
                (
                    "scalarMode_" + Foam::name(modeI),
                    runTime.timeName(), mesh,
                    IOobject::NO_READ, IOobject::NO_WRITE
                ),
                T
            )
        );

        volScalarField& mode = scalarModes[modeI];

        forAll(mode, celli)
        {
            const vector& C = mesh.C()[celli];
            const scalar x = (C.x() - xmin)/Lx;
            const scalar y = (C.y() - ymin)/Ly;
            const scalar z = (C.z() - zmin)/Lz;

            mode[celli] =
                  Foam::sin((modeI + 1)*0.73*pi*x)
                + 0.4*Foam::cos((modeI + 1)*0.51*pi*y)
                + 0.2*Foam::sin((modeI + 1)*0.37*pi*z);
        }

        mode.correctBoundaryConditions();
    }

    fvScalarMatrix scalarAFull(-fvm::laplacian(T));

    forAll(scalarAFull.source(), celli)
    {
        const vector& C = mesh.C()[celli];
        const scalar x = (C.x() - xmin)/Lx;
        const scalar y = (C.y() - ymin)/Ly;
        const scalar z = (C.z() - zmin)/Lz;
        scalarAFull.source()[celli] = 1.0 + 0.4*x + 0.2*y + 0.1*z;
    }

    Eigen::MatrixXd sampledAVScalarFull(targets.size(), nScalarModes);

    for (label modeI = 0; modeI < nScalarModes; ++modeI)
    {
        scalarField Aphi = applyScalarMatrix(scalarAFull, scalarModes[modeI]);
        sampledAVScalarFull.col(modeI) =
            sampleScalarField(Aphi, sampleLocalCell);
    }

    scalarField rhsScalarFull = scalarRhs(scalarAFull);
    Eigen::VectorXd sampledBScalarFull =
        sampleScalarField(rhsScalarFull, sampleLocalCell);

    tmp<volScalarField> tTSub = subset.interpolate(T);
    volScalarField TSub(tTSub);

    fvScalarMatrix scalarASub(-fvm::laplacian(TSub));

    forAll(scalarASub.source(), celli)
    {
        const vector& C = subMesh.C()[celli];
        const scalar x = (C.x() - xmin)/Lx;
        const scalar y = (C.y() - ymin)/Ly;
        const scalar z = (C.z() - zmin)/Lz;
        scalarASub.source()[celli] = 1.0 + 0.4*x + 0.2*y + 0.1*z;
    }

    Eigen::MatrixXd sampledAVScalarSub(targets.size(), nScalarModes);

    for (label modeI = 0; modeI < nScalarModes; ++modeI)
    {
        tmp<volScalarField> tModeSub = subset.interpolate(scalarModes[modeI]);
        volScalarField modeSub(tModeSub);

        scalarField AphiSub = applyScalarMatrix(scalarASub, modeSub);
        sampledAVScalarSub.col(modeI) =
            sampleScalarField(AphiSub, sampleSubCell);
    }

    scalarField rhsScalarSub = scalarRhs(scalarASub);
    Eigen::VectorXd sampledBScalarSub =
        sampleScalarField(rhsScalarSub, sampleSubCell);

    Eigen::MatrixXd scalarArFull =
        sampledAVScalarFull.transpose()*sampledAVScalarFull;
    Eigen::VectorXd scalarBrFull =
        sampledAVScalarFull.transpose()*sampledBScalarFull;

    Eigen::MatrixXd scalarArSub =
        sampledAVScalarSub.transpose()*sampledAVScalarSub;
    Eigen::VectorXd scalarBrSub =
        sampledAVScalarSub.transpose()*sampledBScalarSub;

    Eigen::VectorXd scalarCoeffFull =
        scalarArFull.fullPivLu().solve(scalarBrFull);
    Eigen::VectorXd scalarCoeffSub =
        scalarArSub.fullPivLu().solve(scalarBrSub);

    const scalar scalarAVDiff = relativeDifference(sampledAVScalarSub, sampledAVScalarFull);
    const scalar scalarBDiff = relativeDifference(sampledBScalarSub, sampledBScalarFull);
    const scalar scalarArDiff = relativeDifference(scalarArSub, scalarArFull);
    const scalar scalarBrDiff = relativeDifference(scalarBrSub, scalarBrFull);
    const scalar scalarCoeffDiff = relativeDifference(scalarCoeffSub, scalarCoeffFull);

    // ---------------- VECTOR MODES ----------------
    const label nVectorModes = 3;
    PtrList<volVectorField> vectorModes(nVectorModes);

    for (label modeI = 0; modeI < nVectorModes; ++modeI)
    {
        vectorModes.set
        (
            modeI,
            new volVectorField
            (
                IOobject
                (
                    "vectorMode_" + Foam::name(modeI),
                    runTime.timeName(), mesh,
                    IOobject::NO_READ, IOobject::NO_WRITE
                ),
                U
            )
        );

        volVectorField& mode = vectorModes[modeI];

        forAll(mode, celli)
        {
            const vector& C = mesh.C()[celli];
            const scalar x = (C.x() - xmin)/Lx;
            const scalar y = (C.y() - ymin)/Ly;
            const scalar z = (C.z() - zmin)/Lz;

            mode[celli] =
                vector
                (
                    Foam::sin((modeI + 1)*0.73*pi*x),
                    Foam::cos((modeI + 1)*0.61*pi*y),
                    Foam::sin((modeI + 1)*0.47*pi*z)
                );
        }

        mode.correctBoundaryConditions();
    }

    fvVectorMatrix vectorAFull(-fvm::laplacian(U));

    forAll(vectorAFull.source(), celli)
    {
        const vector& C = mesh.C()[celli];
        const scalar x = (C.x() - xmin)/Lx;
        const scalar y = (C.y() - ymin)/Ly;
        const scalar z = (C.z() - zmin)/Lz;

        vectorAFull.source()[celli] =
            vector
            (
                1.0 + x + 0.10*y,
                2.0 + y + 0.15*z,
                3.0 + z + 0.20*x
            );
    }

    Eigen::MatrixXd sampledAVVectorFull
    (
        vector::nComponents*targets.size(),
        nVectorModes
    );

    for (label modeI = 0; modeI < nVectorModes; ++modeI)
    {
        vectorField Aphi = applyVectorMatrix(vectorAFull, vectorModes[modeI]);
        sampledAVVectorFull.col(modeI) =
            sampleVectorField(Aphi, sampleLocalCell);
    }

    vectorField rhsVectorFull = vectorRhs(vectorAFull);
    Eigen::VectorXd sampledBVectorFull =
        sampleVectorField(rhsVectorFull, sampleLocalCell);

    tmp<volVectorField> tUSub = subset.interpolate(U);
    volVectorField USub(tUSub);

    fvVectorMatrix vectorASub(-fvm::laplacian(USub));

    forAll(vectorASub.source(), celli)
    {
        const vector& C = subMesh.C()[celli];
        const scalar x = (C.x() - xmin)/Lx;
        const scalar y = (C.y() - ymin)/Ly;
        const scalar z = (C.z() - zmin)/Lz;

        vectorASub.source()[celli] =
            vector
            (
                1.0 + x + 0.10*y,
                2.0 + y + 0.15*z,
                3.0 + z + 0.20*x
            );
    }

    Eigen::MatrixXd sampledAVVectorSub
    (
        vector::nComponents*targets.size(),
        nVectorModes
    );

    for (label modeI = 0; modeI < nVectorModes; ++modeI)
    {
        tmp<volVectorField> tModeSub = subset.interpolate(vectorModes[modeI]);
        volVectorField modeSub(tModeSub);

        vectorField AphiSub = applyVectorMatrix(vectorASub, modeSub);
        sampledAVVectorSub.col(modeI) =
            sampleVectorField(AphiSub, sampleSubCell);
    }

    vectorField rhsVectorSub = vectorRhs(vectorASub);
    Eigen::VectorXd sampledBVectorSub =
        sampleVectorField(rhsVectorSub, sampleSubCell);

    Eigen::MatrixXd vectorArFull =
        sampledAVVectorFull.transpose()*sampledAVVectorFull;
    Eigen::VectorXd vectorBrFull =
        sampledAVVectorFull.transpose()*sampledBVectorFull;

    Eigen::MatrixXd vectorArSub =
        sampledAVVectorSub.transpose()*sampledAVVectorSub;
    Eigen::VectorXd vectorBrSub =
        sampledAVVectorSub.transpose()*sampledBVectorSub;

    Eigen::VectorXd vectorCoeffFull =
        vectorArFull.fullPivLu().solve(vectorBrFull);
    Eigen::VectorXd vectorCoeffSub =
        vectorArSub.fullPivLu().solve(vectorBrSub);

    const scalar vectorAVDiff = relativeDifference(sampledAVVectorSub, sampledAVVectorFull);
    const scalar vectorBDiff = relativeDifference(sampledBVectorSub, sampledBVectorFull);
    const scalar vectorArDiff = relativeDifference(vectorArSub, vectorArFull);
    const scalar vectorBrDiff = relativeDifference(vectorBrSub, vectorBrFull);
    const scalar vectorCoeffDiff = relativeDifference(vectorCoeffSub, vectorCoeffFull);

    const scalar tolerance = 1e-10;

    const bool scalarPass =
           scalarAVDiff    < tolerance
        && scalarBDiff     < tolerance
        && scalarArDiff    < tolerance
        && scalarBrDiff    < tolerance
        && scalarCoeffDiff < tolerance;

    const bool vectorPass =
           vectorAVDiff    < tolerance
        && vectorBDiff     < tolerance
        && vectorArDiff    < tolerance
        && vectorBrDiff    < tolerance
        && vectorCoeffDiff < tolerance;

    if (Pstream::master())
    {
        std::cout << nl
             << "========================================" << nl
             << " SCALAR SAMPLED PG" << nl
             << "========================================" << nl
             << "rel diff P_S A V : " << scalarAVDiff << nl
             << "rel diff P_S b   : " << scalarBDiff << nl
             << "rel diff Ar      : " << scalarArDiff << nl
             << "rel diff br      : " << scalarBrDiff << nl
             << "rel diff coeff   : " << scalarCoeffDiff << nl
             << "Result           : " << (scalarPass ? "PASS" : "FAIL") << nl
             << nl
             << "Ar(full sampled) =" << nl << scalarArFull << nl
             << nl
             << "Ar(subset) =" << nl << scalarArSub << nl
             << nl
             << "br(full sampled) =" << nl << scalarBrFull << nl
             << nl
             << "br(subset) =" << nl << scalarBrSub << nl
             << std::endl;

        std::cout << nl
             << "========================================" << nl
             << " VECTOR SAMPLED PG" << nl
             << "========================================" << nl
             << "rel diff P_S A V : " << vectorAVDiff << nl
             << "rel diff P_S b   : " << vectorBDiff << nl
             << "rel diff Ar      : " << vectorArDiff << nl
             << "rel diff br      : " << vectorBrDiff << nl
             << "rel diff coeff   : " << vectorCoeffDiff << nl
             << "Result           : " << (vectorPass ? "PASS" : "FAIL") << nl
             << nl
             << "Ar(full sampled) =" << nl << vectorArFull << nl
             << nl
             << "Ar(subset) =" << nl << vectorArSub << nl
             << nl
             << "br(full sampled) =" << nl << vectorBrFull << nl
             << nl
             << "br(subset) =" << nl << vectorBrSub << nl
             << std::endl;

        std::cout << nl
             << "========================================" << nl
             << " FINAL RESULT" << nl
             << "========================================" << nl
             << "Execution : " << (Pstream::parRun() ? "PARALLEL" : "SERIAL") << nl
             << "MPI ranks : " << Pstream::nProcs() << nl
             << "Scalar PG : " << (scalarPass ? "PASS" : "FAIL") << nl
             << "Vector PG : " << (vectorPass ? "PASS" : "FAIL") << nl
             << "Overall   : " << ((scalarPass && vectorPass) ? "PASS" : "FAIL") << nl
             << "========================================" << nl
             << std::endl;
    }

    return (scalarPass && vectorPass) ? 0 : 1;
}
