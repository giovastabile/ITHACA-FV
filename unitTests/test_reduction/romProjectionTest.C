#include "fvCFD.H"
#include "PstreamReduceOps.H"
#include "Modes.H"

using namespace Foam;


// Helper only needed because addBoundaryDiag() is protected
class projectableFvScalarMatrix
:
public fvScalarMatrix
{
    public:
    
    projectableFvScalarMatrix(const fvScalarMatrix& A)
    :
    fvScalarMatrix(A)
    {}
    
    void prepareForAmul()
    {
        // This is what fvMatrix::solveSegregated() does
        // before calling the lduMatrix solver.
        addBoundaryDiag(diag(), 0);
    }
};


// Apply the complete algebraic FV matrix to one mode:
//
//      Aphi = A * phi
//
// including processor/coupled interfaces.
void applyMatrix
(
    const fvScalarMatrix& A,
    const volScalarField& phi,
    scalarField& Aphi
)
{
    // Work on a copy since addBoundaryDiag modifies diag()
    projectableFvScalarMatrix Awork(A);
    
    Awork.prepareForAmul();
    
    // Local internal values of the mode
    scalarField x(phi.primitiveField());
    
    Aphi.setSize(x.size());
    Aphi = 0.0;
    
    // Processor/cyclic/etc interfaces.
    //
    // For a scalar equation these interface objects contain the
    // communication machinery. Amul uses x as the actual vector.
    lduInterfaceFieldPtrsList interfaces
    (
        A.psi().boundaryField().scalarInterfaces()
    );
    
    Awork.Amul
    (
        Aphi,
        x,
        Awork.boundaryCoeffs(),
        interfaces,
        0
    );
}


// Algebraic Euclidean inner product
//
//      a^T b
//
// with the final sum performed over all MPI ranks.
scalar globalDot
(
    const scalarField& a,
    const scalarField& b
)
{
    scalar value = 0.0;
    
    forAll(a, celli)
    {
        value += a[celli]*b[celli];
    }
    
    reduce(value, sumOp<scalar>());
    
    return value;
}


int main(int argc, char *argv[])
{
    #include "setRootCase.H"
    Foam::Time runTime
    (
        Foam::Time::controlDictName,
        args
    );
    
    Foam::fvMesh mesh
    (
        Foam::IOobject
        (
            Foam::fvMesh::defaultRegion,
            runTime.timeName(),
            runTime,
            Foam::IOobject::MUST_READ
        )
    );
    ITHACAparameters* para = ITHACAparameters::getInstance(mesh, runTime);
    
    
    Info<< nl
    << "Running ROM projection test with "
    << Pstream::nProcs()
    << " MPI rank(s)" << nl << endl;
    
    
    // -------------------------------------------------------------
    // Full-order field
    // -------------------------------------------------------------
    
    volScalarField T
    (
        IOobject
        (
            "T",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        mesh
    );
    
    
    // -------------------------------------------------------------
    // Assemble a completely standard OpenFOAM matrix
    //
    //              A T = 0
    //
    // Here A corresponds to -Laplacian.
    // -------------------------------------------------------------
    
    fvScalarMatrix A
    (
        -fvm::laplacian(T)
    );
    
    
    // -------------------------------------------------------------
    // Build some modes.
    //
    // In your application this PtrList would already exist.
    // -------------------------------------------------------------
    
    const label nModes = 4;
    
    PtrList<volScalarField> modes(nModes);
    
    const scalar pi = constant::mathematical::pi;
    
    // Global x extent of the mesh
    scalar xmin = GREAT;
    scalar xmax = -GREAT;
    
    forAll(mesh.C(), celli)
    {
        xmin = min(xmin, mesh.C()[celli].x());
        xmax = max(xmax, mesh.C()[celli].x());
    }
    
    reduce(xmin, minOp<scalar>());
    reduce(xmax, maxOp<scalar>());
    
    const scalar L = xmax - xmin;
    
    
    for (label modeI = 0; modeI < nModes; ++modeI)
    {
        modes.set
        (
            modeI,
            new volScalarField
            (
                IOobject
                (
                    "mode_" + Foam::name(modeI),
                    runTime.timeName(),
                    mesh,
                    IOobject::NO_READ,
                    IOobject::NO_WRITE
                ),
                T
            )
        );
        
        volScalarField& mode = modes[modeI];
        
        forAll(mode, celli)
        {
            const scalar x =
            (mesh.C()[celli].x() - xmin)/L;
            
            mode[celli] =
            Foam::sin((modeI + 1)*0.73*pi*x);
        }
        
        // Uses same BC type as T.
        // With homogeneous fixedValue boundaries this gives zero
        // boundary values for the modes.
        mode.correctBoundaryConditions();
    }
    
    
    // -------------------------------------------------------------
    //
    //              Ar = V^T A V
    //
    // -------------------------------------------------------------
    
    List<scalarField> Ar(nModes);
    
    forAll(Ar, i)
    {
        Ar[i].setSize(nModes);
        Ar[i] = 0.0;
    }
    
    
    for (label j = 0; j < nModes; ++j)
    {
        scalarField Aphi(mesh.nCells(), 0.0);
        
        applyMatrix
        (
            A,
            modes[j],
            Aphi
        );
        
        
        for (label i = 0; i < nModes; ++i)
        {
            Ar[i][j] =
            globalDot
            (
                modes[i].primitiveField(),
                Aphi
            );
        }
    }
    
    volScalarModes ithacaModes;
    ithacaModes = modes;
    
    
    // -------------------------------------------------------------
    // Print only on master
    // -------------------------------------------------------------
    
    if (Pstream::master())
    {
        Info<< nl
        << "Reduced matrix Ar = V^T A V"
        << nl << endl;
        
        for (label i = 0; i < nModes; ++i)
        {
            for (label j = 0; j < nModes; ++j)
            {
                Info<< Ar[i][j] << " ";
            }
            
            Info<< nl;
        }
        
        Info<< endl;
    }
    
    Eigen::MatrixXd ArEigen = ithacaModes.project(A, nModes, "G")[0];
    

    if (Pstream::master())
    {
            std::cout << "Line: 288 of file romProjectionTest.C" << std::endl;
        std::cout << ArEigen << std::endl; 
    }
    
    
    
    
    Info<< "End" << endl;
    
    return 0;
}