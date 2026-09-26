/*---------------------------------------------------------------------------*\
     ██╗████████╗██╗  ██╗ █████╗  ██████╗ █████╗       ███████╗██╗   ██╗
     ██║╚══██╔══╝██║  ██║██╔══██╗██╔════╝██╔══██╗      ██╔════╝██║   ██║
     ██║   ██║   ███████║███████║██║     ███████║█████╗█████╗  ██║   ██║
     ██║   ██║   ██╔══██║██╔══██║██║     ██╔══██║╚════╝██╔══╝  ╚██╗ ██╔╝
     ██║   ██║   ██║  ██║██║  ██║╚██████╗██║  ██║      ██║      ╚████╔╝
     ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝       ╚═══╝
 * In real Time Highly Advanced Computational Applications for Finite Volumes
 * Copyright (C) 2017 by the ITHACA-FV authors
-------------------------------------------------------------------------------
\*---------------------------------------------------------------------------*/

#include "ReducedSimpleSteadyNS.H"


namespace
{

void accumulateNormalizedResidual
(
    Foam::scalarField& indicator,
    const Foam::vectorField& residual
)
{
    Foam::scalar maxResidual = 0.0;

    forAll(residual, celli)
    {
        maxResidual = Foam::max(maxResidual, Foam::mag(residual[celli]));
    }

    Foam::reduce(maxResidual, Foam::maxOp<Foam::scalar>());

    if (maxResidual <= SMALL)
    {
        return;
    }

    forAll(residual, celli)
    {
        indicator[celli] =
            Foam::max
            (
                indicator[celli],
                Foam::mag(residual[celli])/maxResidual
            );
    }
}


void accumulateNormalizedResidual
(
    Foam::scalarField& indicator,
    const Foam::scalarField& residual
)
{
    Foam::scalar maxResidual = 0.0;

    forAll(residual, celli)
    {
        maxResidual = Foam::max(maxResidual, Foam::mag(residual[celli]));
    }

    Foam::reduce(maxResidual, Foam::maxOp<Foam::scalar>());

    if (maxResidual <= SMALL)
    {
        return;
    }

    forAll(residual, celli)
    {
        indicator[celli] =
            Foam::max
            (
                indicator[celli],
                Foam::mag(residual[celli])/maxResidual
            );
    }
}

} // End anonymous namespace



// * * * * * * * * * * * * Constructors * * * * * * * * * * * * * * * * //

reducedSimpleSteadyNS::reducedSimpleSteadyNS()
{
}


reducedSimpleSteadyNS::reducedSimpleSteadyNS(SteadyNSSimple& FOMproblem)
:
    problem(&FOMproblem)
{
    for (int i = 0; i < problem->inletIndex.rows(); i++)
    {
        ULmodes.append((problem->liftfield[i]).clone());
    }

    for (int i = 0; i < problem->Umodes.size(); i++)
    {
        ULmodes.append((problem->Umodes.toPtrList()[i]).clone());
    }
}


// * * * * * * * * * * Standard full-mesh solver * * * * * * * * * * * * //

void reducedSimpleSteadyNS::solveOnline_Simple
(
    scalar mu_now,
    int NmodesUproj,
    int NmodesPproj,
    int NmodesNut,
    int NmodesSup,
    word Folder
)
{
    ULmodes.resize(0);

    for (int i = 0; i < problem->inletIndex.rows(); i++)
    {
        ULmodes.append((problem->liftfield[i]).clone());
    }

    for (int i = 0; i < NmodesUproj; i++)
    {
        ULmodes.append((problem->Umodes.toPtrList()[i]).clone());
    }

    for (int i = 0; i < NmodesSup; i++)
    {
        ULmodes.append((problem->supmodes.toPtrList()[i]).clone());
    }

    counter++;

    if (NmodesUproj == 0)
    {
        UprojN = ULmodes.size();
    }
    else
    {
        UprojN = NmodesUproj + NmodesSup;
    }

    if (NmodesPproj == 0)
    {
        PprojN = problem->Pmodes.size();
    }
    else
    {
        PprojN = NmodesPproj;
    }

    if (NmodesNut == 0)
    {
        NmodesNut = problem->nutModes.size();
    }

    Eigen::VectorXd uresidualOld = Eigen::VectorXd::Zero(UprojN);
    Eigen::VectorXd presidualOld = Eigen::VectorXd::Zero(PprojN);
    Eigen::VectorXd uresidual = Eigen::VectorXd::Zero(UprojN);
    Eigen::VectorXd presidual = Eigen::VectorXd::Zero(PprojN);

    scalar U_norm_res(1);
    scalar P_norm_res(1);

    Eigen::MatrixXd a = Eigen::VectorXd::Zero(UprojN);
    Eigen::MatrixXd b = Eigen::VectorXd::Zero(PprojN);

    a(0) = vel_now(0, 0);

    float residualJumpLim =
        problem->para->ITHACAdict->lookupOrDefault<float>
        (
            "residualJumpLim",
            1e-5
        );

    float normalizedResidualLim =
        problem->para->ITHACAdict->lookupOrDefault<float>
        (
            "normalizedResidualLim",
            1e-5
        );

    maxIterOn =
        problem->para->ITHACAdict->lookupOrDefault<int>
        (
            "maxIterOn",
            1000
        );

    scalar residual_jump(1 + residualJumpLim);

    volScalarField& P = problem->_p();
    volVectorField& U = problem->_U();
    fvMesh& mesh = problem->_mesh();
    Time& runTime = problem->_runTime();

    if (collectResidualIndicator_)
    {
        residualIndicator_.setSize(mesh.nCells());
        residualIndicator_ = scalar(0);
    }

    P.rename("p");

    surfaceScalarField& phi(problem->_phi());

    ULmodes.reconstruct(U, a, "U");
    problem->Pmodes.reconstruct(P, b, "p");

    phi = fvc::interpolate(U) & U.mesh().Sf();

    int iter = 0;
    simpleControl& simple = problem->_simple();

    if (ITHACAutilities::isTurbulent())
    {
        Eigen::MatrixXd nutCoeff;
        nutCoeff.resize(NmodesNut, 1);

        for (int i = 0; i < NmodesNut; i++)
        {
            Eigen::MatrixXd muEval;
            muEval.resize(1, 1);
            muEval(0, 0) = mu_now;
            nutCoeff(i, 0) = problem->rbfSplines[i]->eval(muEval);
        }

        volScalarField& nut =
            const_cast<volScalarField&>
            (
                problem->_mesh().lookupObject<volScalarField>("nut")
            );

        problem->nutModes.reconstruct(nut, nutCoeff, "nut");
        ITHACAstream::exportSolution(nut, name(counter), Folder);
    }

    PtrList<volVectorField> gradModP;

    for (int i = 0; i < NmodesPproj; i++)
    {
        gradModP.append(fvc::grad(problem->Pmodes[i]));
    }

    projGradModP = ULmodes.project(gradModP, NmodesUproj);

    while
    (
        (
            residual_jump > residualJumpLim
         || std::max(U_norm_res, P_norm_res) > normalizedResidualLim
        )
     && iter < maxIterOn
    )
    {
        iter++;

        Info << "Iteration " << iter << endl;

#if defined(OFVER) && (OFVER == 6)
        simple.loop(runTime);
#else
        simple.loop();
#endif

        volScalarField nueff = problem->turbulence->nuEff().ref();

        fvVectorMatrix UEqn
        (
            fvm::div(phi, U)
          - fvm::laplacian(nueff, U)
          - fvc::div(nueff * dev2(T(fvc::grad(U))))
        );

        UEqn.relax();

        if (collectResidualIndicator_)
        {
            tmp<vectorField> tResidualU = UEqn.residual();
            accumulateNormalizedResidual
            (
                residualIndicator_,
                tResidualU()
            );

            // The standard ROM treats grad(p) separately from UEqn.
            // Include its spatial magnitude in the sampling indicator as
            // a separate normalized contribution, without changing the
            // algebra of the full solver.
            tmp<volVectorField> tGradPIndicator = fvc::grad(P);

            accumulateNormalizedResidual
            (
                residualIndicator_,
                tGradPIndicator().primitiveField()
            );
        }

        List<Eigen::MatrixXd> RedLinSysU =
            ULmodes.project(UEqn, UprojN);

        RedLinSysU[1] =
            RedLinSysU[1]
          - projGradModP * b;

        a =
            reducedProblem::solveLinearSys
            (
                RedLinSysU,
                a,
                uresidual,
                vel_now
            );

        ULmodes.reconstruct(U, a, "U");

        volScalarField rAU(1.0 / UEqn.A());

        volVectorField HbyA
        (
            constrainHbyA
            (
                1.0 / UEqn.A() * UEqn.H(),
                U,
                P
            )
        );

        surfaceScalarField phiHbyA
        (
            "phiHbyA",
            fvc::flux(HbyA)
        );

        adjustPhi(phiHbyA, U, P);

        tmp<volScalarField> rAtU(rAU);

        if (simple.consistent())
        {
            rAtU = 1.0 / (1.0 / rAU - UEqn.H1());

            phiHbyA +=
                fvc::interpolate(rAtU() - rAU)
              * fvc::snGrad(P)
              * mesh.magSf();

            HbyA -=
                (rAU - rAtU())
              * fvc::grad(P);
        }

        List<Eigen::MatrixXd> RedLinSysP;

        while (simple.correctNonOrthogonal())
        {
            fvScalarMatrix pEqn
            (
                fvm::laplacian(rAtU(), P)
             == fvc::div(phiHbyA)
            );

            if (collectResidualIndicator_)
            {
                tmp<scalarField> tResidualP = pEqn.residual();
                accumulateNormalizedResidual
                (
                    residualIndicator_,
                    tResidualP()
                );
            }

            RedLinSysP =
                problem->Pmodes.project
                (
                    pEqn,
                    PprojN
                );

            b =
                reducedProblem::solveLinearSys
                (
                    RedLinSysP,
                    b,
                    presidual
                );

            problem->Pmodes.reconstruct(P, b, "p");

            if (simple.finalNonOrthogonalIter())
            {
                phi = phiHbyA - pEqn.flux();
            }
        }

        P.relax();

        U =
            HbyA
          - rAtU()
          * fvc::grad(P);

        U.correctBoundaryConditions();

        uresidualOld = uresidualOld - uresidual;
        presidualOld = presidualOld - presidual;

        uresidualOld = uresidualOld.cwiseAbs();
        presidualOld = presidualOld.cwiseAbs();

        residual_jump =
            std::max
            (
                uresidualOld.sum(),
                presidualOld.sum()
            );

        uresidualOld = uresidual;
        presidualOld = presidual;

        uresidual = uresidual.cwiseAbs();
        presidual = presidual.cwiseAbs();

        U_norm_res =
            uresidual.sum()
          / (RedLinSysU[1].cwiseAbs()).sum();

        P_norm_res =
            presidual.sum()
          / (RedLinSysP[1].cwiseAbs()).sum();

        if (problem->para->debug)
        {
            Info << "Residual jump = "
                 << residual_jump << endl;

            Info << "Normalized residual = "
                 << std::max(U_norm_res, P_norm_res)
                 << endl;
        }
    }

    Info << "Solution " << counter
         << " converged in " << iter
         << " iterations." << endl;

    Info << "Final normalized residual for velocity: "
         << U_norm_res << endl;

    Info << "Final normalized residual for pressure: "
         << P_norm_res << endl;

    if (collectResidualIndicator_)
    {
        writeResidualIndicator();
    }

    ULmodes.reconstruct(U, a, "Uaux");

    P.rename("Paux");
    problem->Pmodes.reconstruct(P, b, "Paux");

    ITHACAstream::exportSolution(U, name(counter), Folder);
    ITHACAstream::exportSolution(P, name(counter), Folder);

    runTime.setTime(runTime.startTime(), 0);
}



void reducedSimpleSteadyNS::writeResidualIndicator
(
    const word& fieldName
) const
{
    M_Assert
    (
        problem != nullptr,
        "Cannot write residual indicator without a full problem"
    );

    const fvMesh& mesh = problem->_mesh();

    M_Assert
    (
        residualIndicator_.size() == mesh.nCells(),
        "Residual indicator has not been collected yet"
    );

    volScalarField indicatorField
    (
        IOobject
        (
            fieldName,
            problem->_runTime().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
        dimensionedScalar("zero", dimless, 0.0)
    );

    indicatorField.primitiveFieldRef() = residualIndicator_;
    indicatorField.write();

    scalar localMax = 0.0;

    forAll(residualIndicator_, celli)
    {
        localMax = max(localMax, residualIndicator_[celli]);
    }

    reduce(localMax, maxOp<scalar>());

    Info<< "Written residual sampling indicator '"
        << fieldName << "'"
        << " (global max = " << localMax << ")"
        << endl;
}


// * * * * * * * * * * Sampled setup * * * * * * * * * * * * * * * * * //

void reducedSimpleSteadyNS::setupSampled
(
    SampledMesh& sampledMesh,
    int NmodesUproj,
    int NmodesPproj,
    int NmodesSup
)
{
    sampledMeshPtr_ = &sampledMesh;

    // Build exactly the same velocity basis ordering used by the standard
    // SIMPLE ROM: lift functions first, then velocity POD modes and,
    // optionally, supremizer modes.
    ULmodes.resize(0);

    for (int i = 0; i < problem->inletIndex.rows(); ++i)
    {
        ULmodes.append((problem->liftfield[i]).clone());
    }

    for (int i = 0; i < NmodesUproj; ++i)
    {
        ULmodes.append((problem->Umodes.toPtrList()[i]).clone());
    }

    for (int i = 0; i < NmodesSup; ++i)
    {
        ULmodes.append((problem->supmodes.toPtrList()[i]).clone());
    }

    if (NmodesUproj == 0)
    {
        UprojN = ULmodes.size();
    }
    else
    {
        // Preserve the semantics of solveOnline_Simple().
        UprojN = NmodesUproj + NmodesSup;
    }

    if (NmodesPproj == 0)
    {
        PprojN = problem->Pmodes.size();
    }
    else
    {
        PprojN = NmodesPproj;
    }

    sampledUmodes_.reset
    (
        new SampledModes<vector, fvPatchField, volMesh>
        (
            ULmodes,
            sampledMesh.subset(),
            sampledMesh.sampledCells(),
            UprojN
        )
    );

    sampledPmodes_.reset
    (
        new SampledModes<scalar, fvPatchField, volMesh>
        (
            problem->Pmodes,
            sampledMesh.subset(),
            sampledMesh.sampledCells(),
            PprojN
        )
    );

    // ---------------------------------------------------------------
    // Pressure-gradient term for the sampled Galerkin momentum system.
    //
    // The standard solver uses
    //
    //     projGradModP = ULmodes.project(grad(pModes), ...);
    //
    // and subsequently subtracts projGradModP*b from the reduced RHS.
    // Do exactly the same operation here, but integrate only over the
    // sampled cells.  The volume weighting is important because
    // Modes::project(field, ..., "G") uses the FV L2 inner product.
    // ---------------------------------------------------------------
    const PtrList<volVectorField>& UsubModes = sampledUmodes_->subModes();
    const PtrList<volScalarField>& PsubModes = sampledPmodes_->subModes();
    const labelList& sampledSubCells = sampledUmodes_->sampledSubCells();
    const scalarField& Vsub = sampledMesh.subMesh().V();

    sampledProjGradModP_ =
        Eigen::MatrixXd::Zero(UprojN, PprojN);

    for (label pModeI = 0; pModeI < PprojN; ++pModeI)
    {
        tmp<volVectorField> tGradP = fvc::grad(PsubModes[pModeI]);
        const vectorField& gradP = tGradP().primitiveField();

        for (label uModeI = 0; uModeI < UprojN; ++uModeI)
        {
            const vectorField& uMode =
                UsubModes[uModeI].primitiveField();

            scalar value = 0.0;

            forAll(sampledSubCells, sampleI)
            {
                const label celli = sampledSubCells[sampleI];
                value += Vsub[celli]*(uMode[celli] & gradP[celli]);
            }

            reduce(value, sumOp<scalar>());
            sampledProjGradModP_(uModeI, pModeI) = value;
        }
    }

    sampledReady_ = true;

    if (Pstream::master())
    {
        Info << nl
             << "Hyper-reduced SIMPLE setup" << nl
             << "  velocity modes : " << UprojN << nl
             << "  pressure modes : " << PprojN << nl
             << "  sampled cells  : "
             << sampledMesh.sampledCells().size() << " local on proc 0"
             << nl
             << "  submesh cells  : "
             << sampledMesh.globalSubMeshSize() << " global"
             << nl << endl;
    }
}


// * * * * * * * * * * Sampled online solver * * * * * * * * * * * * * * //

void reducedSimpleSteadyNS::solveOnline_SimpleSampled
(
    scalar mu_now,
    int NmodesNut,
    word Folder,
    word projType
)
{
    M_Assert
    (
        sampledReady_ && sampledMeshPtr_,
        "Call setupSampled() before solveOnline_SimpleSampled()"
    );

    M_Assert
    (
        projType == "G",
        "The sampled SIMPLE reference path currently uses Galerkin (G) "
        "to reproduce solveOnline_Simple() exactly before reducing the sample set"
    );

    counter++;

    if (NmodesNut == 0)
    {
        NmodesNut = problem->nutModes.size();
    }

    Eigen::VectorXd uresidualOld = Eigen::VectorXd::Zero(UprojN);
    Eigen::VectorXd pResidualOld = Eigen::VectorXd::Zero(PprojN);
    Eigen::VectorXd uresidual = Eigen::VectorXd::Zero(UprojN);
    Eigen::VectorXd pResidual = Eigen::VectorXd::Zero(PprojN);

    scalar U_norm_res(1);
    scalar P_norm_res(1);

    Eigen::MatrixXd a = Eigen::VectorXd::Zero(UprojN);
    Eigen::MatrixXd b = Eigen::VectorXd::Zero(PprojN);

    a(0) = vel_now(0, 0);

    const float residualJumpLim =
        problem->para->ITHACAdict->lookupOrDefault<float>
        (
            "residualJumpLim",
            1e-5
        );

    const float normalizedResidualLim =
        problem->para->ITHACAdict->lookupOrDefault<float>
        (
            "normalizedResidualLim",
            1e-5
        );

    maxIterOn =
        problem->para->ITHACAdict->lookupOrDefault<int>
        (
            "maxIterOn",
            1000
        );

    scalar residual_jump(1 + residualJumpLim);

    // ============================================================
    // Diagnostics for checking full-mesh/submesh equivalence.
    // ============================================================
    const Switch hrEquivalenceTests =
        problem->para->ITHACAdict->lookupOrDefault<Switch>
        (
            "hrEquivalenceTests",
            false
        );

    const label hrTestMode =
        problem->para->ITHACAdict->lookupOrDefault<label>
        (
            "hrTestMode",
            0
        );

    SampledMesh& sampledMesh = *sampledMeshPtr_;
    fvMesh& subMesh = sampledMesh.subMesh();

    const PtrList<volVectorField>& UsubModes =
        sampledUmodes_->subModes();

    const PtrList<volScalarField>& PsubModes =
        sampledPmodes_->subModes();

    // The online state lives only on the sampled mesh.
    volVectorField USub
    (
        IOobject
        (
            "U",
            problem->_runTime().timeName(),
            subMesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        UsubModes[0]
    );

    USub *= 0.0;

    for (label modeI = 0; modeI < UprojN; ++modeI)
    {
        USub += a(modeI) * UsubModes[modeI];
    }

    volScalarField PSub
    (
        IOobject
        (
            "p",
            problem->_runTime().timeName(),
            subMesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        PsubModes[0]
    );

    PSub *= 0.0;

    for (label modeI = 0; modeI < PprojN; ++modeI)
    {
        PSub += b(modeI) * PsubModes[modeI];
    }

    USub.correctBoundaryConditions();
    PSub.correctBoundaryConditions();

    surfaceScalarField phiSub
    (
        IOobject
        (
            "phiHR",
            problem->_runTime().timeName(),
            subMesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        fvc::interpolate(USub) & subMesh.Sf()
    );

    // nuEff is updated by change_viscosity() before entering this routine.
    // Map it ONCE per online parameter, not once per SIMPLE iteration.
    tmp<volScalarField> tNuEffFull =
        problem->turbulence->nuEff();

    tmp<volScalarField> tNuEffSub =
        sampledMesh.subset().interpolate(tNuEffFull());

    volScalarField nueffSub
    (
        IOobject
        (
            "nuEffHR",
            problem->_runTime().timeName(),
            subMesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        tNuEffSub()
    );

    if (ITHACAutilities::isTurbulent())
    {
        WarningInFunction
            << "The sampled SIMPLE path currently maps nuEff once at the "
            << "beginning of the online solve. For a turbulence model whose "
            << "nuEff changes during SIMPLE iterations, update nueffSub inside "
            << "the loop or add a sampled turbulence closure."
            << endl;
    }

    // Full-mesh fields used only by the optional diagnostic.
    // They are reconstructed from exactly the same coefficients a,b used by
    // the sampled solve.
    volVectorField UFullDiag
    (
        IOobject
        (
            "UFullHRDiag",
            problem->_runTime().timeName(),
            problem->_mesh(),
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        problem->_U()
    );

    volScalarField PFullDiag
    (
        IOobject
        (
            "pFullHRDiag",
            problem->_runTime().timeName(),
            problem->_mesh(),
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        problem->_p()
    );

    volScalarField nueffFullDiag
    (
        IOobject
        (
            "nuEffFullHRDiag",
            problem->_runTime().timeName(),
            problem->_mesh(),
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        tNuEffFull()
    );

    int iter = 0;
    simpleControl& simple = problem->_simple();
    Time& runTime = problem->_runTime();

    List<Eigen::MatrixXd> RedLinSysP(2);

    while
    (
        (
            residual_jump > residualJumpLim
         || std::max(U_norm_res, P_norm_res) > normalizedResidualLim
        )
     && iter < maxIterOn
    )
    {
        ++iter;

        Info << "HR iteration " << iter << endl;

        // Required by field relaxation on the dynamically-created submesh
        // fields (in particular PSub.relax()).
        USub.storePrevIter();
        PSub.storePrevIter();

#if defined(OFVER) && (OFVER == 6)
        simple.loop(runTime);
#else
        simple.loop();
#endif

        // Keep the sampled formulation algebraically identical to the
        // standard SIMPLE ROM.  In particular, pressure is NOT inserted
        // directly in UEqnSub: the original solver projects the momentum
        // matrix first and subtracts the projected pressure gradient from
        // the reduced RHS afterwards.
        // ------------------------------------------------------------
        // Optional full-vs-sampled diagnostic at the SAME reduced state.
        // ------------------------------------------------------------
        autoPtr<surfaceScalarField> phiFullDiagPtr;

        if (hrEquivalenceTests && iter == 1)
        {
            UFullDiag *= 0.0;

            for (label modeI = 0; modeI < UprojN; ++modeI)
            {
                UFullDiag += a(modeI) * ULmodes[modeI];
            }

            PFullDiag *= 0.0;

            for (label modeI = 0; modeI < PprojN; ++modeI)
            {
                PFullDiag += b(modeI) * problem->Pmodes[modeI];
            }

            UFullDiag.correctBoundaryConditions();
            PFullDiag.correctBoundaryConditions();

            // Synchronize the previous-iteration state used by relax().
            UFullDiag.storePrevIter();
            PFullDiag.storePrevIter();

            phiFullDiagPtr.reset
            (
                new surfaceScalarField
                (
                    IOobject
                    (
                        "phiFullHRDiag",
                        problem->_runTime().timeName(),
                        problem->_mesh(),
                        IOobject::NO_READ,
                        IOobject::NO_WRITE
                    ),
                    fvc::interpolate(UFullDiag) & problem->_mesh().Sf()
                )
            );
        }

        fvVectorMatrix UEqnSub
        (
            fvm::div(phiSub, USub)
          - fvm::laplacian(nueffSub, USub)
          - fvc::div
            (
                nueffSub
              * dev2(T(fvc::grad(USub)))
            )
        );

        // ============================================================
        // FULL/SUBMESH EQUIVALENCE TESTS -- PRE RELAXATION
        // ============================================================
        autoPtr<fvVectorMatrix> UEqnFullDiagPtr;

        if (hrEquivalenceTests && iter == 1)
        {
            M_Assert
            (
                hrTestMode >= 0 && hrTestMode < UprojN,
                "hrTestMode must be in [0,UprojN)"
            );

            UEqnFullDiagPtr.reset
            (
                new fvVectorMatrix
                (
                    fvm::div(phiFullDiagPtr(), UFullDiag)
                  - fvm::laplacian(nueffFullDiag, UFullDiag)
                  - fvc::div
                    (
                        nueffFullDiag
                      * dev2(T(fvc::grad(UFullDiag)))
                    )
                )
            );

            // TEST 1: reduced system before relax()
            List<Eigen::MatrixXd> preFull =
                ULmodes.project
                (
                    UEqnFullDiagPtr(),
                    UprojN,
                    "G"
                );

            List<Eigen::MatrixXd> preSub =
                ULmodes.projectSampled
                (
                    UEqnSub,
                    sampledUmodes_(),
                    "G"
                );

            const scalar preRelA =
                (preSub[0] - preFull[0]).norm()
               /(preFull[0].norm() + SMALL);

            const scalar preRelB =
                (preSub[1] - preFull[1]).norm()
               /(preFull[1].norm() + SMALL);

            if (Pstream::master())
            {
                Info<< nl
                    << "========================================" << nl
                    << " HR EQUIVALENCE TEST 1: PRE-RELAX SYSTEM" << nl
                    << " ||Ar_sub-Ar_full||F / ||Ar_full||F = "
                    << preRelA << nl
                    << " ||br_sub-br_full||2 / ||br_full||2 = "
                    << preRelB << nl
                    << " ||Ar_full||F = " << preFull[0].norm() << nl
                    << " ||Ar_sub||F  = " << preSub[0].norm() << nl
                    << " ||br_full||2 = " << preFull[1].norm() << nl
                    << " ||br_sub||2  = " << preSub[1].norm() << nl
                    << "========================================"
                    << nl << endl;
            }

            // ============================================================
            // TEST 1B: is the sparse sampled operator mainly a scaled
            // version of the full operator?
            //
            // IMPORTANT:
            // A constant weight applied to every sampled cell multiplies
            // BOTH Ar and br by the same number and therefore cancels from
            // the reduced linear solve.  This test is diagnostic only.
            // It tells us whether the missing information is mostly a
            // global scale factor or a genuine change of operator shape.
            // ============================================================

            label nFullGlobal = problem->_mesh().nCells();
            label nSampleGlobal = sampledMesh.sampledCells().size();

            reduce(nFullGlobal, sumOp<label>());
            reduce(nSampleGlobal, sumOp<label>());

            const scalar uniformWeight =
                scalar(nFullGlobal)
               /scalar(std::max<label>(nSampleGlobal, 1));

            const Eigen::MatrixXd uniformAr =
                uniformWeight * preSub[0];

            const Eigen::MatrixXd uniformBr =
                uniformWeight * preSub[1];

            const scalar uniformRelA =
                (uniformAr - preFull[0]).norm()
               /(preFull[0].norm() + SMALL);

            const scalar uniformRelB =
                (uniformBr - preFull[1]).norm()
               /(preFull[1].norm() + SMALL);

            // Best possible SINGLE scalar multiplying the whole sampled
            // reduced matrix/vector in a least-squares sense.
            const scalar denomA =
                preSub[0].squaredNorm();

            const scalar denomB =
                preSub[1].squaredNorm();

            const scalar alphaA =
                denomA > SMALL
              ? (
                    preSub[0].cwiseProduct(preFull[0]).sum()
                   /denomA
                )
              : 0.0;

            const scalar alphaB =
                denomB > SMALL
              ? (
                    preSub[1].cwiseProduct(preFull[1]).sum()
                   /denomB
                )
              : 0.0;

            const scalar optimalScalarRelA =
                (alphaA*preSub[0] - preFull[0]).norm()
               /(preFull[0].norm() + SMALL);

            const scalar optimalScalarRelB =
                (alphaB*preSub[1] - preFull[1]).norm()
               /(preFull[1].norm() + SMALL);

            // Cosine/alignment: 1 means same direction in matrix/vector
            // space, irrespective of scale.
            const scalar alignmentA =
                preSub[0].cwiseProduct(preFull[0]).sum()
               /(
                    preSub[0].norm()*preFull[0].norm()
                  + SMALL
                );

            const scalar alignmentB =
                preSub[1].cwiseProduct(preFull[1]).sum()
               /(
                    preSub[1].norm()*preFull[1].norm()
                  + SMALL
                );

            if (Pstream::master())
            {
                Info<< nl
                    << "========================================" << nl
                    << " HR EQUIVALENCE TEST 1B: SCALAR WEIGHTS" << nl
                    << "----------------------------------------" << nl
                    << " Nfull global   = " << nFullGlobal << nl
                    << " Nsample global = " << nSampleGlobal << nl
                    << " uniform weight Nfull/Nsample = "
                    << uniformWeight << nl
                    << nl
                    << " UNIFORM SCALING" << nl
                    << " ||w Ar_sub-Ar_full||F / ||Ar_full||F = "
                    << uniformRelA << nl
                    << " ||w br_sub-br_full||2 / ||br_full||2 = "
                    << uniformRelB << nl
                    << nl
                    << " BEST SINGLE SCALAR FIT" << nl
                    << " alpha_A = " << alphaA << nl
                    << " alpha_b = " << alphaB << nl
                    << " best scalar Ar relative error = "
                    << optimalScalarRelA << nl
                    << " best scalar br relative error = "
                    << optimalScalarRelB << nl
                    << nl
                    << " ALIGNMENT (1 = same shape/direction)" << nl
                    << " alignment_A = " << alignmentA << nl
                    << " alignment_b = " << alignmentB << nl
                    << "========================================"
                    << nl << endl;
            }

            // TEST 2: direct A*v before relax()
            class projectableVectorMatrix
            :
                public fvVectorMatrix
            {
            public:

                projectableVectorMatrix(const fvVectorMatrix& A)
                :
                    fvVectorMatrix(A)
                {}

                void prepareComponent
                (
                    const scalarField& originalDiag,
                    const direction cmpt
                )
                {
                    diag() = originalDiag;
                    addBoundaryDiag(diag(), cmpt);
                }
            };

            projectableVectorMatrix AFull(UEqnFullDiagPtr());
            projectableVectorMatrix ASub(UEqnSub);

            const scalarField fullOriginalDiag(AFull.diag());
            const scalarField subOriginalDiag(ASub.diag());

            const vectorField& vFull =
                ULmodes[hrTestMode].primitiveField();

            const vectorField& vSub =
                sampledUmodes_()[hrTestMode].primitiveField();

            const labelList& cellMap =
                sampledMesh.subset().cellMap();

            scalar totalFull2 = 0.0;
            scalar totalSub2  = 0.0;
            scalar totalDiff2 = 0.0;

            for
            (
                direction cmpt = 0;
                cmpt < vector::nComponents;
                ++cmpt
            )
            {
                AFull.prepareComponent(fullOriginalDiag, cmpt);

                scalarField xFull(vFull.size(), 0.0);

                forAll(xFull, celli)
                {
                    xFull[celli] = vFull[celli][cmpt];
                }

                scalarField AxFull(xFull.size(), 0.0);

                const lduInterfaceFieldPtrsList fullInterfaces
                (
                    UEqnFullDiagPtr()
                        .psi()
                        .boundaryField()
                        .scalarInterfaces()
                );

                FieldField<Field, scalar> fullBouCoeffs
                (
                    AFull.boundaryCoeffs().component(cmpt)
                );

                AFull.Amul
                (
                    AxFull,
                    xFull,
                    fullBouCoeffs,
                    fullInterfaces,
                    cmpt
                );

                ASub.prepareComponent(subOriginalDiag, cmpt);

                scalarField xSub(vSub.size(), 0.0);

                forAll(xSub, subCelli)
                {
                    xSub[subCelli] = vSub[subCelli][cmpt];
                }

                scalarField AxSub(xSub.size(), 0.0);

                const lduInterfaceFieldPtrsList subInterfaces
                (
                    UEqnSub
                        .psi()
                        .boundaryField()
                        .scalarInterfaces()
                );

                FieldField<Field, scalar> subBouCoeffs
                (
                    ASub.boundaryCoeffs().component(cmpt)
                );

                ASub.Amul
                (
                    AxSub,
                    xSub,
                    subBouCoeffs,
                    subInterfaces,
                    cmpt
                );

                scalar compFull2 = 0.0;
                scalar compSub2  = 0.0;
                scalar compDiff2 = 0.0;

                forAll(AxSub, subCelli)
                {
                    const label fullCelli = cellMap[subCelli];

                    const scalar vf = AxFull[fullCelli];
                    const scalar vs = AxSub[subCelli];

                    compFull2 += sqr(vf);
                    compSub2  += sqr(vs);
                    compDiff2 += sqr(vs - vf);
                }

                reduce(compFull2, sumOp<scalar>());
                reduce(compSub2,  sumOp<scalar>());
                reduce(compDiff2, sumOp<scalar>());

                totalFull2 += compFull2;
                totalSub2  += compSub2;
                totalDiff2 += compDiff2;

                if (Pstream::master())
                {
                    Info<< "A*v pre-relax, mode " << hrTestMode
                        << ", component " << cmpt
                        << ": relative error = "
                        << Foam::sqrt(compDiff2)
                          /(Foam::sqrt(compFull2) + SMALL)
                        << ", ||Afull*v|| = "
                        << Foam::sqrt(compFull2)
                        << ", ||Asub*v|| = "
                        << Foam::sqrt(compSub2)
                        << endl;
                }
            }

            if (Pstream::master())
            {
                Info<< nl
                    << "========================================" << nl
                    << " HR EQUIVALENCE TEST 2: DIRECT A*v PRE-RELAX" << nl
                    << " mode = " << hrTestMode << nl
                    << " total relative A*v error = "
                    << Foam::sqrt(totalDiff2)
                      /(Foam::sqrt(totalFull2) + SMALL)
                    << nl
                    << " ||Afull*v|| = " << Foam::sqrt(totalFull2) << nl
                    << " ||Asub*v||  = " << Foam::sqrt(totalSub2) << nl
                    << "========================================"
                    << nl << endl;
            }
        }

        // Apply relaxation.
        UEqnSub.relax();

        if (hrEquivalenceTests && iter == 1)
        {
            UEqnFullDiagPtr().relax();

            // TEST 3: reduced system after relax()
            List<Eigen::MatrixXd> postFull =
                ULmodes.project
                (
                    UEqnFullDiagPtr(),
                    UprojN,
                    "G"
                );

            List<Eigen::MatrixXd> postSub =
                ULmodes.projectSampled
                (
                    UEqnSub,
                    sampledUmodes_(),
                    "G"
                );

            const scalar postRelA =
                (postSub[0] - postFull[0]).norm()
               /(postFull[0].norm() + SMALL);

            const scalar postRelB =
                (postSub[1] - postFull[1]).norm()
               /(postFull[1].norm() + SMALL);

            if (Pstream::master())
            {
                Info<< nl
                    << "========================================" << nl
                    << " HR EQUIVALENCE TEST 3: POST-RELAX SYSTEM" << nl
                    << " ||Ar_sub-Ar_full||F / ||Ar_full||F = "
                    << postRelA << nl
                    << " ||br_sub-br_full||2 / ||br_full||2 = "
                    << postRelB << nl
                    << " ||Ar_full||F = " << postFull[0].norm() << nl
                    << " ||Ar_sub||F  = " << postSub[0].norm() << nl
                    << " ||br_full||2 = " << postFull[1].norm() << nl
                    << " ||br_sub||2  = " << postSub[1].norm() << nl
                    << "========================================"
                    << nl << endl;
            }
        }

        List<Eigen::MatrixXd> RedLinSysU =
            ULmodes.projectSampled
            (
                UEqnSub,
                sampledUmodes_(),
                projType
            );

        // Same split used by solveOnline_Simple():
        //     br <- br - <velocity test modes, grad(p)> b
        RedLinSysU[1] -= sampledProjGradModP_ * b;

        // TEST 4: pressure-gradient reduced contribution.
        if (hrEquivalenceTests && iter == 1)
        {
            PtrList<volVectorField> gradPDiag;

            for (label pModeI = 0; pModeI < PprojN; ++pModeI)
            {
                gradPDiag.append
                (
                    fvc::grad(problem->Pmodes[pModeI]).ptr()
                );
            }

            Eigen::MatrixXd fullProjGradP =
                ULmodes.project
                (
                    gradPDiag,
                    UprojN
                );

            const scalar gradRel =
                (sampledProjGradModP_ - fullProjGradP).norm()
               /(fullProjGradP.norm() + SMALL);

            if (Pstream::master())
            {
                Info<< nl
                    << "========================================" << nl
                    << " HR EQUIVALENCE TEST 4: PRESSURE GRADIENT" << nl
                    << " ||G_sub-G_full||F / ||G_full||F = "
                    << gradRel << nl
                    << " ||G_full||F = " << fullProjGradP.norm() << nl
                    << " ||G_sub||F  = "
                    << sampledProjGradModP_.norm() << nl
                    << "========================================"
                    << nl << endl;
            }
        }

        a =
            reducedProblem::solveLinearSys
            (
                RedLinSysU,
                a,
                uresidual,
                vel_now
            );

        // Reconstruct U only on the submesh.
        USub *= 0.0;

        for (label modeI = 0; modeI < UprojN; ++modeI)
        {
            USub += a(modeI) * UsubModes[modeI];
        }

        USub.correctBoundaryConditions();

        // Keep the diagnostic full state synchronized with the newly solved
        // velocity coefficients before forming the pressure equation.
        if (hrEquivalenceTests && iter == 1)
        {
            UFullDiag *= 0.0;

            for (label modeI = 0; modeI < UprojN; ++modeI)
            {
                UFullDiag += a(modeI) * ULmodes[modeI];
            }

            UFullDiag.correctBoundaryConditions();
        }

        volScalarField rAUSub
        (
            1.0 / UEqnSub.A()
        );

        volVectorField HbyASub
        (
            constrainHbyA
            (
                1.0 / UEqnSub.A() * UEqnSub.H(),
                USub,
                PSub
            )
        );

        surfaceScalarField phiHbyASub
        (
            IOobject
            (
                "phiHbyAHR",
                problem->_runTime().timeName(),
                subMesh,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            fvc::flux(HbyASub)
        );

        adjustPhi(phiHbyASub, USub, PSub);

        tmp<volScalarField> rAtUSub(rAUSub);

        if (simple.consistent())
        {
            rAtUSub =
                1.0
              / (
                    1.0 / rAUSub
                  - UEqnSub.H1()
                );

            phiHbyASub +=
                fvc::interpolate(rAtUSub() - rAUSub)
              * fvc::snGrad(PSub)
              * subMesh.magSf();

            HbyASub -=
                (rAUSub - rAtUSub())
              * fvc::grad(PSub);
        }

        // Full-mesh SIMPLE pressure ingredients corresponding to the same
        // momentum matrix/state, used only for the first-iteration diagnostic.
        autoPtr<volScalarField> rAtUFullDiagPtr;
        autoPtr<surfaceScalarField> phiHbyAFullDiagPtr;

        if (hrEquivalenceTests && iter == 1)
        {
            volScalarField rAUFullDiag
            (
                1.0 / UEqnFullDiagPtr().A()
            );

            volVectorField HbyAFullDiag
            (
                constrainHbyA
                (
                    1.0 / UEqnFullDiagPtr().A() * UEqnFullDiagPtr().H(),
                    UFullDiag,
                    PFullDiag
                )
            );

            phiHbyAFullDiagPtr.reset
            (
                new surfaceScalarField
                (
                    IOobject
                    (
                        "phiHbyAFullHRDiag",
                        problem->_runTime().timeName(),
                        problem->_mesh(),
                        IOobject::NO_READ,
                        IOobject::NO_WRITE
                    ),
                    fvc::flux(HbyAFullDiag)
                )
            );

            adjustPhi
            (
                phiHbyAFullDiagPtr(),
                UFullDiag,
                PFullDiag
            );

            rAtUFullDiagPtr.reset
            (
                new volScalarField(rAUFullDiag)
            );

            if (simple.consistent())
            {
                rAtUFullDiagPtr() =
                    1.0
                  / (
                        1.0 / rAUFullDiag
                      - UEqnFullDiagPtr().H1()
                    );

                phiHbyAFullDiagPtr() +=
                    fvc::interpolate
                    (
                        rAtUFullDiagPtr() - rAUFullDiag
                    )
                  * fvc::snGrad(PFullDiag)
                  * problem->_mesh().magSf();

                HbyAFullDiag -=
                    (rAUFullDiag - rAtUFullDiagPtr())
                  * fvc::grad(PFullDiag);
            }
        }

        while (simple.correctNonOrthogonal())
        {
            // Pressure equation assembled ONLY on the sampled submesh.
            fvScalarMatrix pEqnSub
            (
                fvm::laplacian(rAtUSub(), PSub)
             == fvc::div(phiHbyASub)
            );

            RedLinSysP =
                problem->Pmodes.projectSampled
                (
                    pEqnSub,
                    sampledPmodes_(),
                    projType
                );

            // TEST 5: pressure reduced system.
            if (hrEquivalenceTests && iter == 1)
            {
                fvScalarMatrix pEqnFullDiag
                (
                    fvm::laplacian
                    (
                        rAtUFullDiagPtr(),
                        PFullDiag
                    )
                 == fvc::div(phiHbyAFullDiagPtr())
                );

                List<Eigen::MatrixXd> pFull =
                    problem->Pmodes.project
                    (
                        pEqnFullDiag,
                        PprojN,
                        "G"
                    );

                const scalar pRelA =
                    (RedLinSysP[0] - pFull[0]).norm()
                   /(pFull[0].norm() + SMALL);

                const scalar pRelB =
                    (RedLinSysP[1] - pFull[1]).norm()
                   /(pFull[1].norm() + SMALL);

                if (Pstream::master())
                {
                    Info<< nl
                        << "========================================" << nl
                        << " HR EQUIVALENCE TEST 5: PRESSURE SYSTEM" << nl
                        << " ||Ar_sub-Ar_full||F / ||Ar_full||F = "
                        << pRelA << nl
                        << " ||br_sub-br_full||2 / ||br_full||2 = "
                        << pRelB << nl
                        << " ||Ar_full||F = " << pFull[0].norm() << nl
                        << " ||Ar_sub||F  = " << RedLinSysP[0].norm() << nl
                        << " ||br_full||2 = " << pFull[1].norm() << nl
                        << " ||br_sub||2  = " << RedLinSysP[1].norm() << nl
                        << "========================================"
                        << nl << endl;
                }
            }

            b =
                reducedProblem::solveLinearSys
                (
                    RedLinSysP,
                    b,
                    pResidual
                );

            // Reconstruct pressure only on the submesh.
            PSub *= 0.0;

            for (label modeI = 0; modeI < PprojN; ++modeI)
            {
                PSub += b(modeI) * PsubModes[modeI];
            }

            PSub.correctBoundaryConditions();

            if (simple.finalNonOrthogonalIter())
            {
                phiSub =
                    phiHbyASub
                  - pEqnSub.flux();
            }
        }

        PSub.relax();

        USub =
            HbyASub
          - rAtUSub()
          * fvc::grad(PSub);

        USub.correctBoundaryConditions();

        uresidualOld = (uresidualOld - uresidual).cwiseAbs();
        pResidualOld = (pResidualOld - pResidual).cwiseAbs();

        residual_jump =
            std::max
            (
                uresidualOld.sum(),
                pResidualOld.sum()
            );

        uresidualOld = uresidual;
        pResidualOld = pResidual;

        const scalar rhsUNorm =
            RedLinSysU[1].cwiseAbs().sum();

        const scalar rhsPNorm =
            RedLinSysP[1].cwiseAbs().sum();

        U_norm_res =
            uresidual.cwiseAbs().sum()
          / max(rhsUNorm, SMALL);

        P_norm_res =
            pResidual.cwiseAbs().sum()
          / max(rhsPNorm, SMALL);

        if (problem->para->debug)
        {
            Info << "HR residual jump = "
                 << residual_jump << endl;

            Info << "HR normalized residual = "
                 << std::max(U_norm_res, P_norm_res)
                 << endl;
        }
    }

    Info << "HR solution " << counter
         << " converged in " << iter
         << " iterations." << endl;

    Info << "Final HR normalized residual for velocity: "
         << U_norm_res << endl;

    Info << "Final HR normalized residual for pressure: "
         << P_norm_res << endl;

    // Full-order reconstruction happens ONCE, only for output.
    volVectorField& U = problem->_U();
    volScalarField& P = problem->_p();

    ULmodes.reconstruct(U, a, "Uaux");

    P.rename("Paux");
    problem->Pmodes.reconstruct(P, b, "Paux");

    ITHACAstream::exportSolution(U, name(counter), Folder);
    ITHACAstream::exportSolution(P, name(counter), Folder);

    runTime.setTime(runTime.startTime(), 0);
}


// * * * * * * * * * * Boundary conditions * * * * * * * * * * * * * * * //

void reducedSimpleSteadyNS::setOnlineVelocity(Eigen::MatrixXd vel)
{
    M_Assert
    (
        problem->inletIndex.rows() == vel.size(),
        "Imposed boundary conditions dimensions do not match given values matrix dimensions"
    );

    Eigen::MatrixXd vel_scal;
    vel_scal.resize(vel.rows(), vel.cols());

    for (int k = 0; k < problem->inletIndex.rows(); k++)
    {
        int p = problem->inletIndex(k, 0);
        int l = problem->inletIndex(k, 1);

        scalar area =
            gSum
            (
                problem->liftfield[0].mesh().magSf().boundaryField()[p]
            );

        scalar u_lf =
            gSum
            (
                problem->liftfield[k].mesh().magSf().boundaryField()[p]
              * problem->liftfield[k].boundaryField()[p]
            ).component(l)
          / area;

        vel_scal(k, 0) =
            vel(k, 0)
          / u_lf;
    }

    vel_now = vel_scal;
}
