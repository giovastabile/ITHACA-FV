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

    ULmodes.reconstruct(U, a, "Uaux");

    P.rename("Paux");
    problem->Pmodes.reconstruct(P, b, "Paux");

    ITHACAstream::exportSolution(U, name(counter), Folder);
    ITHACAstream::exportSolution(P, name(counter), Folder);

    runTime.setTime(runTime.startTime(), 0);
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
        projType == "G" || projType == "PG",
        "projType must be G or PG"
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

    // Store previous iteration fields for relaxation
    USub.storePrevIter();
    PSub.storePrevIter();

#if defined(OFVER) && (OFVER == 6)
    simple.loop(runTime);
#else
    simple.loop();
#endif

        // Momentum equation assembled ONLY on the sampled submesh.
        //
        // The pressure gradient is included directly in the sampled residual,
        // so no full-mesh grad(p_modes) projection is required.
        fvVectorMatrix UEqnSub
        (
            fvm::div(phiSub, USub)
          - fvm::laplacian(nueffSub, USub)
          - fvc::div
            (
                nueffSub
              * dev2(T(fvc::grad(USub)))
            )
         ==
           -fvc::grad(PSub)
        );

        UEqnSub.relax();

        List<Eigen::MatrixXd> RedLinSysU =
            ULmodes.projectSampled
            (
                UEqnSub,
                sampledUmodes_(),
                projType
            );

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

