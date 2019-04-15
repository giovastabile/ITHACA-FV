/*---------------------------------------------------------------------------*\
     ██╗████████╗██╗  ██╗ █████╗  ██████╗ █████╗       ███████╗██╗   ██╗
     ██║╚══██╔══╝██║  ██║██╔══██╗██╔════╝██╔══██╗      ██╔════╝██║   ██║
     ██║   ██║   ███████║███████║██║     ███████║█████╗█████╗  ██║   ██║
     ██║   ██║   ██╔══██║██╔══██║██║     ██╔══██║╚════╝██╔══╝  ╚██╗ ██╔╝
     ██║   ██║   ██║  ██║██║  ██║╚██████╗██║  ██║      ██║      ╚████╔╝
     ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝      ╚═╝       ╚═══╝

 * In real Time Highly Advanced Computational Applications for Finite Volumes
 * Copyright (C) 2017 by the ITHACA-FV authors
-------------------------------------------------------------------------------

License
    This file is part of ITHACA-FV

    ITHACA-FV is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    ITHACA-FV is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with ITHACA-FV. If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

/// \file
/// Source file of the reducedSteadyNS class

#include "reducedSimpleSteadyNS.H"

// * * * * * * * * * * * * * * * Constructors * * * * * * * * * * * * * * * * //

// Constructor
reducedSimpleSteadyNS::reducedSimpleSteadyNS()
{
}

reducedSimpleSteadyNS::reducedSimpleSteadyNS(steadyNS_simple& FOMproblem)
    :
    problem(&FOMproblem)
{
    // Create a new Umodes set where the first ones are the lift functions
    for (int i = 0; i < problem->inletIndex.rows(); i++)
    {
        ULmodes.append(problem->liftfield[i]);
    }

    for (int i = 0; i < problem->Umodes.size(); i++)
    {
        ULmodes.append(problem->Umodes.toPtrList()[i]);
    }
}

// * * * * * * * * * * * * * * * Solve Functions  * * * * * * * * * * * * * //

void reducedSimpleSteadyNS::solveOnline_Simple(scalar mu_now,
        int NmodesUproj, int NmodesPproj, int NmodesSup, word Folder)
{
    ULmodes.resize(0);

    for (int i = 0; i < problem->inletIndex.rows(); i++)
    {
        ULmodes.append(problem->liftfield[i]);
    }

    for (int i = 0; i < NmodesUproj; i++)
    {
        ULmodes.append(problem->Umodes.toPtrList()[i]);
    }

    for (int i = 0; i < NmodesSup; i++)
    {
        ULmodes.append(problem->supmodes.toPtrList()[i]);
    }

    counter++;
    scalar UprojN;
    scalar PprojN;

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

    Eigen::VectorXd uresidualOld = Eigen::VectorXd::Zero(UprojN);
    Eigen::VectorXd presidualOld = Eigen::VectorXd::Zero(PprojN);
    Eigen::VectorXd uresidual;
    Eigen::VectorXd presidual;
    scalar U_norm_res(1);
    scalar P_norm_res(1);
    Eigen::MatrixXd a = Eigen::VectorXd::Zero(UprojN);
    Eigen::MatrixXd b = Eigen::VectorXd::Zero(PprojN);
    a(0) = vel_now(0, 0);
    ITHACAparameters para;
    float residualJumpLim =
        para.ITHACAdict->lookupOrDefault<float>("residualJumpLim", 1e-5);
    float normalizedResidualLim =
        para.ITHACAdict->lookupOrDefault<float>("normalizedResidualLim", 1e-5);
    scalar residual_jump(1 + residualJumpLim);
    problem->restart();
    volVectorField Uaux("U", problem->_U());
    volScalarField Paux("p", problem->_p());
    volScalarField& p = problem->_p();
    volVectorField& U = problem->_U();
    surfaceScalarField phi(problem->_phi());
    phi = fvc::interpolate(U) & U.mesh().Sf();
    int iter = 0;
    simpleControl& simple = problem->_simple();
    setRefCell(p, simple.dict(), problem->pRefCell, problem->pRefValue);

    while (residual_jump > residualJumpLim
            || std::max(U_norm_res, P_norm_res) > normalizedResidualLim)
    {
        iter++;
        Paux.storePrevIter();
        //Uaux = ULmodes.reconstruct(a, "Uaux");
        //problem->_phi() = linearInterpolate(Uaux) & problem->_U().mesh().Sf();
        //fvVectorMatrix Au(get_Umatrix_Online(Uaux, Paux));
        fvVectorMatrix UEqn
        (
            fvm::div(phi, Uaux)
            + problem->turbulence->divDevReff(Uaux)
            == -fvc::grad(Paux)
        );
        UEqn.relax();
        //UEqn.solve();
        List<Eigen::MatrixXd> RedLinSysU = ULmodes.project(UEqn, UprojN);
        a = reducedProblem::solveLinearSys(RedLinSysU, a, uresidual, vel_now);
        ULmodes.reconstruct(Uaux, a, "U");
        //phi = fvc::interpolate(Uaux) & Uaux.mesh().Sf();
        phi = fvc::flux(UEqn.H() / UEqn.A());
        //Au.relax();
        //problem->_phi() = linearInterpolate(Uaux) & problem->_U().mesh().Sf();
        //fvScalarMatrix Ap(get_Pmatrix_Online(Uaux, Paux));
        // dimensionedScalar fake("fake", dimensionSet(0,0,1,0,0,0,0), scalar(1.0));
        // fvScalarMatrix pEqn
        // (
        //     fvm::laplacian(1 / UEqn.A(), Paux) == - fvc::div(fvc::div(phi,Uaux))*fake
        // );
        fvScalarMatrix pEqn
        (
            fvm::laplacian(1 / UEqn.A(), Paux) == fvc::div(phi)
        );
        //pEqn.solve();
        List<Eigen::MatrixXd> RedLinSysP = problem->Pmodes.project(pEqn, PprojN);
        b = reducedProblem::solveLinearSys(RedLinSysP, b, presidual);
        problem->Pmodes.reconstruct(Paux, b, "p");
        //Uaux += 1/UEqn.A()*fvc::grad(Paux);
        phi -= pEqn.flux();
// #include "continuityErrs.H"
        Paux.relax();
        // //
        // //Paux = Paux*0.3;
        // //problem->_phi() = problem->phiHbyA_global() - Ap.flux();
        // //Info << presidual.norm() << endl;
        //
        //
        uresidualOld = uresidualOld - uresidual;
        presidualOld = presidualOld - presidual;
        uresidualOld = uresidualOld.cwiseAbs();
        presidualOld = presidualOld.cwiseAbs();
        // //std::cout << uresidualOld.sum() << std::endl;
        // //std::cout << presidualOld.sum() << std::endl;
        residual_jump = std::max(uresidualOld.sum(), presidualOld.sum());
        uresidualOld = uresidual;
        presidualOld = presidual;
        uresidual = uresidual.cwiseAbs();
        presidual = presidual.cwiseAbs();
        U_norm_res = uresidual.sum() / (RedLinSysU[1].cwiseAbs()).sum();
        P_norm_res = presidual.sum() / (RedLinSysP[1].cwiseAbs()).sum();

        if (para.debug)
        {
            std::cout << "Residual jump = " << residual_jump << std::endl;
            std::cout << "Normalized residual = " << std::max(U_norm_res,
                      P_norm_res) << std::endl;
        }
    }

    std::cout << "Solution " << counter << " converged in " << iter <<
              " iterations." << std::endl;
    std::cout << "Final normalized residual for velocity: " << U_norm_res <<
              std::endl;
    std::cout << "Final normalized residual for pressure: " << P_norm_res <<
              std::endl;
    ULmodes.reconstruct(Uaux, a, "Uaux");
    problem->Pmodes.reconstruct(Paux, b, "Paux");
    Uaux.rename("Uaux");
    Paux.rename("Paux");
    ITHACAstream::exportSolution(Uaux, name(counter), Folder);
    ITHACAstream::exportSolution(Paux, name(counter), Folder);
}

fvVectorMatrix reducedSimpleSteadyNS::get_Umatrix_Online(volVectorField& U,
        volScalarField& p)
{
    surfaceScalarField& phi = problem->_phi();
    fvVectorMatrix Ueqn
    (
        fvm::div(phi, U)
        + problem->turbulence->divDevReff(U)
        ==
        -fvc::grad(p)
    );
    Ueqn.relax();
    Ueqn.solve();
    problem->_phi() = fvc::interpolate(U) & U.mesh().Sf();
    problem->Ueqn_global = &Ueqn;
    return Ueqn;
}

fvScalarMatrix reducedSimpleSteadyNS::get_Pmatrix_Online(volVectorField& U,
        volScalarField& p)
{
    fvScalarMatrix pEqn
    (
        fvm::laplacian(1 / problem->Ueqn_global->A(), p) == fvc::div(problem->_phi())
    );
    pEqn.setReference(0, 0.0);
    pEqn.solve();
    problem->_phi() -= pEqn.flux();
    p.relax();
    return pEqn;
}


void reducedSimpleSteadyNS::setOnlineVelocity(Eigen::MatrixXd vel)
{
    assert(problem->inletIndex.rows() == vel.rows()
           && "Imposed boundary conditions dimensions do not match given values matrix dimensions");
    vel_now.resize(vel.rows(), vel.cols());

    for (int k = 0; k < problem->inletIndex.rows(); k++)
    {
        label p = problem->inletIndex(k, 0);
        label l = problem->inletIndex(k, 1);
        scalar area = gSum(problem->liftfield[0].mesh().magSf().boundaryField()[p]);
        scalar u_lf = gSum(problem->liftfield[k].mesh().magSf().boundaryField()[p] *
                           problem->liftfield[k].boundaryField()[p]).component(l) / area;
        vel_now(k, 0) = vel(k, 0) / u_lf;
    }
}