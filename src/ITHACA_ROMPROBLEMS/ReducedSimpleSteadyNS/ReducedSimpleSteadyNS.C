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

#include "ReducedSimpleSteadyNS.H"

// * * * * * * * * * * * * * * * Constructors * * * * * * * * * * * * * * * * //

// Constructor
reducedSimpleSteadyNS::reducedSimpleSteadyNS()
{
}

reducedSimpleSteadyNS::reducedSimpleSteadyNS(SteadyNSSimple& FOMproblem)
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

    ULmodes.toEigen();
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
    Eigen::VectorXd uresidual = Eigen::VectorXd::Zero(UprojN);
    Eigen::VectorXd presidual = Eigen::VectorXd::Zero(PprojN);
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
    volScalarField& P = problem->_p();
    volVectorField& U = problem->_U();
    P.rename("p");
    surfaceScalarField phi(problem->_phi());
    phi = fvc::interpolate(U) & U.mesh().Sf();
    int iter = 0;
    simpleControl& simple = problem->_simple();

    while (residual_jump > residualJumpLim
            || std::max(U_norm_res, P_norm_res) > normalizedResidualLim)
    {
        iter++;
        P.storePrevIter();
        volScalarField nu = problem->_laminarTransport().nu();
        fvVectorMatrix UEqn1
        (
            fvm::div(phi, U)
        );
        fvVectorMatrix UEqn2
        (
            - fvm::laplacian(nu, U)
        );
        volVectorField UEqn4
        (
            - fvc::div(nu * dev2(T(fvc::grad(U))))
        );
        fvVectorMatrix UEqn = UEqn1 + UEqn2 + UEqn4;
        UEqn.relax();
        //UEqn == -fvc::grad(P);
        List<Eigen::MatrixXd> RedLinSysU = ULmodes.project(UEqn1, UprojN);

        for (int i = 0; i < ReducedOperators.size(); i++)
        {
            RedLinSysU[0] += onlineViscosity * ReducedOperators[i][0];
            RedLinSysU[1] += onlineViscosity * ReducedOperators[i][1];
        }

        RedLinSysU[0] += onlineViscosity * redDivDev2;
        RedLinSysU[1] += redGradP * b;
        a = reducedProblem::solveLinearSys(RedLinSysU, a, uresidual, vel_now);
        ULmodes.reconstruct(U, a, "U");
        volVectorField HbyA(constrainHbyA(1.0 / UEqn.A() * UEqn.H(), U, P));
        surfaceScalarField phiHbyA("phiHbyA", fvc::flux(HbyA));
        List<Eigen::MatrixXd> RedLinSysP;

        while (simple.correctNonOrthogonal())
        {
            fvScalarMatrix pEqn
            (
                fvm::laplacian(1 / UEqn.A(), P) == fvc::div(phiHbyA)
            );
            RedLinSysP = problem->Pmodes.project(pEqn, PprojN);
            b = reducedProblem::solveLinearSys(RedLinSysP, b, presidual);
            problem->Pmodes.reconstruct(P, b, "p");

            if (simple.finalNonOrthogonalIter())
            {
                phi = phiHbyA - pEqn.flux();
            }
        }

        P.relax();
        U = HbyA - 1.0 / UEqn.A() * fvc::grad(P);
        U.correctBoundaryConditions();
        uresidualOld = uresidualOld - uresidual;
        presidualOld = presidualOld - presidual;
        uresidualOld = uresidualOld.cwiseAbs();
        presidualOld = presidualOld.cwiseAbs();
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
    ULmodes.reconstruct(U, a, "Uaux");
    P.rename("Paux");
    problem->Pmodes.reconstruct(P, b, "Paux");
    ITHACAstream::exportSolution(U, name(counter), Folder);
    ITHACAstream::exportSolution(P, name(counter), Folder);
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
    M_Assert(problem->inletIndex.rows() == vel.size(),
             "Imposed boundary conditions dimensions do not match given values matrix dimensions");
    Eigen::MatrixXd vel_scal;
    vel_scal.resize(vel.rows(), vel.cols());

    for (int k = 0; k < problem->inletIndex.rows(); k++)
    {
        label p = problem->inletIndex(k, 0);
        label l = problem->inletIndex(k, 1);
        scalar area = gSum(problem->liftfield[0].mesh().magSf().boundaryField()[p]);
        scalar u_lf = gSum(problem->liftfield[k].mesh().magSf().boundaryField()[p] *
                           problem->liftfield[k].boundaryField()[p]).component(l) / area;
        vel_scal(k, 0) = vel(k, 0) / u_lf;
    }

    vel_now = vel_scal;
}

void reducedSimpleSteadyNS::project(int nModesU, int nModesP)
{
    ULmodes.resize(0);
    ReducedOperators.resize(0);

    for (int i = 0; i < problem->inletIndex.rows(); i++)
    {
        ULmodes.append(problem->liftfield[i]);
    }

    for (int i = 0; i < nModesU - problem->liftfield.size(); i++)
    {
        ULmodes.append(problem->Umodes.toPtrList()[i]);
    }

    ULmodes.toEigen();
    volScalarField& P = problem->_p();
    volVectorField& U = problem->_U();
    //volScalarField nu = problem->turbulence->nu();
    dimensionedScalar nu = ("nu", dimensionSet(0, 1, -1, 0, 0, 0, 0), 1);
    fvVectorMatrix UEqn
    (
        - fvm::laplacian(nu, U)
    );
    PtrList<volVectorField> gradP;
    PtrList<volVectorField> divdev2;
    gradP.resize(0);
    divdev2.resize(0);

    for (int i = 0; i < nModesP; i++)
    {
        gradP.append(-fvc::grad(problem->Pmodes[i]));
    }

    for (int i = 0; i < ULmodes.size(); i++)
    {
        divdev2.append(-fvc::div(nu * dev2(T(fvc::grad(ULmodes[i])))));
    }

    redGradP = ULmodes.project(gradP);
    redDivDev2 = ULmodes.project(divdev2);
    ReducedOperators.append(ULmodes.project(UEqn));
}
