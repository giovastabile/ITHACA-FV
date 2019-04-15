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
/// Source file of the steadyNS class.

#include "steadyNS_simple.H"
#include "viscosityModel.H"

// * * * * * * * * * * * * * * * Constructors * * * * * * * * * * * * * * * * //

// Constructor
steadyNS_simple::steadyNS_simple() {}

steadyNS_simple::steadyNS_simple(int argc, char* argv[])
    :
    steadyNS(argc, argv)
{
    Info << offline << endl;
    phiHbyA_global = autoPtr<surfaceScalarField>(new surfaceScalarField("phiHbyA",
                     fvc::flux(_U())));
}

fvVectorMatrix steadyNS_simple::get_Umatrix(volVectorField& U,
        surfaceScalarField& phi, volScalarField& p)
{
    fvVectorMatrix Ueqn
    (
        fvm::div(phi, U)
        + turbulence->divDevReff(U) == - fvc::grad(p)
    );
    return Ueqn;
}

volVectorField steadyNS_simple::get_gradP(volScalarField& p)
{
    volVectorField gradP("gradP", fvc::grad(p));
    return gradP;
}


fvScalarMatrix steadyNS_simple::get_Pmatrix(volScalarField& p,
        fvVectorMatrix& Ueqn, surfaceScalarField& phi)
{
    phi = fvc::flux(1 / Ueqn.A() * Ueqn.H());
    fvScalarMatrix pEqn
    (
        fvm::laplacian(1 / Ueqn.A(), p) == fvc::div(phi)
    );
    return pEqn;
}

void steadyNS_simple::truthSolve2(List<scalar> mu_now, word Folder)
{
    Time& runTime = _runTime();
    volScalarField& p = _p();
    volVectorField& U = _U();
    surfaceScalarField& phi = _phi();
    fv::options& fvOptions = _fvOptions();
    simpleControl& simple = _simple();
    singlePhaseTransportModel& laminarTransport = _laminarTransport();
    scalar residual = 1;
    scalar uresidual = 1;
    Vector<double> uresidual_v(0, 0, 0);
    scalar presidual = 1;
    scalar csolve = 0;
    // Variable that can be changed
    turbulence->read();
    std::ofstream res_os;
    res_os.open("./ITHACAoutput/Offline/residuals", std::ios_base::app);
    fvVectorMatrix UEqn(U, dimensionSet(0, 4, -2, 0, 0, 0, 0));
    fvScalarMatrix pEqn(p, dimensionSet(0, 3, -1, 0, 0, 0, 0));

#if OFVER == 6

    while (simple.loop(runTime) && residual > tolerance && csolve < maxIter )
#else
    while (simple.loop() && residual > tolerance && csolve < maxIter )
#endif
    {
        UEqn = get_Umatrix(U, phi, p);
        UEqn.relax();
        Vector<double> uresidual_v = UEqn.solve().initialResidual();
        
        scalar C = 0;

        for (label i = 0; i < 3; i++)
        {
            if (C < uresidual_v[i])
            {
                C = uresidual_v[i];
            }
        }

        uresidual = C;
        pEqn = get_Pmatrix(p, UEqn, phi);
        presidual = pEqn.solve().initialResidual();
        phi -= pEqn.flux();
        p.relax();
        residual = max(presidual, uresidual);
    }

    UEqnList.append(UEqn);
    PEqnList.append(pEqn);
    res_os << residual << std::endl;
    res_os.close();
    runTime.setTime(runTime.startTime(), 0);
    ITHACAstream::exportSolution(U, name(counter), Folder);
    ITHACAstream::exportSolution(p, name(counter), Folder);
    ITHACAstream::exportSolution(phi, name(counter), Folder);
    Ufield.append(U);
    Pfield.append(p);
    phiField.append(phi);
    counter++;
    writeMu(mu_now);
    // Fill in the mu_samples with parameters (mu) to be used for the POD sample points
    mu_samples.conservativeResize(mu_samples.rows() + 1, mu_now.size());

    for (int i = 0; i < mu_now.size(); i++)
    {
        mu_samples(mu_samples.rows() - 1, i) = mu_now[i];
    }

    // Resize to Unitary if not initialized by user (i.e. non-parametric problem)
    if (mu.cols() == 0)
    {
        mu.resize(1, 1);
    }

    if (mu_samples.rows() == mu.cols())
    {
        ITHACAstream::exportMatrix(mu_samples, "mu_samples", "eigen",
                                   Folder);
    }
}
