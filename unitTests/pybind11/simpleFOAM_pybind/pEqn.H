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
{
    volScalarField rAU(1.0 / UEqn.A());
    volVectorField HbyA(constrainHbyA(rAU * UEqn.H(), _U(), _p()));
    surfaceScalarField phiHbyA("phiHbyA", fvc::flux(HbyA));
    _MRF().makeRelative(phiHbyA);
    adjustPhi(phiHbyA, _U(), _p());

    tmp<volScalarField> rAtU(rAU);

    if (_simple().consistent())
    {
        rAtU = 1.0 / (1.0 / rAU - UEqn.H1());
        phiHbyA +=
        fvc::interpolate(rAtU() - rAU) * fvc::snGrad(_p()) * _mesh().magSf();
        HbyA -= (rAU - rAtU()) * fvc::grad(_p());
    }

    tUEqn.clear();

    // Update the pressure BCs to ensure flux consistency
    constrainPressure(_p(), _U(), phiHbyA, rAtU(), _MRF());
    label i = 0;
    // Non-orthogonal pressure corrector loop
    while (_simple().correctNonOrthogonal())
    {
        fvScalarMatrix pEqn
        (
            fvm::laplacian(rAtU(), _p()) == fvc::div(phiHbyA)
        );
        pEqn.setReference(pRefCell, pRefValue);

        if (i == 0)
        {
            presidual = pEqn.solve().initialResidual();
        }
        else
        {
            pEqn.solve().initialResidual();
        }

        if (_simple().finalNonOrthogonalIter())
        {
            _phi() = phiHbyA - pEqn.flux();
        }

        i++;
    }

    volScalarField contErr(fvc::div(_phi()));

    scalar sumLocalContErr = _runTime().deltaTValue()*
        mag(contErr)().weightedAverage(_mesh().V()).value();

    scalar globalContErr = _runTime().deltaTValue()*
        contErr.weightedAverage(_mesh().V()).value();
    cumulativeContErr += globalContErr;

    Info<< "time step continuity errors : sum local = " << sumLocalContErr
        << ", global = " << globalContErr
        << ", cumulative = " << cumulativeContErr
        << endl;

    // Explicitly relax pressure for momentum corrector
    _p().relax();

    // Momentum corrector
    _U() = HbyA - rAtU() * fvc::grad(_p());
    _U().correctBoundaryConditions();
    _fvOptions().correct(_U());
}
