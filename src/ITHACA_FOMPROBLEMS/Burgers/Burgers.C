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
/// Source file of the Burgers class.

#include "Burgers.H"

// * * * * * * * * * * * * * * * Constructors * * * * * * * * * * * * * * * * //
// Constructor
Burgers::Burgers() {}

Burgers::Burgers(int argc, char* argv[])
    :
    UnsteadyProblem()
{
    _args = autoPtr<argList>
            (
                new argList(argc, argv)
            );

    if (!_args->checkRootCase())
    {
        Foam::FatalError.exit();
    }

    argList& args = _args();
#include "createTime.H"
#include "createMesh.H"
    _simple = autoPtr<simpleControl>
              (
                  new simpleControl
                  (
                      mesh
                  )
              );
    simpleControl& simple = _simple();
#include "createFields.H"
#include "createFvOptions.H"
    ITHACAdict = new IOdictionary
    (
        IOobject
        (
            "ITHACAdict",
            runTime.system(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    );
    para = ITHACAparameters::getInstance(mesh, runTime);
    offline = ITHACAutilities::check_off();
    podex = ITHACAutilities::check_pod();
    setTimes(runTime);
}

void Burgers::truthSolve(word folder)
{
    Time& runTime = _runTime();
    fvMesh& mesh = _mesh();
    volVectorField& U = _U();
    surfaceScalarField& phi = _phi();
    fv::options& fvOptions = _fvOptions();
    simpleControl& simple = _simple();
    dimensionedScalar& nu = _nu();
    counter = 1;
    ITHACAstream::exportSolution(U, name(counter), folder + name(folderN));
    Ufield.append(U.clone());
    counter++;
    nextWrite = startTime;
    nextWrite += writeEvery;

    while (simple.loop())
    {
        Info << "Time = " << _runTime().timeName() << nl << endl;

        while (simple.correctNonOrthogonal())
        {
            fvVectorMatrix UEqn
            (
                fvm::ddt(U)
                + fvm::div(phi, U)
                - fvm::laplacian(nu, U)
            );
            UEqn.solve();
        }

        phi = linearInterpolate(U) & mesh.Sf();

        if (checkWrite(runTime))
        {
            ITHACAstream::exportSolution(U, name(counter), folder + name(folderN));
            counter++;
            Ufield.append(U.clone());
            nextWrite += writeEvery;
        }
    }

    folderN++;
}

void Burgers::residual(label Nmodes, word folder)
{
    Time& runTime = _runTime();
    fvMesh& mesh = _mesh();
    volVectorField& U = _U();
    surfaceScalarField& phi = _phi();
    fv::options& fvOptions = _fvOptions();
    simpleControl& simple = _simple();
    dimensionedScalar& nu = _nu();
    counter = 1;
    ITHACAstream::exportSolution(U, name(counter), folder + name(folderN));
    counter++;
    nextWrite = startTime;
    nextWrite += writeEvery;

    while (simple.loop())
    {
        Info << "Time = " << _runTime().timeName() << nl << endl;
        volVectorField Uaux(U.clone());
        volVectorField UauxOld(U.oldTime().clone());
        Uaux = Umodes.projectSnapshot(Uaux, Nmodes);
        UauxOld = Umodes.projectSnapshot(UauxOld, Nmodes);
        volVectorField& UauxOld2 = Uaux.oldTime();
        UauxOld2 = UauxOld;

        while (simple.correctNonOrthogonal())
        {
            fvVectorMatrix UEqn
            (
                fvm::ddt(U)
                + fvm::div(phi, U)
                - fvm::laplacian(nu, U)
            );
            UEqn.solve();
        }

        volVectorField res(-fvc::ddt(Uaux) - fvc::div(phi, Uaux) + fvc::laplacian(nu,
                           Uaux));
        volVectorField res2(res.clone());
        dimensionedScalar one ("one", dimVol, 1.0);
        res2.ref() = res * mesh.V() / one;
        res2.rename("res");
        phi = linearInterpolate(U) & mesh.Sf();

        if (checkWrite(runTime))
        {
            ITHACAstream::exportSolution(res2, name(counter), folder + name(folderNres));
            counter++;
            resField.append(res2.clone());
            nextWrite += writeEvery;
        }
    }

    folderNres++;
}



void Burgers::restart()
{
    _U.clear();
    _phi.clear();
    _fvOptions.clear();
    _nu.clear();
    _transportProperties.clear();
    argList& args = _args();
    Time& runTime = _runTime();
    runTime.setTime(0, 1);
    Foam::fvMesh& mesh = _mesh();
#include "createFields.H"
#include "createFvOptions.H"
}
