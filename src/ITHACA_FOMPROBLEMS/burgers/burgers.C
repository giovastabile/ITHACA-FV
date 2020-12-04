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

#include "burgers.H"

// * * * * * * * * * * * * * * * Constructors * * * * * * * * * * * * * * * * //
// Constructor
Burgers::Burgers() {}
Burgers::Burgers(int argc, char* argv[])
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
    simpleControl& simple = _simple();//CHECK
#include "createFields.H"
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
    tolerance = ITHACAdict->lookupOrDefault<scalar>("tolerance", 1e-5);
    maxIter = ITHACAdict->lookupOrDefault<scalar>("maxIter", 1000);
    // bcMethod = ITHACAdict->lookupOrDefault<word>("bcMethod", "lift");
    // M_Assert(bcMethod == "lift" || bcMethod == "penalty",
    //          "The BC method must be set to lift or penalty in ITHACAdict");
    para = ITHACAparameters::getInstance(mesh, runTime);
    offline = ITHACAutilities::check_off();
    podex = ITHACAutilities::check_pod();
}

// * * * * * * * * * * * * * * Full Order Methods * * * * * * * * * * * * * * //

// Method to perform a truthSolve
void Burgers::truthSolve(List<scalar> mu_now, fileName folder)
{
    Time& runTime = _runTime();
    fvMesh& mesh = _mesh();
//#include "initContinuityErrs.H" //CHECK
    simpleControl& simple = _simple();
    // fv::options& fvOptions = _fvOptions();
    surfaceScalarField& phi = _phi();
    volVectorField& U = _U();
    dimensionedScalar& nu = _nu();//CHECK
    // IOMRFZoneList& MRF = _MRF();
    instantList Times = runTime.times();
    runTime.setEndTime(finalTime);
    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_FOMPROBLEMS/burgers/burgers.C, line 101 #################### " << "viscosity: " << nu << endl;
    // Perform a TruthSolve
    runTime.setTime(Times[1], 1);
    runTime.setDeltaT(timeStep);
    nextWrite = startTime;

    // Export and store the initial conditions for velocity and pressure
    ITHACAstream::exportSolution(U, name(counter), folder);
    std::ofstream of(folder + name(counter) + "/" +
                     runTime.timeName());
    Ufield.append(U);
    counter++;
    int write_counter{0};
    nextWrite += writeEvery;

    // Save also the couple (initialTime, mu_now)
    mu_samples.conservativeResize(mu_samples.rows() + 1, mu_now.size() + 1);
    mu_samples(mu_samples.rows() - 1, 0) = atof(runTime.timeName().c_str());
    for (label i = 0; i < mu_now.size(); i++)
    {
        mu_samples(mu_samples.rows() - 1, i + 1) = mu_now[i];
    }

    // Start the time loop
    while (runTime.run())
    {
        Info<< "\nCalculating vector transport\n" << endl;
#include "readTimeControls.H"
#include "CourantNo.H"
// #include "setDeltaT.H"
        Info<< "deltaT = " <<  runTime.deltaTValue() << endl;
        runTime.setEndTime(finalTime);
        // Info << "Time = " << runTime.timeName() << nl << endl;

        // --- Pressure-velocity PIMPLE corrector loop
        while (simple.loop())
        {
            Info<< "Time = " << runTime.timeName() << nl << endl;

            while (simple.correctNonOrthogonal())//CHECK
            {
#include "UEqn.H"
            }

            write_counter++;
            Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_FOMPROBLEMS/burgers/burgers.C, line 147 #################### " << write_counter << " " << writeEvery << " " << counter <<  endl;
            if (write_counter >= writeEvery)
            {
                ITHACAstream::exportSolution(U, name(counter), folder);
                std::ofstream of(folder + name(counter) + "/" +
                                runTime.timeName());
                Ufield.append(U);
                counter++;
                writeMu(mu_now);
                // --- Fill in the mu_samples with parameters (time, mu) to be used for the PODI sample points
                mu_samples.conservativeResize(mu_samples.rows() + 1, mu_now.size() + 1);
                mu_samples(mu_samples.rows() - 1, 0) = atof(runTime.timeName().c_str());

            for (label i = 0; i < mu_now.size(); i++)
                {
                    mu_samples(mu_samples.rows() - 1, i + 1) = mu_now[i];
                }
                write_counter = 0;
            }
        }

        Info << "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
             << "  ClockTime = " << runTime.elapsedClockTime() << " s"
             << nl << endl;
    }

    // Resize to Unitary if not initialized by user (i.e. non-parametric problem)
    if (mu.cols() == 0)
    {
        mu.resize(1, 1);
    }

    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_FOMPROBLEMS/burgers/burgers.C, line 176 #################### " << mu_samples.rows() << " " << counter << " " << mu.cols() << endl;
    // counter+1 because also the initial time was saved
    if (mu_samples.rows() == counter * mu.cols())
    {
        Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_FOMPROBLEMS/burgers/burgers.C, line 183 #################### " << endl;
        ITHACAstream::exportMatrix(mu_samples, "mu_samples", "eigen",
                                   folder);
    }
    ITHACAstream::exportMatrix(mu_samples, "mu_samples", "eigen",
                                   folder);
}

// Method to compute the lifting function
void Burgers::liftSolve()
{
    for (label k = 0; k < inletIndex.rows(); k++)
    {
        Time& runTime = _runTime();
        surfaceScalarField& phi = _phi();
        fvMesh& mesh = _mesh();
        volVectorField U = _U();
        dimensionedScalar nu = _nu();
        IOMRFZoneList& MRF = _MRF();
        label BCind = inletIndex(k, 0);
        volVectorField Ulift("Ulift" + name(k), U);
        instantList Times = runTime.times();
        runTime.setTime(Times[1], 1);
        pisoControl potentialFlow(mesh, "potentialFlow");
        Info << "Solving a lifting Problem" << endl;
        Vector<double> v1(0, 0, 0);
        v1[inletIndex(k, 1)] = 1;
        Vector<double> v0(0, 0, 0);

        for (label j = 0; j < U.boundaryField().size(); j++)
        {
            if (j == BCind)
            {
                assignBC(Ulift, j, v1);
            }
            else if (U.boundaryField()[BCind].type() == "fixedValue")
            {
                assignBC(Ulift, j, v0);
            }
            else//CHECK
            {
            }

            assignIF(Ulift, v0);
            phi = linearInterpolate(Ulift) & mesh.Sf();
        }

        Info << "Constructing velocity potential field Phi\n" << endl;
        volScalarField Phi
        (
            IOobject
            (
                "Phi",
                runTime.timeName(),
                mesh,
                IOobject::READ_IF_PRESENT,
                IOobject::NO_WRITE
            ),
            mesh,
            dimensionedScalar("Phi", dimLength * dimVelocity, 0),
            U.boundaryField().types()//CHECK before U = p
        );
        label PhiRefCell = 0;
        scalar PhiRefValue = 0;
        setRefCell
        (
            Phi,
            potentialFlow.dict(),
            PhiRefCell,
            PhiRefValue
        );
        mesh.setFluxRequired(Phi.name());
        runTime.functionObjects().start();
        MRF.makeRelative(phi);
        //adjustPhi(phi, Ulift, U);//CHECK before U=p

        while (potentialFlow.correctNonOrthogonal())
        {
            fvScalarMatrix PhiEqn
            (
                fvm::laplacian(dimensionedScalar("1", dimless, 1), Phi)
                ==
                fvc::div(phi)
            );
            PhiEqn.setReference(PhiRefCell, PhiRefValue);
            PhiEqn.solve();

            if (potentialFlow.finalNonOrthogonalIter())
            {
                phi -= PhiEqn.flux();
            }
        }

        MRF.makeAbsolute(phi);
        Info << "Continuity error = "
             << mag(fvc::div(phi))().weightedAverage(mesh.V()).value()
             << endl;
        Ulift = fvc::reconstruct(phi);
        Ulift.correctBoundaryConditions();
        Info << "Interpolated velocity error = "
             << (sqrt(sum(sqr((fvc::interpolate(U) & mesh.Sf()) - phi)))
                 / sum(mesh.magSf())).value()
             << endl;
        Ulift.write();
        liftfield.append(Ulift);
    }
}

// * * * * * * * * * * * * * * Projection Methods * * * * * * * * * * * * * * //

void Burgers::project(fileName folder, label NU)
{
    NUmodes = NU;
    L_Umodes.resize(0);
    NL_Umodes = NUmodes;

    if (liftfield.size() != 0)
    {
        for (label k = 0; k < liftfield.size(); k++)
        {
            L_Umodes.append(liftfield[k]);
        }
        NL_Umodes += liftfield.size();
    }
    else
    {
        L_Umodes.append(_U0);
        NL_Umodes += 1;
    }

    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_FOMPROBLEMS/burgers/burgers.C, line 308 #################### " << NL_Umodes << " " << Umodes.size() << endl;

    if (NUmodes != 0)
    {
        for (label k = 0; k < NUmodes; k++)
        {
            L_Umodes.append(Umodes[k]);
        }
    }
    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_FOMPROBLEMS/burgers/burgers.C, line 317 #################### " << endl;

    evaluateMatrices();
    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_FOMPROBLEMS/burgers/burgers.C, line 320 #################### " << endl;
    // Export the matrices
    if (para->exportPython)
    {
        ITHACAstream::exportMatrix(B_matrix, "B", "python", "./ITHACAoutput/Matrices/");
        ITHACAstream::exportMatrix(M_matrix, "M", "python", "./ITHACAoutput/Matrices/");
        ITHACAstream::exportTensor(C_tensor, "C", "python", "./ITHACAoutput/Matrices/");
    }

    if (para->exportMatlab)
    {
        ITHACAstream::exportMatrix(B_matrix, "B", "matlab", "./ITHACAoutput/Matrices/");
        ITHACAstream::exportMatrix(M_matrix, "M", "matlab", "./ITHACAoutput/Matrices/");
        ITHACAstream::exportTensor(C_tensor, "C", "python", "./ITHACAoutput/Matrices/");
    }

    if (para->exportTxt)
    {
        ITHACAstream::exportMatrix(B_matrix, "B", "eigen", "./ITHACAoutput/Matrices/");
        ITHACAstream::exportMatrix(M_matrix, "M", "eigen", "./ITHACAoutput/Matrices/");
        ITHACAstream::exportTensor(C_tensor, "C", "python","./ITHACAoutput/Matrices/C");
    }
}

// * * * * * * * * * * * * * * Momentum Eq. Methods * * * * * * * * * * * * * //

Eigen::MatrixXd Burgers::diffusive_term(label NL_Umodes)
{
    label Bsize = NL_Umodes;

    Eigen::MatrixXd B_matrix;
    B_matrix.resize(Bsize, Bsize);

    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_FOMPROBLEMS/burgers/burgers.C, line 414 #################### " << Bsize << endl;

    // Project everything
    for (label i = 0; i < Bsize; i++)
    {
        for (label j = 0; j < Bsize; j++)
        {
            B_matrix(i, j) = fvc::domainIntegrate(L_Umodes[i] & fvc::laplacian(
                    dimensionedScalar("1", dimless, 1), L_Umodes[j])).value();
        }
    }

    if (Pstream::parRun())
    {
        reduce(B_matrix, sumOp<Eigen::MatrixXd>());
    }

    ITHACAstream::SaveDenseMatrix(B_matrix, "./ITHACAoutput/Matrices/",
                                  "B_" + name(liftfield.size()) + "_" + name(NL_Umodes));

    return B_matrix;
}

List <Eigen::MatrixXd> Burgers::convective_term(label NL_Umodes)
{
    label Csize = NL_Umodes;

    List <Eigen::MatrixXd> C_matrix;
    C_matrix.setSize(Csize);

    for (label j = 0; j < Csize; j++)
    {
        C_matrix[j].resize(Csize, Csize);
    }

    for (label i = 0; i < Csize; i++)
    {
        for (label j = 0; j < Csize; j++)
        {
            for (label k = 0; k < Csize; k++)
            {
                C_matrix[i](j, k) = fvc::domainIntegrate(L_Umodes[i] & fvc::div(
                                        linearInterpolate(L_Umodes[j]) & L_Umodes[j].mesh().Sf(),
                                        L_Umodes[k])).value();
            }
        }
    }

    if (Pstream::parRun())
    {
        for (label i = 0; i < Csize; i++)
        {
            List<double> vec(C_matrix[i].data(), C_matrix[i].data() + C_matrix[i].size());
            reduce(vec, sumOp<List<double>>());
            std::memcpy(C_matrix[i].data(), &vec[0], sizeof (double)*vec.size());
        }
    }

    // Export the matrix
    ITHACAstream::exportMatrix(C_matrix, "C", "python", "./ITHACAoutput/Matrices/");
    ITHACAstream::exportMatrix(C_matrix, "C", "matlab", "./ITHACAoutput/Matrices/");
    ITHACAstream::exportMatrix(C_matrix, "C", "eigen", "./ITHACAoutput/Matrices/C");
    return C_matrix;
}

Eigen::Tensor<double, 3> Burgers::convective_term_tens(label NL_Umodes)
{
    label Csize = NL_Umodes;

    Eigen::Tensor<double, 3> C_tensor;
    C_tensor.resize(Csize, Csize, Csize);

    for (label i = 0; i < Csize; i++)
    {
        for (label j = 0; j < Csize; j++)
        {
            for (label k = 0; k < Csize; k++)
            {
                C_tensor(i, j, k) = fvc::domainIntegrate(L_Umodes[i] & fvc::div(
                                        linearInterpolate(L_Umodes[j]) & L_Umodes[j].mesh().Sf(),
                                        L_Umodes[k])).value();
            }
        }
    }

    if (Pstream::parRun())
    {
        reduce(C_tensor, sumOp<Eigen::Tensor<double, 3>>());
    }

    // Export the tensor
    ITHACAstream::SaveDenseTensor(C_tensor, "./ITHACAoutput/Matrices/",
                                  "C_" + name(liftfield.size()) + "_" + name(NL_Umodes) + "_t");
    return C_tensor;
}

Eigen::MatrixXd Burgers::mass_term(label NL_Umodes)
{
    label Msize = NL_Umodes;

    Eigen::MatrixXd M_matrix(Msize, Msize);

    // Project everything
    for (label i = 0; i < Msize; i++)
    {
        for (label j = 0; j < Msize; j++)
        {
            M_matrix(i, j) = fvc::domainIntegrate(L_Umodes[i] &
                                                  L_Umodes[j]).value();
        }
    }

    if (Pstream::parRun())
    {
        reduce(M_matrix, sumOp<Eigen::MatrixXd>());
    }

    ITHACAstream::SaveDenseMatrix(M_matrix, "./ITHACAoutput/Matrices/",
                                  "M_" + name(liftfield.size()) + "_" + name(NL_Umodes));
    return M_matrix;
}

//CHECK_start
void Burgers::change_viscosity(double mu)
{
    dimensionedScalar& nu = _nu();
    nu = dimensionedScalar(dimViscosity, mu);
}

void Burgers::change_initial_velocity(double mu)
{
    _U = _U0.clone();
    *_U *= mu;
    // ITHACAstream::exportSolution(_U(), name(mu), "./initial_data/");
}

//CHECK_end
// void steadyNS::change_viscosity(double mu)
// {
//     const volScalarField& nu =  _laminarTransport().nu();
//     volScalarField& ciao = const_cast<volScalarField&>(nu);
//     this->assignIF(ciao, mu);

//     for (label i = 0; i < ciao.boundaryFieldRef().size(); i++)
//     {
//         this->assignBC(ciao, i, mu);
//     }
// }

void Burgers::restart()
{
    _runTime().objectRegistry::clear();
    _mesh().objectRegistry::clear();
    // _mesh.clear();
    // _runTime.clear();
    _simple.clear();
    _U.clear();
    _phi.clear();
    _nu.clear();//CHECK
    _fvOptions.clear();
    argList& args = _args();
    Time& runTime = _runTime();
    runTime.setTime(0, 1);
    Foam::fvMesh& mesh = _mesh();
    _simple = autoPtr<simpleControl>
              (
                  new simpleControl
                  (
                      mesh
                  )
              );
    simpleControl& simple = _simple();

    Info << "ReReading field U\n" << endl;
    _U = autoPtr<volVectorField>
         (
             new volVectorField
             (
                 IOobject
                 (
                     "U",
                     runTime.timeName(),
                     mesh,
                     IOobject::MUST_READ,
                     IOobject::AUTO_WRITE
                 ),
                 mesh
             )
         );
    volVectorField& U = _U();
    Info << "ReReading/calculating face flux field phi\n" << endl;
    _phi = autoPtr<surfaceScalarField>
           (
               new surfaceScalarField
               (
                   IOobject
                   (
                       "phi",
                       runTime.timeName(),
                       mesh,
                       IOobject::READ_IF_PRESENT,
                       IOobject::AUTO_WRITE
                   ),
                   linearInterpolate(U) & mesh.Sf()
               )
           );
    surfaceScalarField& phi = _phi();
    //CHECK_comments
    // pRefCell = 0;
    // pRefValue = 0.0;
    // setRefCell(p, simple.dict(), pRefCell, pRefValue);
    // _laminarTransport = autoPtr<singlePhaseTransportModel>
    //                     (
    //                         new singlePhaseTransportModel( U, phi )
    //                     );
    // singlePhaseTransportModel& laminarTransport = _laminarTransport();

    //CHECK_start
    _MRF = autoPtr<IOMRFZoneList>
           (
               new IOMRFZoneList(mesh)
           );
    _fvOptions = autoPtr<fv::options>(new fv::options(mesh));


    Info<< "Reading transportProperties\n" << endl;

    IOdictionary transportProperties
    (
        IOobject
        (
            "transportProperties",
            runTime.constant(),
            mesh,
            IOobject::MUST_READ_IF_MODIFIED,
            IOobject::NO_WRITE
        )
    );


    Info<< "Reading viscosity nu\n" << endl;

    _nu = autoPtr<dimensionedScalar>(new dimensionedScalar("nu", dimViscosity, transportProperties));
    dimensionedScalar& nu = _nu();
    //CHECK_end
}

bool Burgers::checkWrite(Time& timeObject)
{
    scalar diffnow = mag(nextWrite - atof(timeObject.timeName().c_str()));
    // scalar diffnext = mag(nextWrite - atof(timeObject.timeName().c_str()) - timeObject.deltaTValue());
    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_FOMPROBLEMS/burgers/burgers.C, line 637 #################### " << diffnow << " " << timeObject.deltaTValue() << endl;
    if ( diffnow <= timeObject.deltaTValue())
    {
        Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_FOMPROBLEMS/burgers/burgers.C, line 605 #################### " << endl;
        return true;
    }
    else
    {
        return false;
    }
}

void Burgers::evaluateMatrices()
{
    if (ITHACAutilities::check_folder("./ITHACAoutput/Matrices/"))
    {
        Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_FOMPROBLEMS/burgers/burgers.C, line 320 #################### " << endl;
        word B_str = "B_" + name(liftfield.size()) + "_" + name(NL_Umodes);

        if (ITHACAutilities::check_file("./ITHACAoutput/Matrices/" + B_str))
        {
            Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_FOMPROBLEMS/burgers/burgers.C, line 324 #################### " << endl;
            ITHACAstream::ReadDenseMatrix(B_matrix, "./ITHACAoutput/Matrices/", B_str);
        }
        else
        {
            Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_FOMPROBLEMS/burgers/burgers.C, line 328 #################### " << endl;
            B_matrix = diffusive_term(NL_Umodes);
            Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_FOMPROBLEMS/burgers/burgers.C, line 330 #################### " << endl;
        }

        word M_str = "M_" + name(liftfield.size()) + "_" + name(NL_Umodes);

        if (ITHACAutilities::check_file("./ITHACAoutput/Matrices/" + M_str))
        {
            ITHACAstream::ReadDenseMatrix(M_matrix, "./ITHACAoutput/Matrices/", M_str);
        }
        else
        {
            M_matrix = mass_term(NL_Umodes);
        }

        word C_str = "C_" + name(liftfield.size()) + "_" + name(NL_Umodes) + "_t";

        if (ITHACAutilities::check_file("./ITHACAoutput/Matrices/" + C_str))
        {
            ITHACAstream::ReadDenseTensor(C_tensor, "./ITHACAoutput/Matrices/", C_str);
        }
        else
        {
            C_tensor = convective_term_tens(NL_Umodes);
        }

        // if (bcMethod == "penalty")
        // {
        //     bcVelVec = bcVelocityVec(NUmodes);
        //     bcVelMat = bcVelocityMat(NUmodes);
        // }
    }
    else
    {
        B_matrix = diffusive_term(NL_Umodes);
        C_tensor = convective_term_tens(NL_Umodes);
        M_matrix = mass_term(NL_Umodes);

        // if (bcMethod == "penalty")
        // {
        //     bcVelVec = bcVelocityVec(NUmodes);
        //     bcVelMat = bcVelocityMat(NUmodes);
        // }
    }
}