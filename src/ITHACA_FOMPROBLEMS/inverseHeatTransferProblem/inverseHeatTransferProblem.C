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
/// Source file of the inverseHeatTransferProblem class.


#include "inverseHeatTransferProblem.H"

// * * * * * * * * * * * * * * * Constructors * * * * * * * * * * * * * * * * //

// Constructors
inverseHeatTransferProblem::inverseHeatTransferProblem() {}

inverseHeatTransferProblem::inverseHeatTransferProblem(int argc, char* argv[])
    :
    DT("DT", dimensionSet(0, 2, -1, 0, 0, 0, 0), 1.0)
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
#include "createFields.H"
#include "createThermocouples.H"
    thermocouplesPos = TCpos;
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
    nProcs = Pstream::nProcs();
    startTime = runTime.startTime().value();
    deltaTime = runTime.deltaTValue();
    endTime = runTime.endTime().value();
    Ntimes = (endTime - startTime) / deltaTime;
    timeSteps.resize( Ntimes + 1 );
    forAll(timeSteps, timeI)
    {
        timeSteps[timeI] = startTime + (timeI) * deltaTime;
    }
    //Info << "debug: timeSteps = " << timeSteps << endl;
    //Info << "debug: startTime = " << startTime << endl;
    //Info << "debug: endTime = " << endTime << endl;
    //Info << "debug: deltaTime = " << deltaTime << endl;
}

// * * * * * * * * * * * * * * Full Order Methods * * * * * * * * * * * * * * //

void inverseHeatTransferProblem::setDiffusivity(scalar _diff)
{
    diffusivity = _diff;
}

void inverseHeatTransferProblem::set_g()
{
    volScalarField& T = _T();
    g.resize(timeSteps.size());
    forAll(timeSteps, timeI)
    {
        g[timeI].resize(T.boundaryField()[hotSide_ind].size(), 0.0);
        forAll (T.boundaryField()[hotSide_ind], faceI)
        {
            g[timeI][faceI] = 0.0;
        }
    }
}

void inverseHeatTransferProblem::set_gBaseFunctions(word type,
        scalar shapeParameter)
{
    volScalarField& T = _T();
    fvMesh& mesh = _mesh();
    gBasisSize = thermocouplesNum * timeSamplesNum;
    //Info << "debug: thermocouplesNum = " << thermocouplesNum << " , timeSamplesNum = " << timeSamplesNum << endl;

    if (type == "rbf")
    {
        Info << "Radial Basis Functions are used." << endl;
        // The center of each function is the projection of each thermocouple
        // on the boundary hotSide

        if (thermocouplesCellID.size() == 0)
        {
            readThermocouples();
        }

        gBaseFunctions.resize(gBasisSize);
        gWeights.resize(gBasisSize);
        int thermocouplesCounter = 0;
        int samplingTimeI = 0;
	scalar maxX =  Foam::max(mesh.boundaryMesh()[hotSide_ind].faceCentres().component(Foam::vector::X));
	scalar maxZ =  Foam::max(mesh.boundaryMesh()[hotSide_ind].faceCentres().component(Foam::vector::Z));

        forAll(gBaseFunctions, funcI)
        {
            //Info << "debug: Setting up RBF for TC " << thermocouplesCounter << ", samplingTime = " << samplingTime[samplingTimeI] << endl;
            gBaseFunctions[funcI].resize(timeSteps.size());
            scalar thermocoupleX =
                mesh.C()[thermocouplesCellID [thermocouplesCounter]].component(0);
            scalar thermocoupleZ =
                mesh.C()[thermocouplesCellID [thermocouplesCounter]].component(2);
            scalar sTime = samplingTime[samplingTimeI];
            forAll(timeSteps, timeI)
            {
                gBaseFunctions[funcI][timeI].resize(T.boundaryField()[hotSide_ind].size());
                scalar time = timeSteps[timeI];
                forAll (T.boundaryField()[hotSide_ind], faceI)
                {
                    scalar faceX = mesh.boundaryMesh()[hotSide_ind].faceCentres()[faceI].x();
                    scalar faceZ = mesh.boundaryMesh()[hotSide_ind].faceCentres()[faceI].z();

                    scalar radius = Foam::sqrt((faceX - thermocoupleX) * (faceX - thermocoupleX) / maxX / maxX +
                                               (faceZ - thermocoupleZ) * (faceZ - thermocoupleZ) / maxZ / maxZ +
                                               (time - sTime) * (time - sTime) / endTime / endTime);
		    //gBaseFunctions[funcI][timeI][faceI] = 1e6 * Foam::sqrt(1 + (shapeParameter * radius) * (shapeParameter * radius));



                    //scalar radius = Foam::sqrt((faceX - thermocoupleX) * (faceX - thermocoupleX) +
                    //                           (faceZ - thermocoupleZ) * (faceZ - thermocoupleZ) +
                    //                           (time - sTime) * (time - sTime));
                    gBaseFunctions[funcI][timeI][faceI] = 1e6 * Foam::exp(- (shapeParameter *
                                                          shapeParameter
                                                          * radius * radius));
                }
            }
            thermocouplesCounter++;

            if (thermocouplesCounter == thermocouplesNum)
            {
                thermocouplesCounter = 0;
                samplingTimeI++;
            }
        }
    }
    else if (type == "pod")
    {
        Info << "Not yet implemented, exiting" << endl;
        exit(10);
    }
}

void inverseHeatTransferProblem::set_gBaseFunctions(word type,
        scalar shapeParameter_space, scalar shapeParameter_time)
{
    volScalarField& T = _T();
    fvMesh& mesh = _mesh();
    gBasisSize = thermocouplesNum * timeSamplesNum;
    //Info << "debug: thermocouplesNum = " << thermocouplesNum << " , timeSamplesNum = " << timeSamplesNum << endl;

    if (type == "rbf")
    {
        Info << "Radial Basis Functions are used." << endl;
        // The center of each function is the projection of each thermocouple
        // on the boundary hotSide

        if (thermocouplesCellID.size() == 0)
        {
            readThermocouples();
        }

        gBaseFunctions.resize(gBasisSize);
        gWeights.resize(gBasisSize);
        int thermocouplesCounter = 0;
        int samplingTimeI = 0;
        forAll(gBaseFunctions, funcI)
        {
            //Info << "debug: Setting up RBF for TC " << thermocouplesCounter << ", samplingTime = " << samplingTime[samplingTimeI] << endl;
            gBaseFunctions[funcI].resize(timeSteps.size());
            scalar thermocoupleX =
                mesh.C()[thermocouplesCellID [thermocouplesCounter]].component(0);
            scalar thermocoupleZ =
                mesh.C()[thermocouplesCellID [thermocouplesCounter]].component(2);
            scalar sTime = samplingTime[samplingTimeI];
            forAll(timeSteps, timeI)
            {
                gBaseFunctions[funcI][timeI].resize(T.boundaryField()[hotSide_ind].size());
                scalar time = timeSteps[timeI];
                forAll (T.boundaryField()[hotSide_ind], faceI)
                {
                    scalar faceX = mesh.boundaryMesh()[hotSide_ind].faceCentres()[faceI].x();
                    scalar faceZ = mesh.boundaryMesh()[hotSide_ind].faceCentres()[faceI].z();
                    scalar radius_space = Foam::sqrt((faceX - thermocoupleX) * (faceX - thermocoupleX) +
                                               (faceZ - thermocoupleZ) * (faceZ - thermocoupleZ));
                    scalar radius_time = Foam::sqrt((time - sTime) * (time - sTime));
		    scalar exponent = shapeParameter_space * radius_space + shapeParameter_time * radius_time;
                    gBaseFunctions[funcI][timeI][faceI] = Foam::exp(- exponent * exponent);
                }
            }
            thermocouplesCounter++;

            if (thermocouplesCounter == thermocouplesNum)
            {
                thermocouplesCounter = 0;
                samplingTimeI++;
            }
        }
    }
    else if (type == "pod")
    {
        Info << "Not yet implemented, exiting" << endl;
        exit(10);
    }
}

void inverseHeatTransferProblem::set_gParametrized(word baseFuncType,
        scalar shapeParameter)
{
    volScalarField& T = _T();
    set_gBaseFunctions(baseFuncType, shapeParameter);
    g.resize(timeSteps.size());
    forAll (gWeights, weigthI)
    {
        gWeights[weigthI] = 0; //-10000;
    }
    Info << "gWeights = " << gWeights << endl;
    forAll(timeSteps, timeI)
    {
        g[timeI].resize(T.boundaryField()[hotSide_ind].size(), 0.0);
        forAll (T.boundaryField()[hotSide_ind], faceI)
        {
            g[timeI][faceI] = 0.0;
            forAll (gWeights, weigthI)
            {
                g[timeI][faceI] += gWeights[weigthI] * gBaseFunctions[weigthI][timeI][faceI];
            }
        }
    }
}

void inverseHeatTransferProblem::set_gParametrized(word baseFuncType,
        scalar shapeParameter_space, scalar shapeParameter_time)
{
    volScalarField& T = _T();
    set_gBaseFunctions(baseFuncType, shapeParameter_space, shapeParameter_time);
    g.resize(timeSteps.size());
    forAll (gWeights, weigthI)
    {
        gWeights[weigthI] = 0; //-10000;
    }
    Info << "gWeights = " << gWeights << endl;
    forAll(timeSteps, timeI)
    {
        g[timeI].resize(T.boundaryField()[hotSide_ind].size(), 0.0);
        forAll (T.boundaryField()[hotSide_ind], faceI)
        {
            g[timeI][faceI] = 0.0;
            forAll (gWeights, weigthI)
            {
                g[timeI][faceI] += gWeights[weigthI] * gBaseFunctions[weigthI][timeI][faceI];
            }
        }
    }
}

void inverseHeatTransferProblem::update_gParametrized(List<scalar> weights)
{
    M_Assert(weights.size() == gBaseFunctions.size(),
             "weigths size different from basis functions size");
    volScalarField& T = _T();
    forAll(timeSteps, timeI)
    {
        forAll (T.boundaryField()[hotSide_ind], faceI)
        {
            g[timeI][faceI] = 0.0;
            forAll (weights, weightI)
            {
                g[timeI][faceI] += weights[weightI] * gBaseFunctions[weightI][timeI][faceI];
            }
        }
    }
}

volScalarField inverseHeatTransferProblem::list2Field(List<scalar> list,
        scalar innerField)
{
    volScalarField& T = _T();
    fvMesh& mesh = _mesh();
    volScalarField field(T);
    ITHACAutilities::assignIF(field, innerField);
    //Access the mesh information for the boundary
    const polyPatch& cPatch = mesh.boundaryMesh()[hotSide_ind];
    //List of cells close to a boundary
    const labelUList& faceCells = cPatch.faceCells();
    forAll(cPatch, faceI)
    {
        //id of the owner cell having the face
        label faceOwner = faceCells[faceI] ;
        field[faceOwner] = list[faceI];
    }
    return field;
}

Eigen::VectorXd  inverseHeatTransferProblem::TSVD(Eigen::MatrixXd A,
        Eigen::MatrixXd b, label filter)
{
    // Add check on b
    Info << "Using truncated SVD for regularization" << endl;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A,
                                          Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXd U = svd.matrixU();
    Eigen::MatrixXd V = svd.matrixV();
    Eigen::VectorXd x;

    for (label i = 0; i < filter; i++)
    {
        scalar coeff = (U.col(i).transpose() * b)(0, 0);

        if (i == 0)
        {
            // / svd.singularValues()(i) << endl;
            x = coeff / svd.singularValues()(i) * V.col(i);
        }
        else
        {
            x += coeff / svd.singularValues()(i) * V.col(i);
        }
    }

    return x;
}

void inverseHeatTransferProblem::parameterizedBCoffline(bool force)
{
    fvMesh& mesh = _mesh();
    Tbasis.resize(0);

    if (ITHACAutilities::check_file(folderOffline + "/Theta_mat.txt") && force == 0)
    {
        Info << "\nOffline already computed." << endl;
        Info << "Check that the basis used for the parameterized BC are correct (RBF, POD, etc.)"
             << endl;
        Theta = ITHACAstream::readMatrix(folderOffline + "Theta_mat.txt");
        addSol = ITHACAstream::readMatrix(folderOffline + "addSol_mat.txt");
        T0_vector = ITHACAstream::readMatrix(folderOffline + "T0_vector_mat.txt");
        ITHACAstream::read_fields(Tad_time, "Tad", folderOffline);
        ITHACAstream::read_fields(T0_time, "T0_field", folderOffline);

        for (label baseI = 0; baseI < Theta.cols(); baseI++)
        {
            Ttime.resize(0);
            ITHACAstream::read_fields(Ttime, "T" + std::to_string(baseI + 1),
                                      folderOffline);
            Tbasis.append(Ttime.clone());
        }
    }
    else
    {
        Info << "\nComputing offline" << endl;
        solveAdditional();
        solveT0();
        Theta.resize(thermocouplesNum * timeSamplesNum, gWeights.size());

        for (label baseI = 0; baseI < Theta.cols(); baseI++)
        {
            restart();
            Ttime.resize(0);
            gWeights = Foam::zero();
            gWeights[baseI] =  1; //1e5
            update_gParametrized(gWeights);
            Info << "Solving for base = " << baseI << endl;
            solveDirect();

            forAll(timeSteps, timeI)
            {
                volScalarField& T = Ttime[timeI];
                /// Saving basis
                volScalarField gParametrizedField = list2Field(g[timeI]);
                ITHACAstream::exportSolution(gParametrizedField,
                                             std::to_string(timeSteps[timeI]),
                                             folderOffline,
                                             "g" + std::to_string(baseI + 1));
                ITHACAstream::exportSolution(T, std::to_string(timeSteps[timeI]),
                                             folderOffline,
                                             "T" + std::to_string(baseI + 1));
            }
            Tbasis.append(Ttime.clone());
            Tcomp = fieldValueAtThermocouples(Ttime);
            forAll(samplingSteps, sampleI)
            {
                forAll(thermocouplesPos, tcI)
                {
                    label measI = tcI + thermocouplesNum * sampleI;
                    Theta(measI, baseI) = Tcomp(measI) + addSol(measI);
                }
            }
        }

        ITHACAstream::exportMatrix(Theta, "Theta", "eigen", folderOffline);
        ITHACAstream::exportMatrix(addSol, "addSol", "eigen", folderOffline);
        ITHACAstream::exportMatrix(T0_vector, "T0_vector", "eigen", folderOffline);
    }

    Eigen::MatrixXd A = Theta.transpose() * Theta;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A,
                                          Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXd singularValues = svd.singularValues();
    double conditionNumber = singularValues.maxCoeff() / singularValues.minCoeff();
    Info << "Condition number = " << conditionNumber << endl;
    ITHACAstream::exportMatrix(singularValues, "singularValues", "eigen",
                               folderOffline);
    Info << "\nOffline ENDED" << endl;
}

void inverseHeatTransferProblem::reconstrucT(word outputFolder)
{
    Info << "Reconstructing field T" << endl;
    update_gParametrized(gWeights);
    Ttime.resize(0);
    forAll(timeSteps, timeI)
    {
        restart();
        volScalarField T(_T);
        ITHACAutilities::assignIF(T, homogeneousBC);
        forAll(Tbasis, baseI)
        {
            T += gWeights[baseI] * (Tbasis[baseI][timeI] + Tad_time[timeI]);
        }
        T += - Tad_time[timeI] + T0_time[timeI];
        ITHACAstream::exportSolution(T, std::to_string(timeSteps[timeI]),
                                     outputFolder,
                                     "Treconstructed");
        volScalarField gParametrizedField = list2Field(g[timeI]);
        ITHACAstream::exportSolution(gParametrizedField,
                                     std::to_string(timeSteps[timeI]),
                                     outputFolder,
                                     "gReconstructed");
        Ttime.append(T.clone());
    }
    Tcomp = fieldValueAtThermocouples(Ttime);
}

void inverseHeatTransferProblem::parameterizedBC(word outputFolder,
        word linSys_solver,
        label TSVD_filter)
{
    Info << endl << "Using quasilinearity of direct problem" << endl;
    //parameterizedBCoffline(folder, forceOffline);
    List<Eigen::MatrixXd> linSys;
    linSys.resize(2);
    Info << "Theta size = " << Theta.rows() << ", " << Theta.cols() << endl;
    //std::cout << "debug: addsol " << addsol <<  std::endl;
    //std::cout << "debug: T0_vector " << T0_vector <<  std::endl;
    linSys[0] = Theta.transpose() * Theta;
    linSys[1] = Theta.transpose() * (Tmeas + addSol - T0_vector);
    Eigen::VectorXd weigths;

    if (linSys_solver == "fullPivLU")
    {
        weigths = linSys[0].fullPivLu().solve(linSys[1]);
    }
    else if (linSys_solver == "jacobiSvd")
    {
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(linSys[0],
                                              Eigen::ComputeThinU | Eigen::ComputeThinV);
        weigths = svd.solve(linSys[1]);
    }
    else if (linSys_solver == "householderQr")
    {
        weigths = linSys[0].householderQr().solve(linSys[1]);
    }
    else if (linSys_solver == "ldlt")
    {
        weigths = linSys[0].ldlt().solve(linSys[1]);
    }
    else if (linSys_solver == "inverse")
    {
        weigths = linSys[0].inverse() * linSys[1];
    }
    else if (linSys_solver == "TSVD")
    {
        weigths = TSVD(linSys[0], linSys[1], TSVD_filter);
    }
    else
    {
        Info << "Select a linear system solver in this list:" << endl
             << "fullPivLU, jacobiSvd, householderQr, ldlt" << endl;
        exit(1);
    }

    gWeights.resize(weigths.size());
    forAll(gWeights, weightI)
    {
        gWeights[weightI] = weigths(weightI);
    }
    update_gParametrized(gWeights);
    label verbose = 0;
    parameterizedBC_postProcess(linSys, weigths, outputFolder, verbose);
    Info << "End" << endl;
    Info << endl;
}

void inverseHeatTransferProblem::set_valueFraction()
{
    fvMesh& mesh = _mesh();
    valueFraction.resize(mesh.boundaryMesh()["coldSide"].size());
    homogeneousBCcoldSide.resize(mesh.boundaryMesh()["coldSide"].size());
    valueFractionAdj.resize(mesh.boundaryMesh()["coldSide"].size());
    Eigen::VectorXd faceCellDist =
        ITHACAutilities::boudaryFaceToCellDistance(mesh, coldSide_ind);
    forAll (valueFraction, faceI)
    {
        valueFraction[faceI] = 1 / (1 + (k / H / faceCellDist(faceI)));
        valueFractionAdj[faceI] =  1 / (1 + (1 / k / H / faceCellDist(faceI)));
        homogeneousBCcoldSide[faceI] =  0;
    }
    refGrad = homogeneousBCcoldSide;
}

void inverseHeatTransferProblem::assignDirectBC(label timeI)
{
    fvMesh& mesh = _mesh();
    volScalarField& T = _T();
    set_valueFraction();
    forAll(mesh.boundaryMesh(), patchI)
    {
        if (patchI == mesh.boundaryMesh().findPatchID("coldSide"))
        {
            ITHACAutilities::assignMixedBC(T, patchI, Tf, refGrad, valueFraction);
        }
        else if (patchI == mesh.boundaryMesh().findPatchID("hotSide"))
        {
            ITHACAutilities::assignBC(T, patchI, - g[timeI] / k);
        }
        else
        {
            ITHACAutilities::assignBC(T, patchI, homogeneousBC);
        }
    }
}

void inverseHeatTransferProblem::solveT0()
{
    Info << "Solving T0 problem" << endl;
    restart();
    fvMesh& mesh = _mesh();
    simpleControl& simple = _simple();
    fv::options& fvOptions(_fvOptions());
    volScalarField T0_field(_T);
    Foam::Time& runTime = _runTime();
    set_valueFraction();
    List<scalar> RobinBC = Tf * 0.0;
    T0_time.resize(0);
    assignT0_IF(T0_field);
    label timeI = 0;
    forAll(mesh.boundaryMesh(), patchI)
    {
        if (patchI == mesh.boundaryMesh().findPatchID("coldSide"))
        {
            ITHACAutilities::assignMixedBC(T0_field, patchI, RobinBC, refGrad,
                                           valueFraction);
        }
        else
        {
            ITHACAutilities::assignBC(T0_field, patchI, homogeneousBC);
        }
    }
    T0_time.append(T0_field.clone());
    ITHACAstream::exportSolution(T0_field, std::to_string(timeSteps[timeI]),
                                 folderOffline,
                                 "T0_field");

    while (runTime.loop())
    {
        Info << "Time = " << runTime.timeName() << nl << endl;
        timeI++;

        while (simple.correctNonOrthogonal())
        {
            fvScalarMatrix TEqn
            (
                fvm::ddt(T0_field) - fvm::laplacian(DT * diffusivity, T0_field)
            );
            fvOptions.constrain(TEqn);
            TEqn.solve();
            fvOptions.correct(T0_field);
        }

        ITHACAstream::exportSolution(T0_field, std::to_string(timeSteps[timeI]),
                                     folderOffline,
                                     "T0_field");
        T0_time.append(T0_field.clone());
        runTime.printExecutionTime(Info);
        runTime.write();
    }

    T0_vector = fieldValueAtThermocouples(T0_time);
    Info << "END \n" << endl;
}

void inverseHeatTransferProblem::solveAdditional()
{
    Info << "Solving additional problem" << endl;
    restart();
    fvMesh& mesh = _mesh();
    simpleControl& simple = _simple();
    fv::options& fvOptions(_fvOptions());
    volScalarField Tad(_T);
    Foam::Time& runTime = _runTime();
    set_valueFraction();
    Tad_time.resize(0);
    ITHACAutilities::assignIF(Tad, homogeneousBC);
    label timeI = 0;
    assignDirectBC(timeI);
    List<scalar> RobinBC = - Tf;
    forAll(mesh.boundaryMesh(), patchI)
    {
        if (patchI == mesh.boundaryMesh().findPatchID("coldSide"))
        {
            ITHACAutilities::assignMixedBC(Tad, patchI, RobinBC, refGrad, valueFraction);
        }
        else
        {
            ITHACAutilities::assignBC(Tad, patchI, homogeneousBC);
        }
    }
    Tad_time.append(Tad.clone());
    ITHACAstream::exportSolution(Tad, std::to_string(timeSteps[timeI]),
                                 folderOffline,
                                 "Tad");

    while (runTime.loop())
    {
        Info << "Time = " << runTime.timeName() << nl << endl;
        timeI++;
        assignDirectBC(timeI);
        RobinBC = - Tf;
        forAll(mesh.boundaryMesh(), patchI)
        {
            if (patchI == mesh.boundaryMesh().findPatchID("coldSide"))
            {
                ITHACAutilities::assignMixedBC(Tad, patchI, RobinBC, refGrad, valueFraction);
            }
            else
            {
                ITHACAutilities::assignBC(Tad, patchI, homogeneousBC);
            }
        }

        while (simple.correctNonOrthogonal())
        {
            fvScalarMatrix TEqn
            (
                fvm::ddt(Tad) - fvm::laplacian(DT * diffusivity, Tad)
            );
            fvOptions.constrain(TEqn);
            TEqn.solve();
            fvOptions.correct(Tad);
        }

        ITHACAstream::exportSolution(Tad, std::to_string(timeSteps[timeI]),
                                     folderOffline,
                                     "Tad");
        Tad_time.append(Tad.clone());
        runTime.printExecutionTime(Info);
        runTime.write();
    }

    addSol = fieldValueAtThermocouples(Tad_time);
    Info << "END \n" << endl;
}

void inverseHeatTransferProblem::solveDirect()
{
    restart();
    assignDirectBC(0);
    solve("direct");
}

void inverseHeatTransferProblem::solve(const char* problemID, word outputFolder)
{
    M_Assert(diffusivity>1e-36, "Set the diffusivity value");
    volScalarField& T = _T();
    simpleControl& simple = _simple();
    Foam::Time& runTime = _runTime();
    fv::options& fvOptions(_fvOptions());
    label timeI = 0;
    Ttime.resize(0);
    Ttime.append(T.clone());

    if (strcmp( problemID, "direct") == 0)
    {
        while (runTime.loop())
        {
            Info << "Time = " << runTime.timeName() << nl << endl;
            timeI++;
            assignDirectBC(timeI);

            while (simple.correctNonOrthogonal())
            {
                fvScalarMatrix TEqn
                (
                    fvm::ddt(T) - fvm::laplacian(DT * diffusivity, T)
                );
                fvOptions.constrain(TEqn);
                TEqn.solve();
                fvOptions.correct(T);
            }

            Ttime.append(T.clone());

            if (outputFolder != "noFolder")
            {
                ITHACAstream::exportSolution(T, std::to_string(timeSteps[timeI]),
                                             outputFolder,
                                             "T");
            }

            runTime.printExecutionTime(Info);
            runTime.write();
        }
    }
    else
    {
        Info << "Problem name should be direct" << endl;
        exit(10);
    }
}

void inverseHeatTransferProblem::readThermocouples()
{
    Info << "Defining positions of thermocouples" << endl;

    if (!thermocouplesRead)
    {
        word fileName = "./thermocouplesCellsID";

        if (ITHACAutilities::check_file(fileName + "_mat.txt"))
        {
            Info << "Reading thermocouples cells from file" << endl;
            Eigen::MatrixXi TCmatrix = ITHACAstream::readMatrix(fileName + "_mat.txt").cast
                                       <int> ();
            thermocouplesCellID = Foam2Eigen::EigenMatrix2List(TCmatrix);
        }
        else
        {
            Info << "Defining positions of thermocouples" << endl;
            fvMesh& mesh = _mesh();
            volScalarField& T = _T();
            thermocouplesCellID.resize(thermocouplesPos.size());
            forAll(thermocouplesPos, tcI)
            {
                thermocouplesCellID[tcI] = mesh.findCell(thermocouplesPos[tcI]);
            }
            volScalarField thermocouplesField(T);
            ITHACAutilities::assignIF(thermocouplesField, homogeneousBC);
            forAll(thermocouplesCellID, tcI)
            {
                thermocouplesField.ref()[thermocouplesCellID[tcI]] = 1;
            }
            ITHACAstream::exportSolution(thermocouplesField, "1", "./ITHACAoutput/debug/",
                                         "thermocouplesField,");
            Eigen::MatrixXi thermocouplesCellID_eigen = Foam2Eigen::List2EigenMatrix(
                        thermocouplesCellID);
            ITHACAstream::exportMatrix(thermocouplesCellID_eigen, fileName,
                                       "eigen", "./");
        }

        thermocouplesRead = 1;
        samplingTime.resize(timeSamplesNum);
        forAll(samplingTime, timeI)
        {
            samplingTime[timeI] = timeSamplesT0 + timeI * timeSamplesDeltaT;
        }
        sampling2symulationTime();
        Info << "debug: samplingTimes = " << samplingTime << endl;
        residual.resize(thermocouplesNum * timeSamplesNum);
    }
    else
    {
        WarningInFunction << "readThermocouples function called twice." << endl;
        WarningInFunction << "I am not doing the second reading." << endl;
    }
}

Eigen::VectorXd inverseHeatTransferProblem::fieldValueAtThermocouples(
    volScalarField& field)
{
    if (!thermocouplesRead)
    {
        readThermocouples();
    }

    fvMesh& mesh = _mesh();
    dictionary interpolationDict =
        mesh.solutionDict().subDict("interpolationSchemes");
    autoPtr<Foam::interpolation<scalar>> fieldInterp =
                                          Foam::interpolation<scalar>::New(interpolationDict, field);
    Eigen::VectorXd fieldInt;
    fieldInt.resize(thermocouplesPos.size());
    forAll(thermocouplesPos, tcI)
    {
        fieldInt(tcI) = fieldInterp->interpolate(thermocouplesPos[tcI],
                        thermocouplesCellID[tcI]);
    }
    return fieldInt;
}

Eigen::VectorXd inverseHeatTransferProblem::fieldValueAtThermocouples(
    PtrList<volScalarField> fieldList, label fieldI)
{
    Eigen::VectorXd fieldInt = fieldValueAtThermocouples(fieldList[fieldI]);
    return fieldInt;
}

Eigen::VectorXd inverseHeatTransferProblem::fieldValueAtThermocouples(
    PtrList<volScalarField> fieldList)
{
    M_Assert(fieldList.size() == Ntimes + 1,
             "The fieldList must be filled for all the timesteps");
    Eigen::VectorXd fieldInt(timeSamplesNum * thermocouplesNum);
    forAll(samplingSteps, sampleTimeI)
    {
        fieldInt.segment(sampleTimeI * thermocouplesNum, thermocouplesNum) =
            fieldValueAtThermocouples(fieldList, samplingSteps[sampleTimeI]);
    }
    return fieldInt;
}


void inverseHeatTransferProblem::writeFields(label folderNumber,
        const char* folder)
{
    //fvMesh& mesh = _mesh();
    //volScalarField& T = _T();
    //autoPtr<volScalarField> gVolField_
    //(
    //    new volScalarField("g", T)
    //);
    //volScalarField& gVolField = gVolField_();
    //ITHACAutilities::assignIF(gVolField, homogeneousBC);
    ////Access the mesh information for the boundary
    //const polyPatch& cPatch = mesh.boundaryMesh()[hotSide_ind];
    ////List of cells close to a boundary
    //const labelUList& faceCells = cPatch.faceCells();
    //forAll(cPatch, faceI)
    //{
    //    //id of the owner cell having the face
    //    label faceOwner = faceCells[faceI] ;
    //    gVolField[faceOwner] = g[faceI];
    //}
    //ITHACAstream::exportSolution(T, std::to_string(folderNumber + 1), folder,
    //                             "T");
    //ITHACAstream::exportSolution(gVolField, std::to_string(folderNumber + 1),
    //                             folder, "g");
}

void inverseHeatTransferProblem::restart(word fieldName)
{
    Info << "\nResetting time and fields: " << fieldName << "\n" << endl;
    Info << "Reinitializing runTime" << endl;
    Time& runTime = _runTime();
    instantList Times = runTime.times();
    runTime.setTime(Times[1], 0);
    _simple.clear();

    if (fieldName == "T" || fieldName == "all")
    {
        _T.clear();
    }

    Foam::fvMesh& mesh = _mesh();
    _simple = autoPtr<simpleControl>
              (
                  new simpleControl
                  (
                      mesh
                  )
              );

    if (fieldName == "T" || fieldName == "all")
    {
        //Info << "ReReading field T\n" << endl;
        _T = autoPtr<volScalarField>
             (
                 new volScalarField
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
                 )
             );
    }

    Info << "Ready for new computation" << endl;
}

void inverseHeatTransferProblem::sampling2symulationTime()
{
    scalar EPSILON = 2e-16;
    label deltaTimeQuotient = std::floor(timeSamplesDeltaT / deltaTime);
    //Info << "debug: deltaTimeQuotient= " <<deltaTimeQuotient<< endl;
    //Info << "debug: timeSamplesDeltaT= " <<timeSamplesDeltaT<< endl;
    //Info << "debug: deltaTime= " <<deltaTime<< endl;
    M_Assert(std::fabs(timeSamplesDeltaT / deltaTime - std::trunc(
                           timeSamplesDeltaT / deltaTime)) < EPSILON,
             "timeSamplesDeltaT should be a multiple of deltaTime");
    label n0 = (timeSamplesT0 - startTime) / deltaTime;
    //Info << "debug: n0 = " << n0 << endl;
    //Info << "debug: (timeSamplesT0 - startTime) / deltaTime = " << (timeSamplesT0 - startTime) / deltaTime << endl;
    M_Assert(std::fabs(n0 * deltaTime - timeSamplesT0) < EPSILON,
             "The first sampling time must coincide with a symulation timestep");
    scalar samplingEndTime = timeSamplesDeltaT * (timeSamplesNum - 1) + timeSamplesT0;
    //Info << "debug: samplingEndTime = " << samplingEndTime << endl;
    //Info << "debug: EndTime = " << endTime << endl;
    M_Assert(!(endTime + EPSILON < samplingEndTime
               && std::fabs(endTime - samplingEndTime) > EPSILON),
             "The samplingEndTime cannot be later than the symulation endTime");
    samplingSteps.resize(timeSamplesNum);
    forAll(samplingTime, sampleI)
    {
        samplingSteps[sampleI] = n0 + sampleI * deltaTimeQuotient;
    }
    //Info << "debug: samplingSteps = " << samplingSteps << endl;
}

void inverseHeatTransferProblem::parameterizedBC_postProcess(
    List<Eigen::MatrixXd> linSys, Eigen::VectorXd weigths, word outputFolder,
    label verbose)
{
    label debug = 0;
    if (debug)
    {
        // Printing outputs at screen
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(linSys[0],
                                              Eigen::ComputeThinU | Eigen::ComputeThinV);
        std::cout << "Singular values of Theta.transpose() * Theta are " << std::endl;
        std::cout << svd.singularValues() << std::endl;
        std::cout << "weigths = " << std::endl;
        std::cout << weigths << std::endl;
        std::cout << "linSys[1] = " << std::endl;
        std::cout << linSys[1] << std::endl;
        std::cout << "Theta = " << std::endl;
        std::cout << Theta << std::endl;
        residual =  linSys[0] * weigths - linSys[1];
        //std::cout << "Residual  = " << std::endl;
        //std::cout << residual << std::endl;
        std::cout << "Residual 2-norm = " << std::endl;
        std::cout << residual.squaredNorm() << std::endl;
        std::cout << "\n addSol = " << std::endl;
        std::cout << addSol << std::endl;
        std::cout << "T0_vector = " << std::endl;
        std::cout << T0_vector << std::endl;
        std::cout << "Tmeas = " << std::endl;
        std::cout << Tmeas << std::endl;
    }

    reconstrucT(outputFolder);
    J = 0.5 * (Tcomp - Tmeas).dot(Tcomp - Tmeas);
    Info << "J = " << J << endl;
}
