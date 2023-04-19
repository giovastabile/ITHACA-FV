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
/// Source file of the sequentialIHTPtest class.


#include "sequentialIHTPtest.H"

// * * * * * * * * * * * * * * * Constructors * * * * * * * * * * * * * * * * //

// Constructors
sequentialIHTPtest::sequentialIHTPtest() {}

sequentialIHTPtest::sequentialIHTPtest(int argc, char* argv[])
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

void sequentialIHTPtest::setDiffusivity(scalar _diff)
{
    diffusivity = _diff;
}

void sequentialIHTPtest::setSpaceBasis(word type,
        scalar shapeParameter)
{
    if (!thermocouplesRead)
    {
        readThermocouples();
    }
    volScalarField& T = _T();
    fvMesh& mesh = _mesh();
    NbasisInSpace = thermocouplesNum;
    heatFluxSpaceBasis.resize(NbasisInSpace);

    if (type == "rbf")
    {
        Info << "Radial Basis Functions are used." << endl;
        Info << "The center of each function is at the projection " << endl;
        Info << "of each thermocouple on the boundary hotSide.\n\n";

        int thermocouplesCounter = 0;
        int rbfCenterTimeI = 0;
	scalar maxX =  Foam::max(mesh.boundaryMesh()[hotSide_ind].faceCentres().component(Foam::vector::X));
	scalar maxZ =  Foam::max(mesh.boundaryMesh()[hotSide_ind].faceCentres().component(Foam::vector::Z));

        forAll(heatFluxSpaceBasis, funcI)
        {
            scalar thermocoupleX =
                mesh.C()[thermocouplesCellID [thermocouplesCounter]].component(0);
            scalar thermocoupleZ =
                mesh.C()[thermocouplesCellID [thermocouplesCounter]].component(2);
            heatFluxSpaceBasis[funcI].resize(T.boundaryField()[hotSide_ind].size());
            forAll (T.boundaryField()[hotSide_ind], faceI)
            {
                scalar faceX = mesh.boundaryMesh()[hotSide_ind].faceCentres()[faceI].x();
                scalar faceZ = mesh.boundaryMesh()[hotSide_ind].faceCentres()[faceI].z();

                scalar radius = Foam::sqrt((faceX - thermocoupleX) * (faceX - 
                        thermocoupleX) / maxX / maxX + (faceZ - thermocoupleZ) * 
                        (faceZ - thermocoupleZ) / maxZ / maxZ);
	        heatFluxSpaceBasis[funcI][faceI] = Foam::sqrt(1 + (shapeParameter *
                            radius) * (shapeParameter * radius));

            }
            thermocouplesCounter++;
        }
    }
    else if (type == "pod")
    {
        Info << "Not yet implemented, exiting" << endl;
        exit(10);
    }
}

void sequentialIHTPtest::set_gParametrized(word spaceBaseFuncType,
        scalar shapeParameter_space, scalar shapeParameter_time)
{
    volScalarField& T = _T();
    setSpaceBasis(spaceBaseFuncType, shapeParameter_space);

    Info << "Using LINEAR basis in time" << endl;
    NbasisInTime = 2;
    NsamplesWindow = 2;
    gBaseFunctionsOld.resize(NbasisInSpace);
    gBaseFunctionsNew.resize(NbasisInSpace);
    label spaceBaseI = 0;

    // seting old basis
    forAll(gBaseFunctionsOld, baseI)
    {
        double baseCenter = 0;
        gBaseFunctionsOld[baseI].resize(NtimeStepsBetweenSamples);

        for(int timeI = 0; timeI < NtimeStepsBetweenSamples; timeI++)
        {
            scalar timeBase = 1 - std::abs(timeSteps[timeI + 1] - baseCenter) / timeSamplesDeltaT;
            if(timeBase < 0)
            {
                Info << "Something wrong in sequentialIHTPtest::set_gParametrized. Exiting" << endl;
                exit(89);
            }
            //gBaseFunctionsOld[baseI][timeI] = timeBase * heatFluxSpaceBasis[spaceBaseI];
            gBaseFunctionsOld[baseI][timeI] = heatFluxSpaceBasis[spaceBaseI];
        }
        spaceBaseI++;
    }

    // seting new basis
    spaceBaseI = 0;
    forAll(gBaseFunctionsNew, baseI)
    {
        double baseCenter = timeSamplesDeltaT;
        gBaseFunctionsNew[baseI].resize(NtimeStepsBetweenSamples);

        for(int timeI = 0; timeI < NtimeStepsBetweenSamples; timeI++)
        {
            scalar timeBase = 1 - std::abs(timeSteps[timeI + 1] - baseCenter) / timeSamplesDeltaT;
            if(timeBase < 0)
            {
                Info << "Something wrong in sequentialIHTPtest::set_gParametrized. Exiting" << endl;
                exit(89);
            }
            gBaseFunctionsNew[baseI][timeI] = timeBase * heatFluxSpaceBasis[spaceBaseI];
        }
        spaceBaseI++;
    }

    g.resize(timeSteps.size());
    gWeightsOld = (NbasisInSpace);
    forAll (gWeightsOld, weigthI)
    {
        gWeightsOld[weigthI] = 0;
    }
    gWeightsNew = gWeightsOld;
    forAll(timeSteps, timeI)
    {
        g[timeI].resize(T.boundaryField()[hotSide_ind].size(), 0.0);
    }
}

void sequentialIHTPtest::update_gParametrized(List<scalar> wOld, List<scalar> wNew)
{
    // g = wOld * spaceBase * timeBaseOld + wNew * spaceBase * timeBaseNew
    M_Assert(wOld.size() == wNew.size(),
             "weigths size different from basis functions size");
    volScalarField& T = _T();
    label firstTimeI = timeSampleI * NtimeStepsBetweenSamples;
    List<List<scalar>> interpolatedWeights;
    label shortTime = 0;
    Info << "debug : wOld = \n" << wOld << endl;
    Info << "debug : wNew = \n" << wNew << endl;
    if(offlineFlag)
    {
        Info << "Offline heat flux update" << endl;
        for(int timeI = 0; timeI < NtimeStepsBetweenSamples; timeI++)
        {
            forAll (T.boundaryField()[hotSide_ind], faceI)
            {
                g[timeI + 1][faceI] = 0.0;
                forAll (wOld, weightI)
                {
                    g[timeI + 1][faceI] += wOld[weightI] * gBaseFunctionsOld[weightI][timeI][faceI] + wNew[weightI] * gBaseFunctionsNew[weightI][timeI][faceI];
                }
            }
        }
    }
    else
    {
        Info << "Online heat flux update" << endl;
        int t0I = samplingSteps[timeSampleI - 1] + 1;
        int tfI = t0I + NtimeStepsBetweenSamples;
        Info << "In between the timestep " << t0I << " and " << tfI << endl;  
        int shortTimeI = 0;
        for(int timeI = t0I; timeI < tfI; timeI++)
        {
            forAll (T.boundaryField()[hotSide_ind], faceI)
            {
                g[timeI][faceI] = 0.0;
                forAll (wOld, weightI)
                {
                        g[timeI][faceI] += wOld[weightI] * gBaseFunctionsOld[weightI][shortTimeI][faceI] + wNew[weightI] * gBaseFunctionsNew[weightI][shortTimeI][faceI];
                }
            }
            shortTimeI++;
        }
    }
}

volScalarField sequentialIHTPtest::list2Field(List<scalar> list,
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

void sequentialIHTPtest::parameterizedBCoffline(bool force)
{
    fvMesh& mesh = _mesh();
    TbasisOld.resize(0);
    TbasisNew.resize(0);
    M_Assert(diffusivity > 0.0, "Call setDiffusivity to set up the diffusivity");

    offlineEndTime = NtimeStepsBetweenSamples * deltaTime;

    Info << "\nComputing offline" << endl;
    ThetaOld.resize(NbasisInSpace, NbasisInSpace);
    ThetaNew.resize(NbasisInSpace, NbasisInSpace);
    gWeightsOld.resize(NbasisInSpace);
    gWeightsNew.resize(NbasisInSpace);
    offlineFlag = 1;
    Info << "ThetaNew size = " << ThetaNew.rows() << ", " << ThetaNew.cols() << endl;
    solveAdditional();

    Info << "Offline for OLD time base" << endl;
    for (label baseI = 0; baseI < ThetaOld.cols(); baseI++)
    {
        Info << "\n--------------------------------------\n" << endl;
        Info << "Base " << baseI + 1 << " of " << ThetaOld.cols() << endl;
        Info << "\n--------------------------------------\n" << endl;
        restart();
        Ttime.resize(0);
        gWeightsOld = Foam::zero();
        gWeightsNew = Foam::zero();
        gWeightsOld[baseI] =  1;
        timeSampleI = 0;
        update_gParametrized(gWeightsOld, gWeightsNew);
        solveDirect();

        for(int timeI = 0; timeI < NtimeStepsBetweenSamples; timeI++)
        {
            volScalarField& T = Ttime[timeI];
            /// Saving basis
            volScalarField gParametrizedField = list2Field(g[timeI + 1]);
            ITHACAstream::exportSolution(gParametrizedField,
                                         std::to_string(timeSteps[timeI + 1]),
                                         folderOfflineOld,
                                         "g" + std::to_string(baseI + 1));
            ITHACAstream::exportSolution(T, std::to_string(timeSteps[timeI + 1]),
                                         folderOfflineOld,
                                         "T" + std::to_string(baseI + 1));
        }
        TbasisOld.append(Ttime.clone());
        Tcomp = fieldValueAtThermocouples(Ttime[NtimeStepsBetweenSamples - 1]);
        M_Assert(Tcomp.size() == addSol.size(), "Something wrong in reading values at the observations points");
        for(int i = 0; i < Tcomp.size(); i++)
        {
            ThetaOld(i, baseI) = Tcomp(i) + addSol(i);
        }
    }
    ITHACAstream::exportMatrix(ThetaOld, "ThetaOld", "eigen", folderOfflineOld);
    ITHACAstream::exportMatrix(addSol, "addSol", "eigen", folderOfflineOld);
    

    Eigen::MatrixXd A = ThetaOld.transpose() * ThetaOld;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A,
                                          Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXd singularValues = svd.singularValues();
    ITHACAstream::exportMatrix(singularValues, "singularValues", "eigen",
                               folderOfflineOld);

    Info << "Offline for NEW time base" << endl;
    for (label baseI = 0; baseI < ThetaNew.cols(); baseI++)
    {
        Info << "\n--------------------------------------\n" << endl;
        Info << "Base " << baseI + 1 << " of " << ThetaNew.cols() << endl;
        Info << "\n--------------------------------------\n" << endl;
        restart();
        Ttime.resize(0);
        gWeightsNew = Foam::zero();
        gWeightsOld = Foam::zero();
        gWeightsNew[baseI] =  1;
        timeSampleI = 0;
        update_gParametrized(gWeightsOld, gWeightsNew);
        solveDirect();

        for(int timeI = 0; timeI < NtimeStepsBetweenSamples; timeI++)
        {
            volScalarField& T = Ttime[timeI];
            /// Saving basis
            volScalarField gParametrizedField = list2Field(g[timeI + 1]);
            ITHACAstream::exportSolution(gParametrizedField,
                                         std::to_string(timeSteps[timeI + 1]),
                                         folderOfflineNew,
                                         "g" + std::to_string(baseI + 1));
            ITHACAstream::exportSolution(T, std::to_string(timeSteps[timeI + 1]),
                                         folderOfflineNew,
                                         "T" + std::to_string(baseI + 1));
        }
        TbasisNew.append(Ttime.clone());
        Info << "debug : Ttime.size() = " << Ttime.size() << endl;
        Tcomp = fieldValueAtThermocouples(Ttime[NtimeStepsBetweenSamples - 1]);
        M_Assert(Tcomp.size() == addSol.size(), "Something wrong in reading values at the observations points");
        for(int i = 0; i < Tcomp.size(); i++)
        {
            ThetaNew(i, baseI) = Tcomp(i) + addSol(i);
        }
    }

    ITHACAstream::exportMatrix(ThetaNew, "ThetaNew", "eigen", folderOfflineNew);
    ITHACAstream::exportMatrix(addSol, "addSol", "eigen", folderOfflineNew);
    

    A = ThetaOld.transpose() * ThetaOld;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd2(A,
                                          Eigen::ComputeThinU | Eigen::ComputeThinV);
    singularValues = svd2.singularValues();
    ITHACAstream::exportMatrix(singularValues, "singularValues", "eigen",
                               folderOfflineNew);
    offlineFlag = 0;
    Info << "\nOffline ENDED" << endl;
}

void sequentialIHTPtest::reconstrucT(word outputFolder, volScalarField initialField)
{
    restartOffline();
    M_Assert(diffusivity>1e-36, "Set the diffusivity value");
    volScalarField& T = _T();
    T = initialField;
    assignDirectBC(0);
    simpleControl& simple = _simple();
    Foam::Time& runTime = _runTime();
    fv::options& fvOptions(_fvOptions());
    label timeI = 0;
    Ttime.resize(0);


    while (runTime.loop())
    {
        Info << "Time = " << runTime.timeName() << nl << endl;
        timeI++;
        scalar realTime = timeSamplesT0 + (timeSampleI - 1) * timeSamplesDeltaT + deltaTime * (timeI + 1);
        int realTimeI = samplingSteps[timeSampleI - 1] + timeI;
        Info << "realTimeI = " << realTimeI << endl;
        assignDirectBC(realTimeI);

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
        ITHACAstream::exportSolution(T, std::to_string(realTime),
                                     outputFolder,
                                     "Treconstructed");
        volScalarField gParametrizedField = list2Field(g[realTimeI]);
        ITHACAstream::exportSolution(gParametrizedField,
                                     std::to_string(realTime),
                                     outputFolder,
                                     "gReconstructed");

        runTime.printExecutionTime(Info);
        runTime.write();
    }
}
/*
void sequentialIHTPtest::reconstrucT(word outputFolder)
{
    Info << "Reconstructing field T" << endl;
    Ttime.resize(0);
    Info << "\nReconstructing solution in the time domain (" << timeSteps[NtimeStepsBetweenSamples * timeSampleI] << ", " << timeSteps[NtimeStepsBetweenSamples + NtimeStepsBetweenSamples * timeSampleI] << "]\n" << endl;
    restart();

    Info << "debug : TbasisOld.size() = " << TbasisOld.size() << endl;
    Info << "debug : TbasisNew.size() = " << TbasisNew.size() << endl;
    Info << "debug : Tad_time.size() = " << Tad_time.size() << endl;
    Info << "debug : T0_time.size() = " << T0_time.size() << endl;
    int timestepWindow = NsamplesWindow * NtimeStepsBetweenSamples + 1;
    for(int timeI = 0; timeI < NtimeStepsBetweenSamples; timeI++)
    {
        volScalarField T(_T);
        ITHACAutilities::assignIF(T, homogeneousBC);
        forAll(TbasisOld, baseI)
        {
            T += gWeightsOld[baseI] * (TbasisOld[baseI][timeI] + Tad_time[timeI]);
        }
        forAll(TbasisNew, baseI)
        {
            T += gWeightsNew[baseI] * (TbasisNew[baseI][timeI] + Tad_time[timeI]);
        }
        T += - Tad_time[timeI] + T0_time[timeI];

        scalar realTime = timeSamplesT0 + (timeSampleI - 1) * timeSamplesDeltaT + deltaTime * (timeI + 1);
        ITHACAstream::exportSolution(T, std::to_string(realTime),
                                     outputFolder,
                                     "Treconstructed");
        int realTimeI = 1 + NtimeStepsBetweenSamples + (timeSampleI - 1) * NtimeStepsBetweenSamples + timeI;
        volScalarField gParametrizedField = list2Field(g[realTimeI]);
        ITHACAstream::exportSolution(gParametrizedField,
                                     std::to_string(realTime),
                                     outputFolder,
                                     "gReconstructed");
        Ttime.append(T.clone());
    }
}
*/
void sequentialIHTPtest::parameterizedBC(word outputFolder, PtrList<volScalarField> initialField,
        word linSys_solver,
        label TSVD_filter)
{
    Info << endl << "Using quasilinearity of direct problem" << endl;
    //parameterizedBCoffline(folder, forceOffline);

    for(int timeI = 0; timeI < samplingSteps[0] + 1; timeI++)
    {
        volScalarField T(_T);
        ITHACAutilities::assignIF(T, homogeneousBC);
        ITHACAstream::exportSolution(T, std::to_string(timeSteps[timeI]),
                                     outputFolder,
                                     "Treconstructed");
        volScalarField gParametrizedField = list2Field(g[timeI]);
        ITHACAstream::exportSolution(gParametrizedField,
                                     std::to_string(timeSteps[timeI]),
                                     outputFolder,
                                     "gReconstructed");
    }

    timeSampleI = 1;
    gWeightsNew = Foam::zero();
    while(timeSampleI < timeSamplesNum)
    {
        Info << "\nTime sample " << timeSampleI + 1 << endl;

        Info << "Time step = " << samplingSteps[timeSampleI] << endl;

	solveT0(initialField[samplingSteps[timeSampleI - 1]]);

	TmeasShort = Tmeas.segment(thermocouplesNum * timeSampleI, thermocouplesNum);
        gWeightsOld = gWeightsNew;
        Eigen::VectorXd weigthsOld(gWeightsOld.size());
        forAll(gWeightsOld, wI)
        {
            weigthsOld(wI) = gWeightsOld[wI];
        }

        List<Eigen::MatrixXd> linSys;
        linSys.resize(2);
        linSys[0] = ThetaNew.transpose() * ThetaNew;
        linSys[1] = ThetaNew.transpose() * (TmeasShort + addSol - T0_vector - ThetaOld * weigthsOld);
        std::cout << "debug: addSol \n" << addSol <<  std::endl;
        std::cout << "debug: T0_vector \n" << T0_vector <<  std::endl;
        std::cout << "debug: weigthsOld \n" << weigthsOld <<  std::endl;
        std::cout << "debug: TmeasShort \n" << TmeasShort <<  std::endl;
        std::cout << "debug: linSys[1] \n" << linSys[1] <<  std::endl;
        std::cout << "debug: linSys[0] \n" << linSys[0] <<  std::endl;
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
        else if (linSys_solver == "TSVD")
        {
            weigths = ITHACAregularization::TSVD(linSys[0], linSys[1], TSVD_filter);
        }
        else
        {
            Info << "Select a linear system solver in this list:" << endl
                 << "fullPivLU, jacobiSvd, householderQr, ldlt" << endl;
            exit(1);
        }
        M_Assert(gWeightsNew.size() == weigths.size(), "Wrong weights dimention");
        forAll(gWeightsNew, weightI)
        {
            gWeightsNew[weightI] = weigths(weightI);
        }
        Info << "New weights = \n" << gWeightsNew << endl;
        update_gParametrized(gWeightsOld, gWeightsNew);
        label verbose = 0;
        parameterizedBC_postProcess(linSys, weigths, outputFolder, initialField[samplingSteps[timeSampleI - 1]], verbose);
	timeSampleI++;
    }
    ITHACAstream::exportMatrix(Jlist, "costFunction", "eigen", outputFolder);
    Info << "End" << endl;
    Info << endl;
}

void sequentialIHTPtest::set_valueFraction()
{
    fvMesh& mesh = _mesh();
    valueFraction.resize(mesh.boundaryMesh()["coldSide"].size());
    homogeneousBCcoldSide.resize(mesh.boundaryMesh()["coldSide"].size());
    Eigen::VectorXd faceCellDist =
        ITHACAutilities::boudaryFaceToCellDistance(mesh, coldSide_ind);
    forAll (valueFraction, faceI)
    {
        valueFraction[faceI] = 1 / (1 + (k / H / faceCellDist(faceI)));
        homogeneousBCcoldSide[faceI] =  0;
    }
    refGrad = homogeneousBCcoldSide;
}

void sequentialIHTPtest::assignDirectBC(label timeI)
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

void sequentialIHTPtest::solveT0(volScalarField initialField)
{
    Info << "Solving T0 problem" << endl;
    restartOffline();
    fvMesh& mesh = _mesh();
    simpleControl& simple = _simple();
    fv::options& fvOptions(_fvOptions());
    volScalarField T0_field(_T);
    Foam::Time& runTime = _runTime();
    set_valueFraction();
    List<scalar> RobinBC = Tf * 0.0;
    word outputFolder = "./ITHACAoutput/T0/";
        ITHACAutilities::assignIF(T0_field, initialField);
    //if(timeSampleI == 1)
    //{
    //    Info << "debug : First timestep. initialField = " << initialField.internalField()[0] << endl;
    //    ITHACAutilities::assignIF(T0_field, initialField);
    //}
    //else
    //{
    //    Info << "debug : other timesteps" << endl;
    //    Info << "debug : Ttime.size() = " << Ttime.size() << endl;
    //    Info << "debug : NtimeStepsBetweenSamples = " << NtimeStepsBetweenSamples << endl;
    //    Info << "debug : initialField = " << Ttime[NtimeStepsBetweenSamples - 1].internalField()[0] << endl;

    //    ITHACAutilities::assignIF(T0_field, Ttime[NtimeStepsBetweenSamples - 1]);
    //}
    std::cout << "debug : T0 at thermocouples = " << fieldValueAtThermocouples(T0_field) << std::endl;

    T0_time.resize(0);
    label timeI = 0;
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

        T0_time.append(T0_field.clone());
        int realTimeI = 1 + NtimeStepsBetweenSamples + (timeSampleI - 1) * NtimeStepsBetweenSamples + timeI - 1;
        ITHACAstream::exportSolution(T0_field, std::to_string(timeSteps[realTimeI]),
                                     outputFolder,
                                     "T0_field");
        runTime.printExecutionTime(Info);
        runTime.write();
    }

    T0_vector = fieldValueAtThermocouples(T0_field);
    Info << "SolveT0 ENDED\n" << endl;
}

void sequentialIHTPtest::solveAdditional()
{
    Info << "Solving additional problem" << endl;
    restartOffline();
    fvMesh& mesh = _mesh();
    simpleControl& simple = _simple();
    fv::options& fvOptions(_fvOptions());
    volScalarField Tad(_T);
    Foam::Time& runTime = _runTime();
    set_valueFraction();
    Tad_time.resize(0);
    ITHACAutilities::assignIF(Tad, homogeneousBC);
    label timeI = 0;


    while (runTime.loop())
    {
        Info << "Time = " << runTime.timeName() << nl << endl;
        timeI++;
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

        Tad_time.append(Tad.clone());
        ITHACAstream::exportSolution(Tad, std::to_string(timeSteps[timeI]),
                                     folderOfflineNew,
                                     "Tad");
        runTime.printExecutionTime(Info);
        runTime.write();
    }

    Info << "Tad_time.size = " << Tad_time.size() << endl;
    Info << "Ntime = " << Ntimes << endl;
    addSol = fieldValueAtThermocouples(Tad);
    Info << "END \n" << endl;
}

void sequentialIHTPtest::solveDirect()
{
    if(offlineFlag)
    {
        restartOffline();
    }
    else
    {
        restart();
    }
    M_Assert(diffusivity>1e-36, "Set the diffusivity value");
    volScalarField& T = _T();
    assignDirectBC(0);
    if(offlineFlag)
    {
        ITHACAutilities::assignIF(T, homogeneousBC);
    }
    simpleControl& simple = _simple();
    Foam::Time& runTime = _runTime();
    fv::options& fvOptions(_fvOptions());
    label timeI = 0;
    Ttime.resize(0);


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

        runTime.printExecutionTime(Info);
        runTime.write();
    }
    Info << "debug : timeI = " << timeI << endl;
    Info << "Direct computation ENDED" << endl;
    
}

void sequentialIHTPtest::readThermocouples()
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
	NtimeStepsBetweenSamples = timeSamplesDeltaT / deltaTime;
        residual.resize(thermocouplesNum * timeSamplesNum);
    }
    else
    {
        WarningInFunction << "readThermocouples function called twice." << endl;
        WarningInFunction << "I am not doing the second reading." << endl;
    }
}

Eigen::VectorXd sequentialIHTPtest::fieldValueAtThermocouples(
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

Eigen::VectorXd sequentialIHTPtest::fieldValueAtThermocouples(
    PtrList<volScalarField> fieldList, label fieldI)
{
    Eigen::VectorXd fieldInt = fieldValueAtThermocouples(fieldList[fieldI]);
    return fieldInt;
}

Eigen::VectorXd sequentialIHTPtest::fieldValueAtThermocouples(
    PtrList<volScalarField> fieldList)
{
    Eigen::VectorXd fieldInt;
    if ( fieldList.size() == Ntimes + 1 )
    {
        Info << "\n Sampling for ALL sampling times \n\n" << endl;
        fieldInt.resize(timeSamplesNum * thermocouplesNum);
        forAll(samplingSteps, sampleTimeI)
        {
            fieldInt.segment(sampleTimeI * thermocouplesNum, thermocouplesNum) =
                fieldValueAtThermocouples(fieldList, samplingSteps[sampleTimeI]);
        }
    }
    else if ( fieldList.size() == NtimeStepsBetweenSamples + 1 )
    {
        fieldInt.resize(NsamplesWindow * thermocouplesNum);
        if(NsamplesWindow == 1)
        {
            Info << "\nField size = " << fieldList.size() << ".\nSampling ONLY the last timestep\n\n" << endl;
            fieldInt.segment(0, thermocouplesNum) =
                fieldValueAtThermocouples(fieldList, NtimeStepsBetweenSamples);
        }
        else if(NsamplesWindow == 2)
        {
            Info << "\nField size = " << fieldList.size() << ".\nSampling first AND last timesteps\n" << endl;
            for(int i = 0; i < NsamplesWindow; i++)
            {
                fieldInt.segment(thermocouplesNum * i, thermocouplesNum) =
                    fieldValueAtThermocouples(fieldList, NtimeStepsBetweenSamples * i);
            }
        }
    }
    else
    {
        Info << "The input fieldList of sequentialIHTPtest::fieldValueAtThermocouples can have size Ntimes + 1 (=" << Ntimes + 1 << ") or\n";
        Info << " NtimeStepsBetweenSamples + 1 (=" <<  NtimeStepsBetweenSamples + 1<< ") but has size " << fieldList.size() << endl;
        Info << "Exiting." << endl;
        exit(23);
    }
    Info << "\nSampling done \n" << endl;
    return fieldInt;
}


void sequentialIHTPtest::restart()
{
    Time& runTime = _runTime();
    instantList Times = runTime.times();
    runTime.setTime(Times[1], 0);
    _simple.clear();
   _T.clear();

    Foam::fvMesh& mesh = _mesh();
    _simple = autoPtr<simpleControl>
              (
                  new simpleControl
                  (
                      mesh
                  )
              );

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
    

    Info << "Ready for new computation" << endl;
}

void sequentialIHTPtest::restartOffline()
{
    Info << "Setting endTime to offlineEndTime" << endl;
    restart();
    Time& runTime = _runTime();
    instantList Times = runTime.times();
    runTime.setTime(0.0, 0);
    runTime.setEndTime(offlineEndTime);
    Info << "Ready for new offline computation" << endl;
}

void sequentialIHTPtest::restartT0()
{
    Info << "Setting endTime to offlineEndTime" << endl;
    restart();
    Time& runTime = _runTime();
    instantList Times = runTime.times();
    runTime.setTime(0.0, 0);
    runTime.setEndTime(timeSamplesDeltaT);
    Info << "Ready for new T0 computation" << endl;
}

void sequentialIHTPtest::sampling2symulationTime()
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
    M_Assert(n0 > 0, "First sampling step cannot be 0");
    //Info << "debug: n0 = " << n0 << endl;
    //Info << "debug: (timeSamplesT0 - startTime) / deltaTime = " << (timeSamplesT0 - startTime) / deltaTime << endl;
    M_Assert(std::fabs(n0 * deltaTime - timeSamplesT0) < EPSILON,
             "The first sampling time must coincide with a simulation timestep");
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

void sequentialIHTPtest::parameterizedBC_postProcess(
    List<Eigen::MatrixXd> linSys, Eigen::VectorXd weigths, word outputFolder, volScalarField initialField,
    label verbose)
{
    if (verbose)
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
        std::cout << "ThetaNew = " << std::endl;
        std::cout << ThetaNew << std::endl;
        std::cout << "ThetaOld = " << std::endl;
        std::cout << ThetaOld << std::endl;
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

    reconstrucT(outputFolder, initialField);
    Info << "debug : Ttime.size() = " << Ttime.size() << endl;
    Tcomp = fieldValueAtThermocouples(Ttime[Ttime.size() - 1]);
    std::cout << "Tcomp = \n" << Tcomp.transpose() << std::endl;
    std::cout << "TmeasShort = \n" << TmeasShort.transpose() << std::endl;
    J = 0.5 * Foam::sqrt((Tcomp - TmeasShort).dot(Tcomp - TmeasShort));
    Info << "J = " << J << endl;
    Jlist.conservativeResize(Jlist.size() + 1);
    Jlist(Jlist.size() - 1) = J;
}
