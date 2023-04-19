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
/// Source file of the sequentialIHTP class.


#include "sequentialIHTP.H"

// * * * * * * * * * * * * * * * Constructors * * * * * * * * * * * * * * * * //

// Constructors
sequentialIHTP::sequentialIHTP() {}

sequentialIHTP::sequentialIHTP(int argc, char* argv[])
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

void sequentialIHTP::setDiffusivity(scalar _diff)
{
    diffusivity = _diff;
}

void sequentialIHTP::setSpaceBasis(word type,
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

    //gBasisSize = thermocouplesNum * (NtimeStepsBetweenSamples);
    //gBasisSize = thermocouplesNum * 1;
    //

    if (type == "rbf")
    {
        Info << "Radial Basis Functions are used." << endl;
        Info << "The center of each function is at the projection " << endl;
        Info << "of each thermocouple on the boundary hotSide.\n\n";


        //gBaseFunctions.resize(gBasisSize);
        //gWeights.resize(gBasisSize);
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

void sequentialIHTP::set_gBaseFunctions(word type,
        scalar shapeParameter_space, scalar shapeParameter_time)
{
    volScalarField& T = _T();
    fvMesh& mesh = _mesh();
    gBasisSize = thermocouplesNum * basisDeltaSample;
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
            gBaseFunctions[funcI].resize(timeSteps.size());
            scalar thermocoupleX =
                mesh.C()[thermocouplesCellID [thermocouplesCounter]].component(0);
            scalar thermocoupleZ =
                mesh.C()[thermocouplesCellID [thermocouplesCounter]].component(2);
            scalar sTime = samplingTime[samplingTimeI];
            
	    for(int timeI = 0; timeI < NtimeStepsBetweenSamples + 1; timeI++)
            {
                gBaseFunctions[funcI][timeI].resize(T.boundaryField()[hotSide_ind].size());
                scalar time = timeSteps[timeI];
		scalar timeBase = 0;

		if(timeBasisType == "constant")
		{
		    Info << "\nUsing CONSTANT time basis\n";
		    timeBase = 1;
		}
		else if(timeBasisType == "linear")
		{
		    Info << "\nUsing LINEAR time basis\n";
		    if(std::abs(time - sTime) < timeSamplesDeltaT)
		    {
		        Info << "time = " <<time << endl;
		        Info << "sTime = " <<sTime << endl;
		        Info << "std::abs(time - sTime) = " << std::abs(time - sTime) << endl;
		        timeBase = 1 - std::abs(time - sTime) / timeSamplesDeltaT;
		    }
		}
		else if(timeBasisType == "rbf")
		{
		    Info << "\nUsing RBF time basis\n";
                    scalar radius_time = Foam::sqrt((time - sTime) * (time - sTime));
		    timeBase = Foam::exp( - (shapeParameter_time * radius_time) * (shapeParameter_time * radius_time) );
		}
		else
		{
		    Info << "Type of time base for the heat flux not defined. EXITING" << endl;
		    exit(101);
		}


                forAll (T.boundaryField()[hotSide_ind], faceI)
                {
                    scalar faceX = mesh.boundaryMesh()[hotSide_ind].faceCentres()[faceI].x();
                    scalar faceZ = mesh.boundaryMesh()[hotSide_ind].faceCentres()[faceI].z();
                    scalar radius_space = Foam::sqrt((faceX - thermocoupleX) * (faceX - thermocoupleX) +
                                               (faceZ - thermocoupleZ) * (faceZ - thermocoupleZ));

		    scalar exponent = shapeParameter_space * radius_space;// + shapeParameter_time * radius_time;
                    gBaseFunctions[funcI][timeI][faceI] = Foam::exp(- exponent * exponent) * timeBase;
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

void sequentialIHTP::set_gParametrized(word spaceBaseFuncType,
        scalar shapeParameter_space, word timeBaseFuncType, scalar shapeParameter_time)
{
    volScalarField& T = _T();
    M_Assert(timeBaseFuncType == "linear" || timeBaseFuncType == "constant", "Time basis can be linear or constant");
    setSpaceBasis(spaceBaseFuncType, shapeParameter_space);

    if(timeBaseFuncType == "constant")
    {
        Info << "Using CONSTANT basis in time" << endl;
        NbasisInTime = 1;
        NsamplesWindow = 1;
        Nbasis = NbasisInTime * NbasisInSpace;
        gBaseFunctions.resize(Nbasis);
        NtimestepsInSequence = NtimeStepsBetweenSamples * NbasisInTime + 1;
        forAll(gBaseFunctions, baseI)
        {
            gBaseFunctions[baseI].resize(NtimestepsInSequence);
            for(int timeI = 0; timeI < NtimestepsInSequence; timeI++)
            {
                gBaseFunctions[baseI][timeI] = heatFluxSpaceBasis[baseI];
            }
        }
    }
    else if(timeBaseFuncType == "linear")
    {
        Info << "Using LINEAR basis in time" << endl;
        NbasisInTime = 2;
        NsamplesWindow = 2;
        Nbasis = NbasisInTime * NbasisInSpace;
        gBaseFunctions.resize(Nbasis);
        NtimestepsInSequence = NtimeStepsBetweenSamples * NsamplesWindow + 1;
        label spaceBaseI = 0;
        forAll(gBaseFunctions, baseI)
        {
            double baseCenter;
            gBaseFunctions[baseI].resize(NtimestepsInSequence);
            if(baseI < NbasisInSpace)
            {
                // I am in the first time base
                baseCenter = timeSteps[0]; // TODO this only works if the first sampling time is at the same distance from 0 as from the other samples
            }
            else
            {
                // I am in the second time base
                baseCenter = samplingTime[1];
            }

            if(spaceBaseI == NbasisInSpace)
            {
                spaceBaseI = 0;
            }

            for(int timeI = 0; timeI < NtimestepsInSequence; timeI++)
            {
                scalar timeBase = 1 - std::abs(timeSteps[timeI] - baseCenter) / (2 * timeSamplesDeltaT);
                if(timeBase < 0)
                {
                    timeBase = 0;
                }
                gBaseFunctions[baseI][timeI] = timeBase * heatFluxSpaceBasis[spaceBaseI];
            }
            spaceBaseI++;
        }
    }
    else
    {
        Info << "Time basis can be linear or constant, exiting" << endl;
        exit(13);
    }

    g.resize(timeSteps.size());
    gWeights.resize(Nbasis);
    forAll (gWeights, weigthI)
    {
        gWeights[weigthI] = 0; //-10000;
    }
    forAll(timeSteps, timeI)
    {
        g[timeI].resize(T.boundaryField()[hotSide_ind].size(), 0.0);
    }
}

List<List<scalar>> sequentialIHTP::interpolateWeights(List<scalar> Wold, List<scalar> Wnew)
{
    M_Assert(Wold.size() == Wnew.size(), "Input weights vectors must have the same size");

    double t0 = 0;
    double t1 = NtimeStepsBetweenSamples * deltaTime;
    List<List<scalar>> Wout;
    Wout.resize(Wold.size());
    forAll (Wold, wI)
    {
        Wout[wI].resize(NtimeStepsBetweenSamples + 1);
        for(int timeI = 0; timeI < NtimeStepsBetweenSamples + 1; timeI++)
        {
            double time = (timeI + 1) * deltaTime;
            double a = Wold[wI] - (Wnew[wI] - Wold[wI]) / (t1 - t0) * t0;
            double b = (Wnew[wI] - Wold[wI]) / (t1 - t0);
            Wout[wI][timeI] = a + b * time;
        }
    }
    return Wout;
}

void sequentialIHTP::update_gParametrized(List<scalar> weights)
{
    M_Assert(weights.size() == Nbasis,
             "weigths size different from basis functions size");
    volScalarField& T = _T();
    label firstTimeI = timeSampleI * NtimeStepsBetweenSamples;
    List<List<scalar>> interpolatedWeights;
    if(NbasisInTime == 1 && offlineFlag || (NbasisInTime == 2 && offlineFlag))
    {
        Info << "debug : update offline" << endl;
    }
    else if(NbasisInTime == 2)
    {
        Info << "debug : update online" << endl;
    }
    else if(NbasisInTime == 1 && !offlineFlag)
    {
        Info << "debug : I am using the interpolation update" << endl;
        if(timeSampleI > 0)
        {
            interpolatedWeights = interpolateWeights(gWeightsOld, weights);
        }
    }

    int lastTimeStep = firstTimeI + NtimeStepsBetweenSamples + 1;
    if(offlineFlag && NbasisInTime == 2)
    {
        lastTimeStep = firstTimeI + NbasisInTime * NtimeStepsBetweenSamples + 1;
    }

    label shortTime = 0;
    for(int timeI = firstTimeI; timeI < lastTimeStep; timeI++)
    {
        forAll (T.boundaryField()[hotSide_ind], faceI)
        {
            g[timeI][faceI] = 0.0;
            forAll (weights, weightI)
            {
                if(NbasisInTime == 1 && offlineFlag || (NbasisInTime == 2 && offlineFlag))
                {
                    g[timeI][faceI] += weights[weightI] * gBaseFunctions[weightI][shortTime][faceI];
                }
                else if(NbasisInTime == 2)
                {
                    g[timeI][faceI] += weights[weightI] * gBaseFunctions[weightI][shortTime + NtimeStepsBetweenSamples][faceI];
                }
                else if(NbasisInTime == 1 && !offlineFlag)
                {
                    if(timeSampleI > 0)
                    {
                        g[timeI][faceI] += interpolatedWeights[weightI][shortTime] * gBaseFunctions[weightI][shortTime][faceI];
                    }
                    else
                    {
                        g[timeI][faceI] += weights[weightI] * gBaseFunctions[weightI][shortTime][faceI];
                    }
                }
            }
        }
	shortTime++;
    }
}

volScalarField sequentialIHTP::list2Field(List<scalar> list,
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

Eigen::VectorXd  sequentialIHTP::TSVD(Eigen::MatrixXd A,
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
            x = coeff / svd.singularValues()(i) * V.col(i);
        }
        else
        {
            x += coeff / svd.singularValues()(i) * V.col(i);
        }
    }

    return x;
}

void sequentialIHTP::parameterizedBCoffline(bool force)
{
    fvMesh& mesh = _mesh();
    Tbasis.resize(0);
    M_Assert(diffusivity > 0.0, "Call setDiffusivity to set up the diffusivity");

    offlineTimestepsSize = NtimeStepsBetweenSamples + 1;
    offlineEndTime = NtimeStepsBetweenSamples * NbasisInTime * deltaTime;

    if (ITHACAutilities::check_file(folderOffline + "/Theta_mat.txt") && force == 0)
    {
        Info << "\nOffline already computed." << endl;
        Info << "Check that the basis used for the parameterized BC are correct (RBF, POD, etc.)"
             << endl;
        Theta = ITHACAstream::readMatrix(folderOffline + "Theta_mat.txt");
        addSol = ITHACAstream::readMatrix(folderOffline + "addSol_mat.txt");
        ITHACAstream::read_fields(Tad_time, "Tad", folderOffline);

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
        Theta.resize(Nbasis, gWeights.size());
	offlineFlag = 1;
        Info << "Theta size = " << Theta.rows() << ", " << Theta.cols() << endl;
        solveAdditional();

        for (label baseI = 0; baseI < Theta.cols(); baseI++)
        {
            Info << "\n--------------------------------------\n" << endl;
            Info << "Base " << baseI + 1 << " of " << Theta.cols() << endl;
            Info << "\n--------------------------------------\n" << endl;
            restart();
            Ttime.resize(0);
            gWeights = Foam::zero();
            gWeights[baseI] =  1;
	    timeSampleI = 0;
            update_gParametrized(gWeights);
            solveDirect(offlineFlag);

            for(int timeI = 0; timeI < offlineTimestepsSize; timeI++)
            {
                Info << "timeI = " << timeI << ". debug 1" << endl;
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
                Info << "timeI = " << timeI << ". debug 2" << endl;
            }
            Tbasis.append(Ttime.clone());
            Tcomp = fieldValueAtThermocouples(Ttime);
            M_Assert(Tcomp.size() == addSol.size(), "Something wrong in reading values at the observations points");
            for(int i = 0; i < Tcomp.size(); i++)
            {
                Theta(i, baseI) = Tcomp(i) + addSol(i);
            }
        }

        ITHACAstream::exportMatrix(Theta, "Theta", "eigen", folderOffline);
        ITHACAstream::exportMatrix(addSol, "addSol", "eigen", folderOffline);
    }

    Eigen::MatrixXd A = Theta.transpose() * Theta;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A,
                                          Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXd singularValues = svd.singularValues();
       std::cout << "singularValues = " << std::endl;
    double conditionNumber = singularValues.maxCoeff() / singularValues.minCoeff();
    Info << "Condition number = " << conditionNumber << endl;
    ITHACAstream::exportMatrix(singularValues, "singularValues", "eigen",
                               folderOffline);
    offlineFlag = 0;
    Info << "\nOffline ENDED" << endl;
}

void sequentialIHTP::reconstrucT(word outputFolder)
{
    Info << "Reconstructing field T" << endl;
    Ttime.resize(0);
    if(NsamplesWindow > 2)
    {
        Info << "Still to implement sequentialIHTP::reconstrucT for higher order basis in space" << endl;
        Info << "Exiting" << endl;
        exit(10);
    }
    if(NsamplesWindow == 1)
    {
         Info << "\nExporting solution in the time domain (" << timeSteps[NtimeStepsBetweenSamples * timeSampleI] << ", " << timeSteps[NtimeStepsBetweenSamples + NtimeStepsBetweenSamples * timeSampleI] << "]\n" << endl;
         restart();
         int timestepWindow = NsamplesWindow * NtimeStepsBetweenSamples + 1;
         for(int timeI = 0; timeI < timestepWindow; timeI++)
         {
             volScalarField T(_T);
             ITHACAutilities::assignIF(T, homogeneousBC);
             forAll(Tbasis, baseI)
             {
                 T += gWeights[baseI] * (Tbasis[baseI][timeI] + Tad_time[timeI]);
             }
             T += - Tad_time[timeI] + T0_time[timeI];

             label realTimeStep = timeI + NtimeStepsBetweenSamples * timeSampleI;
             if(timeSteps[realTimeStep] >= timeSteps[NtimeStepsBetweenSamples * timeSampleI])
             {
                 ITHACAstream::exportSolution(T, std::to_string(timeSteps[realTimeStep]),
                                              outputFolder,
                                              "Treconstructed");
                 volScalarField gParametrizedField = list2Field(g[timeI]);
                 ITHACAstream::exportSolution(gParametrizedField,
                                              std::to_string(timeSteps[realTimeStep]),
                                              outputFolder,
                                              "gReconstructed");
             }
             Ttime.append(T.clone());
         }
     }
     else if(NsamplesWindow == 2)
     {
         // REMEMBER: timeSampleI stas from 1
         label firstRealTimeStep = 0;
         label firstTimeStep = 0;
         if(timeSampleI > 1)
         {
             firstRealTimeStep = NtimeStepsBetweenSamples * timeSampleI;
             firstTimeStep = NtimeStepsBetweenSamples;
         }
         label lastRealTimeStep = NtimeStepsBetweenSamples + NtimeStepsBetweenSamples * timeSampleI;
         Info << "\nExporting solution in the time domain [" << timeSteps[firstRealTimeStep] << ", " << timeSteps[lastRealTimeStep] << "]\n" << endl;
         restart();
         Info << "Tbasis.size() = " << Tbasis.size() << endl;
         Info << "gWeights.size() = " << gWeights.size() << endl;
         Info << "Tad_time.size() = " << Tad_time.size() << endl;
         Info << "T0_time.size() = " << T0_time.size() << endl;
         Info << "Tbasis[0].size() = " << Tbasis[0].size() << endl;
         if(firstTimeStep == 0)
         {
             for(int timeI = firstTimeStep; timeI < NtimeStepsBetweenSamples; timeI++)
             {
                 volScalarField T(_T);
                 ITHACAutilities::assignIF(T, homogeneousBC);
                 label realTimeStep = timeI;
                 ITHACAstream::exportSolution(T0_time[timeI], std::to_string(timeSteps[realTimeStep]),
                                              outputFolder,
                                              "Treconstructed");
                 volScalarField gParametrizedField = list2Field(g[timeI]);
                 ITHACAutilities::assignIF(gParametrizedField, homogeneousBC);
                 ITHACAstream::exportSolution(gParametrizedField,
                                              std::to_string(timeSteps[realTimeStep]),
                                              outputFolder,
                                              "gReconstructed");
             }
         }

         for(int timeI = 0; timeI < NtimeStepsBetweenSamples + 1; timeI++)
         {
             volScalarField T(_T);
             ITHACAutilities::assignIF(T, homogeneousBC);
             forAll(Tbasis, baseI)
             {
                 T += gWeights[baseI] * (Tbasis[baseI][timeI] + Tad_time[timeI]);
             }
             T += - Tad_time[timeI] + T0_time[timeI];

             label realTimeStep = timeI + NtimeStepsBetweenSamples * timeSampleI;
             //if(timeSteps[realTimeStep] >= timeSteps[NtimeStepsBetweenSamples * timeSampleI])
             //{
                 ITHACAstream::exportSolution(T, std::to_string(timeSteps[realTimeStep]),
                                              outputFolder,
                                              "Treconstructed");
                 volScalarField gParametrizedField = list2Field(g[timeI]);
                 ITHACAstream::exportSolution(gParametrizedField,
                                              std::to_string(timeSteps[realTimeStep]),
                                              outputFolder,
                                              "gReconstructed");
             //}
             Ttime.append(T.clone());
         }
         Info << "reconstructT ENDED" << endl;

     }
}

void sequentialIHTP::parameterizedBC(word outputFolder, volScalarField initialField,
        word linSys_solver,
        label TSVD_filter)
{
    Info << endl << "Using quasilinearity of direct problem" << endl;
    //parameterizedBCoffline(folder, forceOffline);
    std::cout << "debug: addSol " << addSol <<  std::endl;
    std::cout << "debug: T0_vector " << T0_vector <<  std::endl;

    if(NsamplesWindow == 1)
    {
        timeSampleI = 0;
    }
    else if(NsamplesWindow == 2)
    {
        timeSampleI = 1;
    }
    else
    {
        Info << "sequentialIHTP::parameterizedBC not yet implemented for NsamplesWindow > 2" << endl;
        Info << "Exiting" << endl;
        exit(15);
    }
    while(timeSampleI < timeSamplesNum)
    {
        Info << "\nTime sample " << timeSampleI + 1 << endl;
        //TODO update initial field for T0
	solveT0(initialField);
        List<Eigen::MatrixXd> linSys;
        linSys.resize(2);

        if(NsamplesWindow == 1)
        {
	    TmeasShort = Tmeas.segment(thermocouplesNum * timeSampleI, thermocouplesNum * NsamplesWindow);
        }
        else if(NsamplesWindow == 2)
        {
	    TmeasShort = Tmeas.segment(thermocouplesNum * (timeSampleI - 1), thermocouplesNum * NsamplesWindow);
        }
        linSys[0] = Theta.transpose() * Theta;
        linSys[1] = Theta.transpose() * (TmeasShort + addSol - T0_vector);
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
        gWeightsOld = gWeights;
        gWeights.resize(weigths.size());
        forAll(gWeights, weightI)
        {
            gWeights[weightI] = weigths(weightI);
        }
        Info << "Weights = \n" << gWeights << endl;
        update_gParametrized(gWeights);
        label verbose = 0;
        parameterizedBC_postProcess(linSys, weigths, outputFolder, verbose);
	timeSampleI++;
    }
    ITHACAstream::exportMatrix(Jlist, "costFunction", "eigen", outputFolder);
    Info << "End" << endl;
    Info << endl;
}

void sequentialIHTP::set_valueFraction()
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

void sequentialIHTP::assignDirectBC(label timeI)
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

void sequentialIHTP::solveT0(volScalarField initialField)
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
    word outputFolder = "./ITHACAoutput/debug/";
    bool flagFirstWindow = 0; //with linear basis this does not save the solutions for the first sampling window
    if(NbasisInTime == 2)
    {
        flagFirstWindow = 1;
    }
    if(timeSampleI == 0 || (timeSampleI == 1 && NsamplesWindow == 2))
    {
        Info << "debug : First timestep. initialField = " << initialField.internalField()[0] << endl;
        ITHACAutilities::assignIF(T0_field, initialField);
    }
    else
    {
        reconstrucT(outputFolder);
        Info << "debug : other timesteps" << endl;
        Info << "debug : Ttime.size() = " << endl;
        Info << "debug : NtimeStepsBetweenSamples = " << NtimeStepsBetweenSamples << endl;
        Info << "debug : initialField = " << Ttime[NtimeStepsBetweenSamples].internalField()[0] << endl;

        ITHACAutilities::assignIF(T0_field, Ttime[NtimeStepsBetweenSamples]);
    }
    std::cout << "debug : T0 at thermocouples = " << fieldValueAtThermocouples(T0_field) << std::endl;

    T0_time.resize(0);
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
    if(!flagFirstWindow)
    {
        T0_time.append(T0_field.clone());
        ITHACAstream::exportSolution(T0_field, std::to_string(timeSteps[timeI]),
                                     folderOffline,
                                     "T0_field");
    }

    while (runTime.loop())
    {
        Info << "Time = " << runTime.timeName() << nl << endl;
        timeI++;
        if(samplingSteps[0] == timeI)
        {
            flagFirstWindow = 0;
        }

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

        if(flagFirstWindow == 0)
        {
            T0_time.append(T0_field.clone());
            ITHACAstream::exportSolution(T0_field, std::to_string(timeSteps[timeI]),
                                         folderOffline,
                                         "T0_field");
        }
        runTime.printExecutionTime(Info);
        runTime.write();
    }

    T0_vector = fieldValueAtThermocouples(T0_time);
    Info << "SolveT0 ENDED\n" << endl;
}

void sequentialIHTP::solveAdditional()
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

    int firstTimeStep = 0;
    if(NsamplesWindow == 2)
    {
        firstTimeStep = NtimeStepsBetweenSamples; 
    }

    if(timeI >= firstTimeStep)
    {
        Tad_time.append(Tad.clone());
        ITHACAstream::exportSolution(Tad, std::to_string(timeSteps[timeI]),
                                     folderOffline,
                                     "Tad");
    }

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

        if(timeI >= firstTimeStep)
        {
            Tad_time.append(Tad.clone());
            ITHACAstream::exportSolution(Tad, std::to_string(timeSteps[timeI]),
                                         folderOffline,
                                         "Tad");
        }
        runTime.printExecutionTime(Info);
        runTime.write();
    }

    Info << "Tad_time.size = " << Tad_time.size() << endl;
    Info << "Ntime = " << Ntimes << endl;
    addSol = fieldValueAtThermocouples(Tad_time);
    Info << "END \n" << endl;
}

void sequentialIHTP::solveDirect(label _offline)
{
    if(_offline)
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
    if(_offline)
    {
        ITHACAutilities::assignIF(T, homogeneousBC);
    }
    simpleControl& simple = _simple();
    Foam::Time& runTime = _runTime();
    fv::options& fvOptions(_fvOptions());
    label timeI = 0;
    Ttime.resize(0);
    int firstTimeStep = 0;
    if(NsamplesWindow == 2)
    {
        firstTimeStep = NtimeStepsBetweenSamples; 
    }

    if(timeI >= firstTimeStep || _offline == 0)
    {
        Ttime.append(T.clone());
    }

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

        if(timeI >= firstTimeStep || _offline == 0)
        {
            Ttime.append(T.clone());
            Info << "debug : I am saving this timestep" << endl;
        }
        std::cout << "debug : direct values at thermocouples = \n" << fieldValueAtThermocouples(T) << endl;

        runTime.printExecutionTime(Info);
        runTime.write();
    }
    Info << "Direct computation ENDED" << endl;
    
}

void sequentialIHTP::readThermocouples()
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

Eigen::VectorXd sequentialIHTP::fieldValueAtThermocouples(
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

Eigen::VectorXd sequentialIHTP::fieldValueAtThermocouples(
    PtrList<volScalarField> fieldList, label fieldI)
{
    Eigen::VectorXd fieldInt = fieldValueAtThermocouples(fieldList[fieldI]);
    return fieldInt;
}

Eigen::VectorXd sequentialIHTP::fieldValueAtThermocouples(
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
        Info << "The input fieldList of sequentialIHTP::fieldValueAtThermocouples can have size Ntimes + 1 (=" << Ntimes + 1 << ") or\n";
        Info << " NtimeStepsBetweenSamples + 1 (=" <<  NtimeStepsBetweenSamples + 1<< ") but has size " << fieldList.size() << endl;
        Info << "Exiting." << endl;
        exit(23);
    }
    Info << "\nSampling done \n" << endl;
    return fieldInt;
}


void sequentialIHTP::restart()
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

void sequentialIHTP::restartOffline()
{
    Info << "Setting endTime to offlineEndTime" << endl;
    restart();
    Time& runTime = _runTime();
    instantList Times = runTime.times();
    runTime.setTime(0.0, 0);
    runTime.setEndTime(offlineEndTime);
    Info << "Ready for new offline computation" << endl;
}

void sequentialIHTP::restartT0()
{
    Info << "Setting endTime to offlineEndTime" << endl;
    restart();
    Time& runTime = _runTime();
    instantList Times = runTime.times();
    runTime.setTime(0.0, 0);
    runTime.setEndTime(timeSamplesDeltaT);
    Info << "Ready for new T0 computation" << endl;
}

void sequentialIHTP::sampling2symulationTime()
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

void sequentialIHTP::parameterizedBC_postProcess(
    List<Eigen::MatrixXd> linSys, Eigen::VectorXd weigths, word outputFolder,
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
    Tcomp = fieldValueAtThermocouples(Ttime);
    std::cout << "Tcomp = \n" << Tcomp.transpose() << std::endl;
    std::cout << "TmeasShort = \n" << TmeasShort.transpose() << std::endl;
    J = 0.5 * Foam::sqrt((Tcomp - TmeasShort).dot(Tcomp - TmeasShort));
    Info << "J = " << J << endl;
    Jlist.conservativeResize(Jlist.size() + 1);
    Jlist(Jlist.size() - 1) = J;
}
