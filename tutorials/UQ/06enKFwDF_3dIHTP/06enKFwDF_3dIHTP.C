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
Description
    Example of a state and boundary condition reconstruction in 3D heat
    transfer problem using EnKF
SourceFiles
    06enKFwDF_3dIHTP.C
\*---------------------------------------------------------------------------*/

#include <iostream>
#include "fvCFD.H"
#include "fvOptions.H"
#include "simpleControl.H"
#include "IOmanip.H"
#include "Time.H"
#include "sequentialIHTP.H"
#include "ITHACAutilities.H"
#include <Eigen/Dense>
#define _USE_MATH_DEFINES
#include <cmath>
#include "Foam2Eigen.H"

#include "MUQ/Modeling/Distributions/Gaussian.h"

#include "muq2ithaca.H"
#include "Fang2017filter_wDF.H"

#include "06enKFwDF_3dIHTP.H"

using namespace SPLINTER;

class TutorialUQ5 : public ITHACAmuq::Fang2017filter_wDF
{
    public:
        explicit TutorialUQ5(int argc, char* argv[], int _Nsamples)
            :
            ITHACAmuq::Fang2017filter_wDF(_Nsamples),
            HTproblem(argc, argv)
        {
            setTime(HTproblem.startTime, HTproblem.deltaTime, HTproblem.endTime);
            setObservationSize(HTproblem.getObservationSize());
            setStateSize(HTproblem.getStateSize());
            setObservationTime(HTproblem.observationStartTimestep,
                               HTproblem.observationDeltaTimesteps);
            HTproblem.setProbe(1, Foam::vector(1.0, 0.02, 0.6));
        }
        inverseHeatTransfer_3D HTproblem;

        //--------------------------------------------------------------------------
        /// Project the state and adds the model error
        void stateProjection()
        {
            Info << "\nState projection start" << endl;
            for (int sampI = 0; sampI < getNumberOfSamples(); sampI++)
            {
                Info << "Sample " << sampI + 1 << ", time = " << getTime() << endl;;
                Eigen::VectorXd newState = HTproblem.projectState(
                        stateEns.getSample(sampI), parameterEns.getSample(sampI), 
                        getTime(), getTimeStep(), getTime() + HTproblem.deltaTime, 
                        modelErrorDensity);
                stateEns.assignSample(sampI, newState);
            }
            Info << "\nState projection end" << endl;
        };

        //--------------------------------------------------------------------------
        /// Observe the state ensamble
        void observeState()
        {
            Eigen::MatrixXd observedState(HTproblem.observe(stateEns.getSample(0)).size(),
                                          getNumberOfSamples());

            for (int sampI = 0; sampI < getNumberOfSamples(); sampI++)
            {
                observedState.col(sampI) = HTproblem.observe(stateEns.getSample(
                                               sampI)) + measNoiseDensity->Sample();
            }

            observationEns.assignSamples(observedState);
        };

        //--------------------------------------------------------------------------
        /// Set parameter prior projecting initial gTrue on the parameterized space
        Eigen::VectorXd setParameterPriorMean()
        {
            HTproblem.set_gTrue();
            Eigen::MatrixXd Temp(HTproblem.heatFluxSpaceBasis[0].size(), 
                    HTproblem.heatFluxSpaceBasis.size());
            forAll(HTproblem.heatFluxSpaceBasis, baseI)
            {
                Temp.col(baseI) = Foam2Eigen::List2EigenMatrix(
                        HTproblem.heatFluxSpaceBasis[baseI]);
            }

            Eigen::MatrixXd Btemp = Foam2Eigen::List2EigenMatrix(HTproblem.gTrue[0]);
            Eigen::MatrixXd B = Temp.transpose() * Btemp;
            Temp = Temp.transpose() * Temp;
            return Temp.fullPivLu().solve(B);
        };

        //--------------------------------------------------------------------------
        /// Post-processing
        void postProcessing(word outputFolder)
        {
            volScalarField T(HTproblem._T());
            PtrList<volScalarField> TtrueList;
            ITHACAstream::read_fields(TtrueList, "Tdirect",
                                      "./ITHACAoutput/true/");
            Eigen::VectorXd probe_rec(getTimeVector().size() - 1);
            Eigen::VectorXd probeState_maxConf(getTimeVector().size() - 1);
            Eigen::VectorXd probeState_minConf(getTimeVector().size() - 1);
            Eigen::MatrixXd gTrue_probe(1,getTimeVector().size() - 1);
            Eigen::MatrixXd gRec_probe(1,getTimeVector().size() - 1);
            Foam::vector hotSide_probeLocation(1.0, 0.0, 0.6);

            for (int timeI = 0; timeI < getTimeVector().size() - 1; timeI++)
            {
                Eigen::VectorXd mean = getStateMean().col(timeI);
                probe_rec(timeI) = HTproblem.fieldValueAtProbe(mean, 
                        HTproblem.probePosition);
                probeState_maxConf(timeI) = HTproblem.fieldValueAtProbe(state_maxConf.col(
                                                timeI), HTproblem.probePosition);
                probeState_minConf(timeI) = HTproblem.fieldValueAtProbe(state_minConf.col(
                                                timeI), HTproblem.probePosition);
                volScalarField meanField = Foam2Eigen::Eigen2field(T, mean);
                ITHACAstream::exportSolution(meanField,
                                             std::to_string(getTime(timeI)), outputFolder,
                                             "stateMean");
                ITHACAstream::exportSolution(TtrueList[timeI],
                                             std::to_string(getTime(timeI)), outputFolder,
                                             "trueState");
                volScalarField diff = TtrueList[timeI] - meanField;
                ITHACAstream::exportSolution(diff,
                                             std::to_string(getTime(timeI)), outputFolder,
                                             "error");
                volScalarField relativeErrorField(meanField);
                double EPS = 1e-16;

                for (label i = 0; i < relativeErrorField.internalField().size(); i++)
                {
                    if (std::abs(TtrueList[timeI].ref()[i]) < EPS)
                    {
                        relativeErrorField.ref()[i] = (std::abs(diff.ref()[i])) / EPS;
                    }
                    else
                    {
                        relativeErrorField.ref()[i] = (std::abs(diff.ref()[i])) /
                                                      TtrueList[timeI].ref()[i];
                    }
                }
                volScalarField gTrueField = HTproblem.list2Field(HTproblem.gTrue[timeI]);
                ITHACAstream::exportSolution(gTrueField,
                                             std::to_string(HTproblem.timeSteps[timeI]), 
                                             outputFolder, "gTrue");
                volScalarField gField = HTproblem.list2Field(HTproblem.updateHeatFlux(
                            getParameterMean().col(timeI)));
                ITHACAstream::exportSolution(gField,
                                             std::to_string(HTproblem.timeSteps[timeI]), 
                                             outputFolder, "gRec");
                gTrue_probe.col(timeI) = HTproblem.fieldValueAtProbe(gTrueField, 
                        hotSide_probeLocation);
                gRec_probe.col(timeI) = HTproblem.fieldValueAtProbe(gField, 
                        hotSide_probeLocation);


                ITHACAstream::exportSolution(relativeErrorField,
                                             std::to_string(getTime(timeI)), outputFolder,
                                             "relativeErrorField");
            }
            ITHACAstream::exportMatrix(probe_rec, "probe_rec", "eigen", outputFolder);
            ITHACAstream::exportMatrix(probeState_maxConf, "probeState_maxConf", 
                    "eigen", outputFolder);
            ITHACAstream::exportMatrix(probeState_minConf, "probeState_minConf", 
                    "eigen", outputFolder);
            ITHACAstream::exportMatrix(gTrue_probe, "gTrue_probe", 
                    "eigen", outputFolder);
            ITHACAstream::exportMatrix(gRec_probe, "gRec_probe", 
                    "eigen", outputFolder);
        }

};

int main(int argc, char* argv[])
{
    int Nsamples = 100;
    TutorialUQ5 example(argc, argv, Nsamples);
    // Reading parameters from file
    ITHACAparameters* para = ITHACAparameters::getInstance(
                                 example.HTproblem._mesh(),
                                 example.HTproblem._runTime());
    
    example.HTproblem.a = para->ITHACAdict->lookupOrDefault<scalar>("a", 0);
    example.HTproblem.b = para->ITHACAdict->lookupOrDefault<scalar>("b", 0);
    example.HTproblem.c = para->ITHACAdict->lookupOrDefault<scalar>("c", 0);
    example.HTproblem.maxFrequency = 
        para->ITHACAdict->lookupOrDefault<scalar>("maxFrequency", 0);

    example.HTproblem.HTC = 
        para->ITHACAdict->lookupOrDefault<scalar>("heatTranferCoeff", 0);
    example.HTproblem.thermalCond = 
        para->ITHACAdict->lookupOrDefault<scalar>("thermalConductivity", 0.0);
    example.HTproblem.density = 
        para->ITHACAdict->lookupOrDefault<scalar>("density", 0.0);
    example.HTproblem.specificHeat = 
        para->ITHACAdict->lookupOrDefault<scalar>("specificHeat", 0.0);
    example.HTproblem.initialField = 
        para->ITHACAdict->lookupOrDefault<scalar>("initialField", 0);

    scalar measNoiseCov = para->ITHACAdict->lookupOrDefault<scalar>(
            "measNoiseCov", 0);
    scalar modelErrorCov = para->ITHACAdict->lookupOrDefault<scalar>(
            "modelErrorCov", 0);
    scalar stateCov = para->ITHACAdict->lookupOrDefault<scalar>(
            "stateInitialCov", 0);
    scalar parameterCov = para->ITHACAdict->lookupOrDefault<scalar>(
            "parameterPriorCov", 0);

    label NheatFluxPODbasis = 
        para->ITHACAdict->lookupOrDefault<label>("NheatFluxPODbasis", 0);
    label innerLoops = para->ITHACAdict->lookupOrDefault<label>("EnKF_innerLoop", 1);

    word reconstructionFolder = "ITHACAoutput/reconstruction";

    scalar basisShapeParameter = 0.6;
    example.HTproblem.setSpaceBasis("rbf", basisShapeParameter, NheatFluxPODbasis);
    example.HTproblem.Nbasis = NheatFluxPODbasis;

    const int stateSize = example.getStateSize();
    example.setParameterSize(NheatFluxPODbasis);
    const int parameterSize = example.getParameterSize();
    Eigen::VectorXd stateInitialMean =
        Eigen::VectorXd::Ones(1) * example.HTproblem.initialField;
    Eigen::MatrixXd stateInitialCov = Eigen::MatrixXd::Identity(1,1) * stateCov;
    Eigen::VectorXd parameterPriorMean = example.setParameterPriorMean();
    Eigen::MatrixXd parameterPriorCov = Eigen::MatrixXd::Identity(parameterSize,
                                        parameterSize) * parameterCov;
    
    // Add noise to measurements
    Eigen::MatrixXd measurementsMat = example.HTproblem.solveDirect();
    example.setMeasNoise(measNoiseCov * measurementsMat.mean());
    ITHACAstream::exportMatrix(measurementsMat, "measurementsMat_noNoise", "eigen", 
            reconstructionFolder);
    for(int i = 0; i < measurementsMat.cols(); i++)
    {
        measurementsMat.col(i) = 
            measurementsMat.col(i) + example.measNoiseDensity->Sample(); 
    }
    ITHACAstream::exportMatrix(measurementsMat, "measurementsMat_noise", "eigen", 
            reconstructionFolder);
    example.setObservations(measurementsMat);


    bool univariateInitStateDensFlag = 1;
    example.setInitialStateDensity(stateInitialMean, stateInitialCov,
            univariateInitStateDensFlag);
    std::cout << "debug: parameterPriorMean = " << parameterPriorMean << std::endl;
    example.setParameterPriorDensity(parameterPriorMean, parameterPriorCov);
    bool univariateModelErrorDistribution = 1;
    example.setModelError(0.1, univariateModelErrorDistribution);
    example.setMeasNoise(0.1);
    example.run(innerLoops, reconstructionFolder);
    example.postProcessing(reconstructionFolder);
    return 0;
}
