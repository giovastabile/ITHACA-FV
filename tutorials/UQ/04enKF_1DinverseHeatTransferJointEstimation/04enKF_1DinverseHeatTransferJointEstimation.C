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
    Example of a state and boundary condition reconstruction in 1D heat
    transfer problem using EnKF
SourceFiles
    03enKF_1DinverseHeatTransferJointEstimation.C
\*---------------------------------------------------------------------------*/

#include <iostream>
#include "fvCFD.H"
#include "fvOptions.H"
#include "simpleControl.H"
#include "IOmanip.H"
#include "Time.H"
#include "laplacianProblem.H"
#include "ITHACAutilities.H"
#include <Eigen/Dense>
#define _USE_MATH_DEFINES
#include <cmath>
#include "Foam2Eigen.H"

#include "MUQ/Modeling/Distributions/Gaussian.h"

#include "muq2ithaca.H"
#include "Pagani2016filter.H"

#include "04enKF_1DinverseHeatTransferJointEstimation.H"

using namespace SPLINTER;

class TutorialUQ4 : public ITHACAmuq::Pagani2016filter
{
    public:
        explicit TutorialUQ4(int argc, char* argv[], int _Nsamples)
            :
            ITHACAmuq::Pagani2016filter(_Nsamples),
            HTproblem(argc, argv)
        {
            setTime(HTproblem.startTime, HTproblem.deltaTime, HTproblem.endTime);
            setObservationSize(HTproblem.getObservationSize());
            setStateSize(HTproblem.getStateSize());
            setParameterSize(HTproblem.getBoundarySize());
            setObservationTime(HTproblem.observationStartTimestep,
                               HTproblem.observationDeltaTimesteps);
            HTproblem.k = 3.0;
            HTproblem.rho = 5.0;
            HTproblem.Cp = 2.0;
            HTproblem.setProbe(1, Foam::vector(0.5, 0.1, 0.005));
        }
        inverseHeatTransfer_1D HTproblem;
        int Nobservations = 0;

        //--------------------------------------------------------------------------
        /// Project the state and adds the model error
        void stateProjection()
        {
            Nobservations++;
            std::cout << "debug : stateMean before proj = \n" << 
                stateEns.mean().transpose() << std::endl;

            int endTimestepProj = getObservationTimestep(getObservationCounter());
            double endTimeProj = getTime(endTimestepProj); 
            double startTime = 0;
            int startTimestep = 0;
            if(getObservationCounter() > 0)
            {
                startTimestep = getObservationTimestep(getObservationCounter() - 1);
                startTime = getTime(startTimestep);
            }
            List<Eigen::MatrixXd> newStateList(getNumberOfSamples());
            for (int sampI = 0; sampI < getNumberOfSamples(); sampI++)
            {
                Info << "************************************************" << endl;
                Info << "\nProjecting sample " << sampI +1 << endl;

                Eigen::VectorXd BCsample = parameterEns.getSample(sampI) + 
                    parameterErrorDensity->Sample();
                newStateList[sampI] = HTproblem.projectState(
                        oldStateEns.getSample(sampI), BCsample,
                        startTime, startTimestep, endTimeProj);
            }
            for(int tI = 0; tI < newStateList[0].cols(); tI++)
            {
                for (int sampI = 0; sampI < getNumberOfSamples(); sampI++)
                {
                    stateEns.assignSample(sampI, newStateList[sampI].col(tI));
                }
                int realTimestep = startTimestep + tI;
                computeStateMean(realTimestep);

                computeParameterMean(realTimestep);

                state_maxConf.col(realTimestep) = ITHACAmuq::muq2ithaca::quantile(
                        stateEns.getSamples(), 0.95);
                state_minConf.col(realTimestep) = ITHACAmuq::muq2ithaca::quantile(
                        stateEns.getSamples(), 0.05);
                parameter_maxConf.col(realTimestep) = ITHACAmuq::muq2ithaca::quantile(
                        parameterEns.getSamples(), 0.95);
                parameter_minConf.col(realTimestep) = ITHACAmuq::muq2ithaca::quantile(
                        parameterEns.getSamples(), 0.05);

            }
            observeState();

            Info << "End time = " << getTime() << endl;
        };

        //--------------------------------------------------------------------------
        /// Observe the state ensamble
        void observeState()
        {
            Eigen::MatrixXd observedState(getObservationSize(),
                                          getNumberOfSamples());

            Info << "stateEns.getSamples().cols() = " << stateEns.getSamples().cols() 
                << endl;
            for (int sampI = 0; sampI < getNumberOfSamples(); sampI++)
            {
                observedState.col(sampI) = HTproblem.observe(stateEns.getSample(
                                               sampI));
            }

            observationEns.assignSamples(observedState);
        };

        //--------------------------------------------------------------------------
        /// Post-processing
        void postProcessing(word outputFolder)
        {
            Info << "Postprocessing the results" << endl;
            volScalarField T(HTproblem._T());
            PtrList<volScalarField> TtrueList;
            ITHACAstream::read_fields(TtrueList, "Tdirect",
                                      "./ITHACAoutput/direct/");

            int Ntimesteps = state_maxConf.cols();
            Eigen::VectorXd probe_rec(Ntimesteps);
            Eigen::VectorXd probeState_maxConf(Ntimesteps);
            Eigen::VectorXd probeState_minConf(Ntimesteps);

            Info << "debug : TtrueList.size() = " <<  TtrueList.size() << endl;

            for(int timeI = 0; timeI < Ntimesteps; timeI++)
            {
                Info << "timeI = " << timeI << endl;
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

                ITHACAstream::exportSolution(relativeErrorField,
                                             std::to_string(getTime(timeI)), outputFolder,
                                             "relativeErrorField");
                ITHACAstream::exportMatrix(probe_rec, "probe_rec", "eigen", outputFolder);
                ITHACAstream::exportMatrix(probeState_maxConf, "probeState_maxConf", 
                        "eigen", outputFolder);
                ITHACAstream::exportMatrix(probeState_minConf, "probeState_minConf", 
                        "eigen", outputFolder);
            }
            Nobservations = 0;
        }
};

int main(int argc, char* argv[])
{
    int Nsamples = 100;
    TutorialUQ4 example(argc, argv, Nsamples);
    // Reading parameters from file
    ITHACAparameters* para = ITHACAparameters::getInstance(
                                 example.HTproblem._mesh(),
                                 example.HTproblem._runTime());
    const int stateSize = example.getStateSize();
    const int parameterSize = example.getParameterSize();
    Eigen::VectorXd stateInitialMean = Eigen::VectorXd::Zero(stateSize);
    Eigen::MatrixXd stateInitialCov = Eigen::MatrixXd::Identity(stateSize,
                                      stateSize) * 5;
    Eigen::VectorXd parameterPriorMean = Eigen::VectorXd::Zero(parameterSize);
    Eigen::MatrixXd parameterPriorCov = Eigen::MatrixXd::Identity(parameterSize,
                                        parameterSize) * 10;
    Eigen::MatrixXd trueMeasurements = example.HTproblem.solveDirect();
    //TODO add noise
    example.setInitialStateDensity(stateInitialMean, stateInitialCov);
    example.setParameterPriorDensity(parameterPriorMean, parameterPriorCov);
    example.setMeasNoise(0.01);
    example.setParameterError(1);
    example.setTrueObservations(trueMeasurements);
    word reconstructionFolder = "ITHACAoutput/reconstruction";
    example.run(reconstructionFolder);
    example.postProcessing(reconstructionFolder);
    return 0;
}
