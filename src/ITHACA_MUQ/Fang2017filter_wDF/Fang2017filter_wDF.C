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
/// Source file of the Fang2017filter_wDF class.

#include "Fang2017filter_wDF.H"

namespace ITHACAmuq
{
// * * * * * * * * * * * * * * * Constructors * * * * * * * * * * * * * * * * //
// Constructor
Fang2017filter_wDF::Fang2017filter_wDF() {}
Fang2017filter_wDF::Fang2017filter_wDF(int _Nsamples)
{
    Nsamples = _Nsamples;
}

// * * * * * * * * * * * * * * Filter Methods * * * * * * * * * * * * * * //

// Return number of samples per ensamble
int Fang2017filter_wDF::getNumberOfSamples()
{
    return Nsamples;
}

// Return time
double Fang2017filter_wDF::getTime()
{
    return timeVector(timeStepI);
}

//--------------------------------------------------------------------------
/// Return time for input timestep
double Fang2017filter_wDF::getTime(int _timeStepI)
{
    return timeVector(_timeStepI);
}

//--------------------------------------------------------------------------
/// Return timestep
int Fang2017filter_wDF::getTimeStep()
{
    return timeStepI;
}

//--------------------------------------------------------------------------
/// Return time vector
Eigen::VectorXd Fang2017filter_wDF::getTimeVector()
{
    return timeVector;
}

//--------------------------------------------------------------------------
/// Return state mean
Eigen::MatrixXd Fang2017filter_wDF::getStateMean()
{
    return stateMean;
}

//--------------------------------------------------------------------------
/// Return parameter mean
Eigen::MatrixXd Fang2017filter_wDF::getParameterMean()
{
    return parameterMean;
}

//--------------------------------------------------------------------------
/// Set the observations matrix
void Fang2017filter_wDF::setObservations(Eigen::MatrixXd _observations)
{
    observations = _observations;
}

//--------------------------------------------------------------------------
/// Setup the time vector 
void Fang2017filter_wDF::setTime(double _startTime, double _deltaTime, double _endTime)
{
    M_Assert(_endTime > _startTime, "endTime must be bigger than startTime");
    startTime = _startTime;
    deltaTime = _deltaTime;
    endTime = _endTime;
    Ntimes = (endTime - startTime) / deltaTime;
    Info << "startTime = " << startTime << endl;
    Info << "endTime = " << endTime << endl;
    Info << "deltaTime = " << deltaTime << endl;
    Info << "Ntimes = " << Ntimes << endl;
    timeVector = Eigen::VectorXd::LinSpaced(Ntimes + 1, startTime, endTime);
}

//--------------------------------------------------------------------------
/// Setup the observation vector 
void Fang2017filter_wDF::setObservationTime(int _observationStart, int _observationDelta)
{
    M_Assert(timeVector.size() > 0, "Setup the timeVector before setting up the observations vector");
    M_Assert(_observationStart > 0, "First observation timestep can't be 0");
    observationStart = _observationStart;
    observationDelta = _observationDelta;
    Info << "First observation at time = " << timeVector(observationStart) << " s" << endl;
    Info << "Observations taken every " << observationDelta << " timesteps" << endl;
    observationBoolVec = Eigen::VectorXi::Zero(timeVector.size() - 1);
    for(int i = observationStart - 1; i < Ntimes; i += observationDelta)
    {
        observationBoolVec(i) = 1;
    }
}

//--------------------------------------------------------------------------
/// Setup of the model error distribution
void Fang2017filter_wDF::setModelError(double cov, bool univariate)
{
    M_Assert(stateSize > 0, "Set the stateSize before setting up the model error");

    Eigen::VectorXd modelError_mu;
    Eigen::MatrixXd modelError_cov;
    if(univariate)
    {
        modelError_mu = Eigen::VectorXd::Zero(1);
        modelError_cov = Eigen::MatrixXd::Identity(1,
                                         1) * cov;
    }
    else
    {
        modelError_mu = Eigen::VectorXd::Zero(stateSize);
        modelError_cov = Eigen::MatrixXd::Identity(stateSize,
                                         stateSize) * cov;
    }
    modelErrorDensity = std::make_shared<muq::Modeling::Gaussian>(modelError_mu,
                        modelError_cov);
    modelErrorFlag = 1;
}

//--------------------------------------------------------------------------
/// Setup of the measurement noise distribution
void Fang2017filter_wDF::setMeasNoise(double cov)
{
    M_Assert(observationSize > 0, "Read measurements before setting up the measurement noise");
    Eigen::VectorXd measNoise_mu = Eigen::VectorXd::Zero(observationSize);
    Eigen::MatrixXd measNoise_cov = Eigen::MatrixXd::Identity(observationSize,
                                    observationSize) * cov;
    measNoiseDensity = std::make_shared<muq::Modeling::Gaussian>(measNoise_mu,
                       measNoise_cov);
    measurementNoiseFlag = 1;
}

//--------------------------------------------------------------------------
/// Create initial state ensemble
void Fang2017filter_wDF::setInitialStateDensity(Eigen::VectorXd _mean,
        Eigen::MatrixXd _cov, bool _univariateFlag)
{
    M_Assert(!(stateSize == 0 && _univariateFlag),
            "If stateSize is not set you cannot use univariate InitialStateDensity");
    if(_univariateFlag == 0)
    {
        if(stateSize == 0)
        {
            stateSize = _mean.size();
        }
        else
        {
            std::string message = "State has size = " + std::to_string(stateSize)
                + " while input mean vector has size = "
                + std::to_string(_mean.size());

            M_Assert(stateSize == _mean.size(), message.c_str());
        }
        M_Assert(_cov.rows() == stateSize && _cov.cols() == stateSize,
                "To initialize the state ensemble use mean and cov with same dimentions");
    }
    else
    {
        univariateInitStateDensFlag = 1;
        M_Assert(_mean.size() == 1, "If univariateInitStateDensFlag is 1 mean size is 1");
        M_Assert(_cov.size() == 1, "If univariateInitStateDensFlag is 1 cov size is 1");
    }
    initialStateDensity = std::make_shared<muq::Modeling::Gaussian>(_mean, _cov);

    initialStateFlag = 1;
}

//--------------------------------------------------------------------------
/// Create initial state ensemble
void Fang2017filter_wDF::sampleInitialState()
{
    Info << "Setting initial state ensamble" << endl;
    M_Assert(initialStateFlag == 1, "Initialize the initial state density before sampling it");
    if(univariateInitStateDensFlag)
    {
        M_Assert(stateSize > 0, "Set stateSize before sampleInitialState");
        M_Assert(Nsamples > 0, "Set Nsamples before sampleInitialState");
        Eigen::MatrixXd _stateEns(stateSize, Nsamples);
        for (int i = 0; i < Nsamples; i++)
        {
            for (int j = 0; j < stateSize; j++)
            {
                _stateEns(j,i) = initialStateDensity->Sample()(0,0);
            }
        }
        stateEns.assignSamples(_stateEns);
    }
    else
    {
        stateEns.assignSamples(ensembleFromDensity(initialStateDensity));
    }
    Info << "debug: State ensamble size = " << stateEns.getSize() << endl;
}

//--------------------------------------------------------------------------
/// Create parameter ensemble 
void Fang2017filter_wDF::setParameterPriorDensity(Eigen::VectorXd _mean, Eigen::MatrixXd _cov)
{
    if(parameterSize == 0)
    {
        parameterSize = _mean.size();
    }
    else
    {
        std::string message = "The input mean has size = " + std::to_string(_mean.size())
            + " but the parameterSize is "
            + std::to_string(parameterSize);

        M_Assert(parameterSize == _mean.size(), message.c_str());
    }
    M_Assert(_cov.rows() == parameterSize && _cov.cols() == parameterSize, "Use mean and cov with same dimentions");
    parameterPriorMean = _mean;
    parameterPriorCov = _cov;
    parameterPriorDensity = std::make_shared<muq::Modeling::Gaussian>(_mean, _cov);
    parameterPriorFlag = 1;
}

//--------------------------------------------------------------------------
/// Create parameter ensemble 
void Fang2017filter_wDF::sampleParameterDist()
{
    M_Assert(parameterPriorFlag == 1, "Set up the parameter prior density");
    parameterEns.assignSamples(ensembleFromDensity(parameterPriorDensity));
    parameterMean.col(timeStepI) = parameterEns.mean(); 
}

//--------------------------------------------------------------------------
/// General class to sample from an input density 
Eigen::MatrixXd Fang2017filter_wDF::ensembleFromDensity(std::shared_ptr<muq::Modeling::Gaussian> _density)
{
    M_Assert(Nsamples > 0, "Number of samples not set up correctly");
    Eigen::MatrixXd output(_density->Sample().size(), Nsamples);
    for (int i = 0; i < Nsamples; i++)
    {
        output.col(i) = _density->Sample();
    }

    return output;
}


//--------------------------------------------------------------------------
/// Concatenate state and parameter ensambles to create the joint ensamble 
void Fang2017filter_wDF::buildJointEns()
{
    Eigen::MatrixXd stateSamps = stateEns.getSamples();
    Eigen::MatrixXd paramSamps = parameterEns.getSamples();
    M_Assert(stateSamps.cols() == paramSamps.cols(), "State and parameter ensambles must have same number of samples");
    Eigen::MatrixXd temp(stateSamps.rows()+paramSamps.rows(), stateSamps.cols());
    temp << stateSamps, 
            paramSamps;
    
    jointEns.assignSamples(temp);
}

//--------------------------------------------------------------------------
/// 
void Fang2017filter_wDF::setObservationSize(int _size)
{
    observationSize = _size;
    Info << "Observation size = " << observationSize << endl;
}

//--------------------------------------------------------------------------
/// 
void Fang2017filter_wDF::setStateSize(int _size)
{
    stateSize = _size;
    Info << "State size = " << stateSize << endl;
}

//--------------------------------------------------------------------------
/// 
int Fang2017filter_wDF::getStateSize()
{
    return stateSize;
}

//--------------------------------------------------------------------------
/// 
void Fang2017filter_wDF::setParameterSize(int _size)
{
    parameterSize = _size;
    Info << "Parameter size = " << parameterSize << endl;
}

//--------------------------------------------------------------------------
/// 
int Fang2017filter_wDF::getParameterSize()
{
    return parameterSize;
}

//--------------------------------------------------------------------------
///
void Fang2017filter_wDF::updateJointEns(Eigen::VectorXd _observation)
{
    M_Assert(_observation.size() == observationSize, "Observation has wrong dimentions");
    //TODO deal with invertibility of observationEns.cov()
    int ensSize = jointEns.getSize();
    Eigen::MatrixXd autoCovInverse = observationEns.cov().inverse();
    Eigen::MatrixXd crossCov = jointEns.crossCov(observationEns.getSamples());

    for(int i = 0; i < ensSize; i++)
    {
        Eigen::VectorXd newSamp = jointEns.getSample(i) + crossCov * autoCovInverse *
            (_observation - observationEns.getSample(i));
        jointEns.assignSample(i, newSamp);
    }

}


//--------------------------------------------------------------------------
/// Run the filtering
void Fang2017filter_wDF::run(int innerLoopMax, word outputFolder)
{
    M_Assert(initialStateFlag == 1, "Set up the initial state density");
    M_Assert(parameterPriorFlag == 1, "Set up the parameter prior density");
    M_Assert(modelErrorFlag == 1, "Set up the model error");
    M_Assert(measurementNoiseFlag == 1, "Set up the measurement noise");
    M_Assert(stateSize > 0, "Set state size");
    M_Assert(observationSize > 0, "Set observation size");
    M_Assert(parameterSize > 0, "Set parameter size");

    timeStepI = 0;
    sampleInitialState();
    stateMean.resize(stateSize, Ntimes);
    state_maxConf.resize(stateSize, Ntimes);
    state_minConf.resize(stateSize, Ntimes);
    parameter_maxConf.resize(parameterSize, Ntimes);
    parameter_minConf.resize(parameterSize, Ntimes);
    parameterMean.resize(parameterSize, Ntimes);
    while(timeStepI < Ntimes)
    {
        Info << "timeStep " << timeStepI << endl;
        if(timeStepI == 0)
        {
            setParameterPriorDensity(parameterPriorMean, parameterPriorCov);
            std::cout << "debugInside: parameterPriorMean = " << parameterPriorMean << std::endl;
            sampleParameterDist();
        }
        else
        {
            setParameterPriorDensity(
                    parameterMean.col(timeStepI - 1), parameterPriorCov);
            sampleParameterDist();
        }
        stateProjection();
        if(observationBoolVec(timeStepI) == 1) //There is a measurement for this timestep
        {
            innerLoopI = 0;
            while(innerLoopI < innerLoopMax)
            {
                Info << "Inner loop " << innerLoopI << endl;
                std::cout << "debug: param mean =\n" << parameterEns.mean() << std::endl;
                if(innerLoopI > 0)
                {
                    setParameterPriorDensity(
                            parameterMean.col(timeStepI), parameterPriorCov);
                    sampleParameterDist();
                    std::cout << "\ndebug : parameterMean after loop =\n" << 
                        parameterMean.col(timeStepI) << std::endl;
                }
                buildJointEns();
                observeState();
                updateJointEns(
                        observations.col(
                            observationBoolVec.head(timeStepI + 1).sum() - 1));
                parameterMean.col(timeStepI) = jointEns.mean().tail(parameterSize);
                innerLoopI++;
            }
            // ################### Kabir: Exporting Posterior destribution of each basis function weights at each time step
            std::string WeightPosterior="heatFlux_weightsPosterior";
            std::string saveFileName= WeightPosterior; // + std::to_string(sampleI);

            std::string folderName = "HFWposterior" + std::to_string(timeStepI);
            std::string folderPath = "ITHACAoutput/projection/" + folderName;

            //Eigen::Matrix<double, -1, -1> matrixKabirPost = jointEns.tail(parameterSize);   // error: ‘class ITHACAmuq::ensemble’ has no member named ‘tail’
            Eigen::Matrix<double, -1, -1> matrixKabirPost = parameterEns.getSamples();

             ITHACAstream::exportMatrix(matrixKabirPost, saveFileName, "eigen", folderPath);
            // ################### Kabir: Exporting Posterior destribution of each basis function weights at each time step
        }
        else //No measurement available
        {
            buildJointEns();
        }
        // ################### Kabir: Exporting Posterior destribution of each basis function weights at each time step
        //std::string WeightPosterior="heatFlux_weightsPosterior";
        //std::string saveFileName= WeightPosterior; // + std::to_string(sampleI);

        //std::string folderName = "HFWposterior" + std::to_string(timeStepI);
        //std::string folderPath = "ITHACAoutput/projection/" + folderName;

        //Eigen::Matrix<double, -1, -1> matrixKabirPost = jointEns.tail(parameterSize);   // error: ‘class ITHACAmuq::ensemble’ has no member named ‘tail’
        //Eigen::Matrix<double, -1, -1> matrixKabirPost = parameterEns.getSamples();

        //ITHACAstream::exportMatrix(matrixKabirPost, saveFileName, "eigen", folderPath);
        // ################### Kabir: Exporting Posterior destribution of each basis function weights at each time step

        parameterMean.col(timeStepI) = jointEns.mean().tail(parameterSize);
        stateMean.col(timeStepI) = jointEns.mean().head(stateSize);
        state_maxConf.col(timeStepI) = muq2ithaca::quantile(stateEns.getSamples(),
                                         0.95);
        state_minConf.col(timeStepI) = muq2ithaca::quantile(stateEns.getSamples(),
                                         0.05);
        parameter_maxConf.col(timeStepI) = muq2ithaca::quantile(parameterEns.getSamples(),
                                         0.95);
        parameter_minConf.col(timeStepI) = muq2ithaca::quantile(parameterEns.getSamples(),
                                         0.05);
        timeStepI++;
    }
    ITHACAstream::exportMatrix(stateMean, "stateMean", "eigen", outputFolder);
    ITHACAstream::exportMatrix(parameterMean, "parameterMean", "eigen", outputFolder);
    ITHACAstream::exportMatrix(parameter_maxConf, "parameter_maxConf", "eigen", 
            outputFolder);
    ITHACAstream::exportMatrix(parameter_minConf, "parameter_minConf", "eigen", 
            outputFolder);
}
}
