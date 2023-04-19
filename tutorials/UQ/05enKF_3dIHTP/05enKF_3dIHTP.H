#ifndef ENKF_3DIHTP_H
#define ENKF_3DIHTP_H
#include "simpleControl.H"


/// \brief Class where the 3D inverse heat transfer problem is implemented
/// \details It is a child of the laplacianProblem class and some of its
/// functions are overridden to be adapted to the specific case.
class inverseHeatTransfer_3D: public sequentialIHTP
{
        bool measurementPointsRead = 0;
        List<vector> measurementPoints;
    public:
        explicit inverseHeatTransfer_3D(int argc, char* argv[])
            :
            sequentialIHTP(argc, argv)
        {
            fvMesh& mesh = _mesh();
            _simple = autoPtr<simpleControl>
                      (
                          new simpleControl
                          (
                              mesh
                          )
                      );
            simpleControl& simple = _simple();
            Time& runTime = _runTime();
#include "createFvOptions.H"
            hotSide_ind = mesh.boundaryMesh().findPatchID("hotSide");
            coldSide_ind = mesh.boundaryMesh().findPatchID("coldSide");
            volScalarField& T(_T());
            stateSize = T.size();
            readObservationTimes();
        }


        word directOutputFolder = "./ITHACAoutput/directProblem/";

        autoPtr<simpleControl> _simple;

        autoPtr<fv::options> _fvOptions;

        /// Diffusivity
        double k;

        /// Density
        double rho;

        /// Heat capacity
        double Cp;

        /// initialField
        scalar initialField = 0.0;

        Eigen::MatrixXd measurements;

        int stateSize;

        label timeStepID;

        /// Number of probes
        int Nprobes;

        /// Probles position, probes are used only for visualization and postprocessing
        /// The observation points are defined in the measurementDict
        Foam::vector probePosition;

        Eigen::MatrixXd probe_true;

        Eigen::MatrixXd probe_mean;

        Eigen::MatrixXd probe_MaxConfidence;

        Eigen::MatrixXd probe_minConfidence;

        Eigen::VectorXd trueBC;

        /// True heat flux at hotSide [W/m2]
        List<List<scalar>> gTrue;
        List<scalar> heatFlux_steady;

        /// List of true temperature field in time
        PtrList<volScalarField> Ttrue;

        label samplingTimeI = 0;
        label sampleI = 0;
        label sampleFlag = 0;
        label observationDeltaTimesteps = 0;
        label observationStartTimestep = 0;

        /// Heat flux parameters
        scalar a = 0;
        scalar b = 0;
        scalar c = 0;
        scalar maxFrequency; // Hz

        // Functions

        //--------------------------------------------------------------------------
        /// Setup probes for post processing
        //TODO only works for 1 probe
        void setProbe(int _Nprobes, Foam::vector _probePosition)
        {
            Nprobes = _Nprobes;
            probePosition = _probePosition;
            probe_mean = Eigen::MatrixXd::Zero(Nprobes, Ntimes);
            probe_true = probe_mean;
            probe_MaxConfidence = probe_mean;
            probe_minConfidence = probe_mean;
        }

        //--------------------------------------------------------------------------
        /// Set steady component of heat flux, g
        void set_steadyHeatFlux()
        {
            fvMesh& mesh(_mesh());
            volScalarField& T(_T());
            hotSide_ind = mesh.boundaryMesh().findPatchID("hotSide");
            label hotSideSize = T.boundaryField()[hotSide_ind].size();
            heatFlux_steady.resize(hotSideSize);
            forAll(heatFlux_steady, faceI)
            {
                scalar faceX =
                    mesh.boundaryMesh()[hotSide_ind].faceCentres()[faceI].x();
                scalar faceZ =
                    mesh.boundaryMesh()[hotSide_ind].faceCentres()[faceI].z();
                heatFlux_steady[faceI] = - k * (a * faceX * faceX + b * faceZ + c);
            }
        }

        //--------------------------------------------------------------------------
        /// Set the true boundary heat flux
        void set_gTrue()
        {
            fvMesh& mesh(_mesh());
            volScalarField& T(_T());
            label hotSideSize = T.boundaryField()[hotSide_ind].size();
            gTrue.resize(timeSteps.size());
            set_steadyHeatFlux();
            forAll(timeSteps, timeI)
            {
                // Setting gTrue
                gTrue[timeI].resize(hotSideSize);
                forAll(gTrue[timeI], faceI)
                {
                    scalar faceX =
                        mesh.boundaryMesh()[hotSide_ind].faceCentres()[faceI].x();
                    scalar faceZ =
                        mesh.boundaryMesh()[hotSide_ind].faceCentres()[faceI].z();


                    scalar g1 = - k * (b * faceZ * faceZ  + c);
                    scalar g2 = - k * c / (1 + (faceX - 1) * (faceX - 1) + faceZ * faceZ);
                    gTrue[timeI][faceI] = g1 + 
                        g1 / 2 * (std::sin ( 2 * M_PI * maxFrequency * 
                                    timeSteps[timeI] * timeSteps[timeI] / endTime)) +
                        g2 * std::exp(- 0.1 * timeSteps[timeI]);
                }
            }
        }


        //--------------------------------------------------------------------------
        /// Set the robin boundary condition values on the coldSide patch
        void set_Tf()
        {
            fvMesh& mesh = _mesh();
            volScalarField& T(_T());
            // Setting BC at the cold side
            label coldSideSize = T.boundaryField()[coldSide_ind].size();
            Tf.resize(coldSideSize);
            forAll(Tf, faceI)
            {
                Tf[faceI] = 300;
            }
            set_valueFraction();
        }

        //--------------------------------------------------------------------------
        /// Assign BC to the direct problem
        void assignTrueIF()
        {
            volScalarField& T(_T());
            M_Assert(initialField > 0, "Initial field is 0");
            ITHACAutilities::assignIF(T, initialField);
        }

        //--------------------------------------------------------------------------
        /// Dumb member to have the cluss not abstract
        void assignT0_IF(volScalarField& T0_init){};

        //--------------------------------------------------------------------------
        /// Assign BC to the direct problem
        void assignDirectBC(label timeI)
        {
            fvMesh& mesh = _mesh();
            volScalarField& T(_T());
            set_Tf();
            forAll(mesh.boundaryMesh(), patchI)
            {
                if (patchI == coldSide_ind)
                {
                    ITHACAutilities::assignMixedBC(T, patchI, Tf, refGrad, valueFraction);
                }
                else if (patchI == hotSide_ind)
                {
                    ITHACAutilities::assignBC(T, patchI, - g[timeI] / k);
                }
                else
                {
                    ITHACAutilities::assignBC(T, patchI, homogeneousBC);
                }
            }
        }


        //--------------------------------------------------------------------------
        /// Updates the boundary conditions according to runTime.
        double updateBC()
        {
            Time& runTime = _runTime();
            scalar time = runTime.value();
            double BC = Foam::sin(time) * 1 + 2;
            trueBC.conservativeResize(trueBC.size() + 1);
            trueBC(trueBC.size() - 1) = BC;
            return BC;
        }

        //--------------------------------------------------------------------------
        /// Set all the sampling counters to 0
        void resetSamplingCounters()
        {
            samplingTimeI = 0;
            sampleI = 0;
            sampleFlag = 0;
        }

        //--------------------------------------------------------------------------
        ///
        int getObservationSize()
        {
            fvMesh& mesh = _mesh();
            Time& runTime = _runTime();
            IOdictionary measurementsDict
            (
                IOobject
                (
                    "measurementsDict",
                    runTime.constant(),
                    mesh,
                    IOobject::MUST_READ,
                    IOobject::NO_WRITE
                )
            );
            List<vector> measurementPoints(measurementsDict.lookup("positions"));
            return measurementPoints.size();
        }

        //--------------------------------------------------------------------------
        /// Read observation delta time form dictionary
        void readObservationTimes()
        {
            fvMesh& mesh = _mesh();
            Time& runTime = _runTime();
            IOdictionary measurementsDict
            (
                IOobject
                (
                    "measurementsDict",
                    runTime.constant(),
                    mesh,
                    IOobject::MUST_READ,
                    IOobject::NO_WRITE
                )
            );
            int deltaTime(readLabel(measurementsDict.lookup("observationDeltaTimesteps")));
            int startTimestep(readLabel(
                                  measurementsDict.lookup("observationStartTimestep")));
            observationDeltaTimesteps = deltaTime;
            observationStartTimestep = startTimestep;
        }

        //--------------------------------------------------------------------------
        /// Return the state size
        int getStateSize()
        {
            volScalarField T(_T());
            return T.internalField().size();
        }

        //--------------------------------------------------------------------------
        /// Return the boundary condition size
        int getBoundarySize()
        {
            fvMesh& mesh = _mesh();
            volScalarField T(_T());
            int BCsize = 
                T.boundaryField()[hotSide_ind].size();
            return BCsize;
        }

        //--------------------------------------------------------------------------
        /// Returns the input field values at the measurement points defined in the 
        /// measurementsDict
        Eigen::VectorXd observe(volScalarField field)
        {
            fvMesh& mesh = _mesh();
            Time& runTime = _runTime();

            if (!measurementPointsRead)
            {
                IOdictionary measurementsDict
                (
                    IOobject
                    (
                        "measurementsDict",
                        runTime.constant(),
                        mesh,
                        IOobject::MUST_READ,
                        IOobject::NO_WRITE
                    )
                );
                List<vector> temp(measurementsDict.lookup("positions"));
                measurementPoints = temp;
            }

            Eigen::VectorXd measures(measurementPoints.size());
            forAll(measurementPoints, pntI)
            {
                //TODO
                //add check if I put two measurements on the same cell
                measures(pntI) = field[mesh.findCell(measurementPoints[pntI])];
            }
            return measures;
        }

        //--------------------------------------------------------------------------
        /// Returns the input field values at the measurement points defined in the 
        /// measurementsDict
        Eigen::VectorXd observe(Eigen::VectorXd field)
        {
            fvMesh& mesh = _mesh();
            Time& runTime = _runTime();
            IOdictionary measurementsDict
            (
                IOobject
                (
                    "measurementsDict",
                    runTime.constant(),
                    mesh,
                    IOobject::MUST_READ,
                    IOobject::NO_WRITE
                )
            );
            List<vector> measurementPoints(measurementsDict.lookup("positions"));
            Eigen::VectorXd measures(measurementPoints.size());
            forAll(measurementPoints, pntI)
            {
                //TODO
                //add check if I put two measurements on the same cell
                measures(pntI) = field(mesh.findCell(measurementPoints[pntI]));
            }
            return measures;
        }

        //--------------------------------------------------------------------------
        /// Project the state ahead of 1 timestep
        Eigen::VectorXd projectState(Eigen::VectorXd previousState,
                Eigen::VectorXd _heatFluxWeights, double startTime_, 
                int startIndex_, double endTime_,
                std::shared_ptr<muq::Modeling::Gaussian> _modelErrorDensity)
        {
            word outputFolder = "ITHACAoutput/projection";
            word fieldName = "T";
            volScalarField T(_T());
            assignIF(T, previousState);
            resetRunTime(startTime_, startIndex_, endTime_);
            timeStepID = startIndex_;
            solve(T, _heatFluxWeights, outputFolder, fieldName);
            Eigen::VectorXd stateOut = Foam2Eigen::field2Eigen(T);
            for(int i = 0; i < stateOut.size(); i++)
            {
                stateOut(i) += _modelErrorDensity->Sample()(0,0);
            }
            return stateOut;
        }

        //--------------------------------------------------------------------------
        /// Update the heat flux, g
        List<scalar> updateHeatFlux(Eigen::VectorXd _weights) 
        {
            M_Assert(_weights.size() == Nbasis, 
                    "weigths size different from basis functions size");
            volScalarField& T = _T();
            List<scalar> out(T.boundaryField()[hotSide_ind].size());
            forAll (T.boundaryField()[hotSide_ind], faceI)
            {
                out[faceI] = 0.0;
                for(int i = 0; i < _weights.size(); i++)
                {
                    out[faceI] += _weights(i) * heatFluxSpaceBasis[i][faceI];
                }
            }
            return out;
        }

        //--------------------------------------------------------------------------
        /// Solve the heat transfer problem for the reconstruction case
        void solve(volScalarField& T, Eigen::VectorXd _heatFlux_weights, word outputFolder,
                   word outputFieldName)
        {
            fvMesh& mesh = _mesh();
            Foam::Time& runTime = _runTime();
            simpleControl& simple = _simple();
            fv::options& fvOptions(_fvOptions());
            dimensionedScalar diffusivity("diffusivity", 
                    dimensionSet(0, 2, -1, 0, 0, 0, 0), k / (rho * Cp));
            Info << "Thermal diffusivity = " << diffusivity << " m2/s" << endl;


            ITHACAutilities::assignBC(T, hotSide_ind,
                                      - updateHeatFlux(_heatFlux_weights) / k);
            std::cout << "debug : _heatFlux_weights = " << _heatFlux_weights << std::endl;

            while (runTime.loop())
            {
                scalar time = runTime.value();

                // Solve
                while (simple.correctNonOrthogonal())
                {
                    fvScalarMatrix TEqn
                    (
                        fvm::ddt(T) - fvm::laplacian(diffusivity, T)
                    );
                    fvOptions.constrain(TEqn);
                    TEqn.solve();
                    fvOptions.correct(T);
                }

                runTime.write();
            }

            ITHACAstream::exportSolution(T, runTime.timeName(),
                                         outputFolder,
                                         outputFieldName);
        }

        //--------------------------------------------------------------------------
        /// Solve the heat transfer problem for the true BC
        Eigen::MatrixXd solve(volScalarField& T, word outputFolder,
                              word outputFieldName)
        {
            Info << "Performing true computation" << endl;
            restart();
            Foam::Time& runTime = _runTime();
            simpleControl& simple = _simple();
            fv::options& fvOptions(_fvOptions());
            label timeI = 0;
            assignTrueIF();
            g = gTrue;
            assignDirectBC(timeI);
            ITHACAstream::exportSolution(T, std::to_string(timeSteps[timeI]),
                                         directOutputFolder,
                                         "Ttrue");
            fvMesh& mesh = _mesh();
            volScalarField gTrueField = list2Field(gTrue[timeI]);
            ITHACAstream::exportSolution(gTrueField,
                                         std::to_string(timeSteps[timeI]), 
                                         directOutputFolder, "gTrue");
            Ttrue.resize(0);
            Ttrue.append(T.clone());

            Eigen::MatrixXd obsMat;
            dimensionedScalar diffusivity("diffusivity", 
                    dimensionSet(0, 2, -1, 0, 0, 0, 0), k / (rho * Cp));
            Info << "Thermal diffusivity = " << diffusivity << " m2/s" << endl;
            sampleFlag = 0;
            timeI = 0;
            sampleI = 0;
            Eigen::VectorXd trueTimeVec(Ntimes);
            M_Assert(observationDeltaTimesteps > 0,
                     "Observation delta timesteps not properly set up");

            while (runTime.loop())
            {
                Info << "Time = " << runTime.timeName() << nl << endl;
                timeI++;
                ITHACAstream::exportSolution(T, std::to_string(timeSteps[timeI]),
                                             directOutputFolder,
                                             "Ttrue");
                gTrueField = list2Field(gTrue[timeI]);
                ITHACAstream::exportSolution(gTrueField,
                                             std::to_string(timeSteps[timeI]), 
                                             directOutputFolder, "gTrue");
                scalar time = runTime.value();

                if (timeI == observationStartTimestep)
                {
                    sampleFlag = observationDeltaTimesteps;
                }
                else if (timeI > observationStartTimestep)
                {
                    sampleFlag++;
                }

                trueTimeVec(timeI - 1) = time;

                // Update BC
                assignDirectBC(timeI);

                // Solve
                while (simple.correctNonOrthogonal())
                {
                    fvScalarMatrix TEqn
                    (
                        fvm::ddt(T) - fvm::laplacian(diffusivity, T)
                    );
                    fvOptions.constrain(TEqn);
                    TEqn.solve();
                    fvOptions.correct(T);
                }

                probe_true.col(timeI - 1) = fieldValueAtProbe(T, probePosition);
                ITHACAstream::exportSolution(T, runTime.timeName(),
                                             outputFolder,
                                             outputFieldName);

                if (sampleFlag == observationDeltaTimesteps)
                {
                    Info << "Sampling at time = " << runTime.timeName() << nl << endl;
                    obsMat.conservativeResize(observe(T).size(), obsMat.cols() + 1);
                    obsMat.col(sampleI) = observe(T);
                    sampleFlag = 0;
                    sampleI++;
                }

                runTime.write();
            }

            ITHACAstream::exportMatrix(trueTimeVec, "trueTimeVec", "eigen", outputFolder);
            ITHACAstream::exportMatrix(probe_true, "probe_true", "eigen", outputFolder);
            ITHACAstream::exportMatrix(trueBC, "trueBC", "eigen", outputFolder);
            resetSamplingCounters();
            return obsMat;
        }

        //--------------------------------------------------------------------------
        /// Preforming a true solution
        Eigen::MatrixXd solveDirect()
        {
            Info << "\n****************************************\n" << endl;
            Info << "\nPerforming true solution\n" << endl;
            word outputFolder = "./ITHACAoutput/direct/";
            restart();
            volScalarField& T(_T());
            set_gTrue();
            /// The measurements are obtained as observation of the true solution
            measurements = solve(T, outputFolder, "Tdirect");
            ITHACAstream::exportMatrix(measurements, "trueMeasurements", "eigen",
                                       outputFolder);
            std::cout << "Number of samples in time = " << measurements.cols() << 
                std::endl;
            std::cout << "Number of sample in space = " << measurements.rows() << 
                std::endl;
            resetSamplingCounters();
            Info << "\nEND true solution\n" << endl;
            Info << "\n****************************************\n" << endl;
            return measurements;
        }


        //--------------------------------------------------------------------------
        /// Assign internalField given an Eigen Vector
        void assignIF(volScalarField& field_, Eigen::VectorXd internalField_)
        {
            for (int i = 0; i < internalField_.size(); i++)
            {
                field_.ref()[i] = internalField_(i);
            }
        }

        //--------------------------------------------------------------------------
        /// Return field value in a given point
        Eigen::VectorXd fieldValueAtProbe(volScalarField field_,
                                          Foam::vector probeLocation_)
        {
            Foam::fvMesh& mesh = _mesh();
            Eigen::VectorXd output(1);
            output(0) = field_[mesh.findCell(probeLocation_)];
            return output;
        }

        //--------------------------------------------------------------------------
        /// Return field value in a given point
        double fieldValueAtProbe(Eigen::VectorXd field_, Foam::vector probeLocation_)
        {
            Foam::fvMesh& mesh = _mesh();
            double output;
            output = field_(mesh.findCell(probeLocation_));
            return output;
        }

        //--------------------------------------------------------------------------
        /// Restart before a new solution
        void restart()
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
        }

        //--------------------------------------------------------------------------
        /// Reset runTime to given values
        void resetRunTime(double startTime_, int startIndex_, double endTime_)
        {
            Info << "\ndebug : timeReset, start = " << startTime_ << ", index = " <<
                 startIndex_ << ", endTime = " << endTime_ << endl;
            Time& runTime = _runTime();
            instantList Times = runTime.times();
            runTime.setTime(startTime_, startIndex_);
            runTime.setEndTime(endTime_);
            _simple.clear();
            Foam::fvMesh& mesh = _mesh();
            _simple = autoPtr<simpleControl>
                      (
                          new simpleControl
                          (
                              mesh
                          )
                      );
        }
};

#endif
