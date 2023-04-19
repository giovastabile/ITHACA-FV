#ifndef inverseHeatTransfer_1D_H
#define inverseHeatTransfer_1D_H
#include "simpleControl.H"
#include "fvOptions.H"


/// \brief Class where the 1D heat transfer problem is implemented
/// \details It is a child of the laplacianProblem class and some of its
/// functions are overridden to be adapted to the specific case.
class inverseHeatTransfer_1D: public laplacianProblem
{
        bool measurementPointsRead = 0;
        List<vector> measurementPoints;
    public:
        explicit inverseHeatTransfer_1D(int argc, char* argv[])
            :
            laplacianProblem(argc, argv)
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
            left_ind = mesh.boundaryMesh().findPatchID("left");
            volScalarField& T(_T());
            stateSize = T.size();
            startTime = runTime.startTime().value();
            deltaTime = runTime.deltaTValue();
            endTime = runTime.endTime().value();
            Ntimes = (endTime - startTime) / deltaTime;
            timeVector.resize(Ntimes);
            readObservationTimes();
        }

        autoPtr<simpleControl> _simple;

        autoPtr<fv::options> _fvOptions;

        /// Diffusivity
        double k;

        /// Density
        double rho;

        /// Heat capacity
        double Cp;

        /// Index of the left patch
        label left_ind;

        Eigen::MatrixXd measurements;

        int stateSize;

        int timeI;
        double startTime;
        double deltaTime;
        double endTime;
        int Ntimes;

        /// Timesteps vector
        Eigen::VectorXd timeVector;

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

        int samplingTimeI = 0;
        int sampleI = 0;
        int sampleFlag = 0;
        int observationDeltaTimesteps = 0;
        int observationStartTimestep = 0;

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
            int BCsize = T.boundaryField()[mesh.boundaryMesh().findPatchID("left")].size();
            return BCsize;
        }

        //--------------------------------------------------------------------------
        /// Returns the input field values at the measurement points defined in the measurementsDict
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
        /// Returns the input field values at the measurement points defined in the measurementsDict
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
                                     Eigen::VectorXd BCsample, double startTime_, int startIndex_, double endTime_,
                                     Eigen::VectorXd _modelError)
        {
            M_Assert(previousState.size() == _modelError.size(),
                     "Model error vector must have same size of the state");
            word outputFolder = "ITHACAoutput/projection";
            word fieldName = "T";
            volScalarField T(_T());
            assignIF(T, previousState);
            timeI = startIndex_;
            resetRunTime(startTime_, startIndex_, endTime_);
            solve(T, BCsample, outputFolder, fieldName);
            return Foam2Eigen::field2Eigen(T) + _modelError;
        }

        //--------------------------------------------------------------------------
        /// Solve the heat transfer problem for the reconstruction case
        void solve(volScalarField& T, Eigen::VectorXd BCsample, word outputFolder,
                   word outputFieldName)
        {
            fvMesh& mesh = _mesh();
            Foam::Time& runTime = _runTime();
            simpleControl& simple = _simple();
            fv::options& fvOptions(_fvOptions());
            dimensionedScalar diffusivity("diffusivity", dimensionSet(0, 2, -1, 0, 0, 0, 0),
                                          k / (rho * Cp));
            Info << "Thermal diffusivity = " << diffusivity << " m2/s" << endl;
            ITHACAutilities::assignBC(T, mesh.boundaryMesh().findPatchID("left"),
                                      - k * BCsample);
            std::cout << "debug : BCsample = " << BCsample << std::endl;

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
            Eigen::MatrixXd obsMat;
            fvMesh& mesh = _mesh();
            Foam::Time& runTime = _runTime();
            simpleControl& simple = _simple();
            fv::options& fvOptions(_fvOptions());
            dimensionedScalar diffusivity("diffusivity", dimensionSet(0, 2, -1, 0, 0, 0, 0),
                                          k / (rho * Cp));
            Info << "Thermal diffusivity = " << diffusivity << " m2/s" << endl;
            sampleFlag = 0;
            timeI = 0;
            sampleI = 0;
            Eigen::VectorXd trueTimeVec(Ntimes);
            M_Assert(observationDeltaTimesteps > 0,
                     "Observation delta timesteps not properly set up");
            trueBC.resize(0);

            while (runTime.loop())
            {
                timeI++;
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
                ITHACAutilities::assignBC(T, mesh.boundaryMesh().findPatchID("left"),
                                          - k * updateBC());

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
            /// The measurements are obtained as observation of the true solution
            measurements = solve(T, outputFolder, "Tdirect");
            ITHACAstream::exportMatrix(measurements, "trueMeasurements", "eigen",
                                       outputFolder);
            std::cout << "Number of samples in time = " << measurements.cols() << std::endl;
            std::cout << "Number of sample in space = " << measurements.rows() << std::endl;
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
