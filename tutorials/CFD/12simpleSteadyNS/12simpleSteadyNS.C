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
    Example of steady NS Reduction Problem solved by the use of the SIMPLE algorithm
SourceFiles
    12simpleSteadyNS.C
\*---------------------------------------------------------------------------*/

#include "SteadyNSSimple.H"
#include "ITHACAstream.H"
#include "ITHACAPOD.H"
#include "ReducedSimpleSteadyNS.H"
#include "forces.H"
#include "IOmanip.H"
#include "SampledMesh.H"
#include "cellSet.H"


class tutorial12 : public SteadyNSSimple
{
    public:
        /// Constructor
        explicit tutorial12(int argc, char* argv[])
            :
            SteadyNSSimple(argc, argv),
            U(_U()),
            p(_p()),
            phi(_phi())
        {}

        /// Velocity field
        volVectorField& U;
        /// Pressure field
        volScalarField& p;
        ///
        surfaceScalarField& phi;

        /// Perform an Offline solve
        void offlineSolve()
        {
            Vector<double> inl(0, 0, 0);
            List<scalar> mu_now(1);

            // if the offline solution is already performed read the fields
            if (offline)
            {
                ITHACAstream::read_fields(Ufield, U, "./ITHACAoutput/Offline/");
                ITHACAstream::read_fields(Pfield, p, "./ITHACAoutput/Offline/");
                mu_samples =
                    ITHACAstream::readMatrix("./ITHACAoutput/Offline/mu_samples_mat.txt");
            }
            else
            {
                Vector<double> Uinl(1, 0, 0);
                label BCind = 0;

                for (label i = 0; i < mu.cols(); i++)
                {
                    mu_now[0] = mu(0, i);
                    change_viscosity(mu(0, i));
                    assignIF(U, Uinl);
                    truthSolve2(mu_now);
                }
            }
        }

};

int main(int argc, char* argv[])
{
    // Construct the tutorial object
    tutorial12 example(argc, argv);
    // Read some parameters from file
    ITHACAparameters* para = ITHACAparameters::getInstance(example._mesh(),
        example._runTime());
    int NmodesUout = para->ITHACAdict->lookupOrDefault<int>("NmodesUout", 15);
    int NmodesPout = para->ITHACAdict->lookupOrDefault<int>("NmodesPout", 15);
    int NmodesUproj = para->ITHACAdict->lookupOrDefault<int>("NmodesUproj", 10);
    int NmodesPproj = para->ITHACAdict->lookupOrDefault<int>("NmodesPproj", 10);
    // Read the par file where the parameters are stored
    word filename("./par");
    example.mu = ITHACAstream::readMatrix(filename);
    // Set the inlet boundaries patch 0 directions x and y
    example.inletIndex.resize(1, 2);
    example.inletIndex(0, 0) = 0;
    example.inletIndex(0, 1) = 0;
    // Perform the offline solve
    example.offlineSolve();
    ITHACAstream::read_fields(example.liftfield, example.U, "./lift/");
    // Homogenize the snapshots
    example.computeLift(example.Ufield, example.liftfield, example.Uomfield);
    // Perform POD on velocity and pressure and store the first 10 modes
    ITHACAPOD::getModes(example.Uomfield, example.Umodes, example._U().name(),
                        example.podex, 0, 0,
                        NmodesUout);
    ITHACAPOD::getModes(example.Pfield, example.Pmodes, example._p().name(),
                        example.podex, 0, 0,
                        NmodesPout);
    // Create the reduced object
    reducedSimpleSteadyNS reduced(example);

    PtrList<volVectorField> U_rec_list;
    PtrList<volScalarField> P_rec_list;

    // Read inlet velocities boundary conditions.
    word vel_file(para->ITHACAdict->lookup("online_velocities"));
    Eigen::MatrixXd vel = ITHACAstream::readMatrix(vel_file);

    // ------------------------------------------------------------
    // Hyper-reduction sampling parameters.
    // ------------------------------------------------------------
    const Switch useResidualSampling =
        para->ITHACAdict->lookupOrDefault<Switch>
        (
            "useResidualSampling",
            true
        );

    const scalar hrSampleFraction =
        para->ITHACAdict->lookupOrDefault<scalar>
        (
            "hrSampleFraction",
            0.10
        );

    const label hrSpacingLayers =
        para->ITHACAdict->lookupOrDefault<label>
        (
            "hrSpacingLayers",
            1
        );

    const label hrLayers =
        para->ITHACAdict->lookupOrDefault<label>
        (
            "hrLayers",
            1
        );

    labelList sampledCells;

    if (useResidualSampling)
    {
        // --------------------------------------------------------
        // Training solve:
        // run the standard/full ROM for one parameter value while
        // accumulating a normalized cell-wise residual indicator.
        // --------------------------------------------------------
        const label trainingParameter =
            para->ITHACAdict->lookupOrDefault<label>
            (
                "hrTrainingParameter",
                0
            );

        M_Assert
        (
            trainingParameter >= 0
         && trainingParameter < example.mu.cols(),
            "hrTrainingParameter is outside the parameter matrix"
        );

        const scalar muTrain =
            example.mu(0, trainingParameter);

        Info<< nl
            << "========================================" << nl
            << " Building residual-based HR sample set" << nl
            << " training parameter index = "
            << trainingParameter << nl
            << " mu = " << muTrain << nl
            << "========================================"
            << nl << endl;

        example.change_viscosity(muTrain);
        reduced.setOnlineVelocity(vel);

        reduced.setCollectResidualIndicator(true);

        reduced.solveOnline_Simple
        (
            muTrain,
            NmodesUproj,
            NmodesPproj,
            0,
            0,
            "./ITHACAoutput/ResidualTraining/"
        );

        // Pick the cells with the largest full-ROM residual indicator.
        sampledCells =
            SampledMesh::residualBasedSamples
            (
                reduced.residualIndicator(),
                example._mesh(),
                hrSampleFraction,
                hrSpacingLayers
            );

        // Always retain cells adjacent to physical boundaries.
        sampledCells =
            SampledMesh::addPhysicalBoundaryCells
            (
                example._mesh(),
                sampledCells
            );

        // --------------------------------------------------------
        // Write/overwrite the actual residual-based sampling set.
        //
        // This keeps constant/polyMesh/sets/hrSamples consistent
        // with the cells actually used by the hyper-reduced solver.
        // --------------------------------------------------------
        cellSet writtenSampleSet
        (
            example._mesh(),
            "hrSamples",
            sampledCells.size()
        );

        forAll(sampledCells, i)
        {
            writtenSampleSet.insert(sampledCells[i]);
        }

        writtenSampleSet.write();

        Info<< "Written residual-based cellSet hrSamples with "
            << sampledCells.size()
            << " cells"
            << endl;

        // Write the accumulated indicator as a field for ParaView.
        reduced.writeResidualIndicator("hrResidualIndicator");

        reduced.setCollectResidualIndicator(false);
    }
    else
    {
        // --------------------------------------------------------
        // Legacy/manual path: read an existing cellSet.
        // --------------------------------------------------------
        const word hrSampleSetName =
            para->ITHACAdict->lookupOrDefault<word>
            (
                "hrSampleSet",
                "hrSamples"
            );

        cellSet hrSampleSet
        (
            example._mesh(),
            hrSampleSetName
        );

        sampledCells = hrSampleSet.toc();
    }

    // ------------------------------------------------------------
    // Build the actual assembly submesh:
    // sampled cells + hrLayers face-neighbour halo.
    // ------------------------------------------------------------
    SampledMesh sampledMesh
    (
        example._mesh(),
        sampledCells,
        hrLayers
    );

    sampledMesh.writeMask("hrMask");

    reduced.setupSampled
    (
        sampledMesh,
        NmodesUproj,
        NmodesPproj
    );

    // ------------------------------------------------------------
    // Hyper-reduced online solutions.
    // Galerkin is intentionally used here to remain algebraically
    // consistent with the original solveOnline_Simple().
    // ------------------------------------------------------------
    for (label k = 0; k < example.mu.cols(); ++k)
    {
        const scalar mu_now = example.mu(0, k);

        example.change_viscosity(mu_now);
        reduced.setOnlineVelocity(vel);

        reduced.solveOnline_SimpleSampled
        (
            mu_now,
            0,
            "./ITHACAoutput/ReconstructHR/",
            "G"
        );
    }

    return 0;
}
