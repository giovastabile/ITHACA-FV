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
    Steady SIMPLE ROM with empirical cubature hyper-reduction
SourceFiles
    12simpleSteadyNS_ECP.C
\*---------------------------------------------------------------------------*/

#include "SteadyNSSimple.H"
#include "ITHACAstream.H"
#include "ITHACAPOD.H"
#include "ReducedSimpleSteadyNS.H"
#include "forces.H"
#include "IOmanip.H"
#include "SampledMesh.H"
#include "hyperReduction.templates.H"
#include "Pstream.H"


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

    const fvMesh& mesh = example._mesh();

    M_Assert
    (
        !Pstream::parRun(),
        "This ECP tutorial currently requires a serial mesh"
    );

    const label nCells = mesh.nCells();
    const label ecpNodes =
        para->ITHACAdict->lookupOrDefault<label>("ecpNodes", 100);
    const label hrLayers =
        para->ITHACAdict->lookupOrDefault<label>("ecpLayers", 2);

    M_Assert
    (
        ecpNodes > 0 && ecpNodes <= nCells,
        "ecpNodes must be positive and no larger than the cell count"
    );

    const scalarField& cellVolumes = mesh.V();
    Eigen::VectorXd ecpVolumes(nCells);

    forAll(cellVolumes, celli)
    {
        ecpVolumes(celli) = cellVolumes[celli];
    }

    const label nVelocityFeatures =
        example.Umodes.size()
      + example.liftfield.size()
      + example.Ufield.size();
    const label nPressureFeatures =
        example.Pmodes.size()
      + example.Pfield.size();
    const label nECPFeatures = 1 + 3*nVelocityFeatures + nPressureFeatures;

    Eigen::MatrixXd ecpFeatures =
        Eigen::MatrixXd::Zero(nCells, nECPFeatures);
    label featureI = 0;

    // Keep the constant function in the cubature training space.
    ecpFeatures.col(featureI++).setOnes();

    auto appendVelocityFeatures =
        [&](const volVectorField& field)
        {
            for (direction cmpt = 0; cmpt < vector::nComponents; ++cmpt)
            {
                forAll(cellVolumes, celli)
                {
                    ecpFeatures(celli, featureI) = field[celli][cmpt];
                }

                ++featureI;
            }
        };

    auto appendPressureFeatures =
        [&](const volScalarField& field)
        {
            forAll(cellVolumes, celli)
            {
                ecpFeatures(celli, featureI) = field[celli];
            }

            ++featureI;
        };

    for (label modeI = 0; modeI < example.Umodes.size(); ++modeI)
    {
        appendVelocityFeatures(example.Umodes.toPtrList()[modeI]);
    }

    for (label liftI = 0; liftI < example.liftfield.size(); ++liftI)
    {
        appendVelocityFeatures(example.liftfield[liftI]);
    }

    for (label snapshotI = 0; snapshotI < example.Ufield.size(); ++snapshotI)
    {
        appendVelocityFeatures(example.Ufield[snapshotI]);
    }

    for (label modeI = 0; modeI < example.Pmodes.size(); ++modeI)
    {
        appendPressureFeatures(example.Pmodes.toPtrList()[modeI]);
    }

    for (label snapshotI = 0; snapshotI < example.Pfield.size(); ++snapshotI)
    {
        appendPressureFeatures(example.Pfield[snapshotI]);
    }

    M_Assert
    (
        featureI == nECPFeatures,
        "Internal ECP feature count does not match the training matrix"
    );

    Eigen::VectorXi initialSeeds(0);
    Eigen::VectorXd unitNormalization = Eigen::VectorXd::Ones(nCells);
    HyperReduction<PtrList<volVectorField>&> ecp
    (
        nECPFeatures,
        ecpNodes,
        1,
        nCells,
        initialSeeds,
        "12simpleSteadyNS_ECP",
        ecpVolumes
    );

    ecp.offlineECP(ecpFeatures, unitNormalization);

    labelList sampledCells(ecp.nodePoints().size());

    forAll(sampledCells, sampleI)
    {
        sampledCells[sampleI] = ecp.nodePoints()[sampleI];
    }

    M_Assert
    (
        ecp.quadratureWeights.size() == sampledCells.size(),
        "ECP returned a different number of weights and selected cells"
    );

    Info<< "ECP selected " << sampledCells.size()
        << " cells from " << nCells
        << "; weight sum = " << ecp.quadratureWeights.sum()
        << endl;

    // ------------------------------------------------------------
    // Build the actual assembly submesh:
    // sampled cells + hrLayers face-neighbour halo.
    // ------------------------------------------------------------
    SampledMesh sampledMesh
    (
        mesh,
        sampledCells,
        hrLayers
    );

    sampledMesh.writeMask("ecpMask");

    reduced.setupSampled
    (
        sampledMesh,
        NmodesUproj,
        NmodesPproj,
        0,
        &ecp.quadratureWeights
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
            "./ITHACAoutput/ReconstructECP/",
            "G"
        );
    }

    return 0;
}
