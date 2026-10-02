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
#include "labelIOList.H"
#include <unordered_map>
#include <cstdint>
#include <sstream>


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
                if (para->ITHACAdict->lookupOrDefault<bool>("middleExport", true))
                {
                    for (label sampleI = 0; sampleI < mu.cols(); ++sampleI)
                    {
                        word sampleFolder = "./ITHACAoutput/Offline/"
                                            + name(sampleI + 1) + "/";
                        ITHACAstream::read_fields(Ufield, U, sampleFolder);
                        ITHACAstream::read_fields(Pfield, p, sampleFolder);
                    }
                }
                else
                {
                    ITHACAstream::read_fields(Ufield, U, "./ITHACAoutput/Offline/");
                    ITHACAstream::read_fields(Pfield, p, "./ITHACAoutput/Offline/");
                }

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
    argList::addBoolOption("ecpPrepare", "Train/export the global rule and basis for parallel online runs");
    // Construct the tutorial object
    tutorial12 example(argc, argv);
    // Read some parameters from file
    ITHACAparameters* para = ITHACAparameters::getInstance(example._mesh(),
        example._runTime());
    int NmodesUout = para->ITHACAdict->lookupOrDefault<int>("NmodesUout", 15);
    int NmodesPout = para->ITHACAdict->lookupOrDefault<int>("NmodesPout", 15);
    int NmodesUproj = para->ITHACAdict->lookupOrDefault<int>("NmodesUproj", 5);
    int NmodesPproj = para->ITHACAdict->lookupOrDefault<int>("NmodesPproj", 5);
    const bool prepareOnly = example._args().found("ecpPrepare");
    M_Assert(!prepareOnly || !Pstream::parRun(), "Run -ecpPrepare in serial before decomposition");
    // Read the par file where the parameters are stored
    word filename("./par");
    example.mu = ITHACAstream::readMatrix(filename);
    // Set the inlet boundaries patch 0 directions x and y
    example.inletIndex.resize(1, 2);
    example.inletIndex(0, 0) = example._mesh().boundaryMesh().findPatchID("inlet");
    M_Assert(example.inletIndex(0, 0) >= 0, "The case requires an inlet patch");
    example.inletIndex(0, 1) = 0;
    if (Pstream::parRun())
    {
        // These fields were decomposed with the mesh by Allrun_parallel.
        fvMesh& mesh = example._mesh();
        example.liftfield.append(new volVectorField
        (
            IOobject("ecpLift0", "0", mesh, IOobject::MUST_READ, IOobject::NO_WRITE, false), mesh
        ));
        for (label i = 0; i < NmodesUproj; ++i)
        {
            example.Umodes.append(new volVectorField
            (
                IOobject("ecpU" + name(i), "0", mesh, IOobject::MUST_READ, IOobject::NO_WRITE, false), mesh
            ));
        }
        for (label i = 0; i < NmodesPproj; ++i)
        {
            example.Pmodes.append(new volScalarField
            (
                IOobject("ecpP" + name(i), "0", mesh, IOobject::MUST_READ, IOobject::NO_WRITE, false), mesh
            ));
        }
    }
    else
    {
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
    }
    // Create the reduced object
    reducedSimpleSteadyNS reduced(example);

    // Read inlet velocities boundary conditions.
    word vel_file(para->ITHACAdict->lookup("online_velocities"));
    Eigen::MatrixXd vel = ITHACAstream::readMatrix(vel_file);

    const fvMesh& mesh = example._mesh();

    const label nCells = mesh.nCells();
    const label ecpNodes =
        para->ITHACAdict->lookupOrDefault<label>("ecpNodes", 300);
    const label hrLayers =
        para->ITHACAdict->lookupOrDefault<label>("ecpLayers", 8);

    const label globalCells = returnReduce(nCells, sumOp<label>());
    M_Assert(hrLayers >= 0, "ecpLayers must be nonnegative");
    M_Assert
    (
        ecpNodes > 0 && ecpNodes <= globalCells,
        "ecpNodes must be positive and no larger than the cell count"
    );

    const scalarField& cellVolumes = mesh.V();
    labelList sampledCells;
    Eigen::VectorXd cubatureWeights;
    if (Pstream::parRun())
    {
        const fileName ruleFolder("constant/ecpRule");
        IFstream metadataStream(ruleFolder + "/metadata");
        M_Assert(metadataStream.good(), "Missing prepared ECP metadata; run Allrun_parallel");
        dictionary metadata(metadataStream);
        M_Assert(readLabel(metadata.lookup("nCells")) == globalCells
            && readLabel(metadata.lookup("NmodesUproj")) == NmodesUproj
            && readLabel(metadata.lookup("NmodesPproj")) == NmodesPproj
            && readLabel(metadata.lookup("ecpNodes")) == ecpNodes,
            "Prepared ECP data does not match the mesh/configuration; rerun Allrun_parallel");
        Eigen::VectorXi globalNodes;
        Eigen::VectorXd globalWeights;
        cnpy::load(globalNodes, ruleFolder + "/nodePoints.npy");
        cnpy::load(globalWeights, ruleFolder + "/quadratureWeights.npy");
        M_Assert(globalNodes.size() == globalWeights.size() && globalWeights.allFinite()
            && (globalWeights.array() >= 0).all(), "Invalid global ECP rule");
        std::unordered_map<label, label> positions;
        for (label i = 0; i < globalNodes.size(); ++i)
        {
            const bool unique = positions.emplace(globalNodes(i), i).second;
            M_Assert(globalNodes(i) >= 0 && globalNodes(i) < globalCells
                && unique, "Invalid or duplicate global ECP cell");
        }
        const labelIOList cellMap
        (
            IOobject("cellProcAddressing", mesh.facesInstance(), polyMesh::meshSubDir,
                mesh, IOobject::MUST_READ, IOobject::NO_WRITE, false)
        );
        M_Assert(cellMap.size() == nCells, "Invalid cellProcAddressing");
        DynamicList<label> localNodes;
        DynamicList<scalar> localWeights;
        forAll(cellMap, celli)
        {
            const auto found = positions.find(cellMap[celli]);
            if (found != positions.end())
            {
                localNodes.append(celli);
                localWeights.append(globalWeights(found->second));
            }
        }
        sampledCells = labelList(localNodes);
        cubatureWeights.resize(localWeights.size());
        forAll(localWeights, i) cubatureWeights(i) = localWeights[i];
        const label selected = returnReduce(sampledCells.size(), sumOp<label>());
        M_Assert(selected == globalNodes.size(), "The decomposed rule lost or duplicated cells");
        Info << "Distributed ECP rule: " << selected << " global samples on "
             << Pstream::nProcs() << " ranks" << endl;
    }
    else
    {
    Eigen::VectorXd ecpVolumes(nCells);

    forAll(cellVolumes, celli)
    {
        ecpVolumes(celli) = cellVolumes[celli];
    }

    // Match setupSampled(): the requested velocity dimension includes the lift.
    M_Assert(NmodesUproj > 1 && NmodesUproj <= example.Umodes.size()
        && NmodesPproj > 0 && NmodesPproj <= example.Pmodes.size(),
        "Invalid ECP projection dimensions");
    PtrList<volVectorField> velocityBasis(NmodesUproj);
    velocityBasis.set(0, example.liftfield[0].clone());
    for (label i = 1; i < NmodesUproj; ++i)
    {
        velocityBasis.set(i, example.Umodes[i - 1].clone());
    }
    Eigen::MatrixXd V = Foam2Eigen::PtrList2Eigen(velocityBasis);
    Eigen::MatrixXd Q = Foam2Eigen::PtrList2Eigen(example.Pmodes.toPtrList())
        .leftCols(NmodesPproj);
    const label nParameters = example.mu.cols();
    M_Assert(nParameters > 0 && example.Ufield.size() > 0
        && example.Ufield.size() == example.Pfield.size()
        && example.Ufield.size() % nParameters == 0,
        "Expected matching snapshots grouped evenly by parameter");
    const label snapshotsPerParameter = example.Ufield.size()/nParameters;
    const label requestedSnapshots = para->ITHACAdict->lookupOrDefault<label>
        ("ecpTrainingSnapshots", 2);
    M_Assert(requestedSnapshots > 0, "ecpTrainingSnapshots must be positive");
    const label operatorSnapshotsPerParameter =
        min(snapshotsPerParameter, requestedSnapshots);
    const label nOperatorSnapshots = nParameters*(operatorSnapshotsPerParameter + 1);
    const label nECPFeatures = 1 + NmodesUproj*NmodesPproj
        + nOperatorSnapshots*(NmodesUproj*(NmodesUproj + 1)
                              + NmodesPproj*(NmodesPproj + 1)
                              + NmodesPproj*(NmodesUproj + 1));
    Eigen::MatrixXd ecpFeatures(nCells, nECPFeatures);
    label featureI = 0;
    ecpFeatures.col(featureI++) = ecpVolumes/ecpVolumes.norm();

    // Each column is a cell's contribution to an entry of V^T A V or V^T b.
    // fvMatrix already contains volume integration and boundary coefficients.
    auto appendSystem = [&](auto& equation, const Eigen::MatrixXd& basis,
                            label components)
    {
        Eigen::SparseMatrix<double> A;
        Eigen::VectorXd rhs;
        Foam2Eigen::fvMatrix2Eigen(equation, A, rhs);
        Eigen::MatrixXd Abasis = A*basis;
        const label start = featureI;
        for (label i = 0; i < basis.cols(); ++i)
        {
            for (label j = 0; j <= basis.cols(); ++j)
            {
                for (label celli = 0; celli < nCells; ++celli)
                {
                    double value = 0;
                    for (label c = 0; c < components; ++c)
                    {
                        const label row = components*celli + c;
                        value += basis(row, i)*(j == basis.cols()
                            ? rhs(row) : Abasis(row, j));
                    }
                    ecpFeatures(celli, featureI) = value;
                }
                ++featureI;
            }
        }
        // Balance systems without amplifying individual near-zero entries.
        auto block = ecpFeatures.middleCols(start, featureI - start);
        const double scale = block.norm();
        if (scale > SMALL) block /= scale;
    };
    const label gradientStart = featureI;
    for (label j = 0; j < NmodesPproj; ++j)
    {
        tmp<volVectorField> gradient = fvc::grad(example.Pmodes[j]);
        for (label i = 0; i < NmodesUproj; ++i)
        {
            forAll(cellVolumes, celli)
            {
                ecpFeatures(celli, featureI) = cellVolumes[celli]
                    *(velocityBasis[i][celli] & gradient()[celli]);
            }
            ++featureI;
        }
    }
    auto gradientBlock = ecpFeatures.middleCols(gradientStart, featureI-gradientStart);
    if (gradientBlock.norm() > SMALL) gradientBlock /= gradientBlock.norm();
    Info << "ECP training: " << nCells << " cells x " << nECPFeatures
         << " reduced-operator features" << endl;

    simpleControl& simple = example._simple();

    for (label sampleI = 0; sampleI < nParameters; ++sampleI)
    {
        example.change_viscosity(example.mu(0, sampleI));

        for
        (
            label operatorSampleI = 0;
            operatorSampleI <= operatorSnapshotsPerParameter;
            ++operatorSampleI
        )
        {
            const label localSnapshotI =
                operatorSnapshotsPerParameter == 1
              ? snapshotsPerParameter - 1
              : max(label(0), operatorSampleI - 1) * (snapshotsPerParameter - 1)
                / (operatorSnapshotsPerParameter - 1);
            const label snapshotI =
                sampleI * snapshotsPerParameter + localSnapshotI;
            volVectorField UTraining("U", example.Ufield[snapshotI]);
            volScalarField pTraining("p", example.Pfield[snapshotI]);
            if (operatorSampleI == 0)
            {
                UTraining = velocityBasis[0]*vel(0, 0);
                pTraining *= 0.0;
            }
            UTraining.correctBoundaryConditions();
            pTraining.correctBoundaryConditions();
            UTraining.storePrevIter();

            surfaceScalarField phiTraining
            (
                IOobject
                (
                    "phiECPTraining",
                    example._runTime().timeName(),
                    mesh,
                    IOobject::NO_READ,
                    IOobject::NO_WRITE
                ),
                fvc::interpolate(UTraining) & mesh.Sf()
            );

            tmp<volScalarField> tNuEffTraining =
                example.turbulence->nuEff();
            volScalarField nuEffTraining
            (
                IOobject
                (
                    "nuEffECPTraining",
                    example._runTime().timeName(),
                    mesh,
                    IOobject::NO_READ,
                    IOobject::NO_WRITE
                ),
                tNuEffTraining()
            );

            fvVectorMatrix UEqnTraining
            (
                fvm::div(phiTraining, UTraining)
              - fvm::laplacian(nuEffTraining, UTraining)
              - fvc::div
                (
                    nuEffTraining
                  * dev2(T(fvc::grad(UTraining)))
                )
            );
            UEqnTraining.relax();
            appendSystem(UEqnTraining, V, 3);

            volScalarField rAUTraining(1.0/UEqnTraining.A());
            volVectorField HbyATraining
            (
                constrainHbyA
                (
                    1.0/UEqnTraining.A()*UEqnTraining.H(),
                    UTraining,
                    pTraining
                )
            );
            volScalarField pTrainingForFlux
            (
                IOobject
                (
                    "pECPTraining",
                    example._runTime().timeName(),
                    mesh,
                    IOobject::NO_READ,
                    IOobject::NO_WRITE
                ),
                pTraining
            );
            surfaceScalarField phiHbyATraining
            (
                IOobject
                (
                    "phiHbyAECPTraining",
                    example._runTime().timeName(),
                    mesh,
                    IOobject::NO_READ,
                    IOobject::NO_WRITE
                ),
                fvc::flux(HbyATraining)
            );
            adjustPhi(phiHbyATraining, UTraining, pTrainingForFlux);
            tmp<volScalarField> rAtUTraining(rAUTraining);

            if (simple.consistent())
            {
                rAtUTraining =
                    1.0/(1.0/rAUTraining - UEqnTraining.H1());
                phiHbyATraining +=
                    fvc::interpolate(rAtUTraining() - rAUTraining)
                  * fvc::snGrad(pTraining)
                  * mesh.magSf();
                HbyATraining -=
                    (rAUTraining - rAtUTraining())
                  * fvc::grad(pTraining);
            }

            fvScalarMatrix pEqnTraining
            (
                fvm::laplacian(rAtUTraining(), pTraining)
             == fvc::div(phiHbyATraining)
            );
            appendSystem(pEqnTraining, Q, 1);

            // H() uses the newly solved velocity while the relaxed matrix
            // and its source stay frozen. Train this affine RHS dependence
            // on every velocity basis direction, not just the old snapshot.
            for (label probe = -1; probe < NmodesUproj; ++probe)
            {
                UTraining *= 0.0;
                if (probe >= 0) UTraining += velocityBasis[probe];
                UTraining.correctBoundaryConditions();
                volVectorField probeHbyA
                (
                    constrainHbyA(rAUTraining*UEqnTraining.H(), UTraining, pTraining)
                );
                surfaceScalarField probeFlux("phiECPProbe", fvc::flux(probeHbyA));
                adjustPhi(probeFlux, UTraining, pTrainingForFlux);
                if (simple.consistent())
                {
                    probeFlux += fvc::interpolate(rAtUTraining()-rAUTraining)
                        *fvc::snGrad(pTraining)*mesh.magSf();
                }
                fvScalarMatrix probeEquation
                (
                    fvm::laplacian(rAtUTraining(), pTraining) == fvc::div(probeFlux)
                );
                Eigen::SparseMatrix<double> probeA;
                Eigen::VectorXd probeRhs;
                Foam2Eigen::fvMatrix2Eigen(probeEquation, probeA, probeRhs);
                for (label i = 0; i < NmodesPproj; ++i)
                {
                    ecpFeatures.col(featureI++) = Q.col(i).cwiseProduct(probeRhs);
                }
            }
        }
    }

    M_Assert
    (
        featureI == nECPFeatures,
        "Internal ECP feature count does not match the training matrix"
    );
    // Give small, cancellation-sensitive continuity terms the same fitting
    // priority as large momentum terms. Exactly zero columns remain zero.
    for (label i = 0; i < ecpFeatures.cols(); ++i)
    {
        const double norm = ecpFeatures.col(i).norm();
        if (norm > SMALL) ecpFeatures.col(i) /= norm;
    }
    M_Assert(ecpFeatures.allFinite(), "Non-finite ECP training features");
    Info << "ECP feature matrix assembled; beginning cubature selection"
         << endl;

    // ECP expects a basis for the training space. Remove the strongly
    // dependent operator columns before NNLS, retaining their cell span.
    const scalar basisTolerance = para->ITHACAdict->lookupOrDefault<scalar>
        ("ecpBasisTolerance", 1e-6);
    M_Assert(basisTolerance > 0 && basisTolerance < 1, "Invalid ecpBasisTolerance");
    Eigen::MatrixXd gram = ecpFeatures.transpose()*ecpFeatures;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(gram);
    M_Assert(eig.info() == Eigen::Success, "ECP training eigendecomposition failed");
    const double cutoff = sqr(basisTolerance)*eig.eigenvalues().maxCoeff();
    label rank = 0;
    for (label i = 0; i < eig.eigenvalues().size(); ++i)
    {
        if (eig.eigenvalues()(i) > cutoff) ++rank;
    }
    M_Assert(rank > 0, "Empty ECP training basis");
    Eigen::MatrixXd compressed = ecpFeatures*eig.eigenvectors().rightCols(rank);
    compressed.array().rowwise() /= eig.eigenvalues().tail(rank).array().sqrt().transpose();
    ecpFeatures.swap(compressed);
    Info << "ECP compressed training rank = " << rank << endl;

    Eigen::VectorXi initialSeeds(0);
    Eigen::VectorXd unitNormalization = Eigen::VectorXd::Ones(nCells);
    HyperReduction<PtrList<volVectorField>&> ecp
    (
        ecpFeatures.cols(),
        ecpNodes,
        1,
        nCells,
        initialSeeds,
        "12simpleSteadyNS_ECP",
        ecpVolumes
    );

    // Invalidate quadrature when the mesh, modes or training states change.
    std::uint64_t fingerprint = 14695981039346656037ULL;
    const auto* bytes = reinterpret_cast<const unsigned char*>(ecpFeatures.data());
    for (std::size_t i = 0; i < ecpFeatures.size()*sizeof(double); ++i)
    {
        fingerprint = (fingerprint ^ bytes[i])*1099511628211ULL;
    }
    std::ostringstream cacheKey;
    cacheKey << std::hex << fingerprint;
    word ecpCacheFolder = "ITHACAoutput/12simpleSteadyNS_ECP/ECP_projected_v1_"
        + name(ecpNodes) + "_" + word(cacheKey.str()) + "_tol"
        + name(para->ITHACAdict->lookupOrDefault<scalar>("ecpTolerance", 0.0));
    if (ecpNodes == nCells)
    {
        // Exact reference rule for checking sampled/full-mesh equivalence.
        sampledCells.setSize(nCells);
        forAll(sampledCells, i) sampledCells[i] = i;
        cubatureWeights = Eigen::VectorXd::Ones(nCells);
    }
    else
    {
        ecp.offlineECP(ecpFeatures, unitNormalization, ecpCacheFolder);
        sampledCells.setSize(ecp.nodePoints().size());
        forAll(sampledCells, i) sampledCells[i] = ecp.nodePoints()[i];
        cubatureWeights = ecp.quadratureWeights;
    }
    M_Assert(cubatureWeights.size() == sampledCells.size()
        && cubatureWeights.allFinite() && (cubatureWeights.array() >= 0).all(),
        "ECP requires one finite nonnegative weight per sampled cell");
    Eigen::VectorXd fitted = Eigen::VectorXd::Zero(ecpFeatures.cols());
    forAll(sampledCells, i)
    {
        fitted += cubatureWeights(i)*ecpFeatures.row(sampledCells[i]).transpose();
    }
    Eigen::VectorXd target = ecpFeatures.colwise().sum().transpose();
    Info << "ECP relative training error = " << (fitted-target).norm()/target.norm() << endl;
    Info << "ECP selected " << sampledCells.size() << " cells from " << nCells
         << "; weight sum = " << cubatureWeights.sum() << endl;

    // Export the rule actually used, independently of historical caches.
    const word activeRuleFolder("ITHACAoutput/12simpleSteadyNS_ECP/active");
    mkDir(activeRuleFolder);
    Eigen::VectorXi activeCells(sampledCells.size());
    forAll(sampledCells, i) activeCells(i) = sampledCells[i];
    cnpy::save(activeCells, activeRuleFolder + "/nodePoints.npy");
    cnpy::save(cubatureWeights, activeRuleFolder + "/quadratureWeights.npy");
    cnpy::save(ecpVolumes, activeRuleFolder + "/cellVolumes.npy");
    OFstream activeCache(activeRuleFolder + "/cache.txt");
    activeCache << (ecpNodes == nCells ? word("allCells") : ecpCacheFolder) << endl;

    OFstream metadata(activeRuleFolder + "/metadata");
    metadata << "FoamFile { version 2.0; format ascii; class dictionary; object metadata; }\n"
             << "nCells " << nCells << ";\nNmodesUproj " << NmodesUproj
             << ";\nNmodesPproj " << NmodesPproj << ";\necpNodes " << ecpNodes << ";\n";
    const word basisFolder("ITHACAoutput/12simpleSteadyNS_ECP/parallelBasis");
    ITHACAstream::exportSolution(example.liftfield[0], "0", basisFolder, "ecpLift0");
    for (label i = 0; i < NmodesUproj; ++i)
        ITHACAstream::exportSolution(example.Umodes[i], "0", basisFolder, "ecpU" + name(i));
    for (label i = 0; i < NmodesPproj; ++i)
        ITHACAstream::exportSolution(example.Pmodes[i], "0", basisFolder, "ecpP" + name(i));
    if (prepareOnly)
    {
        Info << "ECP parallel input prepared" << endl;
        return 0;
    }
    }

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
        &cubatureWeights
    );

    // ------------------------------------------------------------
    // Hyper-reduced online solutions.
    // Galerkin is intentionally used here to remain algebraically
    // consistent with the original solveOnline_Simple().
    // ------------------------------------------------------------
    const bool validate = para->ITHACAdict->lookupOrDefault<bool>("ecpValidate", false);
    reducedSimpleSteadyNS reference(example);
    bool validationPassed = true;
    const scalar validationTolerance = para->ITHACAdict->lookupOrDefault<scalar>
        ("ecpValidationTolerance", 0.05);
    M_Assert(validationTolerance > 0, "ecpValidationTolerance must be positive");
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
        validationPassed = validationPassed && reduced.lastSolveConverged;
        if (validate)
        {
            const Eigen::MatrixXd sampledU = Foam2Eigen::field2Eigen(example.U);
            const Eigen::MatrixXd sampledP = Foam2Eigen::field2Eigen(example.p);
            reference.setOnlineVelocity(vel);
            reference.solveOnline_Simple(mu_now, NmodesUproj, NmodesPproj, 0, 0,
                "./ITHACAoutput/ReconstructFullROM/");
            validationPassed = validationPassed && reference.lastSolveConverged;
            const Eigen::MatrixXd fullU = Foam2Eigen::field2Eigen(example.U);
            const Eigen::MatrixXd fullP = Foam2Eigen::field2Eigen(example.p);
            double uError = 0, uNorm = 0, pError = 0, pNorm = 0;
            forAll(cellVolumes, celli)
            {
                uError += cellVolumes[celli]*(sampledU.middleRows(3*celli, 3)
                    - fullU.middleRows(3*celli, 3)).squaredNorm();
                uNorm += cellVolumes[celli]*fullU.middleRows(3*celli, 3).squaredNorm();
                pError += cellVolumes[celli]*sqr(sampledP(celli)-fullP(celli));
                pNorm += cellVolumes[celli]*sqr(fullP(celli));
            }
            reduce(uError, sumOp<double>());
            reduce(uNorm, sumOp<double>());
            reduce(pError, sumOp<double>());
            reduce(pNorm, sumOp<double>());
            const double relU = std::sqrt(uError/std::max(uNorm, double(SMALL)));
            const double relP = std::sqrt(pError/std::max(pNorm, double(SMALL)));
            Info << "ECP validation mu=" << mu_now << ": relative L2 U=" << relU
                 << ", p=" << relP << endl;
            validationPassed = validationPassed && std::isfinite(relU)
                && std::isfinite(relP) && relU <= validationTolerance
                && relP <= validationTolerance;
        }
    }

    Info << "ECP test " << (validationPassed ? "PASSED" : "FAILED") << endl;
    return validationPassed ? 0 : 1;
}
