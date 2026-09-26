/*---------------------------------------------------------------------------*\
     ██╗████████╗██╗  ██╗ █████╗  ██████╗ █████╗       ███████╗██╗   ██╗
     ██║╚══██╔══╝██║  ██║██╔══██╗██╔════╝██╔══██╗      ██╔════╝██║   ██║
     ██║   ██║   ███████║███████║██║     ███████║█████╗█████╗  ██║   ██║
     ██║   ██║   ██╔══██║██╔══██║██║     ██╔══██║╚════╝██╔══╝  ╚██╗ ██╔╝
     ██║   ██║   ██║  ██║██║  ██║╚██████╗██║  ██║      ██║      ╚████╔╝
     ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝       ╚═══╝
 * In real Time Highly Advanced Computational Applications for Finite Volumes
 * Copyright (C) 2017 by the ITHACA-FV authors
-------------------------------------------------------------------------------
License
    This file is part of ITHACA-FV
\*---------------------------------------------------------------------------*/

#ifndef ReducedSimpleSteadyNS_H
#define ReducedSimpleSteadyNS_H

#include "fvCFD.H"
#include "IOmanip.H"
#include "ReducedProblem.H"
#include "SteadyNSSimple.H"
#include "ITHACAutilities.H"
#include "EigenFunctions.H"
#include <Eigen/Eigen>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>
#include "Modes.H"
#include "SampledMesh.H"
#include "SampledModes.H"

/*---------------------------------------------------------------------------*\
                   Class reducedSimpleSteadyNS Declaration
\*---------------------------------------------------------------------------*/

class reducedSimpleSteadyNS: public reducedProblem
{
    private:

        // Sampled/hyper-reduced objects are built once by setupSampled().
        SampledMesh* sampledMeshPtr_ = nullptr;

        autoPtr<SampledModes<vector, fvPatchField, volMesh>>
            sampledUmodes_;

        autoPtr<SampledModes<scalar, fvPatchField, volMesh>>
            sampledPmodes_;

        bool sampledReady_ = false;

        // Sampled Galerkin projection of the pressure-mode gradients.
        // Rows correspond to sampled velocity test modes, columns to pressure modes.
        Eigen::MatrixXd sampledProjGradModP_;

        // Optional full-ROM residual indicator used to generate HR samples.
        bool collectResidualIndicator_ = false;
        scalarField residualIndicator_;


    public:

        reducedSimpleSteadyNS();

        explicit reducedSimpleSteadyNS(SteadyNSSimple& problem);

        ~reducedSimpleSteadyNS() {};


        // ------------------------------------------------------------------
        // Standard full-mesh online SIMPLE ROM
        // ------------------------------------------------------------------
        void solveOnline_Simple
        (
            scalar mu_now,
            int NmodesUproj,
            int NmodesPproj,
            int NmodesNut = 0,
            int NmodesSup = 0,
            word Folder = "./ITHACAoutput/Reconstruct/"
        );


        // ------------------------------------------------------------------
        // Hyper-reduced SIMPLE ROM
        //
        // setupSampled() performs all mode interpolation only once.
        // solveOnline_SimpleSampled() assembles UEqn and pEqn directly on
        // sampledMesh.subMesh() and uses Modes::projectSampled().
        // ------------------------------------------------------------------

        void setupSampled
        (
            SampledMesh& sampledMesh,
            int NmodesUproj,
            int NmodesPproj,
            int NmodesSup = 0
        );

        void solveOnline_SimpleSampled
        (
            scalar mu_now,
            int NmodesNut = 0,
            word Folder = "./ITHACAoutput/ReconstructHR/",
            word projType = "G"
        );



        // Enable/disable accumulation of a normalized cell-wise residual
        // indicator during the standard full-mesh SIMPLE ROM.
        void setCollectResidualIndicator(const bool enable = true)
        {
            collectResidualIndicator_ = enable;
        }

        const scalarField& residualIndicator() const
        {
            return residualIndicator_;
        }

        void writeResidualIndicator
        (
            const word& fieldName = "hrResidualIndicator"
        ) const;

        /// Set inlet velocity for the online solve.
        void setOnlineVelocity(Eigen::MatrixXd vel);


        /// Lifted velocity modes.
        volVectorModes ULmodes;

        /// Full problem.
        SteadyNSSimple* problem;

        /// Projected gradient of the pressure modes (standard solver).
        Eigen::MatrixXd projGradModP;

        /// Imposed boundary conditions.
        Eigen::MatrixXd vel_now;

        /// Maximum iterations number for the online step.
        int maxIterOn = 1000;

        /// Counter.
        int counter = 0;

        int UprojN;
        int PprojN;
};

#endif
