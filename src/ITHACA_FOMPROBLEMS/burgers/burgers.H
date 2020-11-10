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
Class
    Burgers
Description
    Reduction class for Burgers' equation
SourceFiles
    burgers.C
\*---------------------------------------------------------------------------*/

/// \file
/// Header file of the Burgers class.
/// \dir
/// Directory containing the header and source files for the Burgers class.

#ifndef burgers_H
#define burgers_H
#include "fvCFD.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include "simpleControl.H"
#include "pisoControl.H"
#include "fvOptions.H"
#include "reductionProblem.H"
#include "ITHACAstream.H"
#include "ITHACAparameters.H"
#if OPENFOAM >= 1812
#include "ITHACAforces18.H"
#else
#include "ITHACAforces.H"
#endif
#include "volFields.H"
#include "IOMRFZoneList.H"
#include <iostream>
#include "IPstream.H"
#include "OPstream.H"
#include "Modes.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

/*---------------------------------------------------------------------------*\
                        Class Burgers Declaration
\*---------------------------------------------------------------------------*/

/// Implementation of a parametrized full order <b> unsteady NS problem </b> and preparation of the the reduced matrices for the online solve.
/** In this class are implemented the methods for the offline solve of a unsteady NS problem
and the for the generation of the reduced matrices for subsequent online solve, this class is a son
of the Burgers class */
class Burgers: public reductionProblem
{


    public:
        // Constructors
        /// Null constructor
        Burgers();

        /// Construct with argc and argv
        Burgers(int argc, char* argv[]);
        ~Burgers() {};

        ITHACAparameters* para;

        // Member Functions
        /// List of pointers used to form the velocity snapshots matrix
        PtrList<volVectorField> Ufield;

        /// List of pointers used to form the velocity modes
        volVectorModes Umodes;

        /// List of pointers used to form the list of lifting functions
        PtrList<volVectorField> liftfield;

        /// List of pointers used to form the homogeneous velocity snapshots
        PtrList<volVectorField> Uomfield;

        /// List of pointers containing the total number of lift, supremizer and velocity modes
        volVectorModes L_Umodes;

        /// Tolerance for the residual of the stationary problems, there is the same tolerance for velocity and pressure
        scalar tolerance;

        /// Number of maximum iterations to be done for the computation of the truth solution
        scalar maxIter;

        /// Number of velocity modes to be calculated
        label NUmodesOut;

        /// Number of velocity modes used for the projection
        label NUmodes;

        // Reduced Matrices

        /// Diffusion term
        Eigen::MatrixXd B_matrix;

        /// Mass Matrix
        Eigen::MatrixXd M_matrix;

        /// Non linear term
        List <Eigen::MatrixXd> C_matrix;
        Eigen::Tensor<double, 3 > C_tensor;

        /// Boundary term for penalty method - vector
        List <Eigen::MatrixXd> bcVelVec;

        /// Boundary term for penalty method - matrix
        List <Eigen::MatrixXd> bcVelMat;

        // Other Variables
        /// Counter used for the output of the full order solutions
        label counter = 1;

        /// Velocity field
        autoPtr<volVectorField> _U;

        /// Initial Velocity field (for restart purposes)
        autoPtr<volVectorField> _U0;

        /// Viscosity
        autoPtr<dimensionedScalar> _nu;//CHECK


        /// Mesh
        mutable autoPtr<fvMesh> _mesh;

        /// simpleControl
        autoPtr<simpleControl> _simple;

        /// fvOptions
        autoPtr<fv::options> _fvOptions;

        /// Time
        autoPtr<Time> _runTime;

        /// Flux
        autoPtr<surfaceScalarField> _phi;

        /// Initial Flux (for restart purposes)
        autoPtr<surfaceScalarField> _phi0;

        /// MRF variable
        autoPtr<IOMRFZoneList> _MRF;

        /// continuity error
        scalar cumulativeContErr = 0;

        /// Boundary Method
        word bcMethod;

        // Specific variable for the unstationary case
        /// Start Time (initial time to start storing the snapshots)
        scalar startTime;

        /// Final time (final time of the simulation and consequently of the acquisition of the snapshots)
        scalar finalTime;

        /// Time step of the simulation
        scalar timeStep;

        /// Time step of the writing procedure
        scalar writeEvery = timeStep;

        /// Auxiliary variable to store the next writing instant
        scalar nextWrite;

        /// adjustTimeStep
        bool adjustTimeStep;

        /// maxCourant
        scalar maxCo;

        /// maxDeltaT
        scalar maxDeltaT;

        /// Time-dependent Boundary Method
        word timedepbcMethod;

        // Matrix containing the time-dependent boundary conditions for each time step
        Eigen::MatrixXd timeBCoff;

        // Time derivative numerical scheme order
        word timeDerivativeSchemeOrder;


        // Functions

        //--------------------------------------------------------------------------
        /// @brief      Perform a truthsolve
        /// @param[in]  mu_now  The actual value of the parameter for this truthSolve. Used only
        /// to construct mu_interp matrix which is written out in the Offline folder, also for par file in
        /// the Parameters folder.
        ///
        void truthSolve(List<scalar> mu_now, fileName folder = "./ITHACAoutput/Offline/");

        /// Perform a lift solve
        void liftSolve();

        //--------------------------------------------------------------------------
        //  Projection Methods Momentum Equation
        /// Diffusive Term
        ///
        /// @param[in]  NUmodes    The number of velocity modes.
        /// @param[in]  NPmodes    The number of pressure modes.
        /// @param[in]  NSUPmodes  The number of supremizer modes.
        ///
        /// @return     reduced matrix in Eigen::MatrixXd format for the diffusion term.
        ///
        Eigen::MatrixXd diffusive_term(label NUmodes);

        //--------------------------------------------------------------------------
        /// Convective Term
        ///
        /// @param[in]  NUmodes    The number of velocity modes.
        /// @param[in]  NPmodes    The number of pressure modes.
        /// @param[in]  NSUPmodes  The number of supremizer modes.
        ///
        /// @return     reduced third order tensor in List <Eigen::MatrixXd> format for the convective term.
        ///
        List < Eigen::MatrixXd >  convective_term(label NUmodes);

        //--------------------------------------------------------------------------
        /// @brief      Export convective term as a tensor
        ///
        /// @param[in]  NUmodes    The N of velocity modes
        /// @param[in]  NPmodes    The N of pressure modes
        /// @param[in]  NSUPmodes  The N of supremizer modes
        ///
        /// @return     tensor_C
        ///
        Eigen::Tensor<double, 3 > convective_term_tens(label NUmodes);

        //--------------------------------------------------------------------------
        /// Mass Term
        ///
        /// @param[in]  NUmodes    The number of velocity modes.
        /// @param[in]  NPmodes    The number of pressure modes.
        /// @param[in]  NSUPmodes  The number of supremizer modes.
        ///
        /// @return     reduced matrix in Eigen::MatrixXd format for the mass matrix.
        ///
        Eigen::MatrixXd  mass_term(label NUmodes);

        //--------------------------------------------------------------------------
        /// @brief      Boundary integral modes on boundary used by the penaly method
        ///
        /// @param[in]  NUmodes    The number of velocity modes.
        /// @param[in]  NSUPmodes  The number of supremizer modes.
        ///
        /// @return     reduced third order tensor in List <Eigen::MatrixXd> format for the penalty method.
        ///
        List< Eigen::MatrixXd > bcVelocityVec(label NUmodes);

        //--------------------------------------------------------------------------
        /// @brief      Boundary integral modes on boundary used by the penaly method
        ///
        /// @param[in]  NUmodes    The number of velocity modes.
        /// @param[in]  NSUPmodes  The number of supremizer modes.
        ///
        /// @return     reduced third order tensor in List <Eigen::MatrixXd> format for the penalty method.
        ///
        List< Eigen::MatrixXd > bcVelocityMat(label NUmodes);

        //--------------------------------------------------------------------------
        /// Function to change the viscosity
        ///
        /// @param[in]  mu    viscosity (scalar)
        ///
        void change_viscosity(double mu);

        /// set U and P back to the values into the 0 folder
        void restart();

        //--------------------------------------------------------------------------
        /// Function to check if the solution must be exported.
        ///
        /// @param      timeObject  The time object of OpenFOAM.
        ///
        /// @return     1 if we must write 0 elsewhere.
        ///
        bool checkWrite(Time& timeObject);
};

#endif











