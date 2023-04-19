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
    sequentialIHTPwithTimeInterpolation
Description
    A general full order implementation of an inverse problem
SourceFiles
    sequentialIHTPwithTimeInterpolation.C
\*---------------------------------------------------------------------------*/

/// \file
/// Header file of the sequentialIHTPwithTimeInterpolation class.
/// \dir
/// Directory containing the header and source files for the sequentialIHTPwithTimeInterpolation class.


#ifndef sequentialIHTPwithTimeInterpolation_H
#define sequentialIHTPwithTimeInterpolation_H
#include <iostream>
#include "fvCFD.H"
#include "interpolation.H"
#include "fvOptions.H"
#include "simpleControl.H"
#include "IOmanip.H"
#include "Time.H"
#include "laplacianProblem.H"
// #include "reducedLaplacian.H"
#include "ITHACAPOD.H"
#include "ITHACAutilities.H"
#include <Eigen/Dense>
#include <cmath>
#include "Foam2Eigen.H"
#include "cnpy.H"
#include "mixedFvPatchFields.H"
#include "cellDistFuncs.H"
#include "reductionProblem.H"
#include "Modes.H"
#define _USE_MATH_DEFINES

using namespace SPLINTER;

/// Class to implement a full order inverse parameterized laplacian problem
class sequentialIHTPwithTimeInterpolation: public laplacianProblem
{

    public:
        // Constructors
        sequentialIHTPwithTimeInterpolation();
        /// Construct with argc and argv
        sequentialIHTPwithTimeInterpolation(int argc, char* argv[]);
        //sequentialIHTPwithTimeInterpolation(int argc, char* argv[], scalar _diffusivity);
        virtual ~sequentialIHTPwithTimeInterpolation() {};

        ITHACAparameters* para;

        // Dummy variables to transform inverseLaplacianFoam into a class
        /// Temperature field
        autoPtr<volScalarField> _T;
        PtrList<volScalarField> Ttime;
        List<PtrList<volScalarField>> Tbasis;

        /// Additional temperature field
        autoPtr<volScalarField> _Tad;
        PtrList<volScalarField> Tad_time;
        scalar T0;
        PtrList<volScalarField> T0_time;

        /// Mesh
        mutable autoPtr<fvMesh> _mesh;

        /// simpleControl
        autoPtr<simpleControl> _simple;

        /// fvOptions
        autoPtr<fv::options> _fvOptions;

        /// Time
        autoPtr<Time> _runTime;
        autoPtr<Time> _runTimeShort;

        /// Dummy thermal diffusivity with unitary value
        dimensionedScalar DT;

        /// Diffusivity value
	scalar diffusivity = 0.0; 

        /// List of snapshots for the solutions
        PtrList<volScalarField> Tfield;

        /// List of POD modes
        volScalarModes Tmodes;

        /// Number of POD modes
        int NmodesT;

        /// Time discretization
        scalar startTime;
        scalar deltaTime;
        scalar endTime;
        label Ntimes;
        List<scalar> timeSteps;

        /// List of time at which I acquire the measurements
        List<scalar> samplingTime;
        scalar timeSamplesDeltaT;
        scalar timeSamplesT0;
        label timeSamplesNum;
	label NtimeStepsBetweenSamples;
	label NtimeBasis; // The number of basis used is NtimeStepsBetweenSamples + 1
	label timeSampleI;
        label basisDeltaSample = 2; //Numer of sampling steps to consider when computing offline phase
        List<label> samplingSteps; //symulation timesteps at which there are mesurements


        int interpolation;
        int offlineSolutionI;
        int offlinePhase;
        int thermocouplesRead;
        int thermocouplesNum;
        int gBasisSize;
        label saveSolInLists = 0;
        double J; //cost funtion  [K^2]
        Eigen::MatrixXd Jlist; //cost funtion  [K^2]
        double gradJ_L2norm;
        double Jtol;
        double JtolRel;
        double k;  //thermal diffusivity [W/(m K)]
        double H;  //heat transfer coefficient [W/(m2 K)]
        double density; // [kg /m3]
        double specificHeat; // [J/kg/K]
        double L2norm;
        double LinfNorm;
        double gamma;
        double gamma_den;
        double beta;
        double castWidth = 0;
        word folderOffline = "./ITHACAoutput/offlineParamBC/";

        scalar v = 0.0;
        scalar homogeneousBC = 0.0;
        List<scalar> homogeneousBCcoldSide;
        List<scalar> Tf; //temperature at coldSide [K]
        List<scalar> refGrad;
        List<scalar> valueFraction;
        List<scalar> valueFractionAdj;

        label nProcs;
        label hotSide_ind;
        label coldSide_ind;

        List<List<scalar>>
                        g;           //Heat flux at hotSide for the direct problem [W/m2]

        List<List<List<scalar>>> gBaseFunctions;
        List<scalar> gWeights;
        List<List<scalar>> gTrue;       //True heat flux at hotSide [w/m2]
        List<scalar> faceCellArea; //[m2]
        List<scalar> err;         //relative error []
        List<scalar> gradJ;

        // Parametrized BC
        Eigen::VectorXd residual;
        Eigen::VectorXd addSol;
        Eigen::VectorXd T0_vector;
        Eigen::VectorXd Tcomp;
        Eigen::MatrixXd Theta;
        Eigen::MatrixXd gPODmodes;
	word timeBasisType = "None";


        List<vector> thermocouplesPos;
        List<int> thermocouplesCellID;
        List<int> thermocouplesCellProc;
        List<Foam::vector> thermocouplesCellC;
        Eigen::VectorXd Tmeas;   //Temperature at the thermocouples locations [K]
	Eigen::VectorXd TmeasShort; //Temperature at the thermocouples locations in only two sampling times [K]
        Eigen::VectorXd Tdirect;
        Eigen::VectorXd Tdiff;

        // Reduced Matrices
        /// A matrices
        List<Eigen::MatrixXd> ArbT;


        // Functions

        //--------------------------------------------------------------------------

        /// Set diffusivity
        void setDiffusivity(scalar _diff);

        //--------------------------------------------------------------------------

        /// Set initial heat flux for the conjugate gradient method
        void set_g();

        //--------------------------------------------------------------------------

        /// Define the base functions used for the parametrization of g
        void set_gBaseFunctions(word type, scalar shapeParameter_space);

        //--------------------------------------------------------------------------

        /// Define the base functions used for the parametrization of g
        void set_gBaseFunctions(word type, scalar shapeParameter_space, scalar shapeParameter_time);

        //--------------------------------------------------------------------------

        /// Set parameterized heat flux defining the basis
        void set_gParametrized(word baseFuncType, scalar shapeParameter_space, scalar shapeParameter_time = 0.0);

        //--------------------------------------------------------------------------

        /// Convert list of boundary heat flux into field
        volScalarField list2Field(List<scalar> list, scalar innerField = 0.0);

        //--------------------------------------------------------------------------

        /// Update the boundary condition g when g is parameterized
        ///
        /// @param[in]  weigths  New values of the weights of the basis functions
        ///
        void update_gParametrized(List<scalar> weights);

        //--------------------------------------------------------------------------

        ///
        ///
        Eigen::VectorXd TSVD(Eigen::MatrixXd A, Eigen::MatrixXd b, label filter);

        //--------------------------------------------------------------------------

        /// Performs offline computation for the parameterized BC method, if
        /// the offline directory ""./ITHACAoutputs/offlineParamBC" exists,
        /// it reads the solution from there
        ///
        /// @param[in]  force   If 1, force the offline phase to be computed
        ///
        void parameterizedBCoffline(bool force = 0);

        //--------------------------------------------------------------------------

        /// Reconstructs the temperature field using superposition of effects
        ///
        void reconstrucT(word outputFolder);

        //--------------------------------------------------------------------------

        ///
        ///
        void parameterizedBC(word folder, word linSys_solver = "fullPivLU",
                             label TSVD_filter = 0);

        //--------------------------------------------------------------------------

        /// interpolating Tmeas in between the measurements
	//
	/// @param[in]  timeSampleI Index of the first of the two sampling time to interpolate in between 
	Eigen::VectorXd TmeasInterpolation(label timeSampleI);

        //--------------------------------------------------------------------------

        /// Set valueFraction list values for Robin condition
        void set_valueFraction();

        //--------------------------------------------------------------------------

        /// Set BC of the direct problem
        virtual void assignDirectBC(label timeI);

        //--------------------------------------------------------------------------

        /// Set IF of the T0 problem
        ///
        virtual void assignT0_IF(volScalarField& T0_field) = 0;

        //--------------------------------------------------------------------------

        /// Solve the T0 problem
        ///
        void solveT0();

        //--------------------------------------------------------------------------

        /// Set BC and IF of the additional problem for the parameterized heat flux
        ///
        void solveAdditional();

        //--------------------------------------------------------------------------

        /// Solve direct problem
        ///
        void solveDirect(label offline = 0);

        //--------------------------------------------------------------------------

        /// Identifies in the mesh the cells corresponding to the
        /// termocouples locations.
        ///
        virtual void readThermocouples();

        //--------------------------------------------------------------------------

        /// Interpolates the field value at the thermocouples points
        /// NOTE: do NOT call whe field is an element of a PtrList
        ///
        /// @param[in]  field    Field to read the values
        ///
        /// @param[in]  Vector of field values at thermocouples points
        ///
        Eigen::VectorXd fieldValueAtThermocouples(volScalarField& field);

        //--------------------------------------------------------------------------

        /// Interpolates the field value at the thermocouples points
        /// Always use thisone for elementes of PtrList
        ///
        /// @param[in]  field    Field to read the values
        ///
        /// @param[in]  Vector of field values at thermocouples points
        ///
        Eigen::VectorXd fieldValueAtThermocouples(PtrList<volScalarField> fieldList,
                label fieldI);

        //--------------------------------------------------------------------------

        /// Interpolates the field value at the thermocouples points
        ///
        /// @param[in]  field    Field to read the values
        ///
        /// @param[in]  Vector of field values at thermocouples points
        ///
        Eigen::VectorXd fieldValueAtThermocouples(PtrList<volScalarField> fieldList);

        //--------------------------------------------------------------------------

        /// Restart fields
        ///
        void restart(word fieldName = "all");

        //--------------------------------------------------------------------------

        /// Restart fields
        ///
        void restartOffline();

        //--------------------------------------------------------------------------

        /// Fills the vector samplingSteps which contains the timesteps at which the measurements are taken
        ///
        void sampling2symulationTime();

        //--------------------------------------------------------------------------

        ///
        ///
        void parameterizedBC_postProcess(List<Eigen::MatrixXd> linSys,
                                         Eigen::VectorXd weigths, word outputFolder, label verbose = 0);

};

#endif
