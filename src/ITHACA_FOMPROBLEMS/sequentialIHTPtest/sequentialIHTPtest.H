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
    sequentialIHTPtest
Description
    A general full order implementation of an inverse problem
SourceFiles
    sequentialIHTPtest.C
\*---------------------------------------------------------------------------*/

/// \file
/// Header file of the sequentialIHTPtest class.
/// \dir
/// Directory containing the header and source files for the sequentialIHTPtest class.


#ifndef sequentialIHTPtest_H
#define sequentialIHTPtest_H
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
#include "ITHACAregularization.H"
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
class sequentialIHTPtest: public laplacianProblem
{

    public:
        // Constructors
        sequentialIHTPtest();
        /// Construct with argc and argv
        sequentialIHTPtest(int argc, char* argv[]);
        //sequentialIHTPtest(int argc, char* argv[], scalar _diffusivity);
        virtual ~sequentialIHTPtest() {};

        ITHACAparameters* para;

        // Dummy variables to transform inverseLaplacianFoam into a class
        /// Temperature field
        autoPtr<volScalarField> _T;
        PtrList<volScalarField> Ttime;
        List<PtrList<volScalarField>> TbasisOld;
        List<PtrList<volScalarField>> TbasisNew;

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

        /// Time discretization (filled in the constructor)
        scalar startTime;
        scalar deltaTime;
        scalar endTime;
        label Ntimes;
        List<scalar> timeSteps;

        /// List of time at which I acquire the measurements (filled in readThermocouples())
        List<scalar> samplingTime;

        /// Time interval in between samples (read from thermocouplesDict)
        scalar timeSamplesDeltaT;

        /// First sampling time (read from thermocouplesDict)
        scalar timeSamplesT0;

        /// Number of time samples (filled in createThermocouples.H)
        label timeSamplesNum;

        /// Number of timesteps between two samples
	label NtimeStepsBetweenSamples;

        /// Time sample index
	label timeSampleI;

        /// Number of sampling steps to consider when computing offline phase
        label basisDeltaSample = 2; 

        /// List of timesteps at which measurements are available
        List<label> samplingSteps; 

        /// 1 if readThermocouples() was called, 0 elsewise
        bool thermocouplesRead = 0;

        /// Number of thermocouples
        int thermocouplesNum;

        /// Number of heat flux bases (both in time and space)
        int gBasisSize;

        /// Cost funtion  [K^2]
        double J;

        /// List of cost funtions  [K^2]
        Eigen::VectorXd Jlist;

        /// Thermal diffusivity [W/(m K)]
        double k;

        /// Heat transfer coefficient [W/(m2 K)]
        double H;

        /// Density [kg /m3]
        double density;

        /// Specific heat capacity [J/kg/K]
        double specificHeat;

        /// Folder where the offline solutions are saved
        word folderOfflineOld = "./ITHACAoutput/offlineParamBCold/";
        word folderOfflineNew = "./ITHACAoutput/offlineParamBCnew/";

        /// Number of timestep to solve for during offline phase
        label offlineTimestepsSize = 0;

        /// End time for the ofline computation
        scalar offlineEndTime = 0.0;

        bool offlineFlag = 0;

        scalar homogeneousBC = 0.0;
        List<scalar> homogeneousBCcoldSide;
        List<scalar> Tf; //temperature at coldSide [K]
        List<scalar> refGrad;
        List<scalar> valueFraction;

        /// Index of the hotSide patch
        label hotSide_ind;

        /// Index of the coldSide patch
        label coldSide_ind;

        /// Heat flux at hotSide
        List<List<scalar>> g;

        /// Heat flux at hotSide
        List<List<scalar>> gShort;

        /// Heat flux space basis
        List<List<scalar>> heatFluxSpaceBasis;

        /// Bases of the heat flux
        List<List<List<scalar>>> gBaseFunctionsOld;

        /// Bases of the heat flux
        List<List<List<scalar>>> gBaseFunctionsNew;

        /// Number of basis
        label Nbasis = 0;

        /// Number of basis in time
        label NbasisInTime = 0;

        /// Number of basis in space
        label NbasisInSpace = 0;

        /// Number of samples considered in the offline phase
        label NsamplesWindow = 0;

        /// Weights of the parameterization
        List<scalar> gWeightsOld;
        List<scalar> gWeightsNew;

        /// True heat flux at hotSide [W/m2]
        List<List<scalar>> gTrue;

        /// Parametrized BC
        Eigen::VectorXd residual;
        Eigen::VectorXd addSol;
        Eigen::VectorXd T0_vector;
        Eigen::VectorXd Tcomp;
        Eigen::MatrixXd ThetaOld;
        Eigen::MatrixXd ThetaNew;
	word timeBasisType = "None";


        List<vector> thermocouplesPos;
        List<int> thermocouplesCellID;
        List<int> thermocouplesCellProc;
        List<Foam::vector> thermocouplesCellC;
        Eigen::VectorXd Tmeas;   //Temperature at the thermocouples locations [K]
	Eigen::VectorXd TmeasShort; //Temperature at the thermocouples locations in only two sampling times [K]
        Eigen::VectorXd Tdirect;
        Eigen::VectorXd Tdiff;

        // Functions

        //--------------------------------------------------------------------------

        /// Set diffusivity
        void setDiffusivity(scalar _diff);

        //--------------------------------------------------------------------------

        /// Define the base functions used for the parametrization of g
        void setSpaceBasis(word type, scalar shapeParameter);

        //--------------------------------------------------------------------------

        /// Set parameterized heat flux defining the basis
        void set_gParametrized(word spaceBaseFuncType,
            scalar shapeParameter_space, 
            scalar shapeParameter_time = 0);

        //--------------------------------------------------------------------------

        /// Convert list of boundary heat flux into field
        volScalarField list2Field(List<scalar> list, scalar innerField = 0.0);

        //--------------------------------------------------------------------------

        /// Update the boundary condition g when g is parameterized
        ///
        /// @param[in]  weigths  New values of the weights of the basis functions
        ///
        void update_gParametrized(List<scalar> wOld, List<scalar> wNew);

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
        void reconstrucT(word outputFolder, volScalarField initialField);

        //--------------------------------------------------------------------------

        ///
        ///
        void parameterizedBC(word folder, PtrList<volScalarField> initialField,word linSys_solver = "fullPivLU",
                             label TSVD_filter = 0);

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
        virtual void solveT0(volScalarField initialField);

        //--------------------------------------------------------------------------

        /// Set BC and IF of the additional problem for the parameterized heat flux
        ///
        void solveAdditional();

        //--------------------------------------------------------------------------

        /// Solve direct problem
        ///
        void solveDirect();

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

        /// Restart temperature field
        ///
        void restart();

        //--------------------------------------------------------------------------

        /// Restart fields
        ///
        void restartOffline();

        //--------------------------------------------------------------------------

        /// Restart fields
        ///
        void restartT0();

        //--------------------------------------------------------------------------

        /// Fills the vector samplingSteps which contains the timesteps at which the measurements are taken
        ///
        void sampling2symulationTime();

        //--------------------------------------------------------------------------

        ///
        ///
        void parameterizedBC_postProcess(List<Eigen::MatrixXd> linSys,
                                         Eigen::VectorXd weigths, word outputFolder, volScalarField initialField, label verbose = 0);

};

#endif
