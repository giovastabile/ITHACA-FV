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
    reducedSteadyNS
Description
    A reduced problem for the stationary NS equations
SourceFiles
    reducedSteadyNS.C
\*---------------------------------------------------------------------------*/

/// \file
/// Header file of the reducedSteadyNS class.
/// \dir
/// Directory containing the header and source files for the reducedSteadyNS class

#ifndef NonlinearReducedBurgers_H
#define NonlinearReducedBurgers_H

#include <torch/script.h>
#include <torch/torch.h>
#include "torch2Eigen.H"
#include "torch2Foam.H"

#include "Foam2Eigen.H"
#include "fvCFD.H"
#include "IOmanip.H"
#include "ReducedProblem.H"
#include "burgers.H"
#include "ITHACAutilities.H"
#include "EigenFunctions.H"
#include <Eigen/Eigen>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>
#include <torch/script.h>
#include <utility>

class Embedding
{
public:
    /// latent dimension
    int latent_dim;

    /// number of cells
    int output_dim;

    /// decoder module
    autoPtr<torch::jit::script::Module> decoder;

    /// Initial condition
    autoPtr<volVectorField> _U0;

    /// value of the decoder at the zero vector
    autoPtr<volVectorField> _g0;

    ///
    Eigen::MatrixXd latent_initial;

    /// DEBUG commodity counter
    int counter = 0;

    Embedding() {Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.H, line 73 #################### " << endl;}
    Embedding(int latent_dim, fileName decoder_path, volVectorField &U0, Eigen::MatrixXd lat_init);

    Embedding(Embedding &embedding) : latent_dim{embedding.latent_dim},
                                      output_dim{embedding.output_dim},
                                      decoder{embedding.decoder},
                                      _U0{embedding._U0},
                                      _g0{embedding._g0}
    {
        Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.H, line 81 #################### " << endl;
    }

    Embedding(Embedding &&embedding) : latent_dim{std::move(embedding.latent_dim)},
                                       output_dim{embedding.output_dim},
                                      decoder{std::move(embedding.decoder)},
                                      _U0{std::move(embedding._U0)},
                                      _g0{std::move(embedding._g0)}
    {
        Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.H, line 81 #################### " << endl;
    }

    autoPtr<volVectorField> forward(const Eigen::VectorXd &x, const scalar mu);
    autoPtr<Eigen::MatrixXd> jacobian(const Eigen::VectorXd &x, const scalar mu);

    std::pair<autoPtr<volVectorField>, autoPtr<Eigen::MatrixXd>> forward_with_gradient(const Eigen::VectorXd &x, const scalar mu);

    autoPtr<volVectorField> embedding_ref(const scalar mu);

    ~Embedding(){Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.H, line 96 #################### " << endl;};
};

/// Structure to implement a newton object for a non-stationary NS problem
struct newton_nmlspg_burgers : public newton_argument<double>
{
public:
    newton_nmlspg_burgers() {}
    newton_nmlspg_burgers(int Nx, int Ny, Burgers &problem, Embedding &embedding, volVectorField& field) :
    newton_argument<double>(Nx, Ny),
    problem(&problem),
    N_BC(problem.inletIndex.rows()),
    embedding(&embedding)
    {

        if (problem.liftfield.size() != 0)
        {
            Nphi_u = problem.NUmodes + problem.liftfield.size();
        }
        else
        {
            Nphi_u = problem.NUmodes + 1;
        }
    }

    int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const;
    int df(const Eigen::VectorXd &x, Eigen::MatrixXd &fjac) const;

    Burgers *problem;
    int Nphi_u;
    int N_BC;
    dimensionedScalar nu;
    scalar dt;
    scalar mu;

    Embedding *embedding;

    Eigen::VectorXd y_old;
    Eigen::VectorXd yOldOld;
    Eigen::VectorXd BC;
    Eigen::MatrixXd tauU;
};

/*---------------------------------------------------------------------------*\
                        Class reducedProblem Declaration
\*---------------------------------------------------------------------------*/

/// Class where it is implemented a reduced problem for the <b> steady Navier-stokes </b> problem.
/** In this class are implemented the methods to solve a reduced version of the stationary Navier-stokes
equations. This class must be used together with the steadyNS class */
class NonlinearReducedBurgers : public reducedProblem
{
private:
public:
    // Constructors
    /// Construct Null
    NonlinearReducedBurgers();

    /// Construct Null
    ///
    /// @param[in]  problem  a full order steadyNS problem
    /// @param[in]  tipo     Type of pressure stabilisation method you want to use "SUP" for supremizer, "PPE" for pressure Poisson equation.
    ///
    explicit NonlinearReducedBurgers(Burgers &problem, fileName decoder_path, int latent_dim, Eigen::MatrixXd lat_init);

    ~NonlinearReducedBurgers(){};

    /// parameters to be read from the ITHACAdict file
    ITHACAparameters *para;

    /// Decoder
    autoPtr<Embedding> embedding;

    /// Online inlet velocity vector
    Eigen::MatrixXd vel_now;

    /// Reduced matrix for tangent forces
    Eigen::MatrixXd fTau;

    /// Reduced matrix for normal forces
    Eigen::MatrixXd fN;

    /// Vector to store the previous solution during the Newton procedure
    Eigen::VectorXd y_old;

    /// Vector to store the solution during the Newton procedure
    Eigen::VectorXd y;

    /// Reduced viscosity in case of parametrized viscosity
    dimensionedScalar nu;

    /// List of Eigen matrices to store the online solution
    List<Eigen::MatrixXd> online_solution;

    /// List of pointers to store the modes for velocity
    PtrList<volVectorField> Umodes;

    /// List of pointers to store the snapshots for velocity
    PtrList<volVectorField> Usnapshots;

    /// Recontructed velocity field
    PtrList<volVectorField> UREC;

    /// Recontructed velocity fields list
    PtrList<volVectorField> uRecFields;

    /// Newton object used to solve the non linear problem
    newton_nmlspg_burgers newton_object;

    /// Pointer to the FOM problem
    Burgers *problem;

    /// Number of velocity modes
    int Nphi_u;

    /// Number of parametrized boundary conditions
    int N_BC;

    /// Counter to count the online solutions
    int count_online_solve = 1;

    /// Penalty Factor
    Eigen::MatrixXd tauU;

    /// Scalar to store the current time
    scalar time;

    /// Scalar to store the time increment
    double dt;

    /// Scalar to store the final time if the online simulation
    scalar finalTime;

    /// Scalar to store the initial time if the online simulation
    scalar tstart;

    /// A variable for storing the reduced coefficients
    double storeEvery;

    /// A variable for exporting the fields
    double exportEvery;

    /// A variable for starting solving the reduced system from zero or not
    bool startFromZero = false;

    /// Number of timesteps calculated for the iterative penalty method
    int timeStepPenalty = 1;

    /// Penalty Factor determined with iterative solver
    Eigen::MatrixXd tauIter;

    /// Maximum number of iterations to be done for the computation of the penalty factor
    scalar maxIterPenalty;

    /// Tolerance for the residual of the boundary values, there is the same tolerance for velocity and temperature
    scalar tolerancePenalty;

    // Functions

    /// Method to determine the penalty factors iteratively.
    ///
    /// @param[in]  vel_now   The vector of online velocity. It is defined in
    /// with an Eigen::MatrixXd and must have one col or as many cols as timesteps to be performed
    /// in case of time-dependent BCs and as many rows as the number of parametrized boundary conditions.
    ///
    /// @param[in]  tauIter   The penalty factors determined one iteration step
    ///
    /// @param[in]  startSnap The first snapshot taken from the offline snapshots
    /// and used to get the reduced initial condition.
    ///
    Eigen::MatrixXd penalty(Eigen::MatrixXd &vel_now, Eigen::MatrixXd &tauIter,
                            int startSnap = 0);

    /// Method to perform an online solve using a supremizer stabilisation method
    ///
    /// @param[in]  vel_now   The vector of online velocity. It is defined in
    /// with an Eigen::MatrixXd and must have one col and as many rows as the number
    /// of parametrized boundary conditions.
    /// @param[in]  startSnap The first snapshot taken from the offline snapshots
    /// and used to get the reduced initial condition.
    ///
    //void solveOnline(Eigen::MatrixXd vel_now, int startSnap = 0);

    /// Method to perform an online solve using a supremizer stabilisation method
    ///
    /// @param[in]  vel_now   The vector of online velocity. It is defined in
    /// with an Eigen::MatrixXd and must have one col and as many rows as the number
    /// of parametrized boundary conditions.
    /// @param[in]  startSnap The first snapshot taken from the offline snapshots
    /// and used to get the reduced initial condition.
    ///
    void solveOnline(int startSnap = 0);

    /// Method to perform an online solve using a supremizer stabilisation method
    ///
    /// @param[in]  vel_now   The vector of online velocity. It is defined in
    /// with an Eigen::MatrixXd and must have one col and as many rows as the number
    /// of parametrized boundary conditions.
    /// @param[in]  startSnap The first snapshot taken from the offline snapshots
    /// and used to get the reduced initial condition.
    ///
    void solveOnline(Eigen::MatrixXd mu, int startSnap = 0);

    ///
    /// Method to reconstruct the solutions from an online solve.
    ///
    /// @param[in]  exportFields  A boolean variable which determines whether to export fields or not
    /// @param[in]  folder        The folder where to output the solutions in case on wants to
    /// @param[in]  printevery    The printevery variabel, when it is 1 the function outputs all stored fields
    ///
    void reconstruct(bool exportFields = false,
                     fileName folder = "./ITHACAoutput/online_rec");

    ///
    /// Method to reconstruct the solutions from an online solve.
    ///
    /// @param[in]  exportFields  A boolean variable which determines whether to export fields or not
    /// @param[in]  folder        The folder where to output the solutions in case on wants to
    /// @param[in]  printevery    The printevery variabel, when it is 1 the function outputs all stored fields
    ///
    void reconstruct(bool exportFields, fileName folder, Eigen::MatrixXd redCoeff);

    ///
    /// @brief      Sets the online velocity.
    ///
    /// @param[in]  vel   The velocity
    ///
    /// @return     The rescaled velocity vector
    ///
    Eigen::MatrixXd setOnlineVelocity(Eigen::MatrixXd vel);

    ///
    /// @brief      Projects and reconstructs the snapshots in Umodes.
    ///
    /// @param[in]  folder     The folder where to output the solution
    ///
    void trueProjection(fileName folder);
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif
