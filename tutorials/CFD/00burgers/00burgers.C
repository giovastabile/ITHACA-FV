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
    Example of a Burgers' Problem
SourceFiles
    00burgers.C
\*---------------------------------------------------------------------------*/

#include "burgers.H"
#include "ITHACAPOD.H"
#include "ReducedBurgers.H"
#include "ITHACAstream.H"
#include "Foam2Eigen.H"
#include "cnpy.H"
#include <chrono>
#include <math.h>
#include <iomanip>
#include <string>

class tutorial00 : public Burgers
{
public:
    explicit tutorial00(int argc, char *argv[])
        : Burgers(argc, argv),
          U(_U())
    {
    }

    // Fields To Perform
    volVectorField &U;

    void offlineSolveViscosity()
    {
        List<scalar> mu_now(1);

        if (offline)
        {
            ITHACAstream::read_fields(Ufield, U, "./ITHACAoutput/Offline/");
        }
        else
        {
            for (label i = 0; i < mu.cols(); i++)
            {
                mu_now[0] = mu(0, i);
                change_viscosity(mu(0, i));
                truthSolve(mu_now);
            }
        }
    }

    void offlineSolveInitialVelocity()
    {
        List<scalar> mu_now(1);

        if (offline)
        {
            ITHACAstream::read_fields(Ufield, U, "./ITHACAoutput/Offline/");
        }
        else
        {
            for (label i = 0; i < mu.cols(); i++)
            {
                mu_now[0] = mu(0, i);
                change_initial_velocity(mu(0, i));
                truthSolve(mu_now);
            }
        }
    }
};

/*---------------------------------------------------------------------------*\
                               Starting the MAIN
\*---------------------------------------------------------------------------*/

void one_parameter_viscosity(tutorial00);
void one_parameter_initial_velocity(tutorial00);

int main(int argc, char *argv[])
{
    // Construct the tutorial00 object
    tutorial00 example(argc, argv);
    one_parameter_initial_velocity(example);

    // if (argv[1] == std::string("oneinitial"))
    // {
    //     one_parameter_initial_velocity(example);
    // }
    // else if (argv[1] == std::string("oneviscosity"))
    // {
    //     one_parameter_viscosity(example);
    // }
    // else
    // {
    //     std::cout << "Pass oneviscosity or oneinitial as arguments." << endl;
    // }



    exit(0);
}

void one_parameter_viscosity(tutorial00 example)
{
    // Read parameters from ITHACAdict file
    ITHACAparameters *para = ITHACAparameters::getInstance(example._mesh(),
                                                           example._runTime());
    int NmodesUout = para->ITHACAdict->lookupOrDefault<int>("NmodesUout", 15);
    int NmodesUproj = para->ITHACAdict->lookupOrDefault<int>("NmodesUproj", 10);

    /// Set the number of parameters
    example.Pnumber = 1;
    /// Set the dimension of the training set
    example.Tnumber = NmodesUout;
    /// Instantiates a void Pnumber-by-Tnumber matrix mu for the parameters and a void
    /// Pnumber-by-2 matrix mu_range for the ranges
    example.setParameters();
    // Set the parameter ranges
    example.mu_range(0, 0) = 0.0001;
    example.mu_range(0, 1) = 0.01;
    // Generate a number of Tnumber linearly equispaced samples inside the parameter range
    example.genEquiPar();

    // Time parameters
    example.startTime = 0;
    example.finalTime = 2;
    example.timeStep = 0.001;
    example.writeEvery = 0.1;

    // Perform The Offline Solve;
    example.offlineSolveViscosity();

    // Perform a POD decomposition for velocity and pressure
    ITHACAPOD::getModes(example.Ufield, example.Umodes, example._U().name(),
                        example.podex, 0, 0, NmodesUout);

    example.project("./Matrices", NmodesUproj);

    ReducedBurgers reduced(example);

    // Set values of the reduced model
    reduced.nu = 0.0001;
    reduced.tstart = 0;
    reduced.finalTime = 2;
    reduced.dt = 0.001;
    reduced.storeEvery = 0.005;
    reduced.exportEvery = 0.1;

    reduced.solveOnline(1);

    // Reconstruct the solution and export it
    reduced.reconstruct(true, "./ITHACAoutput/Reconstruction/");
}

void one_parameter_initial_velocity(tutorial00 example)
{
    // Read parameters from ITHACAdict file
    ITHACAparameters *para = ITHACAparameters::getInstance(example._mesh(),
                                                           example._runTime());
    int NmodesUout = para->ITHACAdict->lookupOrDefault<int>("NmodesUout", 15);
    int NmodesUproj = para->ITHACAdict->lookupOrDefault<int>("NmodesUproj", 10);

    /// Set the number of parameters
    example.Pnumber = 1;
    /// Set the dimension of the training set
    example.Tnumber = NmodesUout;
    /// Instantiates a void Pnumber-by-Tnumber matrix mu for the parameters and a void
    /// Pnumber-by-2 matrix mu_range for the ranges
    example.setParameters();
    // Set the parameter ranges
    example.mu_range(0, 0) = 0.8;
    example.mu_range(0, 1) = 1.2;
    // Generate a number of Tnumber linearly equispaced samples inside the parameter range
    example.genEquiPar();

    // Time parameters
    example.startTime = 0;
    example.finalTime = 2;
    example.timeStep = 0.001;
    example.writeEvery = 0.0013;

    // Perform The Offline Solve;
    example.offlineSolveInitialVelocity();
    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/tutorials/CFD/00burgers/00burgers.C, line 204 #################### " << "Number of total modes" << example.Ufield.size() << endl;

    // Perform a POD decomposition for velocity and pressure
    ITHACAPOD::getModes(example.Ufield, example.Umodes, example._U().name(),
                        example.podex, 0, 0, NmodesUout);

    Eigen::MatrixXd SnapMatrix = Foam2Eigen::PtrList2Eigen(example.Ufield);
    cnpy::save(SnapMatrix, "npSnapshots.npy");

    example.project("./Matrices", NmodesUproj);

    ReducedBurgers reduced(example);

    // Set values of the reduced model
    reduced.nu = 0.0001;
    reduced.tstart = 0;
    reduced.finalTime = 2;
    reduced.dt = 0.001;
    reduced.storeEvery = 0.005;
    reduced.exportEvery = 0.1;

    reduced.solveOnline(1.0, 1);

    // Reconstruct the solution and export it
    reduced.reconstruct(true, "./ITHACAoutput/Reconstruction/");
}

/// \dir 04unsteadyNS Folder of the turorial 4
/// \file
/// \brief Implementation of tutorial 4 for an unsteady Navier-Stokes problem

/// \example 04unsteadyNS.C
/// \section intro_unsreadyNS Introduction to tutorial 4
/// In this tutorial we implement a parametrized unsteady Navier-Stokes 2D problem where the parameter is the kinematic viscosity.
/// The physical problem represents an incompressible flow passing around a very long cylinder. The simulation domain is rectangular
/// with spatial bounds of [-4, 30], and [-5, 5] in the X and Y directions, respectively. The cylinder has a radius of
/// 0.5 unit length and is located at the origin. The system has a prescribed uniform inlet velocity of 1 m/s which is constant through the whole simulation.
///
/// The following image illustrates the simulated system at time = 50 s and Re = 100.
/// \image html cylinder.png
///
/// \section code04 A detailed look into the code
///
/// In this section we explain the main steps necessary to construct the tutorial N°4
///
/// \subsection header ITHACA-FV header files
///
/// First of all let's have a look at the header files that need to be included and what they are responsible for.
///
/// The header files of ITHACA-FV necessary for this tutorial are: <unsteadyNS.H> for the full order unsteady NS problem,
/// <ITHACAPOD.H> for the POD decomposition, <reducedUnsteadyNS.H> for the construction of the reduced order problem,
/// and finally <ITHACAstream.H> for some ITHACA input-output operations.
///
/// \dontinclude 04unsteadyNS.C
/// \skip unsteadyNS
/// \until ITHACAstream
///
/// \subsection classtutorial04 Definition of the tutorial04 class
///
/// We define the tutorial04 class as a child of the unsteadyNS class.
/// The constructor is defined with members that are the fields need to be manipulated
/// during the resolution of the full order problem using pimpleFoam. Such fields are
/// also initialized with the same initial conditions in the solver.
/// \skipline tutorial04
/// \until {}
///
/// Inside the tutorial04 class we define the offlineSolve method according to the
/// specific parametrized problem that needs to be solved. If the offline solve has
/// been previously performed then the method just reads the existing snapshots from the Offline directory.
/// Otherwise it loops over all the parameters, changes the system viscosity with the iterable parameter
/// then performs the offline solve.
///
/// \skipline offlineSolve
/// \until }
/// \skipline else
/// \until }
/// \skipline }
/// \skipline }
///
/// We note that in the commented line we show that it is possible to parametrize the boundary conditions.
/// For further details we refer to the classes: reductionProblem, and unsteadyNS.
///
/// \subsection main Definition of the main function
///
/// In this section we show the definition of the main function.
/// First we construct the object "example" of type tutorial04:
///
/// \skipline example
///
/// Then we parse the ITHACAdict file to determine the number of modes
/// to be written out and also the ones to be used for projection of
/// the velocity, pressure, and the supremizer:
/// \skipline ITHACAparameters
/// \until NmodesSUPproj
///
/// we note that a default value can be assigned in case the parser did
/// not find the corresponding string in the ITHACAdict file.
///
/// Now we would like to perform 10 parametrized simulations where the kinematic viscosity
/// is the sole parameter to change, and it lies in the range of {0.1, 0.01} m^2/s equispaced.
/// Alternatively, we can also think of those simulations as that they are performed for fluid
/// flow that has Re changes from Re=10 to Re=100 with step size = 10. In fact, both definitions
/// are the same since the inlet velocity and the domain geometry are both kept fixed through all
/// simulations.
///
/// In our implementation, the parameter (viscosity) can be defined by specifying that
/// Nparameters=1, Nsamples=10, and the parameter ranges from 0.1 to 0.01 equispaced, i.e.
///
/// \skipline example.Pnumber
/// \until example.genEquiPar()
///
/// After that we set the inlet boundaries where we have the non homogeneous BC:
///
/// \skipline example.inlet
/// \until example.inletIndex(0, 1) = 0;
///
/// And we set the parameters for the time integration, so as to simulate 20 seconds for each
/// simulation, with a step size = 0.01 seconds, and the data are dumped every 1.0 seconds, i.e.
///
/// \skipline example.startTime
/// \until example.writeEvery
///
/// Now we are ready to perform the offline stage:
///
/// \skipline Solve()
///
/// and to solve the supremizer problem:
///
/// \skipline supremizer()
///
/// In order to search and compute the lifting function (which should be a step function of value
/// equals to the unitary inlet velocity), we perform the following:
///
/// \skipline liftSolve()
///
/// Then we create homogenuous basis functions for the velocity:
///
/// \skipline computeLift
///
/// After that, the modes for velocity, pressure and supremizers are obtained:
///
/// \skipline getModes
/// \until supfield
///
/// then the projection onto the POD modes is performed with:
///
/// \skipline projectSUP
///
/// Now that we obtained all the necessary information from the POD decomposition and the reduced matrices,
/// we are now ready to construct the dynamical system for the reduced order model (ROM). We proceed
/// by constructing the object "reduced" of type reducedUnsteadyNS:
///
/// \skipline reducedUnsteadyNS
///
/// And then we can use the new constructed ROM to perform the online procedure, from which we can simulate the
/// problem at new set of parameters. For instance, we solve the problem with a viscosity=0.055 for a 15
/// seconds of physical time:
///
/// \skipline reduced.nu
/// \until reduced.dt
///
/// and then the online solve is performed. In this tutorial, the value of the online velocity
/// is in fact a multiplication factor of the step lifting function for the unitary inlet velocity.
/// Therefore the online velocity sets the new BC at the inlet, hence we solve the ROM at new BC.
///
/// \skipline Eigen::
/// \until solveOnline_sup
///
/// Finally the ROM solution is reconstructed and exported:
///
/// \skipline reconstruct_sup
///
/// We note that all the previous evaluations of the pressure were based on the supremizers approach.
/// We can also use the Pressure Poisson Equation (PPE) instead of SUP so as to be implemented for the
/// projections, the online solve, and the fields reconstructions.
///
///
/// \section plaincode The plain program
/// Here there's the plain code
///
