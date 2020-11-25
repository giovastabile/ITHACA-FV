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
#include <algorithm>

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

    void offlineSolveInitialVelocity(fileName folder = "./ITHACAoutput/Offline/")
    {
        List<scalar> mu_now(1);

        if (offline)
        {
            Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/tutorials/CFD/00burgers/00burgers.C, line 79 #################### " << folder << endl;
            ITHACAstream::read_fields(Ufield, "U", folder);
            ITHACAstream::exportFields(Ufield, "./TRAIN", "uTrain");
            Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/tutorials/CFD/00burgers/00burgers.C, line 81 #################### " << Ufield.size() << endl;
        }
        else
        {
            for (label i = 0; i < mu.cols(); i++)
            {
                mu_now[0] = mu(0, i);
                change_initial_velocity(mu(0, i));
                truthSolve(mu_now, folder);
            }
        }
    }
};

/*---------------------------------------------------------------------------*\
                               Starting the MAIN
\*---------------------------------------------------------------------------*/

void one_parameter_viscosity(tutorial00);
void train_one_parameter_initial_velocity(tutorial00);
void test_one_parameter_initial_velocity(tutorial00);


int main(int argc, char *argv[])
{
    if (argc == 1)
    {
        std::cout << "Pass train or test." << endl;
        exit(0);
    }
    // processed arguments
    int argc_proc = argc-1;
    char* argv_proc[argc_proc];
    argv_proc[0] = argv[0];

    if (argc > 2){std::copy(argv+2, argv+argc, argv_proc+1);}

    if (std::strcmp(argv[1],"train")==0)
    {
        Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/tutorials/CFD/00burgers/00burgers.C, line 113 #################### " << endl;
        // save mu_samples and training snapshots reduced coefficients
        tutorial00 example(argc_proc, argv_proc);
        train_one_parameter_initial_velocity(example);
    }
    else if (std::strcmp(argv[1],"test")==0)
    {
        Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/tutorials/CFD/00burgers/00burgers.C, line 121 #################### " << endl;
        // compute FOM, ROM-intrusive, ROM-nonintrusive and evaluate errors
        tutorial00 test_FOM(argc_proc, argv_proc);
        test_one_parameter_initial_velocity(test_FOM);
    }
    else
    {
        std::cout << "Pass train or test." << endl;
    }

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
    reduced.exportEvery = 0.0013;

    reduced.solveOnline(1);

    // Reconstruct the solution and export it
    reduced.reconstruct(true, "./ITHACAoutput/Reconstruction/");
}

void train_one_parameter_initial_velocity(tutorial00 example)
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
    example.mu_range(0, 0) = 0.5;
    example.mu_range(0, 1) = 1.5;
    // Generate a number of Tnumber linearly equispaced samples inside the parameter range
    example.genEquiPar();
    cnpy::save(example.mu, "parTrain.npy");

    // Time parameters
    example.startTime = 0;
    example.finalTime = 2;
    example.timeStep = 0.001;
    example.writeEvery = 0.01;

    // Perform The Offline Solve;
    example.offlineSolveInitialVelocity("./ITHACAoutput/Offline/Training/");

    // Perform a POD decomposition for velocity
    ITHACAPOD::getModes(example.Ufield, example.Umodes, example._U().name(),
                        example.podex, 0, 0, NmodesUout);

    Eigen::MatrixXd SnapMatrix = Foam2Eigen::PtrList2Eigen(example.Ufield);
    Info << "snapshots size: " << SnapMatrix.size() << endl;
    cnpy::save(SnapMatrix, "npSnapshots.npy");

    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/tutorials/CFD/00burgers/00burgers.C, line 231 #################### " << endl;
    example.project("./Matrices", NmodesUproj);
    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/tutorials/CFD/00burgers/00burgers.C, line 233 #################### " << endl;

    // The initial conditions are used as the first mode
    ITHACAstream::exportFields(example.L_Umodes, "./ITHACAoutput/POD_and_initial/", "U");

    Eigen::MatrixXd modes = Foam2Eigen::PtrList2Eigen(example.L_Umodes);
    Info << "snapshots size: " << modes.size() << endl;
    cnpy::save(modes, "npInitialAndModes.npy");

    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/tutorials/CFD/00burgers/00burgers.C, line 236 #################### " << endl;
    ReducedBurgers reduced(example);
    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/tutorials/CFD/00burgers/00burgers.C, line 238 #################### " << endl;

    // Set values of the reduced model
    reduced.nu = 0.0001;
    reduced.tstart = 0;
    reduced.finalTime = 2;
    reduced.dt = 0.001;
    reduced.storeEvery = 0.01;
    reduced.exportEvery = 0.01;

    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/tutorials/CFD/00burgers/00burgers.C, line 247 #################### " << endl;
    reduced.solveOnline(example.mu, 0);
    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/tutorials/CFD/00burgers/00burgers.C, line 248 #################### " << endl;
    ITHACAstream::exportMatrix(reduced.online_solution, "red_coeff", "python", "./ITHACAoutput/red_coeff");

    // sample test set
    example.setParameters();
    // Set the parameter ranges
    example.mu_range(0, 0) = 0.5 - 0.25;
    example.mu_range(0, 1) = 1.5 + 0.25;
    // Generate a number of Tnumber linearly equispaced samples inside the parameter range
    example.genEquiPar();
    cnpy::save(example.mu, "parTest.npy");

}

void test_one_parameter_initial_velocity(tutorial00 test_FOM)
{
    // Read parameters from ITHACAdict file
    ITHACAparameters *para = ITHACAparameters::getInstance(test_FOM._mesh(),
                                                           test_FOM._runTime());
    int NmodesUout = para->ITHACAdict->lookupOrDefault<int>("NmodesUout", 15);
    int NmodesUproj = para->ITHACAdict->lookupOrDefault<int>("NmodesUproj", 10);
    int NmodesUtest = para->ITHACAdict->lookupOrDefault<int>("NmodesUtest", 100);

    /// Set the number of parameters
    test_FOM.Pnumber = 1;
    /// Set the dimension of the test set
    test_FOM.Tnumber = NmodesUtest;
    /// Instantiates a void Pnumber-by-Tnumber matrix mu for the parameters and a void
    /// Pnumber-by-2 matrix mu_range for the ranges
    test_FOM.setParameters();
    // Set the parameter ranges
    test_FOM.mu_range(0, 0) = 0.5;
    test_FOM.mu_range(0, 1) = 1.5;
    // Generate a number of Tnumber linearly equispaced samples inside the parameter range
    Eigen::MatrixXd mu;
    test_FOM.mu = cnpy::load(mu, "parTest.npy");

    // Time parameters
    test_FOM.startTime = 0;
    test_FOM.finalTime = 2;
    test_FOM.timeStep = 0.001;
    test_FOM.writeEvery = 0.01;

    // Perform The Offline Solve;
    if (!ITHACAutilities::check_folder("./ITHACAoutput/Offline/Test/80"))
        {
            test_FOM.offline = false;
            Info << "Offline Test data already exist, reading existing data" << endl;
        }
    test_FOM.offlineSolveInitialVelocity("./ITHACAoutput/Offline/Test/");
    Eigen::MatrixXd trueSnapMatrix = Foam2Eigen::PtrList2Eigen(test_FOM.Ufield);
    Info << "snapshots size: " << trueSnapMatrix.size() << endl;
    cnpy::save(trueSnapMatrix, "npTrueSnapshots.npy");

    test_FOM.NUmodes = NmodesUproj;
    ITHACAstream::read_fields(test_FOM.L_Umodes, "U", "./ITHACAoutput/POD_and_initial/", 0, NmodesUout);

    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/tutorials/CFD/00burgers/00burgers.C, line 291 #################### " << test_FOM.NUmodes << " " << test_FOM.L_Umodes.size()<< endl;
    ITHACAstream::exportFields(test_FOM.L_Umodes, "./TEST", "uTest");

    test_FOM.NL_Umodes = test_FOM.L_Umodes.size();
    test_FOM.evaluateMatrices();

    ReducedBurgers reduced_nonIntrusive(test_FOM);

    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/tutorials/CFD/00burgers/00burgers.C, line 288 #################### " << endl;

    // Set values of the reduced_nonIntrusive model
    reduced_nonIntrusive.nu = 0.0001;
    reduced_nonIntrusive.tstart = 0;
    reduced_nonIntrusive.finalTime = 2;
    reduced_nonIntrusive.dt = 0.001;
    reduced_nonIntrusive.storeEvery = 0.01;
    reduced_nonIntrusive.exportEvery = 0.01;
    //reduced_nonIntrusive.Nphi_u = NmodesUproj;// the initial condition is added to the modes

    Eigen::MatrixXd nonIntrusiveCoeff;

    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/tutorials/CFD/00burgers/00burgers.C, line 298 #################### " << endl;

    nonIntrusiveCoeff = cnpy::load(nonIntrusiveCoeff, "nonIntrusiveCoeff.npy", "rowMajor");

    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/tutorials/CFD/00burgers/00burgers.C, line 299 #################### " << nonIntrusiveCoeff.rows() << " "  << nonIntrusiveCoeff.cols() << endl;

    // Reconstruct the solution and export it
    reduced_nonIntrusive.reconstruct(true, "./ITHACAoutput/Reconstruction/", nonIntrusiveCoeff);

    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/tutorials/CFD/00burgers/00burgers.C, line 310 #################### " << test_FOM.Ufield.size() << " " << reduced_nonIntrusive.uRecFields.size() << endl;

    Eigen::MatrixXd errL2UnonIntrusive = ITHACAutilities::errorL2Rel(test_FOM.Ufield,
                             reduced_nonIntrusive.uRecFields);

    ITHACAstream::exportMatrix(errL2UnonIntrusive, "errL2UnonIntrusive", "matlab",
                               "./ITHACAoutput/ErrorsL2/");
    cnpy::save(errL2UnonIntrusive, "./ITHACAoutput/ErrorsL2/errL2UnonIntrusive.npy");

    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/tutorials/CFD/00burgers/00burgers.C, line 345 #################### " << endl;

    ReducedBurgers reduced_intrusive(test_FOM);

    // Set values of the reduced model
    reduced_intrusive.nu = 0.0001;
    reduced_intrusive.tstart = 0;
    reduced_intrusive.finalTime = 2;
    reduced_intrusive.dt = 0.001;
    reduced_intrusive.storeEvery = 0.01;
    reduced_intrusive.exportEvery = 0.01;

    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/tutorials/CFD/00burgers/00burgers.C, line 341 #################### " << endl;

    reduced_intrusive.solveOnline(test_FOM.mu, 1);
    ITHACAstream::exportMatrix(reduced_intrusive.online_solution, "red_coeff", "python", "./ITHACAoutput/red_coeff_intrusive");

    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/tutorials/CFD/00burgers/00burgers.C, line 344 #################### " << endl;

    // Reconstruct the solution and export it
    reduced_intrusive.reconstruct(true, "./ITHACAoutput/ReconstructionIntrusive/");

    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/tutorials/CFD/00burgers/00burgers.C, line 347 #################### " << endl;

    Eigen::MatrixXd errL2Uintrusive = ITHACAutilities::errorL2Rel(test_FOM.Ufield,
                             reduced_intrusive.uRecFields);

    ITHACAstream::exportMatrix(errL2Uintrusive, "errL2UIntrusive", "matlab",
                               "./ITHACAoutput/ErrorsL2/");
    cnpy::save(errL2Uintrusive, "./ITHACAoutput/ErrorsL2/errL2UIntrusive.npy");

    // Evaluate the true projection error
    reduced_intrusive.trueProjection("./ITHACAoutput/Reconstruction/");

    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/tutorials/CFD/00burgers/00burgers.C, line 310 #################### " << test_FOM.Ufield.size() << " " << reduced_intrusive.uRecFields.size() << endl;

    Eigen::MatrixXd errL2UtrueProjection = ITHACAutilities::errorL2Rel(test_FOM.Ufield,
                             reduced_intrusive.uRecFields);

    ITHACAstream::exportMatrix(errL2UnonIntrusive, "errL2UtrueProjectionROM", "matlab",
                               "./ITHACAoutput/ErrorsL2/");
    cnpy::save(errL2UnonIntrusive, "./ITHACAoutput/ErrorsL2/errL2UtrueProjectionROM.npy");

    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/tutorials/CFD/00burgers/00burgers.C, line 357 #################### " << endl;

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
