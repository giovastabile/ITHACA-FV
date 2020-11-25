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

\*---------------------------------------------------------------------------*/

/// \file
/// Source file of the ReducedBurgers class

#include "ReducedBurgers.H"

// * * * * * * * * * * * * * * * Constructors * * * * * * * * * * * * * * * * //

// Constructor
ReducedBurgers::ReducedBurgers()
{
    para = ITHACAparameters::getInstance();
}

ReducedBurgers::ReducedBurgers(Burgers& FOMproblem)
    :
    problem(&FOMproblem)
{
    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/ReducedBurgers/ReducedBurgers.C, line 48 #################### " << endl;//N_BC = problem->inletIndex.rows();//CHECK
    Nphi_u = problem->B_matrix.rows();
    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/ReducedBurgers/ReducedBurgers.C, line 50 #################### " << endl;
    // for (int k = 0; k < problem->liftfield.size(); k++)
    // {
    //     Umodes.append(problem->liftfield[k]);
    // }

    for (int k = 0; k < problem->NL_Umodes; k++)
    {
        Umodes.append(problem->L_Umodes[k]);
        Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/ReducedBurgers/ReducedBurgers.C, line 59 #################### " << endl;
    }

    newton_object = newton_burgers(Nphi_u, Nphi_u, FOMproblem);
}

// Operator to evaluate the residual for the Supremizer approach
int newton_burgers::operator()(const Eigen::VectorXd& x,
                                      Eigen::VectorXd& fvec) const
{
    Eigen::VectorXd a_dot(Nphi_u);
    Eigen::VectorXd a_tmp(Nphi_u);
    a_tmp = x.head(Nphi_u);

    // Choose the order of the numerical difference scheme for approximating the time derivative
    if (problem->timeDerivativeSchemeOrder == "first")
    {
        a_dot = (x.head(Nphi_u) - y_old.head(Nphi_u)) / dt;
    }
    else
    {
        a_dot = (1.5 * x.head(Nphi_u) - 2 * y_old.head(Nphi_u) + 0.5 * yOldOld.head(
                     Nphi_u)) / dt;
    }

    // Convective term
    Eigen::MatrixXd cc(1, 1);
    // Diffusion term
    Eigen::VectorXd M1 = problem->B_matrix * a_tmp * nu;
    // Mass Term
    Eigen::VectorXd M5 = problem->M_matrix * a_dot;
    // Penalty term
    Eigen::MatrixXd penaltyU = Eigen::MatrixXd::Zero(Nphi_u, N_BC);

    // Term for penalty method
    if (problem->bcMethod == "penalty")
    {
        for (int l = 0; l < N_BC; l++)
        {
            penaltyU.col(l) = tauU(l,
                                   0) * (BC(l) * problem->bcVelVec[l] - problem->bcVelMat[l] *
                                         a_tmp);
        }
    }

    for (int i = 0; i < Nphi_u; i++)
    {
        cc = a_tmp.transpose() * Eigen::SliceFromTensor(problem->C_tensor, 0,
                i) * a_tmp;
        fvec(i) = - M5(i) + M1(i) - cc(0, 0);

        if (problem->bcMethod == "penalty")
        {
            for (int l = 0; l < N_BC; l++)
            {
                fvec(i) += penaltyU(i, l);
            }
        }
    }

    if (problem->bcMethod == "lift")
    {
        for (int j = 0; j < N_BC; j++)
        {
            fvec(j) = x(j) - BC(j);
        }
    }

    return 0;
}


int newton_burgers::df(const Eigen::VectorXd& x,
                        Eigen::MatrixXd& fjac) const
{
    Eigen::NumericalDiff<newton_burgers> numDiff(*this);
    numDiff.df(x, fjac);
    return 0;
}


// * * * * * * * * * * * * * * * Solve Functions  * * * * * * * * * * * * * //

// void ReducedBurgers::solveOnline(Eigen::MatrixXd vel,
//                                         int startSnap)
// {
//     M_Assert(exportEvery >= dt,
//              "The time step dt must be smaller than exportEvery.");
//     M_Assert(storeEvery >= dt,
//              "The time step dt must be smaller than storeEvery.");
//     M_Assert(ITHACAutilities::isInteger(storeEvery / dt) == true,
//              "The variable storeEvery must be an integer multiple of the time step dt.");
//     M_Assert(ITHACAutilities::isInteger(exportEvery / dt) == true,
//              "The variable exportEvery must be an integer multiple of the time step dt.");
//     M_Assert(ITHACAutilities::isInteger(exportEvery / storeEvery) == true,
//              "The variable exportEvery must be an integer multiple of the variable storeEvery.");
//     int numberOfStores = round(storeEvery / dt);

//     if (problem->bcMethod == "lift")
//     {
//         vel_now = setOnlineVelocity(vel);
//     }
//     else if (problem->bcMethod == "penalty")
//     {
//         vel_now = vel;
//     }

//     // Create and resize the solution vector
//     y.resize(Nphi_u, 1);
//     y.setZero();
//     y.head(Nphi_u) = ITHACAutilities::getCoeffs(problem->Ufield[startSnap],
//                      Umodes);

//     int nextStore = 0;
//     int counter2 = 0;

//     // Change initial condition for the lifting function
//     if (problem->bcMethod == "lift")
//     {
//         for (int j = 0; j < N_BC; j++)
//         {
//             y(j) = vel_now(j, 0);
//         }
//     }

//     // Set some properties of the newton object
//     newton_object.nu = nu;
//     newton_object.y_old = y;
//     newton_object.yOldOld = newton_object.y_old;
//     newton_object.dt = dt;
//     newton_object.BC.resize(N_BC);
//     newton_object.tauU = tauU;

//     for (int j = 0; j < N_BC; j++)
//     {
//         newton_object.BC(j) = vel_now(j, 0);
//     }

//     // Set number of online solutions
//     int Ntsteps = static_cast<int>((finalTime - tstart) / dt);
//     int onlineSize = static_cast<int>(Ntsteps / numberOfStores);
//     online_solution.resize(onlineSize);
//     // Set the initial time
//     time = tstart;
//     // Counting variable
//     int counter = 0;
//     // Create vector to store temporal solution and save initial condition as first solution
//     Eigen::MatrixXd tmp_sol(Nphi_u + 1, 1);
//     tmp_sol(0) = time;
//     tmp_sol.col(0).tail(y.rows()) = y;
//     online_solution[counter] = tmp_sol;
//     counter ++;
//     counter2++;
//     nextStore += numberOfStores;
//     // Create nonlinear solver object
//     Eigen::HybridNonLinearSolver<newton_burgers> hnls(newton_object);
//     // Set output colors for fancy output
//     Color::Modifier red(Color::FG_RED);
//     Color::Modifier green(Color::FG_GREEN);
//     Color::Modifier def(Color::FG_DEFAULT);

//     while (time < finalTime)
//     {
//         time = time + dt;

//         // Set time-dependent BCs
//         if (problem->timedepbcMethod == "yes" )
//         {
//             for (int j = 0; j < N_BC; j++)
//             {
//                 newton_object.BC(j) = vel_now(j, counter);
//             }
//         }

//         Eigen::VectorXd res(y);
//         res.setZero();
//         hnls.solve(y);

//         if (problem->bcMethod == "lift")
//         {
//             for (int j = 0; j < N_BC; j++)
//             {
//                 if (problem->timedepbcMethod == "no" )
//                 {
//                     y(j) = vel_now(j, 0);
//                 }
//                 else if (problem->timedepbcMethod == "yes" )
//                 {
//                     y(j) = vel_now(j, counter);
//                 }
//             }
//         }

//         newton_object.operator()(y, res);
//         newton_object.yOldOld = newton_object.y_old;
//         newton_object.y_old = y;
//         std::cout << "################## Online solve N° " << counter <<
//                   " ##################" << std::endl;
//         Info << "Time = " << time << endl;

//         if (res.norm() < 1e-5)
//         {
//             std::cout << green << "|F(x)| = " << res.norm() << " - Minimun reached in " <<
//                       hnls.iter << " iterations " << def << std::endl << std::endl;
//         }
//         else
//         {
//             std::cout << red << "|F(x)| = " << res.norm() << " - Minimun reached in " <<
//                       hnls.iter << " iterations " << def << std::endl << std::endl;
//         }

//         tmp_sol(0) = time;
//         tmp_sol.col(0).tail(y.rows()) = y;

//         if (counter == nextStore)
//         {
//             if (counter2 >= online_solution.size())
//             {
//                 online_solution.append(tmp_sol);
//             }
//             else
//             {
//                 online_solution[counter2] = tmp_sol;
//             }

//             nextStore += numberOfStores;
//             counter2 ++;
//         }

//         counter ++;
//     }

//     // Export the solution
//     ITHACAstream::exportMatrix(online_solution, "red_coeff", "python",
//                                "./ITHACAoutput/red_coeff");
//     ITHACAstream::exportMatrix(online_solution, "red_coeff", "matlab",
//                                "./ITHACAoutput/red_coeff");
// }

void ReducedBurgers::solveOnline(int startSnap)
{
    M_Assert(exportEvery >= dt,
             "The time step dt must be smaller than exportEvery.");
    M_Assert(storeEvery >= dt,
             "The time step dt must be smaller than storeEvery.");
    M_Assert(ITHACAutilities::isInteger(storeEvery / dt) == true,
             "The variable storeEvery must be an integer multiple of the time step dt.");
    M_Assert(ITHACAutilities::isInteger(exportEvery / dt) == true,
             "The variable exportEvery must be an integer multiple of the time step dt.");
    M_Assert(ITHACAutilities::isInteger(exportEvery / storeEvery) == true,
             "The variable exportEvery must be an integer multiple of the variable storeEvery.");
    int numberOfStores = round(storeEvery / dt);

    //CHECK
    // if (problem->bcMethod == "lift")
    // {
    //     vel_now = setOnlineVelocity(vel);
    // }
    // else if (problem->bcMethod == "penalty")
    // {
    //     vel_now = vel;
    // }

    // Create and resize the solution vector
    y.resize(Nphi_u, 1);
    y.setZero();
    y.head(Nphi_u) = ITHACAutilities::getCoeffs(problem->Ufield[startSnap],Umodes);

    int nextStore = 0;
    int counter2 = 0;

    // Change initial condition for the lifting function
    // if (problem->bcMethod == "lift")
    // {
    //     for (int j = 0; j < N_BC; j++)
    //     {
    //         y(j) = vel_now(j, 0);
    //     }
    // }

    // Set some properties of the newton object
    newton_object.nu = nu;
    newton_object.y_old = y;
    newton_object.yOldOld = newton_object.y_old;
    newton_object.dt = dt;
    // newton_object.BC.resize(N_BC);//CHECK
    newton_object.tauU = tauU;

    // for (int j = 0; j < N_BC; j++)
    // {
    //     newton_object.BC(j) = vel_now(j, 0);
    // }

    // Set number of online solutions
    int Ntsteps = static_cast<int>((finalTime - tstart) / dt);
    int onlineSize = static_cast<int>(Ntsteps / numberOfStores);
    online_solution.resize(onlineSize);
    // Set the initial time
    time = tstart;
    // Counting variable
    int counter = 0;
    // Create vector to store temporal solution and save initial condition as first solution
    Eigen::MatrixXd tmp_sol(Nphi_u + 1, 1);
    tmp_sol(0) = time;
    tmp_sol.col(0).tail(y.rows()) = y;
    online_solution[counter] = tmp_sol;
    counter ++;
    counter2++;
    nextStore += numberOfStores;
    // Create nonlinear solver object
    Eigen::HybridNonLinearSolver<newton_burgers> hnls(newton_object);
    // Set output colors for fancy output
    Color::Modifier red(Color::FG_RED);
    Color::Modifier green(Color::FG_GREEN);
    Color::Modifier def(Color::FG_DEFAULT);

    while (time < finalTime)
    {
        time = time + dt;

        // Set time-dependent BCs
        if (problem->timedepbcMethod == "yes" )
        {
            for (int j = 0; j < N_BC; j++)
            {
                newton_object.BC(j) = vel_now(j, counter);
            }
        }

        Eigen::VectorXd res(y);
        res.setZero();
        hnls.solve(y);

        if (problem->bcMethod == "lift")
        {
            for (int j = 0; j < N_BC; j++)
            {
                if (problem->timedepbcMethod == "no" )
                {
                    y(j) = vel_now(j, 0);
                }
                else if (problem->timedepbcMethod == "yes" )
                {
                    y(j) = vel_now(j, counter);
                }
            }
        }

        newton_object.operator()(y, res);
        newton_object.yOldOld = newton_object.y_old;
        newton_object.y_old = y;
        std::cout << "################## Online solve N° " << counter <<
                  " ##################" << std::endl;
        Info << "Time = " << time << endl;

        if (res.norm() < 1e-5)
        {
            std::cout << green << "|F(x)| = " << res.norm() << " - Minimun reached in " <<
                      hnls.iter << " iterations " << def << std::endl << std::endl;
        }
        else
        {
            std::cout << red << "|F(x)| = " << res.norm() << " - Minimun reached in " <<
                      hnls.iter << " iterations " << def << std::endl << std::endl;
        }

        tmp_sol(0) = time;
        tmp_sol.col(0).tail(y.rows()) = y;

        if (counter == nextStore)
        {
            if (counter2 >= online_solution.size())
            {
                online_solution.append(tmp_sol);
            }
            else
            {
                online_solution[counter2] = tmp_sol;
            }

            nextStore += numberOfStores;
            counter2 ++;
        }

        counter ++;
    }

    // Export the solution
    ITHACAstream::exportMatrix(online_solution, "red_coeff", "python",
                               "./ITHACAoutput/red_coeff");
    ITHACAstream::exportMatrix(online_solution, "red_coeff", "matlab",
                               "./ITHACAoutput/red_coeff");
}

void ReducedBurgers::solveOnline(Eigen::MatrixXd mu, int startSnap)
{
    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/ReducedBurgers/ReducedBurgers.C, line 455 #################### " << endl;
    M_Assert(exportEvery >= dt,
             "The time step dt must be smaller than exportEvery.");
    M_Assert(storeEvery >= dt,
             "The time step dt must be smaller than storeEvery.");
    M_Assert(ITHACAutilities::isInteger(storeEvery / dt) == true,
             "The variable storeEvery must be an integer multiple of the time step dt.");
    M_Assert(ITHACAutilities::isInteger(exportEvery / dt) == true,
             "The variable exportEvery must be an integer multiple of the time step dt.");
    M_Assert(ITHACAutilities::isInteger(exportEvery / storeEvery) == true,
             "The variable exportEvery must be an integer multiple of the variable storeEvery.");

    int numberOfStores = round(storeEvery / dt);

    // Counter of the number of online solutions saved, accounting also time as parameter
    int counter2 = 0;

    // Set number of online solutions
    int Ntsteps = static_cast<int>((finalTime - tstart) / dt);
    int onlineSize = static_cast<int>(Ntsteps / numberOfStores);
    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/ReducedBurgers/ReducedBurgers.C, line 474 #################### " << mu.cols() << " " << onlineSize << " " << Nphi_u << endl;
    online_solution.resize((mu.cols())*(onlineSize+1));

    // Iterate online solution for each parameter saved row-wise in mu
    for (int n_param = 0; n_param < mu.cols(); n_param++)
    {
        // Set the initial time
        time = tstart;

        // Counter of the number of saved time steps for the present parameter with index k
        int counter = 0;
        int nextStore = 0;

        // Create and resize the solution vector
        y.resize(Nphi_u, 1);
        y.setZero();
        Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/ReducedBurgers/ReducedBurgers.C, line 489 #################### " << Umodes.size() << endl;
        //y.head(Nphi_u) = ITHACAutilities::getCoeffs(problem->Ufield[startSnap],Umodes);
        Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/ReducedBurgers/ReducedBurgers.C, line 490 #################### " << endl;
        // set the multypling scalar of the initial velocity
        y(0) = mu(0, n_param);
        Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/ReducedBurgers/ReducedBurgers.C, line 496 #################### " << endl;
        // Set some properties of the newton object
        newton_object.nu = nu;
        newton_object.y_old = y;
        newton_object.yOldOld = newton_object.y_old;
        newton_object.dt = dt;
        newton_object.tauU = tauU;

        // Create vector to store temporal solution and save initial condition as first solution
        Eigen::MatrixXd tmp_sol(Nphi_u + 1, 1);
        tmp_sol(0) = time;
        tmp_sol.col(0).tail(y.rows()) = y;

        online_solution[counter2] = tmp_sol;
        counter2++;
        counter++;
        nextStore += numberOfStores;
        // Create nonlinear solver object
        Eigen::HybridNonLinearSolver<newton_burgers> hnls(newton_object);
        // Set output colors for fancy output
        Color::Modifier red(Color::FG_RED);
        Color::Modifier green(Color::FG_GREEN);
        Color::Modifier def(Color::FG_DEFAULT);

        Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/ReducedBurgers/ReducedBurgers.C, line 520 #################### " << endl;
        while (time < finalTime)
        {
            Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/ReducedBurgers/ReducedBurgers.C, line 523 #################### " << endl;
            time = time + dt;

            // Set time-dependent BCs
            if (problem->timedepbcMethod == "yes" )
            {
                for (int j = 0; j < N_BC; j++)
                {
                    newton_object.BC(j) = vel_now(j, counter);
                }
            }
            Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/ReducedBurgers/ReducedBurgers.C, line 534 #################### " << endl;
            Eigen::VectorXd res(y);
            res.setZero();
            hnls.solve(y);

            if (problem->bcMethod == "lift")
            {
                for (int j = 0; j < N_BC; j++)
                {
                    if (problem->timedepbcMethod == "no" )
                    {
                        y(j) = vel_now(j, 0);
                    }
                    else if (problem->timedepbcMethod == "yes" )
                    {
                        y(j) = vel_now(j, counter);
                    }
                }
            }
            Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/ReducedBurgers/ReducedBurgers.C, line 553 #################### " << endl;
            newton_object.operator()(y, res);
            newton_object.yOldOld = newton_object.y_old;
            newton_object.y_old = y;
            std::cout << "################## Online solve N° " << counter <<
                    " ##################" << std::endl;
            Info << "Time = " << time << endl;

            if (res.norm() < 1e-5)
            {
                std::cout << green << "|F(x)| = " << res.norm() << " - Minimun reached in " <<
                        hnls.iter << " iterations " << def << std::endl << std::endl;
            }
            else
            {
                std::cout << red << "|F(x)| = " << res.norm() << " - Minimun reached in " <<
                        hnls.iter << " iterations " << def << std::endl << std::endl;
            }

            tmp_sol(0) = time;
            tmp_sol.col(0).tail(y.rows()) = y;

            if (counter == nextStore)
            {
                if (counter2 >= online_solution.size())
                {
                    online_solution.append(tmp_sol);
                }
                else
                {
                    online_solution[counter2] = tmp_sol;
                }

                nextStore += numberOfStores;
                counter2 ++;
            }

            counter ++;
        }
    }
}


// * * * * * * * * * * * * * * * Jacobian Evaluation  * * * * * * * * * * * * * //
Eigen::MatrixXd ReducedBurgers::penalty(Eigen::MatrixXd& vel_now,
        Eigen::MatrixXd& tauIter,
        int startSnap)
{
    // Initialize new value on boundaries
    Eigen::MatrixXd valBC = Eigen::MatrixXd::Zero(N_BC, timeStepPenalty);
    // Initialize old values on boundaries
    Eigen::MatrixXd valBC0 = Eigen::MatrixXd::Zero(N_BC, timeStepPenalty);
    int Iter = 0;
    Eigen::VectorXd diffvel =  (vel_now.col(timeStepPenalty - 1) - valBC.col(
                                    timeStepPenalty - 1));
    diffvel = diffvel.cwiseAbs();

    while (diffvel.maxCoeff() > tolerancePenalty && Iter < maxIterPenalty)
    {
        if ((valBC.col(timeStepPenalty - 1) - valBC0.col(timeStepPenalty - 1)).sum() !=
                0)
        {
            for (int j = 0; j < N_BC; j++)
            {
                tauIter(j, 0) = tauIter(j, 0) * diffvel(j) / tolerancePenalty;
            }
        }

        std::cout << "Solving for penalty factor(s): " << tauIter << std::endl;
        std::cout << "number of iterations: " << Iter << std::endl;
        //  Set the old boundary value to the current value
        valBC0  = valBC;
        y.resize(Nphi_u, 1);
        y.setZero();
        y.head(Nphi_u) = ITHACAutilities::getCoeffs(problem->Ufield[startSnap],
                         Umodes);

        // Set some properties of the newton object
        newton_object.nu = nu;
        newton_object.y_old = y;
        newton_object.dt = dt;
        newton_object.BC.resize(N_BC);
        newton_object.tauU = tauIter;

        // Set boundary conditions
        for (int j = 0; j < N_BC; j++)
        {
            newton_object.BC(j) = vel_now(j, 0);
        }

        // Create nonlinear solver object
        Eigen::HybridNonLinearSolver<newton_burgers> hnls(newton_object);
        // Set output colors for fancy output
        Color::Modifier red(Color::FG_RED);
        Color::Modifier green(Color::FG_GREEN);
        Color::Modifier def(Color::FG_DEFAULT);
        // Set initially for convergence check
        Eigen::VectorXd res(y);
        res.setZero();

        // Start the time loop
        for (int i = 1; i < timeStepPenalty; i++)
        {
            // Set boundary conditions
            for (int j = 0; j < N_BC; j++)
            {
                if (problem->timedepbcMethod == "yes" )
                {
                    newton_object.BC(j) = vel_now(j, i);
                }
                else
                {
                    newton_object.BC(j) = vel_now(j, 0);
                }
            }

            Eigen::VectorXd res(y);
            res.setZero();
            hnls.solve(y);
            newton_object.operator()(y, res);
            newton_object.y_old = y;

            if (res.norm() < 1e-5)
            {
                std::cout << green << "|F(x)| = " << res.norm() << " - Minimun reached in " <<
                          hnls.iter << " iterations " << def << std::endl << std::endl;
            }
            else
            {
                std::cout << red << "|F(x)| = " << res.norm() << " - Minimun reached in " <<
                          hnls.iter << " iterations " << def << std::endl << std::endl;
            }

            volVectorField U_rec("U_rec", Umodes[0] * 0);

            for (int j = 0; j < Nphi_u; j++)
            {
                U_rec += Umodes[j] * y(j);
            }

            for (int k = 0; k < problem->inletIndex.rows(); k++)
            {
                int BCind = problem->inletIndex(k, 0);
                int BCcomp = problem->inletIndex(k, 1);
                valBC(k, i) = U_rec.boundaryFieldRef()[BCind][0].component(BCcomp);
            }
        }

        for (int j = 0; j < N_BC; j++)
        {
            diffvel(j) = abs(abs(vel_now(j, timeStepPenalty - 1)) - abs(valBC(j,
                             timeStepPenalty - 1)));
        }

        std::cout << "max error: " << diffvel.maxCoeff() << std::endl;
        // Count the number of iterations
        Iter ++;
    }

    std::cout << "Final penalty factor(s): " << tauIter << std::endl;
    std::cout << "Iterations: " << Iter - 1 << std::endl;
    return tauIter;
}

void ReducedBurgers::reconstruct(bool exportFields, fileName folder)
{
    if (exportFields)
    {
        mkDir(folder);
        ITHACAutilities::createSymLink(folder);
    }
    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/ReducedBurgers/ReducedBurgers.C, line 719 #################### " << Nphi_u << endl;
    int counter = 0;
    int nextwrite = 0;
    List < Eigen::MatrixXd> CoeffU;
    List <double> tValues;
    CoeffU.resize(0);
    tValues.resize(0);
    int exportEveryIndex = round(exportEvery / storeEvery);

    for (int i = 0; i < online_solution.size(); i++)
    {
        if (counter == nextwrite)
        {
            Eigen::MatrixXd currentUCoeff;
            currentUCoeff = online_solution[i].block(1, 0, Nphi_u, 1);
            CoeffU.append(currentUCoeff);
            nextwrite += exportEveryIndex;
            double timeNow = online_solution[i](0, 0);
            tValues.append(timeNow);
        }

        counter++;
    }

    volVectorField uRec("uRec", Umodes[0] * 0);
    uRecFields = problem->L_Umodes.reconstruct(uRec, CoeffU, "uRec");

    if (exportFields)
    {
        ITHACAstream::exportFields(uRecFields, folder,
                                   "uRec");
    }
}

void ReducedBurgers::reconstruct(bool exportFields, fileName folder, Eigen::MatrixXd redCoeff)
{
    if (exportFields)
    {
        mkDir(folder);
        ITHACAutilities::createSymLink(folder);
    }
    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/ReducedBurgers/ReducedBurgers.C, line 759 #################### " << Nphi_u << endl;
    int counter = 0;
    int nextwrite = 0;
    List < Eigen::MatrixXd> CoeffU;
    List <double> tValues;
    CoeffU.resize(0);
    tValues.resize(0);
    int exportEveryIndex = round(exportEvery / storeEvery);

    for (int i = 0; i < redCoeff.rows(); i++)
    {
        if (counter == nextwrite)
        {
            Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/ReducedBurgers/ReducedBurgers.C, line 773 #################### " << redCoeff(i, 0) << " " << redCoeff(i, 1) << " " << redCoeff(i, 2) << " " << redCoeff(i, 3) << endl;

            Eigen::MatrixXd currentUCoeff(Nphi_u, 1);
            // currentUCoeff = Eigen::MatrixXd::Ones(Nphi_u, 1);

            Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/ReducedBurgers/ReducedBurgers.C, line 775 #################### " << currentUCoeff.rows() << " " << redCoeff.row(i).tail(Nphi_u).cols() << " " << redCoeff.row(i).tail(Nphi_u).transpose().rows() << endl;

            currentUCoeff.col(0) = redCoeff.row(i).tail(Nphi_u).transpose();

            Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/ReducedBurgers/ReducedBurgers.C, line 776 #################### " << currentUCoeff.rows() << endl;//currentUCoeff(0, 0) << " " << currentUCoeff(1, 0) << " " << currentUCoeff(2, 0) << endl;

            CoeffU.append(currentUCoeff);
            nextwrite += exportEveryIndex;
            double timeNow = redCoeff(i, 0);
            tValues.append(timeNow);
            Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/ReducedBurgers/ReducedBurgers.C, line 779 #################### " << counter << endl;
        }

        counter++;
    }

    volVectorField uRec("uRec", Umodes[0] * 0);

    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/ReducedBurgers/ReducedBurgers.C, line 786 #################### " << endl;

    uRecFields = problem->L_Umodes.reconstruct(uRec, CoeffU, "uRec");

    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/ReducedBurgers/ReducedBurgers.C, line 787 #################### " << endl;

    if (exportFields)
    {
        ITHACAstream::exportFields(uRecFields, folder,
                                   "uRec");
    }
}

Eigen::MatrixXd ReducedBurgers::setOnlineVelocity(Eigen::MatrixXd vel)
{
    assert(problem->inletIndex.rows() == vel.rows()
           && "Imposed boundary conditions dimensions do not match given values matrix dimensions");
    Eigen::MatrixXd vel_scal;
    vel_scal.resize(vel.rows(), vel.cols());

    for (int k = 0; k < problem->inletIndex.rows(); k++)
    {
        int p = problem->inletIndex(k, 0);
        int l = problem->inletIndex(k, 1);
        scalar area = gSum(problem->liftfield[0].mesh().magSf().boundaryField()[p]);
        scalar u_lf = gSum(problem->liftfield[k].mesh().magSf().boundaryField()[p] *
                           problem->liftfield[k].boundaryField()[p]).component(l) / area;

        for (int i = 0; i < vel.cols(); i++)
        {
            vel_scal(k, i) = vel(k, i) / u_lf;
        }
    }

    return vel_scal;
}

void ReducedBurgers::trueProjection(fileName folder)
{

    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/ReducedBurgers/ReducedBurgers.C, line 841 #################### " << problem->Ufield.size() << " " << Nphi_u << endl;

    // CoeffU size: (Umodes[0].size(), Nphi_u)
    List < Eigen::MatrixXd> CoeffU;
    CoeffU.resize(0);

    for (int n_index = 0; n_index < problem->Ufield.size(); n_index++)
    {
        Eigen::MatrixXd currentUCoeff(Nphi_u, 1);
            // currentUCoeff = Eigen::MatrixXd::Ones(Nphi_u, 1);

        Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/ReducedBurgers/ReducedBurgers.C, line 852 #################### " << currentUCoeff.rows() << endl;

        currentUCoeff.col(0) = ITHACAutilities::getCoeffs(problem->Ufield[n_index],Umodes);

        Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/ReducedBurgers/ReducedBurgers.C, line 856 #################### " << currentUCoeff.rows() << endl;

        CoeffU.append(currentUCoeff);
    }

    volVectorField uRec("uRec", Umodes[0] * 0);

    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/ReducedBurgers/ReducedBurgers.C, line 863 #################### " << endl;

    uRecFields = problem->L_Umodes.reconstruct(uRec, CoeffU, "uRec");

    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/ReducedBurgers/ReducedBurgers.C, line 867 #################### " << uRecFields.size() << endl;

    ITHACAstream::exportFields(uRecFields, folder, "uTrueProjection");
}