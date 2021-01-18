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
/// Source file of the NonlinearReducedBurgers class

#include <torch/script.h>
#include <torch/torch.h>
#include "torch2Eigen.H"
#include "torch2Foam.H"
#include "Foam2Eigen.H"
#include "ITHACAstream.H"
#include <chrono>
#include "cnpy.H"


#include "NonlinearReducedBurgers.H"

using namespace ITHACAtorch;
// * * * * * * * * * * * * * * * Constructors * * * * * * * * * * * * * * * * //

NonlinearReducedBurgers::NonlinearReducedBurgers()
{
    para = ITHACAparameters::getInstance();
}

NonlinearReducedBurgers::NonlinearReducedBurgers(Burgers &FOMproblem, fileName decoder_path, int dim_latent, Eigen::MatrixXd latent_initial)
    : Nphi_u{dim_latent},
      problem{&FOMproblem}
{

    embedding = autoPtr<Embedding>(new Embedding(Nphi_u, decoder_path, problem->L_Umodes[0], latent_initial));

    // FOMproblem is only needed for initial conditions
    newton_object = newton_nmlspg_burgers(Nphi_u, 2*embedding->output_dim, FOMproblem, embedding.ref(), problem->L_Umodes[0]);
}

Embedding::Embedding(int dim, fileName decoder_path, volVectorField &U0, Eigen::MatrixXd lat_init) : latent_dim{dim}, latent_initial{lat_init}
{
    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 65 #################### EMBEDDING CONSTRUCTOR" << latent_initial(0) << endl;

    // get the number of degrees of freedom relative to a single component
    output_dim = U0.size();
    decoder = autoPtr<torch::jit::script::Module>(new torch::jit::script::Module(torch::jit::load(decoder_path)));

    // define initial velocity field and initialize decoder output variable g0
    _U0 = autoPtr<volVectorField>(new volVectorField(U0));
    _g0 = autoPtr<volVectorField>(new volVectorField(U0));

    // declare input of decoder of type IValue since the decoder is loaded from pytorch
    std::vector<torch::jit::IValue> input;
    torch::Tensor latent_initial_tensor = torch2Eigen::eigenMatrix2torchTensor(latent_initial);

    input.push_back(latent_initial_tensor.to(torch::kCUDA));

    auto start = std::chrono::system_clock::now();
    torch::Tensor tensor = decoder->forward(std::move(input)).toTensor().to(torch::kCPU);
    auto end = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 81 #################### constructor " << elapsed.count() << " ms" << endl;

    // tensor must be scalar, tensor components of velocity must be saved
    // contiguously in memory (frist x components, then y), the order of the
    // vector must be the one of cells in the mesh.
    auto tensor_stacked = torch::cat({std::move(tensor).reshape({2, 60, 60}), torch::zeros({1, 60, 60})}, 0).reshape({3, -1}).transpose(0, 1).contiguous();
    auto g0 = torch2Foam::torch2Field<vector>(tensor_stacked);
    _g0.ref().ref().field() = std::move(g0);

    // PtrList<volVectorField> save_field;
    // save_field.append(_g0());
    // auto test_ref_0 = embedding_ref(1);
    // save_field.append(test_ref_0() + _g0());
    // ITHACAstream::exportFields(save_field, "./REF", "g0");
    // counter++;
}

// return reference element of embedding s.t. initial embedding is mu * _U0()
autoPtr<volVectorField> Embedding::embedding_ref(const scalar mu)
{
    return autoPtr<volVectorField>(new volVectorField(mu * _U0() - _g0()));
}

autoPtr<volVectorField> Embedding::forward(const Eigen::VectorXd &x, const scalar mu)
{
    auto start = std::chrono::system_clock::now();
    std::vector<torch::jit::IValue> input;
    Eigen::MatrixXd input_matrix{x};

    input_matrix.resize(1, latent_dim);
    torch::Tensor input_tensor = torch2Eigen::eigenMatrix2torchTensor(std::move(input_matrix));
    input_tensor = input_tensor.reshape({1, latent_dim});
    input_tensor = input_tensor.set_requires_grad(true);
    input.push_back(input_tensor.to(torch::kCUDA));

    torch::Tensor push_forward_tensor = decoder->forward(std::move(input)).toTensor().to(torch::kCPU);


    auto g = autoPtr<volVectorField>(new volVectorField(_U0()));

    auto tensor_stacked = torch::cat({push_forward_tensor.reshape({2, 60, 60}), torch::zeros({1, 60, 60})}, 0).reshape({3, -1}).transpose(0, 1).contiguous();
    // auto test = torch::cat({torch::zeros({3600, 1}), torch::ones({3600, 1}), 2*torch::ones({3600, 1})}, 1);
    auto push_forward = torch2Foam::torch2Field<vector>(tensor_stacked);

    // add reference term
    g.ref().ref().field() = std::move(push_forward);
    g.ref() += embedding_ref(mu).ref();

    // PtrList<volVectorField> save_field;
    // save_field.append(g());
    // ITHACAstream::exportFields(save_field, "./REF", "g"+std::to_string(counter));
    // counter++;

    auto end = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 141 #################### forward call " << elapsed.count() << " ms" << endl;
    return g;
}

autoPtr<Eigen::MatrixXd> Embedding::jacobian(const Eigen::VectorXd &x, const scalar mu)
{
    int jacobian_out_dim = output_dim * 2;
    Eigen::MatrixXd input_matrix{x};

    input_matrix.resize(1, latent_dim);
    torch::Tensor input_tensor = torch2Eigen::eigenMatrix2torchTensor(std::move(input_matrix));
    input_tensor = input_tensor.reshape({1, latent_dim}).set_requires_grad(true);
    input_jac.push_back(input_tensor).to(torch::kCUDA);

    auto grad_output = torch::eye(output_dim * 2);

    // compute the jacobian of out wrt input
    torch::Tensor forward_tensor = decoder->forward(input_jac).toTensor().squeeze();

    for(int i=0; i<10; i++)
    {
        grad_output.to(torch::kCUDA);
        forward_tensor.backward();
        input_tensor.grad();
        input_tensor.grad().zero_();
    }


    auto gradient = torch::autograd::grad({forward_tensor},
                                          {input_repeated},
                                          /*grad_outputs=*/{grad_output},
                                          /*retain_graph=*/true,
                                          /*create_graph=*/true);

    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers_true.C, line 188 #################### " << endl;
    auto grad = gradient[0];

    // TODO: remove this two lines?
    auto J = torch::ones({jacobian_out_dim, latent_dim});
    J.slice(/*dim=*/0, /*start=*/0, /*end=*/jacobian_out_dim) = grad.to(torch::kCPU);

    auto grad_eigen = torch2Eigen::torchTensor2eigenMatrix<double>(J);
    auto dg = autoPtr<Eigen::MatrixXd>(new Eigen::MatrixXd(std::move(grad_eigen)));
    // cnpy::save(dg.ref(), "jacobian.npy");

    return dg;
}

// autoPtr<Eigen::MatrixXd> Embedding::jacobian(const Eigen::VectorXd &x, const scalar mu)
// {
//     int jacobian_out_dim = output_dim * 2;
//     Eigen::MatrixXd input_matrix{x};

//     input_matrix.resize(1, latent_dim);
//     torch::Tensor input_tensor = torch2Eigen::eigenMatrix2torchTensor(std::move(input_matrix));
//     input_tensor = input_tensor.reshape({1, latent_dim}).set_requires_grad(true);

//     std::vector<torch::jit::IValue> input_jac;
//     auto input_repeated = input_tensor.repeat({jacobian_out_dim, 1});
//     input_repeated = input_repeated.set_requires_grad(true);

//     /// to save inputs
//     // auto save_inputs = torch2Eigen::torchTensor2eigenMatrix<double>(input_repeated);
//     // cnpy::save(save_inputs, "x.npy");

//     input_repeated = input_repeated.to(torch::kCUDA);
//     input_jac.push_back(input_repeated);

//     // matrix to left multiply Jacobian matrix with s.t. autograd::grad
//     // computes all the column vectors of matrix Jacobian at the same time
//     auto grad_output = torch::eye(output_dim * 2).to(torch::kCUDA);

//     // compute the jacobian of out wrt input
//     // auto start = std::chrono::system_clock::now();
//     torch::Tensor forward_tensor = decoder->forward(input_jac).toTensor().squeeze();

//     auto start = std::chrono::system_clock::now();
//     forward_tensor.backward(grad_output, true);
//     auto end = std::chrono::system_clock::now();
//     auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

//     Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 137 #################### backward jacobian " << elapsed.count() << " ms" << endl;

//     auto gradient = torch::autograd::grad({forward_tensor},
//                                           {input_repeated},
//                                           /*grad_outputs=*/{grad_output},
//                                           /*retain_graph=*/true,
//                                           /*create_graph=*/true);

//     Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers_true.C, line 188 #################### " << endl;
//     auto grad = gradient[0];

//     // TODO: remove this two lines?
//     auto J = torch::ones({jacobian_out_dim, latent_dim});
//     J.slice(/*dim=*/0, /*start=*/0, /*end=*/jacobian_out_dim) = grad.to(torch::kCPU);

//     auto grad_eigen = torch2Eigen::torchTensor2eigenMatrix<double>(J);
//     auto dg = autoPtr<Eigen::MatrixXd>(new Eigen::MatrixXd(std::move(grad_eigen)));
//     // cnpy::save(dg.ref(), "jacobian.npy");

//     return dg;
// }

std::pair<autoPtr<volVectorField>, autoPtr<Eigen::MatrixXd>> Embedding::forward_with_gradient(const Eigen::VectorXd &x, const scalar mu)
{
    auto g = forward(x, mu);
    auto dg = jacobian(x, mu);
    return std::make_pair(g, dg);
}

// Operator to evaluate the residual
int newton_nmlspg_burgers::operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const
{
    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 227 #################### OPERATOR(), x = " << x(0) << " " << x(1) << " " << x(2) << " " << x(3) << endl;

    auto g = embedding->forward(x.head(Nphi_u), mu);
    volVectorField& a_tmp = g();
    fvMesh& mesh = problem->_mesh();
    auto phi = linearInterpolate(a_tmp) & mesh.Sf();

    //  create fictitious system s.t. the residual is r_n+1 = g_n+1 -g_n - f_n+1
    auto a_old = g_old();
    auto start = std::chrono::system_clock::now();
    volVectorField& tmp = a_tmp.oldTime();
    tmp = a_old;

    fvVectorMatrix resEqn(
        fvm::ddt(a_tmp) + 0.5 * fvm::div(phi, a_tmp) - fvm::laplacian(dimensionedScalar(dimViscosity, nu.value()), a_tmp));

    // resEqn.solve();
    a_tmp.field() = resEqn.residual();

    auto end = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 214 #################### residual " << elapsed.count() << " ms" << endl;

    // Eigen::VectorXd tmp_fvec;
    // Foam2Eigen::fvMatrix2EigenV(resEqn, tmp_fvec);

    // fvec = tmp_fvec.col(0).head(this->embedding->output_dim * 2);
    fvec = Foam2Eigen::field2Eigen(a_tmp).col(0).head(this->embedding->output_dim * 2);
    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 246 #################### RESIDUAL=" << fvec.norm() << endl;

    return 0;
}

int newton_nmlspg_burgers::df(const Eigen::VectorXd &x,
                              Eigen::MatrixXd &fjac) const
{
//     Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 266 #################### DF " << endl;

//    Eigen::NumericalDiff<newton_nmlspg_burgers, Eigen::Central> numDiff(*this, 1.e-04);
//    numDiff.df(x, fjac);

//     Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 266 #################### DF END" << endl;

    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 87 #################### "
     << "fjac call" << endl;

    auto pair = embedding->forward_with_gradient(x.head(Nphi_u), mu);
    volVectorField &a_tmp = pair.first();
    Eigen::MatrixXd a_grad = pair.second();

    fvMesh &mesh = problem->_mesh();
    auto phi = linearInterpolate(a_tmp) & mesh.Sf();

    auto a_old = g_old();
    volVectorField& tmp = a_tmp.oldTime();
    tmp = a_old;

    fvVectorMatrix resEqn(
        fvm::ddt(a_tmp) + 0.5 * fvm::div(phi(), a_tmp) - fvm::laplacian(dimensionedScalar(dimViscosity, nu.value()), a_tmp));

    Eigen::SparseMatrix<double> dres;
    Foam2Eigen::fvMatrix2EigenM<Foam::Vector<double>, decltype(dres)>(resEqn, dres);

    fjac = dres.block(0, 0, this->embedding->output_dim * 2, this->embedding->output_dim * 2) * a_grad;
    cnpy::save(fjac, "jacobian.npy");

    Eigen::NumericalDiff<newton_nmlspg_burgers, Eigen::Central> numDiff2(*this, 1.e-02);
    numDiff2.df(x, fjac);
    cnpy::save(fjac, "fjac_central_2.npy");

    Eigen::NumericalDiff<newton_nmlspg_burgers, Eigen::Central> numDiff3(*this, 1.e-03);
    numDiff3.df(x, fjac);
    cnpy::save(fjac, "fjac_central_3.npy");

    Eigen::NumericalDiff<newton_nmlspg_burgers, Eigen::Central> numDiff4(*this, 1.e-04);
    numDiff4.df(x, fjac);
    cnpy::save(fjac, "fjac_central_4.npy");

    Eigen::NumericalDiff<newton_nmlspg_burgers, Eigen::Central> numDiff5(*this, 1.e-05);
    numDiff5.df(x, fjac);
    cnpy::save(fjac, "fjac_central_5.npy");

    return 0;
}

// * * * * * * * * * * * * * * * Solve Functions  * * * * * * * * * * * * * //
void NonlinearReducedBurgers::solveOnline(Eigen::MatrixXd mu, int startSnap)
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

    // Counter of the number of online solutions saved, accounting also time as parameter
    int counter2 = 0;

    // Set number of online solutions
    int Ntsteps = static_cast<int>((finalTime - tstart) / dt);
    int onlineSizeTimeSeries = static_cast<int>(Ntsteps / numberOfStores);

    // resize the online solution list with the length of n_parameters times
    // length of the time series
    online_solution.resize((mu.cols()) * (onlineSizeTimeSeries));

    // Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 276 #################### " << endl;

    // Iterate online solution for each parameter saved row-wise in mu
    for (int n_param = 0; n_param < mu.cols(); n_param++)
    {
        // Set the initial time
        time = tstart;

        // Counter of the number of saved time steps for the present parameter with index n_param
        int counter = 0;
        int nextStore = 0;

        // Create and resize the solution vector (column vector)
        y.resize(Nphi_u, 1);
        y = embedding->latent_initial.transpose();
        newton_object.g_old = embedding->forward(y, mu(0, n_param));

        auto tmp = newton_object.g_old();
        uRecFields.append(tmp);

        // Set some properties of the newton object
        newton_object.mu = mu(0, n_param);
        newton_object.nu = nu;
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
        // Eigen::NumericalDiff<newton_nmlspg_burgers, Eigen::Central> numDiffobject(newton_object, 1.e-02);
        Eigen::LevenbergMarquardt<decltype(newton_object)> lm(newton_object);
        // Eigen::LevenbergMarquardt<decltype(newton_object)> lm(newton_object);
        lm.parameters.factor = 10;
        lm.parameters.xtol = 1.49012e-08;
        lm.parameters.ftol = 1.49012e-08;
        lm.parameters.maxfev = 2000;

        Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 346 #################### START MINIMIZE" << endl;

        // Set output colors for fancy output
        Color::Modifier red(Color::FG_RED);
        Color::Modifier green(Color::FG_GREEN);
        Color::Modifier def(Color::FG_DEFAULT);

        time = time + dt;

        while (time < finalTime)
        {
            auto start = std::chrono::system_clock::now();
            lm.minimize(y);
            auto end = std::chrono::system_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

            Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 334 #################### END MINIMIZE " << elapsed.count() << " ms, x=(" << y(0) << " " << y(1) << " " << y(2) << " " << y(3) << ")" << endl;

            Eigen::VectorXd res(2 * newton_object.embedding->output_dim);
            res.setZero();

            newton_object(y, res);
            newton_object.g_old = embedding->forward(y, mu(0, n_param));
            auto tmp = newton_object.g_old();
            uRecFields.append(tmp);

            std::cout << "################## Online solve N° " << counter << " ##################" << std::endl;
            Info << "Time = " << time << endl;

            if (res.norm() < 1e-5)
            {
                std::cout << green << "|F(x)| = " << res.norm() << " - Minimun reached in " << lm.iter << " iterations " << def << std::endl
                          << std::endl;
            }
            else
            {
                std::cout << red << "|F(x)| = " << res.norm() << " - Minimun reached in " << lm.iter << " iterations " << def << std::endl
                          << std::endl;
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
                counter2++;
            }

            counter++;
            time = time + dt;
        }
    }
}

// * * * * * * * * * * * * * * *  Evaluation  * * * * * * * * * * * * * //

void NonlinearReducedBurgers::reconstruct(bool exportFields, fileName folder)
{
    if (exportFields)
    {
        mkDir(folder);
        ITHACAutilities::createSymLink(folder);
    }
    // Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 719 #################### " << Nphi_u << endl;
    int counter = 0;
    int nextwrite = 0;
    List<Eigen::MatrixXd> CoeffU;
    List<double> tValues;
    CoeffU.resize(0);
    tValues.resize(0);
    int exportEveryIndex = round(exportEvery / storeEvery);
    std::vector<torch::jit::IValue> input;

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

            // torch::Tensor tensor = torch2Eigen::eigenMatrix2torchTensor(currentUCoeff);
            // input.push_back(tensor);
            // torch::Tensor tensor_forwarded = embedding->forward(std::move(tensor)).toTensor();
            // auto tensor_stacked = torch::cat({std::move(tensor_forwarded).reshape({2, 60, 60}), torch::zeros({1, 60, 60})}, 0).reshape({3, -1}).transpose(0, 1);
            // auto g = torch2Foam::torch2Field<vector>(tensor_stacked);
            // uRecFields.append(g);
        }

        counter++;
    }

    if (exportFields)
    {
        ITHACAstream::exportFields(uRecFields, folder,
                                   "uRec");
    }
}

//TODO: not implemented yet
void NonlinearReducedBurgers::reconstruct(bool exportFields, fileName folder, Eigen::MatrixXd redCoeff)
{
    if (exportFields)
    {
        mkDir(folder);
        ITHACAutilities::createSymLink(folder);
    }
    // Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 759 #################### " << Nphi_u << endl;
    int counter = 0;
    int nextwrite = 0;
    List<Eigen::MatrixXd> CoeffU;
    List<double> tValues;
    CoeffU.resize(0);
    tValues.resize(0);
    int exportEveryIndex = round(exportEvery / storeEvery);

    for (int i = 0; i < redCoeff.rows(); i++)
    {
        if (counter == nextwrite)
        {
            // Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 773 #################### " << redCoeff(i, 0) << " " << redCoeff(i, 1) << " " << redCoeff(i, 2) << " " << redCoeff(i, 3) << endl;

            Eigen::MatrixXd currentUCoeff(Nphi_u, 1);
            // currentUCoeff = Eigen::MatrixXd::Ones(Nphi_u, 1);

            // Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 775 #################### " << currentUCoeff.rows() << " " << redCoeff.row(i).tail(Nphi_u).cols() << " " << redCoeff.row(i).tail(Nphi_u).transpose().rows() << endl;

            currentUCoeff.col(0) = redCoeff.row(i).tail(Nphi_u).transpose();

            // Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 776 #################### " << currentUCoeff.rows() << endl; //currentUCoeff(0, 0) << " " << currentUCoeff(1, 0) << " " << currentUCoeff(2, 0) << endl;

            CoeffU.append(currentUCoeff);
            nextwrite += exportEveryIndex;
            double timeNow = redCoeff(i, 0);
            tValues.append(timeNow);
            // Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 779 #################### " << counter << endl;
        }

        counter++;
    }

    volVectorField uRec("uRec", Umodes[0] * 0);

    // Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 786 #################### " << endl;

    uRecFields = problem->L_Umodes.reconstruct(uRec, CoeffU, "uRec");

    // Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 787 #################### " << endl;

    if (exportFields)
    {
        ITHACAstream::exportFields(uRecFields, folder,
                                   "uRec");
    }
}

Eigen::MatrixXd NonlinearReducedBurgers::setOnlineVelocity(Eigen::MatrixXd vel)
{
    assert(problem->inletIndex.rows() == vel.rows() && "Imposed boundary conditions dimensions do not match given values matrix dimensions");
    Eigen::MatrixXd vel_scal;
    vel_scal.resize(vel.rows(), vel.cols());

    for (int k = 0; k < problem->inletIndex.rows(); k++)
    {
        int p = problem->inletIndex(k, 0);
        int l = problem->inletIndex(k, 1);
        scalar area = gSum(problem->liftfield[0].mesh().magSf().boundaryField()[p]);
        scalar u_lf = gSum(problem->liftfield[k].mesh().magSf().boundaryField()[p] *
                           problem->liftfield[k].boundaryField()[p])
                          .component(l) /
                      area;

        for (int i = 0; i < vel.cols(); i++)
        {
            vel_scal(k, i) = vel(k, i) / u_lf;
        }
    }

    return vel_scal;
}

void NonlinearReducedBurgers::trueProjection(fileName folder)
{

    // Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 841 #################### " << problem->Ufield.size() << " " << Nphi_u << endl;

    // CoeffU size: (Umodes[0].size(), Nphi_u)
    List<Eigen::MatrixXd> CoeffU;
    CoeffU.resize(0);

    for (int n_index = 0; n_index < problem->Ufield.size(); n_index++)
    {
        Eigen::MatrixXd currentUCoeff(Nphi_u, 1);
        // currentUCoeff = Eigen::MatrixXd::Ones(Nphi_u, 1);

        // Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 852 #################### " << currentUCoeff.rows() << endl;

        currentUCoeff.col(0) = ITHACAutilities::getCoeffs(problem->Ufield[n_index], Umodes);

        // Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 856 #################### " << currentUCoeff.rows() << endl;

        CoeffU.append(currentUCoeff);
    }

    volVectorField uRec("uRec", Umodes[0] * 0);

    // Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 863 #################### " << endl;

    uRecFields = problem->L_Umodes.reconstruct(uRec, CoeffU, "uRec");

    // Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 867 #################### " << uRecFields.size() << endl;

    ITHACAstream::exportFields(uRecFields, folder, "uTrueProjection");
}