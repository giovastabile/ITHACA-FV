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

#include "NonlinearReducedBurgers.H"

// * * * * * * * * * * * * * * * Constructors * * * * * * * * * * * * * * * * //

// Constructor
NonlinearReducedBurgers::NonlinearReducedBurgers()
{
    para = ITHACAparameters::getInstance();
}

NonlinearReducedBurgers::NonlinearReducedBurgers(Burgers &FOMproblem, fileName decoder_path, int dim_latent)
    : Nphi_u{dim_latent},
      problem{&FOMproblem}
{
    embedding = autoPtr<Embedding>(new Embedding(Nphi_u, decoder_path, problem->L_Umodes[0]));
    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 61 #################### " << embedding.ref().latent_dim << endl;
    // FOMproblem is only needed for boundary conditions
    newton_object = newton_nmlspg_burgers(Nphi_u, Nphi_u, FOMproblem, embedding.ref());
    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 61 #################### " << newton_object.embedding->latent_dim << " " << newton_object.embedding->output_dim << endl;

    // Embedding embedding_test(embedding.ref());
    // Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 67 #################### " << embedding_test.latent_dim << " " << embedding_test.output_dim << endl;
}

Embedding::Embedding(int dim, fileName decoder_path, volVectorField &U0) : latent_dim{dim}
{
    output_dim = U0.size();
    decoder = autoPtr<torch::jit::script::Module>(new torch::jit::script::Module(torch::jit::load(decoder_path)));

    // define initial velocity field
    _U0 = autoPtr<volVectorField>(new volVectorField(U0));
    _g0 = autoPtr<volVectorField>(new volVectorField(U0));

    // define the value the assumes at the zero vector
    std::vector<torch::jit::IValue> input;
    input.push_back(torch::zeros({1, latent_dim}));
    at::Tensor tensor = decoder->forward(std::move(input)).toTensor();

    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.H, line 78 #################### " << latent_dim << " " << output_dim <<  endl;

    // tensor must be scalar, tensor components of velocity must be saved
    // contiguously in memory, the order of the vector must be the one of cells
    // in the mesh.
    tensor = tensor.reshape({-1});
    auto g0 = ITHACAtorch::torch2Foam::torch2Field<vector>(tensor);
    _g0.ref().ref().field() = std::move(g0);

    PtrList<volVectorField> save_field;
    save_field.append(_g0());
    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 91 #################### " << endl;

    auto test_ref_0 = embedding_ref(1);
    save_field.append(test_ref_0() - _g0());
    ITHACAstream::exportFields(save_field, "./REF", "g0");
}

// Embedding::Embedding(Embedding& embedding)
// {

//     /// Initial condition
//     autoPtr<volVectorField> _U0;

//     /// value of the decoder at the zero vector
//     autoPtr<volVectorField> _g0;
// }

autoPtr<volVectorField> Embedding::embedding_ref(const scalar mu)
{
    return autoPtr<volVectorField>(new volVectorField(mu * _U0() + _g0()));
}

std::pair<autoPtr<volVectorField>, autoPtr<Eigen::MatrixXd>> Embedding::forward_with_gradient(const Eigen::VectorXd &x, const scalar mu)
{
    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 106 #################### " << endl;
    std::vector<torch::jit::IValue> input;
    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 108 #################### " << endl;
    Eigen::MatrixXd input_matrix{x};
    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 110 #################### " << latent_dim << " " << output_dim <<  endl;
    input_matrix.resize(1, latent_dim);
    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 112 #################### " << endl;
    // input_matrix.row(0) = x;
    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 111 #################### " << endl;
    torch::Tensor input_tensor = ITHACAtorch::torch2Eigen::eigenMatrix2torchTensor(input_matrix);
    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 111 #################### " << std::to_string(input_tensor.dim()) << endl;
    input_tensor = input_tensor.reshape({1, latent_dim});
    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 130 #################### " << endl;
    input_tensor = input_tensor.set_requires_grad(true);
    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 132 #################### " << endl;
    input.push_back(input_tensor);
    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 134 #################### " << endl;
    torch::Tensor push_forward_tensor = decoder->forward(input).toTensor().squeeze();
    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 115 #################### " << endl;

    // preallocate jacobian
    int jacobian_out_dim = output_dim * 2;
    auto J = torch::ones({jacobian_out_dim, latent_dim});
    // preallocate time variables
    auto start = std::chrono::system_clock::now();
    auto end = std::chrono::system_clock::now();
    auto elapsed = end - start;

    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 143 #################### " << jacobian_out_dim << " " << push_forward_tensor.sizes()[0] << " " << push_forward_tensor.sizes()[1] << " " << push_forward_tensor.sizes()[2] << endl;

    std::vector<torch::jit::IValue> input_jac;
    auto input_repeated = input_tensor.repeat({jacobian_out_dim, 1});
    input_repeated = input_repeated.set_requires_grad(true);
    input_jac.push_back(input_repeated);
    auto grad_output = torch::eye(output_dim*3).slice(/*dim=*/0,/*start=*/0, jacobian_out_dim) ;

    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 149 #################### " << endl;
    // compute the jacobian of out wrt input
    start = std::chrono::system_clock::now();
    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 152 #################### " << endl;
    torch::Tensor forward_tensor = decoder->forward(input_jac).toTensor().squeeze();
    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 154 #################### " << forward_tensor.dim() << " " << forward_tensor.sizes()[0] << " " << forward_tensor.sizes()[1] << endl;

    // forward_tensor.backward(grad_output);


    auto gradient = torch::autograd::grad({forward_tensor},
                                        {input_repeated},
                                        /*grad_outputs=*/{grad_output},
                                        /*retain_graph=*/true,
                                        /*create_graph=*/true);
    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 160 #################### " << endl;
    // Info << "gradient size: " << gradient.size() << std::endl;
    auto grad = gradient[0];


    J.slice(/*dim=*/0,/*start=*/0, /*end=*/jacobian_out_dim) = grad;// input_repeated.grad();


    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 161 #################### " << J.dim() << " " << J.sizes()[0] << " " << J.sizes()[1] << endl;
    // for (int i = 0; i < jacobian_out_dim; i++)
    // {
    //     Info << "Jacobian of embedding teration : " << i << endl;
    //     auto grad_output = torch::ones({1});

    //     auto gradient = torch::autograd::grad({push_forward_tensor.slice(
    //                                           /*dim=*/0,
    //                                           /*start=*/i,
    //                                           /*end=*/i + 1)},
    //                                           {input_tensor},
    //                                           /*grad_outputs=*/{grad_output},
    //                                           /*retain_graph=*/true,
    //                                           /*create_graph=*/true);
    //     // Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 160 #################### " << endl;
    //     // Info << "gradient size: " << std::to_string(gradient.size()) << std::endl;
    //     auto grad = gradient[0];
    //     // Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 161 #################### " << grad.dim() << " " << grad.sizes()[0] << " " << grad.sizes()[1] << " "  << grad.sizes()[2] << " "  << grad.sizes()[3] << endl;
    //     // Info << "grad size: " << grad.size(1) << std::endl;
    //     J.slice(/*dim=*/0,/*start=*/i, /*end=*/i + 1) = grad;
    // }
    end = std::chrono::system_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    Info << "Jacobian: " << endl;
    //std::cout << J << std::endl;
    Info << "duration: " << elapsed.count() << endl;

    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 140 #################### " << endl;

    auto g = autoPtr<volVectorField>(new volVectorField(_U0()));
    auto push_forward = ITHACAtorch::torch2Foam::torch2Field<vector>(push_forward_tensor);

    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 150 #################### " << endl;
    g.ref().ref().field() = std::move(push_forward);
    g.ref() += embedding_ref(mu).ref();

    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 153 #################### " << endl;
    auto grad_eigen = ITHACAtorch::torch2Eigen::torchTensor2eigenMatrix<double>(J);
    auto dg = autoPtr<Eigen::MatrixXd>(new Eigen::MatrixXd(std::move(grad_eigen)));
    return std::make_pair(g, dg);
}

// Operator to evaluate the residual
int newton_nmlspg_burgers::operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const
{
    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 64 #################### "
         << "residual call " << this->embedding->latent_dim << " " << this->embedding->output_dim << endl;

    auto pair = embedding->forward_with_gradient(x.head(Nphi_u), mu);
    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 136 #################### " << endl;
    Foam::volVectorField& a_tmp = pair.first();
    Eigen::MatrixXd a_grad = pair.second();

    Foam::fvMesh &mesh = problem->_mesh();
    auto phi = linearInterpolate(a_tmp) & mesh.Sf();

    Foam::fvVectorMatrix resEqn(
        fvm::ddt(a_tmp) + 0.5 * fvm::div(phi(), a_tmp) - fvm::laplacian(nu, a_tmp));

    a_tmp.ref().field() = resEqn.residual();
    Eigen::VectorXd res = Foam2Eigen::field2Eigen(a_tmp);
    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 179 #################### " << res.rows() << "x" << res.cols() << endl;
    fvec = a_grad.transpose() * res;

    return 0;
}

int newton_nmlspg_burgers::df(const Eigen::VectorXd &x,
                              Eigen::MatrixXd &fjac) const
{
    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 87 #################### "
         << "fjac call" << endl;

    auto pair = embedding->forward_with_gradient(x.head(Nphi_u), mu);
    volVectorField &a_tmp = pair.first();
    Eigen::MatrixXd a_grad = pair.second();

    fvMesh &mesh = problem->_mesh();
    auto phi = linearInterpolate(a_tmp) & mesh.Sf();

    fvVectorMatrix resEqn(
        fvm::ddt(a_tmp) + 0.5 * fvm::div(phi(), a_tmp) - fvm::laplacian(nu, a_tmp));

    Eigen::SparseMatrix<double> dres;
    Foam2Eigen::fvMatrix2EigenM<Foam::Vector<double>, decltype(dres)>(resEqn, dres);
    Eigen::MatrixXd dresJac = dres * a_grad;
    fjac = dresJac.transpose() * dresJac;

    return 0;
}

// * * * * * * * * * * * * * * * Solve Functions  * * * * * * * * * * * * * //
void NonlinearReducedBurgers::solveOnline(Eigen::MatrixXd mu, int startSnap)
{
     Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 261 #################### " << newton_object.embedding->output_dim << " " << newton_object.embedding->latent_dim << endl;

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

    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 261 #################### " << newton_object.embedding->output_dim << " " << newton_object.embedding->latent_dim << endl;

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

        Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 261 #################### " << newton_object.embedding->output_dim << " " << newton_object.embedding->latent_dim << endl;
        // Set some properties of the newton object
        newton_object.mu = mu(0, n_param);
        newton_object.nu = nu;
        newton_object.y_old = y;
        newton_object.yOldOld = newton_object.y_old;
        newton_object.dt = dt;
        newton_object.tauU = tauU;
        Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 261 #################### " << newton_object.embedding->output_dim << " " << newton_object.embedding->latent_dim << endl;
        // Create vector to store temporal solution and save initial condition as first solution
        Eigen::MatrixXd tmp_sol(Nphi_u + 1, 1);
        tmp_sol(0) = time;
        tmp_sol.col(0).tail(y.rows()) = y;

        online_solution[counter2] = tmp_sol;
        counter2++;
        counter++;
        nextStore += numberOfStores;

        Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 261 #################### " << newton_object.embedding->output_dim << " " << newton_object.embedding->latent_dim << endl;
        // Create nonlinear solver object
        Eigen::HybridNonLinearSolver<newton_nmlspg_burgers> hnls(newton_object);
        Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 264 #################### " << endl;
        // Set output colors for fancy output
        Color::Modifier red(Color::FG_RED);
        Color::Modifier green(Color::FG_GREEN);
        Color::Modifier def(Color::FG_DEFAULT);

        time = time + dt;

        while (time < finalTime)
        {
            Eigen::VectorXd res(y);
            res.setZero();
            Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 276 #################### " << endl;
            hnls.solve(y);
            Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 278 #################### " << endl;
            newton_object.operator()(y, res);
            newton_object.yOldOld = newton_object.y_old;
            newton_object.y_old = y;

            std::cout << "################## Online solve N° " << counter << " ##################" << std::endl;
            Info << "Time = " << time << endl;

            if (res.norm() < 1e-5)
            {
                std::cout << green << "|F(x)| = " << res.norm() << " - Minimun reached in " << hnls.iter << " iterations " << def << std::endl
                          << std::endl;
            }
            else
            {
                std::cout << red << "|F(x)| = " << res.norm() << " - Minimun reached in " << hnls.iter << " iterations " << def << std::endl
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
    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 719 #################### " << Nphi_u << endl;
    int counter = 0;
    int nextwrite = 0;
    List<Eigen::MatrixXd> CoeffU;
    List<double> tValues;
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

void NonlinearReducedBurgers::reconstruct(bool exportFields, fileName folder, Eigen::MatrixXd redCoeff)
{
    if (exportFields)
    {
        mkDir(folder);
        ITHACAutilities::createSymLink(folder);
    }
    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 759 #################### " << Nphi_u << endl;
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
            Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 773 #################### " << redCoeff(i, 0) << " " << redCoeff(i, 1) << " " << redCoeff(i, 2) << " " << redCoeff(i, 3) << endl;

            Eigen::MatrixXd currentUCoeff(Nphi_u, 1);
            // currentUCoeff = Eigen::MatrixXd::Ones(Nphi_u, 1);

            Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 775 #################### " << currentUCoeff.rows() << " " << redCoeff.row(i).tail(Nphi_u).cols() << " " << redCoeff.row(i).tail(Nphi_u).transpose().rows() << endl;

            currentUCoeff.col(0) = redCoeff.row(i).tail(Nphi_u).transpose();

            Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 776 #################### " << currentUCoeff.rows() << endl; //currentUCoeff(0, 0) << " " << currentUCoeff(1, 0) << " " << currentUCoeff(2, 0) << endl;

            CoeffU.append(currentUCoeff);
            nextwrite += exportEveryIndex;
            double timeNow = redCoeff(i, 0);
            tValues.append(timeNow);
            Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 779 #################### " << counter << endl;
        }

        counter++;
    }

    volVectorField uRec("uRec", Umodes[0] * 0);

    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 786 #################### " << endl;

    uRecFields = problem->L_Umodes.reconstruct(uRec, CoeffU, "uRec");

    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 787 #################### " << endl;

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

    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 841 #################### " << problem->Ufield.size() << " " << Nphi_u << endl;

    // CoeffU size: (Umodes[0].size(), Nphi_u)
    List<Eigen::MatrixXd> CoeffU;
    CoeffU.resize(0);

    for (int n_index = 0; n_index < problem->Ufield.size(); n_index++)
    {
        Eigen::MatrixXd currentUCoeff(Nphi_u, 1);
        // currentUCoeff = Eigen::MatrixXd::Ones(Nphi_u, 1);

        Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 852 #################### " << currentUCoeff.rows() << endl;

        currentUCoeff.col(0) = ITHACAutilities::getCoeffs(problem->Ufield[n_index], Umodes);

        Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 856 #################### " << currentUCoeff.rows() << endl;

        CoeffU.append(currentUCoeff);
    }

    volVectorField uRec("uRec", Umodes[0] * 0);

    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 863 #################### " << endl;

    uRecFields = problem->L_Umodes.reconstruct(uRec, CoeffU, "uRec");

    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/src/ITHACA_ROMPROBLEMS/NonlinearReducedBurgers/NonlinearReducedBurgers.C, line 867 #################### " << uRecFields.size() << endl;

    ITHACAstream::exportFields(uRecFields, folder, "uTrueProjection");
}