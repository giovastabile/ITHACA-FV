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


#include "NonlinearReducedBurgers_true.H"

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

    // problem->L_Umodes is used to create volVectorFields
    embedding = autoPtr<Embedding>(new Embedding(Nphi_u, decoder_path, problem->L_Umodes[0], latent_initial));

    // FOMproblem is only needed for initial conditions
    newton_object = newton_nmlspg_burgers(Nphi_u, 2*embedding->output_dim, FOMproblem, embedding.ref(), problem->L_Umodes[0]);
}

Embedding::Embedding(int dim, fileName decoder_path, volVectorField &U0, Eigen::MatrixXd lat_init) : latent_dim{dim}, latent_initial{lat_init}
{
    // get the number of degrees of freedom relative to a single component
    output_dim = U0.size(); // 3600

    decoder = autoPtr<torch::jit::script::Module>(new torch::jit::script::Module(torch::jit::load(decoder_path)));

    // define initial velocity field _U0 used to define the reference snapshot
    // and initialize decoder output variable g0
    _U0 = autoPtr<volVectorField>(new volVectorField(U0));
    _g0 = autoPtr<volVectorField>(new volVectorField(U0));

    // declare input of decoder of type IValue since the decoder is loaded from pytorch
    std::vector<torch::jit::IValue> input;
    torch::Tensor latent_initial_tensor = torch2Eigen::eigenMatrix2torchTensor(latent_initial);

    // the tensor inputs of the decoder must be of type at::kFloat (not double)
    input.push_back(latent_initial_tensor.to(at::kFloat).to(torch::kCUDA));

    std::cout << "LATENT INITIAL" << latent_initial_tensor << std::endl;

    torch::Tensor tensor = decoder->forward(std::move(input)).toTensor().to(torch::kCPU);

    // add the z component to the tensor as a zero {1, 60, 60} tensor and
    // reshape the tensor s.t. the components x,y,z of a single cell center are
    // contiguous in memory (this is necessary for torch2field method)
    auto tensor_stacked = torch::cat({std::move(tensor).reshape({2, 60, 60}), torch::zeros({1, 60, 60})}, 0).reshape({3, -1}).transpose(0, 1).contiguous();


    auto g0 = torch2Foam::torch2Field<vector>(tensor_stacked);
    _g0.ref().ref().field() = std::move(g0);

    // save_field.append(_g0());
    // ITHACAstream::exportFields(save_field, "./REF", "g0");
}

// private method used only inside Embedding::forward. Return reference element of embedding s.t. initial embedding is mu * _U0()
autoPtr<volVectorField> Embedding::embedding_ref(const scalar mu)
{
    return autoPtr<volVectorField>(new volVectorField(mu * _U0() - _g0()));
}

autoPtr<volVectorField> Embedding::forward(const Eigen::VectorXd &x, const scalar mu)
{
    // declare input of decoder of type IValue since the decoder is loaded from pytorch
    std::vector<torch::jit::IValue> input;
    Eigen::MatrixXd input_matrix{x};

    input_matrix.resize(1, latent_dim);
    torch::Tensor input_tensor = torch2Eigen::eigenMatrix2torchTensor(std::move(input_matrix));
    input_tensor = input_tensor.reshape({1, latent_dim});
    input_tensor = input_tensor.set_requires_grad(true);

    // the tensor inputs of the decoder must be of type at::kFloat (not double)
    input.push_back(input_tensor.to(at::kFloat).to(torch::kCUDA));

    torch::Tensor push_forward_tensor = decoder->forward(std::move(input)).toTensor().to(torch::kCPU);

    auto g = autoPtr<volVectorField>(new volVectorField(_U0()));

    // add the z component to the tensor as a zero {1, 60, 60} tensor and
    // reshape the tensor s.t. the components x,y,z of a single cell center are
    // contiguous in memory (this is necessary for torch2field method)
    auto tensor_stacked = torch::cat({push_forward_tensor.reshape({2, 60, 60}), torch::zeros({1, 60, 60})}, 0).reshape({3, -1}).transpose(0, 1).contiguous();

    auto push_forward = torch2Foam::torch2Field<vector>(tensor_stacked);

    // add reference term
    g.ref().ref().field() = std::move(push_forward);
    g.ref() += embedding_ref(mu).ref();

    // save_field.append(g());
    // ITHACAstream::exportFields(save_field, "./R", "g"+std::to_string(counter));
    return g;
}

/* Since torch::autograd::jacobian is not implemented in libtorch yet, this is
one of among the possible ways to compute the full jacobian with
torch::autograd::grad.The drawback is that 7200-by-4 repeated inputs are
forwarded to obtain an output of dimension 7200-by-7200 and then a costly
backward is computed. Since this operation could require a lot of GPU memory,
the evaluation of the components of the jacobian is split in 2 batches of
3600.*/
autoPtr<Eigen::MatrixXd> Embedding::jacobian(const Eigen::VectorXd &x, const scalar mu)
{
    // dimension of degrees of freedom associated to x and y components
    int jacobian_out_dim = output_dim * 2; // 7200
    Eigen::MatrixXd input_matrix{x};

    input_matrix.resize(1, latent_dim);
    torch::Tensor input_tensor = torch2Eigen::eigenMatrix2torchTensor(std::move(input_matrix));
    input_tensor = input_tensor.reshape({1, latent_dim}).set_requires_grad(true);

    // compute the jacobian with batches of 3600 for a total of 7200 components.
    // Since torch::autograd::
    auto input_repeated = input_tensor.repeat({3600, 1});
    input_repeated = input_repeated.set_requires_grad(true);

    // declare input of decoder of type IValue since the decoder is loaded from
    // pytorch. The tensor inputs of the decoder must be of type at::kFloat (not double)
    std::vector<torch::jit::IValue> input_jac;
    input_jac.push_back(input_repeated.to(at::kFloat).to(torch::kCUDA));

    // term to multiply with matrix-to-matrix product with the Jacobian of the
    // net: since it is the identity 7200-by-7200 matrix we obtainexactly the Jacobian.
    auto grad_output = torch::eye(output_dim * 2);

    // initialize the jacobian of the decoder of size jacobian_out_dim-by-latent_dim
    torch::Tensor forward_tensor = decoder->forward(input_jac).toTensor().squeeze();
    auto J = torch::ones({jacobian_out_dim, latent_dim});

    // compute the jacobian with batches of 3600 for a total of 7200 components
    for(int i=0; i<2; i++)
    {
        auto grad_component = grad_output.slice(0, i*3600, (1+i)*3600).to(torch::kCUDA);
        forward_tensor.backward(grad_component, true);

        auto gradient = torch::autograd::grad({forward_tensor},
                                          {input_repeated},
                                          /*grad_outputs=*/{grad_component},
                                          /*retain_graph=*/true,
                                          /*create_graph=*/true);

        J.slice(/*dim*/0, i*3600, (1+i)*3600) = gradient[0].detach();
    }

    auto grad_eigen = torch2Eigen::torchTensor2eigenMatrix<double>(J);
    auto dg = autoPtr<Eigen::MatrixXd>(new Eigen::MatrixXd(std::move(grad_eigen)));

    return dg;
}

std::pair<autoPtr<volVectorField>, autoPtr<Eigen::MatrixXd>> Embedding::forward_with_gradient(const Eigen::VectorXd &x, const scalar mu)
{
    auto g = forward(x, mu);
    auto dg = jacobian(x, mu);
    return std::make_pair(g, dg);
}

// Operator to evaluate the residual
int newton_nmlspg_burgers::operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const
{
    Info << " residual, x = " << x(0) << " " << x(1) << " " << x(2) << " " << x(3) << endl;

    auto g = embedding->forward(x.head(Nphi_u), mu);
    volVectorField& a_tmp = g();
    fvMesh& mesh = problem->_mesh();
    auto phi = linearInterpolate(a_tmp) & mesh.Sf();

    auto a_old = g_old();
    volVectorField& tmp = a_tmp.oldTime();
    tmp = a_old;

    fvVectorMatrix resEqn(
        fvm::ddt(a_tmp) + 0.5 * fvm::div(phi, a_tmp) - fvm::laplacian(dimensionedScalar(dimViscosity, nu.value()), a_tmp));

    resEqn.solve();
    a_tmp.field() = resEqn.residual();

    fvec = Foam2Eigen::field2Eigen(a_tmp).col(0).head(this->embedding->output_dim * 2);

    // this->embedding->save_field.append(a_tmp);

    // if (this->embedding->counter == 2000) {
    //     std::cout << "SAVED" << std::endl;
    //     ITHACAstream::exportFields(this->embedding->save_field, "./RESIDUAL", "g");
    // }

    // this->embedding->counter++;

    Info << " residual norm: " << fvec.norm() << endl;

    return 0;
}

int newton_nmlspg_burgers::df(const Eigen::VectorXd &x,
                              Eigen::MatrixXd &fjac) const
{
    // cnpy::save(x,/*  */ "x.npy");
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

    fjac = (-1) * dres.block(0, 0, this->embedding->output_dim * 2,
    this->embedding->output_dim * 2) * a_grad;

    // cnpy::save(fjac, "jacobian.npy");
    // cnpy::save(Eigen::MatrixXd(dres), "system_df.npy");
    // cnpy::save(a_grad, "torch_grad.npy");

    // Eigen::NumericalDiff<newton_nmlspg_burgers, Eigen::Central> numDiff5(*this, 1.e-05);
    // numDiff5.df(x, fjac);
    // cnpy::save(fjac, "fjac_central_5.npy");
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

        Eigen::LevenbergMarquardt<decltype(newton_object)> lm(newton_object);

        lm.parameters.factor = 100; //step bound for the diagonal shift, is this related to damping parameter, lambda?
        lm.parameters.maxfev = 5000;//max number of function evaluations
        lm.parameters.xtol = 1.49012e-20; //tolerance for the norm of the solution vector
        lm.parameters.ftol = 1.49012e-20; //tolerance for the norm of the vector function
        lm.parameters.gtol = 0; // tolerance for the norm of the gradient of the error vector
        lm.parameters.epsfcn = 0; //error precision

        // Set output colors for fancy output
        Color::Modifier red(Color::FG_RED);
        Color::Modifier green(Color::FG_GREEN);
        Color::Modifier def(Color::FG_DEFAULT);

        time = time + dt;

        while (time < finalTime)
        {
            Eigen::LevenbergMarquardtSpace::Status ret = lm.minimize(y);

            std::cout << "LM finished with status: " << ret << std::endl;

            Info << " minimum: x=(" << y(0) << " " << y(1) << " " << y(2) << " " << y(3) << ")" << endl;

            Eigen::VectorXd res(2 * newton_object.embedding->output_dim);
            res.setZero();

            // update the old solution for the evaluation of the residual and jacobian
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
            Eigen::MatrixXd currentUCoeff(Nphi_u, 1);

            currentUCoeff.col(0) = redCoeff.row(i).tail(Nphi_u).transpose();

            CoeffU.append(currentUCoeff);
            nextwrite += exportEveryIndex;
            double timeNow = redCoeff(i, 0);
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
    List<Eigen::MatrixXd> CoeffU;
    CoeffU.resize(0);

    for (int n_index = 0; n_index < problem->Ufield.size(); n_index++)
    {
        Eigen::MatrixXd currentUCoeff(Nphi_u, 1);

        currentUCoeff.col(0) = ITHACAutilities::getCoeffs(problem->Ufield[n_index], Umodes);

        CoeffU.append(currentUCoeff);
    }

    volVectorField uRec("uRec", Umodes[0] * 0);

    uRecFields = problem->L_Umodes.reconstruct(uRec, CoeffU, "uRec");

    ITHACAstream::exportFields(uRecFields, folder, "uTrueProjection");
}