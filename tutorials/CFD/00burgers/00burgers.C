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

#include <torch/script.h>
#include <torch/torch.h>
#include "torch2Eigen.H"
#include "torch2Foam.H"
#include "Foam2Eigen.H"

#include "burgers.H"
#include "ITHACAPOD.H"
#include "ReducedBurgers.H"
#include "NonlinearReducedBurgers.H"
#include "ITHACAstream.H"
#include "cnpy.H"
#include <chrono>
#include <math.h>
#include <iomanip>
#include <string>
#include <algorithm>

using namespace ITHACAtorch;


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
            ITHACAstream::read_fields(Ufield, "U", folder);
            // ITHACAstream::exportFields(Ufield, "./TRAIN", "uTrain");
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

// torch::Tensor read_data(const std::string& data_path)
// {
//     Eigen::MatrixXd tensor_eig;
//     torch::Tensor tensor = torch2Eigen::eigenMatrix2torchTensor(cnpy::load(tensor_eig, data_path));
//     return tensor;
// };

// class MyDataset : public torch::data::Dataset<MyDataset>
// {
//     private:
//         torch::Tensor states_;

//     public:
//         explicit MyDataset(const std::string& data_path)
//             : states_(read_data(data_path)) {   };

//         torch::data::Example<> get(size_t index) override;
// };

// torch::data::Example<> MyDataset::get(size_t index)
// {

//     return states_{[index]};
// }

/*---------------------------------------------------------------------------*\
                               Starting the MAIN
\*---------------------------------------------------------------------------*/

void one_parameter_viscosity(tutorial00);
void train_one_parameter_initial_velocity(tutorial00);
void test_one_parameter_initial_velocity(tutorial00);
void nonlinear_one_parameter_initial_velocity(tutorial00);
void nonlinear_test_rec(tutorial00);
void nonlinear_test_data(tutorial00);

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

    tutorial00 example(argc_proc, argv_proc);

    if (std::strcmp(argv[1],"train")==0)
    {
        // save mu_samples and training snapshots reduced coefficients
        train_one_parameter_initial_velocity(example);
    }
    else if (std::strcmp(argv[1],"test")==0)
    {
        // compute FOM, ROM-intrusive, ROM-nonintrusive and evaluate errors
        test_one_parameter_initial_velocity(example);
    }
    else if (std::strcmp(argv[1],"nonlinear")==0)
    {
        // compute NM-LSPG and evaluate errors
        nonlinear_one_parameter_initial_velocity(example);
    }
    else if (std::strcmp(argv[1],"nltestrec")==0)
    {
        // reconstruct NM projected solutions and compute NM projection error
        nonlinear_test_rec(example);
    }
    else if (std::strcmp(argv[1],"nltestdata")==0)
    {
        // reconstruct NM-LSTM predicted solutions and compute rel L2 error
        nonlinear_test_data(example);
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
    example.timeStep = 0.0001;
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

void train_one_parameter_initial_velocity(tutorial00 train_FOM)
{
    // Read parameters from ITHACAdict file
    ITHACAparameters *para = ITHACAparameters::getInstance(train_FOM._mesh(),
                                                           train_FOM._runTime());
    int NmodesUout = para->ITHACAdict->lookupOrDefault<int>("NmodesUout", 15);
    int NmodesUproj = para->ITHACAdict->lookupOrDefault<int>("NmodesUproj", 10);

    /// Set the number of parameters
    train_FOM.Pnumber = 1;
    /// Set the dimension of the training set
    train_FOM.Tnumber = 6;
    /// Instantiates a void Pnumber-by-Tnumber matrix mu for the parameters and a void
    /// Pnumber-by-2 matrix mu_range for the ranges
    train_FOM.setParameters();
    // Set the parameter ranges
    train_FOM.mu_range(0, 0) = 0.8;
    train_FOM.mu_range(0, 1) = 1.2;
    // Generate a number of Tnumber linearly equispaced samples inside the parameter range
    train_FOM.genEquiPar();
    cnpy::save(train_FOM.mu, "parTrain.npy");

    // Time parameters
    train_FOM.startTime = 0;
    train_FOM.finalTime = 2;
    train_FOM.timeStep = 0.001;
    train_FOM.writeEvery = 1;

    // Perform The Offline Solve;
    train_FOM.offlineSolveInitialVelocity("./ITHACAoutput/Offline/Training/");

    // Perform a POD decomposition for velocity
    ITHACAPOD::getModes(train_FOM.Ufield, train_FOM.Umodes, train_FOM._U().name(),
                        train_FOM.podex, 0, 0, NmodesUout);

    Eigen::MatrixXd SnapMatrix = Foam2Eigen::PtrList2Eigen(train_FOM.Ufield);
    Info << "snapshots size: " << SnapMatrix.size() << endl;
    cnpy::save(SnapMatrix, "npSnapshots.npy");

    train_FOM.project("./Matrices", NmodesUproj);

    // The initial conditions are used as the first mode
    ITHACAstream::exportFields(train_FOM.L_Umodes, "./ITHACAoutput/POD_and_initial/", "U");

    Eigen::MatrixXd modes = Foam2Eigen::PtrList2Eigen(train_FOM.L_Umodes);
    Info << "modes size: " << modes.size() << endl;
    cnpy::save(modes, "npInitialAndModes.npy");

    ReducedBurgers reduced(train_FOM);

    // Set values of the reduced model
    reduced.nu = 0.0001;
    reduced.tstart = 0;
    reduced.finalTime = 2;
    reduced.dt = 0.001;
    reduced.storeEvery = 0.001;
    reduced.exportEvery = 0.001;

    reduced.solveOnline(train_FOM.mu, 0);
    ITHACAstream::exportMatrix(reduced.online_solution, "red_coeff", "python", "./ITHACAoutput/red_coeff");

    /// Set the dimension of the training set
    train_FOM.Tnumber = 1;
    // sample test set
    train_FOM.setParameters();
    // Set the parameter ranges
    train_FOM.mu_range(0, 0) = 0.8;
    train_FOM.mu_range(0, 1) = 1.2;
    // Generate a number of Tnumber linearly equispaced samples inside the parameter range
    train_FOM.genRandPar();
    cnpy::save(train_FOM.mu, "parTest.npy");

}

void test_one_parameter_initial_velocity(tutorial00 test_FOM)
{
    // Read parameters from ITHACAdict file
    ITHACAparameters *para = ITHACAparameters::getInstance(test_FOM._mesh(),
                                                           test_FOM._runTime());
    int NmodesUout = para->ITHACAdict->lookupOrDefault<int>("NmodesUout", 15);
    int NmodesUproj = para->ITHACAdict->lookupOrDefault<int>("NmodesUproj", 10);
    int NmodesUtest = para->ITHACAdict->lookupOrDefault<int>("NmodesUtest", 1);

    /// Set the number of parameters
    test_FOM.Pnumber = 1;
    /// Set the dimension of the test set
    test_FOM.Tnumber = NmodesUtest;

    // Generate a number of Tnumber linearly equispaced samples inside the parameter range
    Eigen::MatrixXd mu;
    test_FOM.mu = cnpy::load(mu, "parTest.npy");

    // Time parameters
    test_FOM.startTime = 0;
    test_FOM.finalTime = 2;
    test_FOM.timeStep = 0.001;
    test_FOM.writeEvery = 1;

    // Perform The Offline Solve;
    if (!ITHACAutilities::check_folder("./ITHACAoutput/Offline/Test/"))
    {
        test_FOM.offline = false;
        Info << "Offline Test data already exist, reading existing data" << endl;
    }

    test_FOM.offlineSolveInitialVelocity("./ITHACAoutput/Offline/Test/");
    Eigen::MatrixXd trueSnapMatrix = Foam2Eigen::PtrList2Eigen(test_FOM.Ufield);

    Info << "snapshots size: " << trueSnapMatrix.size() << test_FOM.Ufield.size() << endl;
    cnpy::save(trueSnapMatrix, "npTrueSnapshots.npy");

    test_FOM.NUmodes = NmodesUproj;
    ITHACAstream::read_fields(test_FOM.L_Umodes, "U", "./ITHACAoutput/POD_and_initial/", 1);
    ITHACAstream::exportFields(test_FOM.L_Umodes, "./TEST", "uTest");

    test_FOM.NL_Umodes = test_FOM.L_Umodes.size();
    test_FOM.evaluateMatrices();

    ReducedBurgers reduced_nonIntrusive(test_FOM);

    // Set values of the reduced_nonIntrusive model
    reduced_nonIntrusive.nu = 0.0001;
    reduced_nonIntrusive.tstart = 0;
    reduced_nonIntrusive.finalTime = 2;
    reduced_nonIntrusive.dt = 0.001;
    reduced_nonIntrusive.storeEvery = 0.001;
    reduced_nonIntrusive.exportEvery = 0.001;
    //reduced_nonIntrusive.Nphi_u = NmodesUproj;// the initial condition is added to the modes

    Eigen::MatrixXd nonIntrusiveCoeff;

    nonIntrusiveCoeff = cnpy::load(nonIntrusiveCoeff, "nonIntrusiveCoeff.npy", "rowMajor");

    // Reconstruct the solution and export it
    reduced_nonIntrusive.reconstruct(true, "./ITHACAoutput/ReconstructionNonIntrusive/", nonIntrusiveCoeff);

    Eigen::MatrixXd errL2UnonIntrusive = ITHACAutilities::errorL2Rel(test_FOM.Ufield, reduced_nonIntrusive.uRecFields);

    ITHACAstream::exportMatrix(errL2UnonIntrusive, "errL2UnonIntrusive", "matlab",
                               "./ITHACAoutput/ErrorsL2/");
    cnpy::save(errL2UnonIntrusive, "./ITHACAoutput/ErrorsL2/errL2UnonIntrusive.npy");

    ReducedBurgers reduced_intrusive(test_FOM);

    // Set values of the reduced model
    reduced_intrusive.nu = 0.0001;
    reduced_intrusive.tstart = 0;
    reduced_intrusive.finalTime = 2;
    reduced_intrusive.dt = 0.001;
    reduced_intrusive.storeEvery = 0.001;
    reduced_intrusive.exportEvery = 0.001;

    reduced_intrusive.solveOnline(test_FOM.mu, 1);
    ITHACAstream::exportMatrix(reduced_intrusive.online_solution, "red_coeff", "python", "./ITHACAoutput/red_coeff_intrusive");

    // Reconstruct the solution and export it
    reduced_intrusive.reconstruct(true, "./ITHACAoutput/ReconstructionIntrusive/");
    ITHACAstream::exportFields(reduced_intrusive.uRecFields, "./ITHACAoutput/ReconstructionIntrusive/", "uRec");


    Eigen::MatrixXd errL2Uintrusive = ITHACAutilities::errorL2Rel(test_FOM.Ufield,
                             reduced_intrusive.uRecFields);

    ITHACAstream::exportMatrix(errL2Uintrusive, "errL2UIntrusive", "matlab",
                               "./ITHACAoutput/ErrorsL2/");
    cnpy::save(errL2Uintrusive, "./ITHACAoutput/ErrorsL2/errL2UIntrusive.npy");

    // Evaluate the true projection error
    reduced_intrusive.trueProjection("./ITHACAoutput/Reconstruction/");

    Eigen::MatrixXd errL2UtrueProjection = ITHACAutilities::errorL2Rel(test_FOM.Ufield,
                             reduced_intrusive.uRecFields);

    ITHACAstream::exportMatrix(errL2UtrueProjection, "errL2UtrueProjectionROM", "matlab",
                               "./ITHACAoutput/ErrorsL2/");
    cnpy::save(errL2UtrueProjection, "./ITHACAoutput/ErrorsL2/errL2UtrueProjectionROM.npy");

}

void nonlinear_one_parameter_initial_velocity(tutorial00 test_FOM)
{
    // Read parameters from ITHACAdict file
    ITHACAparameters *para = ITHACAparameters::getInstance(test_FOM._mesh(),
                                                           test_FOM._runTime());
    int NmodesUout = para->ITHACAdict->lookupOrDefault<int>("NmodesUout", 15);
    int NmodesUproj = para->ITHACAdict->lookupOrDefault<int>("NmodesUproj", 10);
    int NmodesUtest = para->ITHACAdict->lookupOrDefault<int>("NmodesUtest", 100);
    int NnonlinearModes = para->ITHACAdict->lookupOrDefault<int>("NnonlinearModes", 4);

    /// Set the number of parameters
    test_FOM.Pnumber = 1;

    /// Set the dimension of the test set
    test_FOM.Tnumber = NmodesUtest;

    // load the test samples
    Eigen::MatrixXd mu;
    test_FOM.mu = cnpy::load(mu, "parTest.npy");

    // Time parameters
    test_FOM.startTime = 0;
    test_FOM.finalTime = 2;
    test_FOM.timeStep = 0.001;
    test_FOM.writeEvery = 1;

    Eigen::MatrixXd initial_latent;
    cnpy::load(initial_latent, "./Autoencoders/ConvolutionalAe/latent_initial_4.npy");

    std::cout << "LATENT INIT " << initial_latent << std::endl;

    // Perform The Offline Solve;
    if (!ITHACAutilities::check_folder("./ITHACAoutput/Offline/Test/"))
    {
        test_FOM.offline = false;
        Info << "Offline Test data already exist, reading existing data" << endl;
    }

    test_FOM.offlineSolveInitialVelocity("./ITHACAoutput/Offline/Test/");

    // load modes from training
    test_FOM.NUmodes = NmodesUproj;
    ITHACAstream::read_fields(test_FOM.L_Umodes, "U", "./ITHACAoutput/POD_and_initial/", 1, NmodesUout);
    test_FOM.NL_Umodes = test_FOM.L_Umodes.size();

    // NM-LSPG
    NonlinearReducedBurgers reduced_nm_lspg(test_FOM, "./Autoencoders/ConvolutionalAe/decoder_gpu_4.pt", NnonlinearModes, initial_latent);

    // Set values of the reduced model
    reduced_nm_lspg.nu = 0.0001;
    reduced_nm_lspg.tstart = 0;
    reduced_nm_lspg.finalTime = 2;
    reduced_nm_lspg.dt = 0.001;
    reduced_nm_lspg.storeEvery = 0.001;
    reduced_nm_lspg.exportEvery = 0.001;

    reduced_nm_lspg.solveOnline(test_FOM.mu, 1);
    ITHACAstream::exportMatrix(reduced_nm_lspg.online_solution, "red_coeff", "python", "./ITHACAoutput/red_coeff_NM_LSPG");

    // Reconstruct the solution and export it
    ITHACAstream::exportFields(reduced_nm_lspg.uRecFields, "./ITHACAoutput/ReconstructionNMLSPG/", "uRec");
    // reduced_nm_lspg.reconstruct(true,"./ITHACAoutput/ReconstructionNMLSPG/");

    Info << " #################### DEBUG ~/OpenFOAM/OpenFOAM-v2006/applications/utilities/ITHACA-FV/tutorials/CFD/00burgers/00burgers.C, line 482 #################### " << test_FOM.Ufield.size() << " " <<  reduced_nm_lspg.uRecFields.size() << " " << reduced_nm_lspg.online_solution.size() << endl;

    Eigen::MatrixXd errL2UNMLSPG = ITHACAutilities::errorL2Rel(test_FOM.Ufield,
                             reduced_nm_lspg.uRecFields);

    ITHACAstream::exportMatrix(errL2UNMLSPG, "errL2UNMLSPG", "matlab",
                               "./ITHACAoutput/ErrorsL2/");
    cnpy::save(errL2UNMLSPG, "./ITHACAoutput/ErrorsL2/errL2UNMLSPG.npy");
}

void nonlinear_test_rec(tutorial00 test_FOM)
{
    // Read parameters from ITHACAdict file
    ITHACAparameters *para = ITHACAparameters::getInstance(test_FOM._mesh(),
                                                           test_FOM._runTime());
    int NmodesUout = para->ITHACAdict->lookupOrDefault<int>("NmodesUout", 15);
    int NmodesUproj = para->ITHACAdict->lookupOrDefault<int>("NmodesUproj", 10);
    int NmodesUtest = para->ITHACAdict->lookupOrDefault<int>("NmodesUtest", 100);
    int NnonlinearModes = para->ITHACAdict->lookupOrDefault<int>("NnonlinearModes", 4);

    /// Set the number of parameters
    test_FOM.Pnumber = 1;

    /// Set the dimension of the test set
    test_FOM.Tnumber = NmodesUtest;

    // load the test samples
    Eigen::MatrixXd mu;
    test_FOM.mu = cnpy::load(mu, "parTest.npy");

    // Time parameters
    test_FOM.startTime = 0;
    test_FOM.finalTime = 2;
    test_FOM.timeStep = 0.001;
    test_FOM.writeEvery = 1;

    Eigen::MatrixXd initial_latent;
    cnpy::load(initial_latent, "./Autoencoders/ConvolutionalAe/latent_initial_4.npy");

    // Perform The Offline Solve;
    if (!ITHACAutilities::check_folder("./ITHACAoutput/Offline/Test/"))
    {
        test_FOM.offline = false;
        Info << "Offline Test data already exist, reading existing data" << endl;
    }

    test_FOM.offlineSolveInitialVelocity("./ITHACAoutput/Offline/Test/");

    // load modes from training
    test_FOM.NUmodes = NmodesUproj;
    ITHACAstream::read_fields(test_FOM.L_Umodes, "U", "./ITHACAoutput/POD_and_initial/", 1, NmodesUout);
    test_FOM.NL_Umodes = test_FOM.L_Umodes.size();

    // load the test snapshots
    // Generate your data set. At this point you can add transforms to you data
    // set, e.g. stack your batches into a single tensor.
    Eigen::MatrixXd dataset_eig;
    torch::Tensor inputs_true = torch2Eigen::eigenMatrix2torchTensor(cnpy::load(dataset_eig, "./npTrueSnapshots_framed.npy"));

    // load autoencoder
    auto ae = autoPtr<torch::jit::script::Module>(new torch::jit::script::Module(torch::jit::load("./Autoencoders/ConvolutionalAe/model_gpu_4.pt")));

    // forward autoencoder
    // auto rec_torch = torch::zeros({2001, 7200});
    std::vector<torch::jit::IValue> input;
    input.push_back(inputs_true.reshape({2001, 2, 60, 60}).to(torch::kCUDA));
    torch::Tensor forward_tensor = ae->forward(input).toTensor().squeeze().detach().to(torch::kCPU);

    auto tensor_stacked = torch::cat({forward_tensor.reshape({2001, 2, 60, 60}), torch::zeros({2001, 1, 60, 60})}, 1).reshape({2001, 3, -1}).transpose(1, 2).contiguous();

    PtrList<volVectorField> rec_fields;
    for(int i=0; i<2001; i++)
    {
        auto g = autoPtr<volVectorField>(new volVectorField(test_FOM._U0()));
        auto single_snap = tensor_stacked.slice(0, i, 1+i).squeeze();
        auto field = torch2Foam::torch2Field<vector>(single_snap);
        g.ref().ref().field() = field;
        rec_fields.append(g);
    }
    ITHACAstream::exportFields(rec_fields, "./REC_AE", "g0");


    // compute LRelError
    Eigen::MatrixXd errConsistency = ITHACAutilities::errorL2Rel(test_FOM.Ufield,
                             rec_fields);

    ITHACAstream::exportMatrix(errConsistency, "errConsistency", "matlab",
                               "./ITHACAoutput/ErrorsL2/");
    cnpy::save(errConsistency, "./ITHACAoutput/ErrorsL2/errConsistency.npy");
}

void nonlinear_test_data(tutorial00 test_FOM)
{
    // Read parameters from ITHACAdict file
    ITHACAparameters *para = ITHACAparameters::getInstance(test_FOM._mesh(),
                                                           test_FOM._runTime());
    int NmodesUout = para->ITHACAdict->lookupOrDefault<int>("NmodesUout", 15);
    int NmodesUproj = para->ITHACAdict->lookupOrDefault<int>("NmodesUproj", 10);
    int NmodesUtest = para->ITHACAdict->lookupOrDefault<int>("NmodesUtest", 100);
    int NnonlinearModes = para->ITHACAdict->lookupOrDefault<int>("NnonlinearModes", 4);

    /// Set the number of parameters
    test_FOM.Pnumber = 1;

    /// Set the dimension of the test set
    test_FOM.Tnumber = NmodesUtest;

    // load the test samples
    Eigen::MatrixXd mu;
    test_FOM.mu = cnpy::load(mu, "parTest.npy");

    // Time parameters
    test_FOM.startTime = 0;
    test_FOM.finalTime = 2;
    test_FOM.timeStep = 0.001;
    test_FOM.writeEvery = 1;

    Eigen::MatrixXd initial_latent;
    initial_latent = cnpy::load(initial_latent, "./Autoencoders/ConvolutionalAe/latent_initial_4.npy", "rowMajor");
    std::cout << "INITIAL" << initial_latent << std::endl;

    // Perform The Offline Solve;
    if (!ITHACAutilities::check_folder("./ITHACAoutput/Offline/Test/"))
    {
        test_FOM.offline = false;
        Info << "Offline Test data already exist, reading existing data" << endl;
    }

    test_FOM.offlineSolveInitialVelocity("./ITHACAoutput/Offline/Test/");

    // load modes from training
    test_FOM.NUmodes = NmodesUproj;
    ITHACAstream::read_fields(test_FOM.L_Umodes, "U", "./ITHACAoutput/POD_and_initial/", 1, NmodesUout);
    test_FOM.NL_Umodes = test_FOM.L_Umodes.size();

    // load the test snapshots
    Eigen::MatrixXd dataset_eig;
    dataset_eig = cnpy::load(dataset_eig, "./nonIntrusiveCoeffConvAe.npy", "rowMajor");
    std::cout << "EIG" << dataset_eig << std::endl;
    torch::Tensor inputs_hidden = torch2Eigen::eigenMatrix2torchTensor(dataset_eig);
    std::cout << "TEST" << inputs_hidden.to(at::kFloat) << std::endl;

    // load autoencoder
    auto decoder = autoPtr<torch::jit::script::Module>(new torch::jit::script::Module(torch::jit::load("./Autoencoders/ConvolutionalAe/decoder_gpu_4.pt")));

    // define initial velocity field and initialize decoder output variable g0
    auto _g0 = autoPtr<volVectorField>(new volVectorField(test_FOM._U0()));

    // declare input of decoder of type IValue since the decoder is loaded from pytorch
    std::vector<torch::jit::IValue> input_init;
    torch::Tensor latent_initial_tensor = torch2Eigen::eigenMatrix2torchTensor(initial_latent);
    std::cout << "TEST INIT" << latent_initial_tensor << std::endl;

    input_init.push_back(latent_initial_tensor.to(at::kFloat).to(torch::kCUDA));

    torch::Tensor tensor = decoder->forward(std::move(input_init)).toTensor().squeeze().detach().to(torch::kCPU);

    auto tensor_stacked_init = torch::cat({std::move(tensor).reshape({2, 60, 60}), torch::zeros({1, 60, 60})}, 0).reshape({3, -1}).transpose(0, 1).contiguous();
    auto g0 = torch2Foam::torch2Field<vector>(tensor_stacked_init);
    _g0.ref().ref().field() = std::move(g0);
    PtrList<volVectorField> save_field;
    save_field.append(_g0());
    ITHACAstream::exportFields(save_field, "./REFtest", "g0");

    // forward autoencoder
    std::vector<torch::jit::IValue> input;
    input.push_back(inputs_hidden.to(at::kFloat).to(torch::kCUDA));

    torch::Tensor forward_tensor = decoder->forward(input).toTensor().squeeze().detach().to(torch::kCPU);

    auto tensor_stacked = torch::cat({forward_tensor.reshape({2001, 2, 60, 60}), torch::zeros({2001, 1, 60, 60})}, 1).reshape({2001, 3, -1}).transpose(1, 2).contiguous();

    PtrList<volVectorField> rec_fields;
    for(int i=0; i<2001; i++)
    {
        auto g = autoPtr<volVectorField>(new volVectorField(test_FOM._U0()));
        auto single_snap = tensor_stacked.slice(0, i, 1+i).squeeze();
        auto field = torch2Foam::torch2Field<vector>(single_snap);
        g.ref().ref().field() = field;
        rec_fields.append(g);
    }
    ITHACAstream::exportFields(rec_fields, "./INTR", "g0");

    // compute LRelError
    Eigen::MatrixXd errConsistency = ITHACAutilities::errorL2Rel(test_FOM.Ufield,
                             rec_fields);

    ITHACAstream::exportMatrix(errConsistency, "errConsistency", "matlab",
                               "./ITHACAoutput/ErrorsL2/");
    cnpy::save(errConsistency, "./ITHACAoutput/ErrorsL2/errConsistency.npy");
}