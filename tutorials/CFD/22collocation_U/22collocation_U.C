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
    train of a heat transfer Reduction Problem
SourceFiles
    02thermalBlock.C
\*---------------------------------------------------------------------------*/

#include <iostream>
#include "fvCFD.H"
#include "IOmanip.H"
#include "Time.H"
#include "laplacianProblem.H"
#include "ReducedLaplacian.H"
#include "ReducedProblem.H"
#include "ITHACAPOD.H"
#include "ITHACAutilities.H"
#include "DEIM.H"
#include <Eigen/Dense>
#define _USE_MATH_DEFINES
#include <cmath>
#include "RBFMotionSolver.H"

/// \brief Class where the tutorial number 2 is implemented.
/// \details It is a child of the laplacianProblem class and some of its
/// functions are overridden to be adapted to the specific case.

class DEIM_function : public DEIM<volScalarField>
{
    public:
        using DEIM::DEIM;
        PtrList<volScalarField> fields;
        autoPtr<volScalarField> subField;
};


class DEIM_function_M : public DEIM<fvScalarMatrix>
{
        using DEIM::DEIM;
    public:
        autoPtr<volScalarField> fieldA;
        autoPtr<volScalarField> fieldB;
};

class tutorial22: public laplacianProblem
{
    public:
        explicit tutorial22(int argc, char* argv[])
            :
            laplacianProblem(argc, argv),
            T(_T()),
            S(_S())
        {
            nu = autoPtr<surfaceTensorField>
                 (
                     new surfaceTensorField
                     (
                         IOobject
                         (
                             "nu",
                             _runTime().timeName(),
                             _mesh(),
                             IOobject::NO_READ,
                             IOobject::AUTO_WRITE
                         ),
                         _mesh(),
                         dimensionedTensor("zero", dimensionSet( 0, 2, -1, 0, 0, 0, 0), tensor(0, 0, 0,
                                           0, 0, 0, 0, 0, 0))
                     )
                 );
            dyndict = new IOdictionary
            (
                IOobject
                (
                    "dynamicMeshDictRBF",
                    "./constant",
                    _mesh(),
                    IOobject::MUST_READ,
                    IOobject::NO_WRITE
                )
            );
            ms = new RBFMotionSolver(_mesh(), *dyndict);
            ITHACAutilities::getPointsFromPatch(_mesh(), 0, x0left, x0left_ind);
            ITHACAutilities::getPointsFromPatch(_mesh(), 1, x0bot, x0bot_ind);
            movingIDs = ms->movingIDs();
            x0 = ms->movingPoints();
            curX = x0;
            curX = ms->movingPoints();
        }
        /// Temperature field
        volScalarField& T;
        labelList movingIDs;
        PtrList<volScalarField> resField;
        /// Diffusivity field
        autoPtr<surfaceTensorField> nu;
        Eigen::MatrixXd Sr;
        vectorField point;
        List<vector> x0left;
        List<vector> x0right;
        List<vector> x0top;
        List<vector> x0bot;
        labelList x0left_ind;
        labelList x0right_ind;
        labelList x0top_ind;
        labelList x0bot_ind;

        List<vector> x0;
        List<vector> curX;

        RBFMotionSolver* ms;
        Eigen::MatrixXd ModesTEig;
        std::vector<Eigen::MatrixXd> ReducedMatricesA;
        std::vector<Eigen::MatrixXd> ReducedVectorsB;

        autoPtr<DEIM_function> DEIMmatrice;
        autoPtr<DEIM_function> DEIMmatriceRes;
        autoPtr<DEIM_function_M> DEIMmatrice_M;

        PtrList<fvScalarMatrix> Mlist;

        /// dictionary to store input output infos
        IOdictionary* dyndict;

        /// Source term field
        volScalarField& S;

        /// It perform an offline Solve
        void offlineSolve(word folder = "./ITHACAoutput/Offline/")
        {
            dimensionedScalar DT
            (
                "DT",
                dimensionSet( 0, 2, -1, 0, 0, 0, 0),
                scalar(1)
            );
            List<scalar> mu_now(2);

            for (label i = 0; i < mu.rows(); i++)
            {
                for (label j = 0; j < mu.cols() ; j++)
                {
                    mu_now[j] = mu(i, j);
                }

                updateMesh(mu_now[0], mu_now[1]);
                SetSource();
                SetDiffusivity(mu_now);
                fvScalarMatrix Teqn(fvm::laplacian(nu(), T) == S );
                Mlist.append((Teqn).clone());
                Teqn.solve();
                Tfield.append(T.clone());
                ITHACAstream::exportSolution(T, name(i + 1), folder);
                ITHACAstream::writePoints(_mesh().points(), folder, name(i + 1) + "/polyMesh/");
            }
        }

        void PODDEIM(int NmodesT, int NmodesDEIMA, int NmodesDEIMB)
        {
            DEIMmatrice_M = autoPtr<DEIM_function_M> (new DEIM_function_M(Mlist,
                            NmodesDEIMA, NmodesDEIMB, "T_matrix"));
            fvMesh& mesh  =  const_cast<fvMesh&>(T.mesh());
            // Differential Operator
            DEIMmatrice_M->fieldA = autoPtr<volScalarField>(new volScalarField(
                                        DEIMmatrice_M->generateSubmeshMatrix(2, mesh, T)));
            DEIMmatrice_M->fieldB = autoPtr<volScalarField>(new volScalarField(
                                        DEIMmatrice_M->generateSubmeshVector(2, mesh, T)));
            // Source Terms
            ModesTEig = Foam2Eigen::PtrList2Eigen(Tmodes);
            ModesTEig.conservativeResize(ModesTEig.rows(), NmodesT);
            ReducedMatricesA.resize(NmodesDEIMA);
            ReducedVectorsB.resize(NmodesDEIMB);

            for (int i = 0; i < NmodesDEIMA; i++)
            {
                ReducedMatricesA[i] = ModesTEig.transpose() * DEIMmatrice_M->MatrixOnlineA[i] *
                                      ModesTEig;
            }

            for (int i = 0; i < NmodesDEIMB; i++)
            {
                ReducedVectorsB[i] = ModesTEig.transpose() * DEIMmatrice_M->MatrixOnlineB;
            }
        };

        void residualComputation(label Nmodes, word folder = "./ITHACAoutput/Offline/")
        {
            List<scalar> mu_now(2);

            for (label i = 0; i < mu.rows(); i++)
            {
                for (label j = 0; j < mu.cols() ; j++)
                {
                    mu_now[j] = mu(i, j);
                }

                updateMesh(mu_now[0], mu_now[1]);
                SetSource();
                SetDiffusivity(mu_now);
                fvScalarMatrix Teqn(fvm::laplacian(nu(), T) == S );
                Teqn.solve();
                volScalarField Taux(T.clone());
                Taux = Tmodes.projectSnapshot(Taux, Nmodes);
                volScalarField resCol(fvc::laplacian(nu(), Taux) - S);
                resCol.rename("res");
                ITHACAstream::exportSolution(resCol, name(i + 1), folder);
                resField.append(resCol.clone());
            }
        }
        /// Define the source term function
        void SetSource()
        {
            volScalarField yPos = T.mesh().C().component(vector::Y).ref();
            volScalarField xPos = T.mesh().C().component(vector::X).ref();
            forAll(S, counter)
            {
                S[counter] = Foam::exp(4 * xPos[counter] * yPos[counter]);
            }
        }

        Eigen::MatrixXd onlineCoeffsA()
        {
            Eigen::MatrixXd theta(DEIMmatrice_M->magicPointsAcol().size(), 1);
            fvScalarMatrix Aof(fvm::laplacian(nu(), T) == S );
            Eigen::SparseMatrix<double> Mr;
            Eigen::VectorXd br;
            Foam2Eigen::fvMatrix2Eigen(Aof, Mr, br);

            for (int i = 0; i < DEIMmatrice_M->magicPointsAcol().size(); i++)
            {
                label ind_row(DEIMmatrice_M->magicPointsArow()[i] +
                              (DEIMmatrice_M->xyz_Arow())[i] *
                              T.size());
                int ind_col(DEIMmatrice_M->magicPointsAcol()[i] + (DEIMmatrice_M->xyz_Acol())[i]
                            *
                            T.size());
                theta(i) = Mr.coeffRef(ind_row, ind_col);
            }

            return theta;
        }

        Eigen::MatrixXd onlineCoeffsB()
        {
            fvScalarMatrix Aof(fvm::laplacian(nu(), T) == S );
            Eigen::MatrixXd theta(DEIMmatrice_M->magicPointsB().size(), 1);
            Eigen::SparseMatrix<double> Mr;
            Eigen::VectorXd br;
            Foam2Eigen::fvMatrix2Eigen(Aof, Mr, br);

            for (int i = 0; i < DEIMmatrice_M->magicPointsB().size(); i++)
            {
                int ind_row = DEIMmatrice_M->magicPointsB()[i] + (DEIMmatrice_M->xyz_B())[i] *
                              T.size();
                theta(i) = br(ind_row);
            }

            return theta;
        }


        void SetDiffusivity(List<scalar> mu_now)
        {
            surfaceScalarField yPos = T.mesh().Cf().component(vector::Y).ref();
            surfaceScalarField xPos = T.mesh().Cf().component(vector::X).ref();
            forAll(nu(), counter)
            {
                nu()[counter][0] = (1 + mu_now[0] * xPos[counter]);
                nu()[counter][4] = (1 + mu_now[1] * yPos[counter]);
            }

            for (label i = 0; i < nu().boundaryField().size(); i++)
            {
                for (label j = 0; j < nu().boundaryField()[i].size(); j++)
                {
                    nu().boundaryFieldRef()[i][j][0] = (1 + mu_now[0] * xPos.boundaryField()[i][j]);
                    nu().boundaryFieldRef()[i][j][4] = (1 + mu_now[1] * yPos.boundaryField()[i][j]);
                }
            }
        }

        void DEIM(label Nmodes)
        {
            DEIMmatrice = autoPtr<DEIM_function> (new DEIM_function(Tfield, Nmodes,
                                                  "Tsolution", T.name()));
            DEIMmatrice->generateSubmesh(1, _mesh(), T);
            DEIMmatriceRes = autoPtr<DEIM_function> (new DEIM_function(resField, Nmodes,
                             "resSolution", "res"));
            DEIMmatriceRes->generateSubmesh(1, _mesh(), T);
        }

        List<Eigen::MatrixXd> collocation(int Nmodes)
        {
            List<Eigen::MatrixXd> collSys(2);
            List<label> magicPoints(DEIMmatrice->magicPoints());
            Eigen::MatrixXd A(magicPoints.size(), Nmodes);
            Eigen::MatrixXd b(magicPoints.size(), 1);

            for (label i = 0; i < Nmodes; i++)
            {
                volScalarField laplI(fvc::laplacian(nu(), Tmodes[i]));

                for (label j = 0; j < magicPoints.size(); j++)
                {
                    A(j, i) = laplI[magicPoints[j]];
                    b(j) = S[magicPoints[j]];
                }
            }

            collSys[0] = A;
            collSys[1] = b;
            return collSys;
        }

        void updateMesh(double dx, double dy)
        {
            fvMesh& mesh = _mesh();
            List<vector> disL = ITHACAutilities::displacedSegment(x0left, dx,
                                0, dy, 0, 0, 0);
            List<vector> disB = ITHACAutilities::displacedSegment(x0bot, dx, 0,
                                dy, 0, 0, 0);
            ITHACAutilities::setIndices2Value(x0left_ind, disL, movingIDs, curX);
            ITHACAutilities::setIndices2Value(x0bot_ind, disB, movingIDs, curX);
            ms->setMotion(curX - (ms->movingPoints() - x0));
            point = ms->curPoints();
            mesh.movePoints(point);
        }
};


int main(int argc, char* argv[])
{
    // Create the train object of the tutorial02 type
    tutorial22 train(argc, argv);
    // List<vector> b(2);
    // b[0] = vector(1,2,3);
    // b[1] = vector(4,5,6);
    // Info << Foam::sqrt(magSqr(b)) << endl;
    // exit(0);
    // Read some parameters from file
    ITHACAparameters* para = ITHACAparameters::getInstance(train._mesh(),
                             train._runTime());
    int NmodesTout = para->ITHACAdict->lookupOrDefault<int>("NmodesTout", 100);
    int NmodesTproj = para->ITHACAdict->lookupOrDefault<int>("NmodesTproj", 10);
    List<vector> points = train._mesh().C();
    Eigen::MatrixXd p = Foam2Eigen::field2Eigen(points);
    cnpy::save(p, "points.npy");
    // Set the number of parameters
    train.Pnumber = 2;
    train.Tnumber = 100;
    // Set the parameters
    train.setParameters();
    // Set the parameter ranges, in all the subdomains the diffusivity varies between
    // 0.001 and 0.1
    train.mu_range(0, 0) = -0.25;
    train.mu_range(0, 1) = 0.25;
    train.mu_range(1, 0) = -0.25;
    train.mu_range(1, 1) = 0.25;

    // Generate the Parameters
    if (std::ifstream("mu_train.npy"))
    {
        Info << "Reading parameters of train from file" << endl;
        cnpy::load(train.mu, "mu_train.npy");
    }
    else
    {
        train.genRandPar(train.Tnumber);
        cnpy::save(train.mu, "mu_train.npy");
    }

    // List<label> lista(3);
    // lista[0] = 1;
    // lista[1] = 2;
    // lista[2] = 3;
    // Info << ITHACAutilities::L2Norm(train.S,&lista) << endl;;
    // exit(0);
    train.offlineSolve();
    ITHACAPOD::getModes(train.Tfield, train.Tmodes, train._T().name(),
                        train.podex, 0, 0,
                        NmodesTout);
    train.residualComputation(NmodesTproj);
    label Nr = NmodesTproj;
    train.DEIM(Nr);
    train.PODDEIM(Nr, Nr, Nr);
    Eigen::MatrixXi mp = Foam2Eigen::List2EigenMatrix(
                             train.DEIMmatrice_M->magicPointsAcol());
    // Eigen::MatrixXi mp = Foam2Eigen::List2EigenMatrix(
    //    train.DEIMmatrice_M->magicPointsArow());
    List<label> mu = train.DEIMmatrice().magicPoints();
    List<label> mr = train.DEIMmatriceRes().magicPoints();
    List<label> xyz1 = train.DEIMmatrice().xyz();
    List<label> xyz2 = train.DEIMmatriceRes().xyz();
    mu.append(mr);
    xyz1.append(xyz2);
    List<label> muTot;
    List<label> xyzTot;
    // for (label i = 0; i < train._U().size(); i++)
    // {
    //     muTot.append(i);
    //     xyzTot.append(0);
    // }
    // train.DEIMmatrice().setMagicPoints(mu, xyz1);
    // train.DEIMmatrice().generateSubmesh(1, train._mesh(), train.T,
    //                                       false);
    // train.DEIMmatrice().setMagicPoints(mu, xyz1);
    //  train.DEIMmatrice().generateSubmesh(1, train._mesh(), train.T,
    //                                       false);
    volScalarField Trec(train.T);
    List<scalar> mu_now(2);
    tutorial22 test(argc, argv);
    // Set the number of parameters
    test.Pnumber = 2;
    test.Tnumber = 100;
    // Set the parameters
    test.setParameters();
    // Set the parameter ranges, in all the subdomains the diffusivity varies between
    // 0.001 and 0.1
    test.mu_range(0, 0) = -0.22;
    test.mu_range(0, 1) = 0.22;
    test.mu_range(1, 0) = -0.22;
    test.mu_range(1, 1) = 0.22;
    PtrList<volScalarField> TlistGal;
    PtrList<volScalarField> TlistCol;

    if (std::ifstream("mu_test.npy"))
    {
        Info << "Reading parameters of test from file" << endl;
        cnpy::load(test.mu, "mu_test.npy");
    }
    else
    {
        test.genRandPar(test.Tnumber);
        cnpy::save(test.mu, "mu_test.npy");
    }

    test.offlineSolve("./ITHACAoutput/test/");

    for (int i = 0; i < test.mu.rows(); i++)
    {
        for (label j = 0; j < test.mu.cols() ; j++)
        {
            mu_now[j] = test.mu(i, j);
        }

        train.updateMesh(mu_now[0], mu_now[1]);
        train.SetSource();
        train.SetDiffusivity(mu_now);
        fvScalarMatrix Teqn(fvm::laplacian(train.nu(), train.T) == train.S );
        List<Eigen::MatrixXd> LinSys;
        // LinSys = train.Tmodes.project(Teqn, Nr, "G");
        Eigen::MatrixXd thetaonA = train.onlineCoeffsA();
        Eigen::MatrixXd thetaonB = train.onlineCoeffsB();
        Eigen::MatrixXd A = EigenFunctions::MVproduct(train.ReducedMatricesA, thetaonA);
        Eigen::VectorXd B = train.ReducedVectorsB[0] * thetaonB;
        LinSys.append(A);
        LinSys.append(B);
        Eigen::MatrixXd a(Nr, 1);
        Eigen::VectorXd ares(Nr);
        // Galerkin
        a = reducedProblem::solveLinearSys(LinSys, a, ares);
        train.Tmodes.reconstruct(Trec, a, "Trec");
        volScalarField resGal(fvc::laplacian(train.nu(), Trec) - train.S);
        TlistGal.append(Trec.clone());
        ITHACAstream::exportSolution(Trec, name(i + 1), "./ITHACAoutput/test/");
        // Collocation
        List<Eigen::MatrixXd> collSys = train.collocation(Nr);
        a = reducedProblem::solveLinearSys(collSys, a, ares,
                                           "completeOrthogonalDecomposition");
        train.Tmodes.reconstruct(Trec, a, "Tcol");
        volScalarField resCol(fvc::laplacian(train.nu(), Trec) - train.S);
        ITHACAstream::exportSolution(Trec, name(i + 1), "./ITHACAoutput/test/");
        TlistCol.append(Trec.clone());
        // Info << "Gal: " << ITHACAutilities::L2Norm(resGal, &train.DEIMmatrice->magicPoints()) << endl;
        // Info << "Col: " << ITHACAutilities::L2Norm(resCol, &train.DEIMmatrice->magicPoints()) << endl;
    }

    Eigen::MatrixXd errGal = ITHACAutilities::errorFrobRel(test.Tfield, TlistGal);
    Eigen::MatrixXd errCol = ITHACAutilities::errorFrobRel(test.Tfield, TlistCol);
    std::cout << "Error with Galerkin Method: " << errGal.norm() / 100 << std::endl;
    std::cout << "Error with Collocation Method: " << errCol.norm() / 100 << std::endl;
    word filename_gal = "Gal_" + name(Nr) + ".npy";
    word filename_col = "Col_" + name(Nr) + ".npy";

    cnpy::save(errGal, filename_gal);
    cnpy::save(errCol, filename_col);

}
