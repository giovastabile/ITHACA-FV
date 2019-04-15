/*---------------------------------------------------------------------------*\
Copyright (C) 2017 by the ITHACA-FV authors

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
    Example of NS-Stokes Reduction Problem

\*---------------------------------------------------------------------------*/

#include "argList.H"
#include "Time.H"
#include "fvMesh.H"
#include "RBFMotionSolver.H"
#include "dictionary.H"
#include <iostream>
#include "fvCFD.H"
#include "IOmanip.H"
#include "steadyNS_simple.H"
#include "ITHACAPOD.H"
#include "ITHACAutilities.H"
#include <Eigen/Dense>
#include "DEIM.H"
#include "EigenFunctions.H"
#define _USE_MATH_DEFINES
#include <cmath>
#include "pointMesh.H" //Perhaps not needed..?
#include "pointFields.H" //Perhaps not needed..?
#include "pointPatchField.H"

// class DEIM_function : public DEIM<PtrList<fvScalarMatrix>, volScalarField>
// {
//     public:

//         using DEIM::DEIM;

//         static fvScalarMatrix evaluate_expression(volScalarField& T,
//                 dimensionedScalar& DT)
//         {
//             fvScalarMatrix TiEqn22
//             (
//                 fvm::laplacian(DT, T)
//             );
//             return TiEqn22;
//         }

//         Eigen::MatrixXd onlineCoeffsA(dimensionedScalar& DT)
//         {
//             Eigen::MatrixXd theta(fieldsA.size(), 1);

//             for (int i = 0; i < fieldsA.size(); i++)
//             {
//                 Eigen::SparseMatrix<double> Mr;
//                 fvScalarMatrix Aof = evaluate_expression(fieldsA[i], DT);
//                 Foam2Eigen::fvMatrix2EigenM(Aof, Mr);
//                 int ind_row = localMagicPointsA[i].first() + xyz_A[i].first() *
//                               fieldsA[i].size();
//                 int ind_col = localMagicPointsA[i].second() + xyz_A[i].second() *
//                               fieldsA[i].size();
//                 theta(i) = Mr.coeffRef(ind_row, ind_col);
//             }

//             return theta;
//         }

//         Eigen::MatrixXd onlineCoeffsB(dimensionedScalar& DT)
//         {
//             Eigen::MatrixXd theta(fieldsB.size(), 1);

//             for (int i = 0; i < fieldsB.size(); i++)
//             {
//                 Eigen::VectorXd br;
//                 fvScalarMatrix Aof = evaluate_expression(fieldsB[i], DT);
//                 Foam2Eigen::fvMatrix2EigenV(Aof, br);
//                 int ind_row = localMagicPointsB[i] + xyz_B[i] * fieldsB[i].size();
//                 theta(i) = br(ind_row);
//             }

//             return theta;
//         }
// };


class NS_geom_par : public steadyNS_simple
{
    public:
        explicit NS_geom_par(int argc, char* argv[])
            :
            steadyNS_simple(argc, argv),
            U(_U()),
            p(_p()),
            phi(_phi())
        {
            fvMesh& mesh = _mesh();
            dyndict = new IOdictionary
            (
                IOobject
                (
                    "dynamicMeshDictRBF",
                    "./constant",
                    mesh,
                    IOobject::MUST_READ,
                    IOobject::NO_WRITE
                )
            );
#include "createFields_aux.H"
#include "prepareRestart.H"
            ITHACAutilities::getPointsFromPatch(mesh, 2, wing0, wing0_ind);
            ms = new RBFMotionSolver(mesh, *dyndict);
            vectorField motion(ms->movingPoints().size(), vector::zero);
            movingIDs = ms->movingIDs();
            x0 = ms->movingPoints();
            curX = x0;
            point0 = ms->curPoints();
            NTmodes = readInt(ITHACAdict->lookup("N_modes_T"));
            NmodesDEIMA = readInt(ITHACAdict->lookup("N_modes_DEIM_A"));
            NmodesDEIMB = readInt(ITHACAdict->lookup("N_modes_DEIM_B"));
            axis = Foam::vector(0, 0, 1);
        }

        int NTmodes;
        int NmodesDEIMA;
        int NmodesDEIMB;

        volScalarField& p;
        volVectorField& U;
        surfaceScalarField& phi;
        vector axis;

        autoPtr<volScalarField> _nut0;
        autoPtr<volScalarField> _nuTilda0;

        autoPtr<volScalarField> _nut;
        autoPtr<volScalarField> _nuTilda;


        std::chrono::duration<double> elapsed;
        std::chrono::duration<double> elapsedON;
        std::chrono::duration<double> elapsedOFF;

        autoPtr<RBFMotionSolver> RBFmotionPtr;
        autoPtr<volScalarField> _cv;
        autoPtr<volScalarField> _NonOrtho;


        /// dictionary to store input output infos
        IOdictionary* dyndict;

        RBFMotionSolver* ms;

        List<vector> wing0;
        vectorField point0;

        vectorField point;

        labelList movingIDs;
        List<vector> x0;
        List<vector> curX;

        labelList wing0_ind;

        PtrList<fvScalarMatrix> Mlist;
        PtrList<volScalarField> Volumes;

        PtrList<volScalarField> Tfield_new;
        PtrList<volScalarField> Volumes_new;



        Eigen::MatrixXd ModesTEig;
        std::vector<Eigen::MatrixXd> ReducedMatricesA;
        Eigen::MatrixXd ReducedVectorsB;

        // DEIM_function* DEIMmatrice;

        void OfflineSolve(Eigen::MatrixXd pars, word Folder)
        {
            fvMesh& mesh = _mesh();
            Time& runTime = _runTime();
            volScalarField& cv = _cv();

            if (offline)
            {
                ITHACAstream::read_fields(Ufield, U, "./ITHACAoutput/Offline/");
            }
            else
            {
                for (int k = 0; k < pars.rows(); k++)
                {
                    List<scalar> par(1);
                    updateMesh(pars.row(k), k);
                    truthSolve2(par);
                    cv.ref() = mesh.V();
                    Volumes.append(cv);
                    ITHACAstream::exportSolution(U, name(k + 1), Folder);
                    ITHACAstream::exportSolution(cv, name(k + 1), Folder);
                    ITHACAstream::exportSolution(p, name(k + 1), Folder);
                    Ufield.append(U);
                    Pfield.append(p);
                    ITHACAstream::writePoints(mesh.points(), Folder, name(k + 1) + "/polyMesh/");
                    restart();
                }
            }
        };

        void restart()
        {
            fvMesh& mesh = _mesh();
            volScalarField& p = _p();
            volVectorField& U = _U();
            surfaceScalarField& phi = _phi();
            volScalarField& nut = const_cast<volScalarField&>
                                  (mesh.objectRegistry::lookupObject<volScalarField>("nut"));
            volScalarField& nuTilda = const_cast<volScalarField&>
                                      (mesh.objectRegistry::lookupObject<volScalarField>("nuTilda"));
            p = _p0();
            U = _U0();
            phi = _phi0();
            nut = _nut0();
            nuTilda = _nuTilda0();
        }

        // void OfflineSolveNew(Eigen::MatrixXd pars, word Folder)
        // {
        //     fvMesh& mesh = _mesh();
        //     Time& runTime = _runTime();
        //     dimensionedScalar& DT = _DT();
        //     volScalarField& T = _T();
        //     volScalarField& cv = _cv();
        //     std::ofstream myfileOFF;
        //     myfileOFF.open("timeOFF" + name(NTmodes) + "_" + name(NmodesDEIMA) + "_" + name(
        //                        NmodesDEIMB) + "_RBF2nd" + ".txt");
        //     for (int k = 0; k < pars.rows(); k++)
        //     {
        //         updateMesh(pars.row(k));
        //         volScalarField& NonOrtho = _NonOrtho();
        //         auto start = std::chrono::high_resolution_clock::now();
        //         fvScalarMatrix Teqn = DEIMmatrice->evaluate_expression(T, DT);
        //         // Solve
        //         Teqn.solve();
        //         auto finish = std::chrono::high_resolution_clock::now();
        //         elapsedOFF += (finish - start);
        //         cv.ref() = mesh.V();
        //         Volumes_new.append(cv);
        //         ITHACAstream::exportSolution(T, name(k + 1), Folder);
        //         ITHACAstream::exportSolution(cv, name(k + 1), Folder);
        //         ITHACAstream::exportSolution(NonOrtho, name(k + 1), Folder);
        //         Tfield_new.append(T);
        //         ITHACAstream::writePoints(mesh.points(), Folder, name(k + 1) + "/polyMesh/");
        //     }
        //     myfileOFF << elapsedOFF.count() << std::endl;
        //     myfileOFF.close();
        // };
        void updateMesh(Eigen::MatrixXd pars, int k)
        {
            fvMesh& mesh = _mesh();
            double r = pars(0);
            mesh.movePoints(point0);
            List<vector> wing0_cur = ITHACAutilities::rotatePoints(wing0, axis, r);
            ITHACAutilities::setIndices2Value(wing0_ind, wing0_cur, movingIDs, curX);
            ms->setMotion(curX - x0);
            point = ms->curPoints();
            mesh.movePoints(point);
        }

        // void PODDEIM()
        // {
        //     PODDEIM(NTmodes, NmodesDEIMA, NmodesDEIMB);
        // }
        // void PODDEIM(int NmodesT, int NmodesDEIMA, int NmodesDEIMB)
        // {
        //     volScalarField& T = _T();
        //     DEIMmatrice = new DEIM_function(Mlist, NmodesDEIMA, NmodesDEIMB, "T_matrix");
        //     fvMesh& mesh  =  const_cast<fvMesh&>(T.mesh());
        //     DEIMmatrice->generateSubmeshesMatrix(2, mesh, T);
        //     DEIMmatrice->generateSubmeshesVector(2, mesh, T);
        //     ModesTEig = Foam2Eigen::PtrList2Eigen(Tmodes);
        //     ModesTEig.conservativeResize(ModesTEig.rows(), NmodesT);
        //     ITHACAPOD::GrammSchmidt(ModesTEig);
        //     ReducedMatricesA.resize(NmodesDEIMA);
        //     for (int i = 0; i < NmodesDEIMA; i++)
        //     {
        //         ReducedMatricesA[i] = ModesTEig.transpose() * DEIMmatrice->MatrixOnlineA[i] *
        //                               ModesTEig;
        //     }
        //     ReducedVectorsB = ModesTEig.transpose() * DEIMmatrice->MatrixOnlineB;
        // };
        // void OnlineSolve(Eigen::MatrixXd par_new, word Folder)
        // {
        //     volScalarField& T = _T();
        //     dimensionedScalar& DT = _DT();
        //     fvMesh& mesh  =  const_cast<fvMesh&>(T.mesh());
        //     std::ofstream myfileON;
        //     myfileON.open ("timeON" + name(NTmodes) + "_" + name(NmodesDEIMA) + "_" + name(
        //                        NmodesDEIMB) + "_RBF2nd" + ".txt");
        //     for (int i = 0; i < par_new.rows(); i++)
        //     {
        //         updateMesh(par_new.row(i));
        //         volScalarField& NonOrtho = _NonOrtho();
        //         DEIMmatrice->generateSubmeshesMatrix(2, mesh, T, 1);
        //         DEIMmatrice->generateSubmeshesVector(2, mesh, T, 1);
        //         auto start = std::chrono::high_resolution_clock::now();
        //         Eigen::MatrixXd thetaonA = DEIMmatrice->onlineCoeffsA(DT);
        //         Eigen::MatrixXd thetaonB = DEIMmatrice->onlineCoeffsB(DT);
        //         Eigen::MatrixXd A = EigenFunctions::MVproduct(ReducedMatricesA, thetaonA);
        //         Eigen::VectorXd x = A.householderQr().solve(ReducedVectorsB * thetaonB);
        //         auto finish = std::chrono::high_resolution_clock::now();
        //         elapsedON += (finish - start);
        //         Eigen::VectorXd full = ModesTEig * x;
        //         volScalarField Tred("Tred", T);
        //         Tred = Foam2Eigen::Eigen2field(Tred, full);
        //         ITHACAstream::exportSolution(Tred, name(i + 1), "./ITHACAoutput/" + Folder);
        //         ITHACAstream::exportSolution(NonOrtho, name(i + 1), "./ITHACAoutput/" + Folder);
        //         ITHACAstream::writePoints(mesh.points(), "./ITHACAoutput/" + Folder,
        //                                   name(i + 1) + "/polyMesh/");
        //         Tonline.append(Tred);
        //     }
        //     myfileON << elapsedON.count() << std::endl;
        //     myfileON.close();
        // }
};


int main(int argc, char* argv[])
{
    Eigen::VectorXd one = Eigen::VectorXd::Ones(100);
    Eigen::DiagonalMatrix<double, Eigen::Dynamic> diag1(100);
    std::cout << diag1 << std::endl;
    // Eigen::MatrixXd aa = one.asDiagonal();
    // std::cout << *(&aa(0,0)+1) << std::endl;
    exit(0);
    NS_geom_par example(argc, argv);
    volScalarField& p = example.p;
    volVectorField& U = example.U;
    auto start = std::chrono::high_resolution_clock::now();
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    start = std::chrono::high_resolution_clock::now();
    Info << U.ref()[0] << endl;
    Info << U.ref()[1] << endl;
    Eigen::VectorXd rv = Eigen::VectorXd::Random(U.internalField().size() * 3);
    // start = std::chrono::high_resolution_clock::now();
    // Foam2Eigen::Eigen2field(U, rv);
    // finish = std::chrono::high_resolution_clock::now();
    // elapsed = finish - start;
    // std::cout << "Elapsed time: " << elapsed.count() << std::endl;
    // start = std::chrono::high_resolution_clock::now();
    double* a = &U.ref()[0][0];
    a = rv.data();
    finish = std::chrono::high_resolution_clock::now();
    elapsed = finish - start;
    std::cout << "Elapsed time: " << elapsed.count() << std::endl;
    Info << U.ref()[0] << endl;
    Info << U.ref()[1] << endl;
    std::cout << rv(0) << std::endl;
    std::cout << rv(1) << std::endl;
    std::cout << rv(2) << std::endl;
    std::cout << rv(3) << std::endl;
    std::cout << rv(4) << std::endl;
    std::cout << rv(5) << std::endl;
    exit(0);

    for (int i = 0; i < p.internalField().size(); i++)
    {
        Info << p.internalField()[i] << endl;
    }

    Eigen::VectorXd r = Eigen::VectorXd::Random(p.internalField().size());
    std::cout << r << std::endl;
    std::memcpy (&p.ref()[0], r.data(), p.ref().size()*sizeof(double));
    finish = std::chrono::high_resolution_clock::now();
    elapsed = finish - start;
    std::cout << "Elapsed time: " << elapsed.count() << std::endl;
    start = std::chrono::high_resolution_clock::now();
    Foam2Eigen::Eigen2field(p, r);
    finish = std::chrono::high_resolution_clock::now();
    elapsed = finish - start;
    std::cout << "Elapsed time: " << elapsed.count() << std::endl;
    exit(0);
    Info << p << endl;
    exit(0);
    Eigen::MatrixXd parx = ITHACAutilities::rand(100, 1, -10, 10);
    example.OfflineSolve(parx, "./ITHACAoutput/Offline/");
    /// Compute the POD modes
    // ITHACAPOD::getModes(example.Tfield, example.Tmodes, example.Volumes,
    //                     example.podex);
    // /// Compute the offline part of the DEIM procedure
    // example.PODDEIM();
    // /// Construct a new set of parameters
    // Eigen::MatrixXd par_new1 = ITHACAutilities::rand(100, 2, -0.28, 0.28);
    // /// Solve the online problem with the new parameters
    // example.OnlineSolve(par_new1, "comparison");
    // ///
    // example.OfflineSolveNew(par_new1, "./ITHACAoutput/comparison/");
    // Eigen::MatrixXd error = ITHACAutilities::error_listfields(example.Tfield_new,
    //                         example.Tonline);
    // ITHACAstream::exportMatrix(error,
    //                            "error_" + name(example.NTmodes) + "_" + name(example.NmodesDEIMA) + "_" + name(
    //                                example.NmodesDEIMA), "python", ".");
    // Info << "End\n" << endl;
    return 0;
}
