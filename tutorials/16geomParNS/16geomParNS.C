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
#include "reducedSimpleSteadyNS.H"
#include "reducedSteadyNS.H"
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

class DEIM_functionU : public DEIM<fvVectorMatrix>
{
    public:
        using DEIM::DEIM;
};

class DEIM_functionP : public DEIM<fvScalarMatrix>
{
    public:
        using DEIM::DEIM;
};


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
            //#include "prepareRestart.H"
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
            ITHACAutilities::createSymLink("./ITHACAoutput/supfield");
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

        DEIM_functionU* DEIMU;
        DEIM_functionP* DEIMP;



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
        PtrList<fvScalarMatrix> Plist;
        PtrList<fvVectorMatrix> Ulist;

        PtrList<volScalarField> Volumes;

        PtrList<volScalarField> Tfield_new;
        PtrList<volScalarField> Volumes_new;



        Eigen::MatrixXd ModesTEig;
        std::vector<Eigen::MatrixXd> ReducedMatricesA;
        Eigen::MatrixXd ReducedVectorsB;

        // DEIM_function* DEIMmatrice;

        void OfflineSolve(Eigen::VectorXd pars, word Folder)
        {
            fvMesh& mesh = _mesh();
            Time& runTime = _runTime();
            volScalarField& cv = _cv();
            surfaceScalarField& phi = _phi();

            if (offline)
            {
                ITHACAstream::read_fields(Ufield, U, "./ITHACAoutput/Offline/");
                ITHACAstream::read_fields(Pfield, p, "./ITHACAoutput/Offline/");
                ITHACAstream::read_fields(Volumes, cv, "./ITHACAoutput/Offline/");
                ITHACAstream::read_fields(phiField, phi, "./ITHACAoutput/Offline/");
                volVectorField Usup("Usup", U);
                ITHACAstream::read_fields(supfield, Usup, "./ITHACAoutput/supfield/");
            }

            else
            {
                for (int k = 0; k < pars.rows(); k++)
                {
                    List<scalar> par(1);
                    updateMesh(pars[k]);
                    truthSolve2(par, Folder);
                    cv.ref() = mesh.V();
                    Volumes.append(cv);
                    ITHACAstream::exportSolution(U, name(k + 1), Folder);
                    ITHACAstream::exportSolution(cv, name(k + 1), Folder);
                    ITHACAstream::exportSolution(p, name(k + 1), Folder);
                    solveOneSup(p, k);
                    Ufield.append(U);
                    Pfield.append(p);
                    ITHACAstream::writePoints(mesh.points(), Folder, name(k + 1) + "/polyMesh/");
                    restart();
                }
            }
        };

        void solveOneSup(volScalarField p, int k)
        {
            volVectorField Usup
            (
                IOobject
                (
                    "Usup",
                    U.time().timeName(),
                    U.mesh(),
                    IOobject::NO_READ,
                    IOobject::AUTO_WRITE
                ),
                U.mesh(),
                dimensionedVector("zero", U.dimensions(), vector::zero)
            );
            dimensionedScalar nu_fake
            (
                "nu_fake",
                dimensionSet(0, 2, -1, 0, 0, 0, 0),
                scalar(1)
            );
            Vector<double> v(0, 0, 0);

            for (label i = 0; i < Usup.boundaryField().size(); i++)
            {
                if (Usup.boundaryField()[i].type() != "processor")
                {
                    ITHACAutilities::changeBCtype(Usup, "fixedValue", i);
                    assignBC(Usup, i, v);
                    assignIF(Usup, v);
                }
            }

            fvVectorMatrix u_sup_eqn
            (
                - fvm::laplacian(nu_fake, Usup)
            );
            solve
            (
                u_sup_eqn == fvc::grad(p)
            );
            supfield.append(Usup);
            ITHACAstream::exportSolution(Usup, name(k + 1), "./ITHACAoutput/supfield/");
        }

        void restart()
        {
            fvMesh& mesh = _mesh();
            volScalarField& p = _p();
            volVectorField& U = _U();
            surfaceScalarField& phi = _phi();
            //volScalarField& nut = const_cast<volScalarField&>
            //                      (mesh.objectRegistry::lookupObject<volScalarField>("nut"));
            //volScalarField& nuTilda = const_cast<volScalarField&>
            //                          (mesh.objectRegistry::lookupObject<volScalarField>("nuTilda"));
            p = _p0();
            U = _U0();
            phi = _phi0();
            // nut = _nut0();
            // nuTilda = _nuTilda0();
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
        void updateMesh(double par)
        {
            fvMesh& mesh = _mesh();
            mesh.movePoints(point0);
            List<vector> wing0_cur = ITHACAutilities::rotatePoints(wing0, axis, par);
            ITHACAutilities::setIndices2Value(wing0_ind, wing0_cur, movingIDs, curX);
            ms->setMotion(curX - x0);
            point = ms->curPoints();
            mesh.movePoints(point);
        }

        void PODDEIM()
        {
            PODDEIM(NTmodes, NmodesDEIMA, NmodesDEIMB);
        }
        void PODDEIM(int NmodesT, int NmodesDEIMA, int NmodesDEIMB)
        {
            volVectorField& U = _U();
            volScalarField& p = _p();
            DEIMU = new DEIM_functionU(UEqnList, NmodesDEIMA, NmodesDEIMB, "U_matrix");
            DEIMP = new DEIM_functionP(PEqnList, NmodesDEIMA, NmodesDEIMB, "P_matrix");
            fvMesh& mesh  =  const_cast<fvMesh&>(U.mesh());
            DEIMU->generateSubmeshesMatrix(1, mesh, U);
            DEIMU->generateSubmeshesVector(1, mesh, U);
            DEIMP->generateSubmeshesMatrix(1, mesh, p);
            DEIMP->generateSubmeshesVector(1, mesh, p);
        }

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
    NS_geom_par example(argc, argv);
    Eigen::MatrixXd parAlpha;
    std::ifstream exFile("./angles_mat.txt");

    if (exFile)
    {
        parAlpha = ITHACAstream::readMatrix("./angles_mat.txt");
    }

    else
    {
        parAlpha = ITHACAutilities::rand(100, 1, -5, 5);
        ITHACAstream::exportMatrix(parAlpha, "angles", "eigen", "./");
    }

    example.OfflineSolve(parAlpha.leftCols(1), "./ITHACAoutput/Offline/");
        example.PODDEIM(40,40,40);

    ITHACAstream::read_fields(example.liftfield, example.U, "./lift/");
    example.inletIndex.resize(1, 2);
    example.inletIndex(0, 0) = 0;
    example.inletIndex(0, 1) = 0;
    ITHACAutilities::normalizeFields(example.liftfield);
    // Homogenize the snapshots
    example.computeLift(example.Ufield, example.liftfield, example.Uomfield);
    // Perform POD on velocity and pressure and store the first 10 modes
    ITHACAPOD::getModes(example.Uomfield, example.Umodes, example.Volumes,
                        example.podex, 0, 0,
                        50);
    ITHACAPOD::getModes(example.Pfield, example.Pmodes, example.Volumes,
                        example.podex, 0, 0, 50);
    ITHACAPOD::getModes(example.supfield, example.supmodes, 0,
                        0, 1, 50);
    ITHACAPOD::getModes(example.phiField, example.phiModes, example.Uomfield, 0,
                        0, 1, 50);
    // PtrList<volScalarField> Proiezioni_P = example.Pmodes.projectSnapshots(example.Pfield,
    //                                      example.Volumes);
    // // PtrList<volVectorField> Proiezioni_U = example.Umodes.projectSnapshots(example.Ufield,
    // //                                      example.Volumes);
    // // ITHACAstream::exportFields(Proiezioni_U, "ITHACAoutput/Offline", "U_proj");
    // ITHACAstream::exportFields(Proiezioni_P, "ITHACAoutput/Offline", "P_proj");
    //exit(0);
    Eigen::MatrixXd vel(1, 1);
    vel(0, 0) = 1;
    Eigen::VectorXd onlineAlpha;
    onlineAlpha.resize(1);
    onlineAlpha(0) = 2.0;
    example.restart();
    reducedSimpleSteadyNS reduced(example);
    example.updateMesh(onlineAlpha(0));
    reduced.setOnlineVelocity(vel);
    //reduced.solveOnline_Simple(1, 10, 10);
    //     ITHACAstream::writePoints(example._mesh().points(), "./ITHACAoutput/Reconstruct", name(i + 1) + "/polyMesh/");
    // }
    //Error check
    NS_geom_par checkOff(argc, argv);
    checkOff.offline = false;
    checkOff.OfflineSolve(onlineAlpha, "./ITHACAoutput/checkOff/");
    checkOff.offline = true;
    Eigen::MatrixXd ErrorU;
    Eigen::MatrixXd ErrorP;
    ErrorU.resize(50, 1);
    ErrorP.resize(50, 1);
    PtrList<volScalarField> onlineP;
    PtrList<volVectorField> onlineU;
    example.updateMesh(onlineAlpha(0));
    ITHACAstream::writePoints(example._mesh().points(), "./ITHACAoutput/checkOn",
                              name(1) + "/polyMesh/");
    reduced.solveOnline_Simple(1, 30, 30, 15, "./ITHACAoutput/checkOn/");
    exit(0);
    example.projectSUP("./Matrices", 5, 5, 5);
    example.C_tensor = example.convective_term_tens_phi(5, 5, 5);
    example.P_matrix = example.divergence_term_phi(5, 5, 5);
    reducedSteadyNS ridotto(example);
    ridotto.solveOnline_sup(vel);
    Eigen::MatrixXd tmp_sol(ridotto.y.rows() + 1, 1);
    std::cout << ridotto.y << std::endl;
    tmp_sol(0) = 1;
    tmp_sol.col(0).tail(ridotto.y.rows()) = ridotto.y;
    ridotto.online_solution.append(tmp_sol);
    ridotto.reconstruct_sup("./ITHACAoutput/Reconstruction/");
    ITHACAstream::writePoints(example._mesh().points(),
                              "./ITHACAoutput/Reconstruction", name(1) + "/polyMesh/");
    exit(0);

    for (int i = 0; i < 50; i++)
    {
        ITHACAstream::writePoints(example._mesh().points(), "./ITHACAoutput/checkOn",
                                  name(1) + "/polyMesh/");
        reduced.solveOnline_Simple(1, i + 1, i + 1, i + 1, "./ITHACAoutput/checkOn/");
    }

    ITHACAstream::read_fields(onlineU, "Uaux", "./ITHACAoutput/checkOn");
    ITHACAstream::read_fields(onlineP, "Paux", "./ITHACAoutput/checkOn");

    for (int i = 0; i < 50; i++)
    {
        ErrorP(i, 0) = ITHACAutilities::error_fields(checkOff.Pfield[0], onlineP[i]);
        ErrorU(i, 0) = ITHACAutilities::error_fields(checkOff.Ufield[0], onlineU[i]);
    }

    cnpy::save(ErrorP, "ErrorP.npy");
    cnpy::save(ErrorU, "ErrorU.npy");
    // /// Compute the offline part of the DEIM procedure
    //example.PODDEIM();
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
