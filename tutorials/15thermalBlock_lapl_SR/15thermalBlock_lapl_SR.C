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
#include "motionSolver.H"
#include "volPointInterpolation.H"
#include "displacementLaplacianFvMotionSolver.H"
#include "inverseDistanceDiffusivity.H"
#include <iostream>
#include "fvCFD.H"
#include "IOmanip.H"
#include "laplacianProblem.H"
#include "reducedLaplacian.H"
#include "reducedProblem.H"
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

class DEIM_function : public DEIM<PtrList<fvScalarMatrix>, volScalarField>
{
    public:

        using DEIM::DEIM;

        static fvScalarMatrix evaluate_expression(volScalarField& T,
                dimensionedScalar& DT)
        {
            std::cout.setstate(std::ios_base::failbit);
            fvScalarMatrix TiEqn22
            (
                fvm::laplacian(DT, T)
            );
            std::cout.clear();
            return TiEqn22;
        }

        Eigen::MatrixXd onlineCoeffsA(dimensionedScalar& DT)
        {
            Eigen::MatrixXd theta(fieldsA.size(), 1);

            for (int i = 0; i < fieldsA.size(); i++)
            {
                Eigen::SparseMatrix<double> Mr;
                std::cout.setstate(std::ios_base::failbit);
                fvScalarMatrix Aof = evaluate_expression(fieldsA[i], DT);
                std::cout.clear();
                Foam2Eigen::fvMatrix2EigenM(Aof, Mr);
                int ind_row = localMagicPointsA[i].first() + xyz_A[i].first() *
                              fieldsA[i].size();
                int ind_col = localMagicPointsA[i].second() + xyz_A[i].second() *
                              fieldsA[i].size();
                theta(i) = Mr.coeffRef(ind_row, ind_col);
            }

            return theta;
        }

        Eigen::MatrixXd onlineCoeffsB(dimensionedScalar& DT)
        {
            Eigen::MatrixXd theta(fieldsB.size(), 1);

            for (int i = 0; i < fieldsB.size(); i++)
            {
                Eigen::VectorXd br;
                fvScalarMatrix Aof = evaluate_expression(fieldsB[i], DT);
                Foam2Eigen::fvMatrix2EigenV(Aof, br);
                int ind_row = localMagicPointsB[i] + xyz_B[i] * fieldsB[i].size();
                theta(i) = br(ind_row);
            }

            return theta;
        }
};

class DEIM_function_geom : public DEIM<PtrList<fvVectorMatrix>, volVectorField>
{
    public:
        using DEIM::DEIM;

        static fvVectorMatrix evaluate_expression(volVectorField& cellDisplacement,
                const surfaceScalarField& faceDiffusivity)
        {
            std::cout.setstate(std::ios_base::failbit);
            fvVectorMatrix TEqn
            (
                fvm::laplacian
                (
                    dimensionedScalar("viscosity", dimViscosity, 1.0)
                    *faceDiffusivity,
                    cellDisplacement,
                    "laplacian(diffusivity,cellDisplacement)"
                )
            );
            std::cout.clear();
            return TEqn;
        }

};


class ThermalGeo : public laplacianProblem
{
    public:
        explicit ThermalGeo(int argc, char* argv[])
        {
            _args = autoPtr<argList>
                    (
                        new argList(argc, argv)
                    );

            if (!_args->checkRootCase())
            {
                Foam::FatalError.exit();
            }

            argList& args = _args();
#include "createTime.H"
#include "createMesh.H"
#include "createFields.H"
            ITHACAdict = new IOdictionary
            (
                IOobject
                (
                    "ITHACAdict",
                    "./system",
                    mesh,
                    IOobject::MUST_READ,
                    IOobject::NO_WRITE
                )
            );
            DynDict = new IOdictionary
            (
                IOobject
                (
                    "dynamicMeshDictLapl",
                    "./constant",
                    mesh,
                    IOobject::MUST_READ,
                    IOobject::NO_WRITE
                )
            );
            /// Create reference boundaries
            ITHACAutilities::getPointsFromPatch(mesh, 1, x0left, x0left_ind);
            ITHACAutilities::getPointsFromPatch(mesh, 3, x0right, x0right_ind);
            ITHACAutilities::getPointsFromPatch(mesh, 5, x0top, x0top_ind);
            ITHACAutilities::getPointsFromPatch(mesh, 7, x0bot, x0bot_ind);
            motionPtr = motionSolver::New(mesh);
            NTmodes = readInt(ITHACAdict->lookup("N_modes_T"));
            NmodesDEIMA = readInt(ITHACAdict->lookup("N_modes_DEIM_A"));
            NmodesDEIMB = readInt(ITHACAdict->lookup("N_modes_DEIM_B"));
            P0 = mesh.points();
            offline = ITHACAutilities::check_off();
            podex = ITHACAutilities::check_pod();
            diffusivityPtr = autoPtr<motionDiffusivity>
                             (motionDiffusivity::New(mesh, DynDict->lookup("diffusivity")));
            volVectorField& cellDisplacement = const_cast<volVectorField&>
                                               (
                                                   mesh.objectRegistry::lookupObject<volVectorField>
                                                   (
                                                           "cellDisplacement"
                                                   )
                                               );
            pointVectorField& PointDisplacement = const_cast<pointVectorField&>
                                                  (
                                                          mesh.objectRegistry::lookupObject<pointVectorField>
                                                          (
                                                                  "pointDisplacement"
                                                          )
                                                  );
            surfaceScalarField& faceDiffusivity = const_cast<surfaceScalarField&>
                                                  (
                                                          mesh.objectRegistry::lookupObject<surfaceScalarField>
                                                          (
                                                                  "faceDiffusivity"
                                                          )
                                                  );
            volPointInterpolation::New(mesh).interpolate
            (
                cellDisplacement,
                PointDisplacement
            );
            x_up_left = max(x0left);
            x_low_left = min(x0left);
            x_up_bot = max(x0bot);
            x_low_bot = min(x0bot);
            x_up_top = max(x0top);
            x_low_top = min(x0top);
        }

        int NTmodes;
        int NmodesDEIMA;
        int NmodesDEIMB;
        IOdictionary* DynDict;
        std::chrono::duration<double> elapsed;
        std::chrono::duration<double> elapsedON;
        std::chrono::duration<double> elapsedOFF;

        /// Temperature field
        autoPtr<motionSolver> motionPtr;
        autoPtr<volScalarField> _cv;
        autoPtr<dimensionedScalar> _DT;
        autoPtr<volScalarField> _NonOrtho;
        autoPtr<pointVectorField> _pointDisplacement;
        // autoPtr<motionInterpolation> interpolationPtr;
        autoPtr<motionDiffusivity> diffusivityPtr;

        vector x_up_left;
        vector x_low_left;
        vector x_up_bot;
        vector x_low_bot;
        vector x_up_top;
        vector x_low_top;

        List<vector> x0left;
        List<vector> x0right;
        List<vector> x0top;
        List<vector> x0bot;

        pointField P0;

        labelList x0left_ind;
        labelList x0right_ind;
        labelList x0top_ind;
        labelList x0bot_ind;

        PtrList<fvScalarMatrix> Mlist;
        PtrList<fvVectorMatrix> GeoList;
        PtrList<volVectorField> GeoField;
        PtrList<volScalarField> Volumes;
        /// Lifted velocity modes.
        Modes<vector> DisModes;

        Eigen::MatrixXd ModesTEig;
        std::vector<Eigen::MatrixXd> ReducedMatricesA;
        Eigen::MatrixXd ReducedVectorsB;

        PtrList<volScalarField> Tfield_new;
        PtrList<volScalarField> Volumes_new;

        DEIM_function* DEIMmatrice;

        void OfflineSolve(Eigen::MatrixXd pars, word Folder)
        {
            fvMesh& mesh = _mesh();
            Time& runTime = _runTime();
            dimensionedScalar& DT = _DT();
            volScalarField& T = _T();
            volScalarField& cv = _cv();
            volScalarField& NonOrtho = _NonOrtho();
            dimensionedScalar dummy("zero", dimensionSet(0, 3, 0, -1, 0, 0, 0), 0.0);

            if (offline)
            {
                ITHACAstream::read_fields(Tfield, T, "./ITHACAoutput/Offline/");
            }
            else
            {
                Volumes.setSize(pars.rows());
                std::ofstream myfile;
                myfile.open("timeGEO" + name(NTmodes) + "_" + name(NmodesDEIMA) + "_" + name(
                                NmodesDEIMB) + "_lapl2nd" + ".txt");

                for (int k = 0; k < pars.rows(); k++)
                {
                    auto start = std::chrono::high_resolution_clock::now();
                    updateMesh(pars.row(k));
                    auto finish = std::chrono::high_resolution_clock::now();
                    elapsed += (finish - start);
                    fvScalarMatrix Teqn = DEIMmatrice->evaluate_expression(T, DT);
                    // Solve
                    Teqn.solve();
                    Mlist.append(Teqn);
                    volScalarField cv("cv", T * dummy);
                    cv.ref() = mesh.V();
                    Volumes.set(k, cv);
                    ITHACAstream::exportSolution(T, name(k + 1), Folder);
                    ITHACAstream::exportSolution(cv, name(k + 1), Folder);
                    ITHACAstream::exportSolution(NonOrtho, name(k + 1), Folder);
                    Tfield.append(T);
                    ITHACAstream::writePoints(mesh.points(), Folder, name(k + 1) + "/polyMesh/");
                }

                myfile << elapsed.count() << std::endl;
                myfile.close();
            }
        };

        void OfflineSolveNew(Eigen::MatrixXd pars, word Folder)
        {
            fvMesh& mesh = _mesh();
            Time& runTime = _runTime();
            dimensionedScalar& DT = _DT();
            volScalarField& T = _T();
            volScalarField& cv = _cv();
            std::ofstream myfileOFF;
            myfileOFF.open("timeOFF" + name(NTmodes) + "_" + name(NmodesDEIMA) + "_" + name(
                               NmodesDEIMB) + "_lapl2nd" + ".txt");

            for (int k = 0; k < pars.rows(); k++)
            {
                updateMesh(pars.row(k));
                volScalarField& NonOrtho = _NonOrtho();
                auto start = std::chrono::high_resolution_clock::now();
                fvScalarMatrix Teqn = DEIMmatrice->evaluate_expression(T, DT);
                // Solve
                Teqn.solve();
                auto finish = std::chrono::high_resolution_clock::now();
                elapsedOFF += (finish - start);
                cv.ref() = mesh.V();
                Volumes_new.append(cv);
                ITHACAstream::exportSolution(T, name(k + 1), Folder);
                ITHACAstream::exportSolution(cv, name(k + 1), Folder);
                ITHACAstream::exportSolution(NonOrtho, name(k + 1), Folder);
                Tfield_new.append(T);
                ITHACAstream::writePoints(mesh.points(), Folder, name(k + 1) + "/polyMesh/");
            }

            myfileOFF << elapsedOFF.count() << std::endl;
            myfileOFF.close();
        };

        void updateMesh(Eigen::MatrixXd pars)
        {
            fvMesh& mesh = _mesh();
            mesh.movePoints(P0);
            volScalarField& NonOrtho = _NonOrtho();
            pointVectorField& PointDisplacement = const_cast<pointVectorField&>
                                                  (
                                                          mesh.objectRegistry::lookupObject<pointVectorField>
                                                          (
                                                                  "pointDisplacement"
                                                          )
                                                  );
            PointDisplacement = PointDisplacement * 0;
            List<vector> disL = ITHACAutilities::displacedSegment(x0left, pars(0, 0),
                                pars(0, 0), -pars(0, 1), pars(0, 1));
            List<vector> disR = ITHACAutilities::displacedSegment(x0right, 0, 0, 0, 0);
            List<vector> disT = ITHACAutilities::displacedSegment(x0top, pars(0, 0), 0,
                                pars(0, 1), 0);
            List<vector> disB = ITHACAutilities::displacedSegment(x0bot, pars(0, 0), 0,
                                -pars(0, 1), 0);
            vectorField Left;
            vectorField Right;
            vectorField Top;
            vectorField Bot;
            Left.resize(PointDisplacement.boundaryFieldRef()[1].size());
            Right.resize(PointDisplacement.boundaryFieldRef()[3].size());
            Top.resize(PointDisplacement.boundaryFieldRef()[5].size());
            Bot.resize(PointDisplacement.boundaryFieldRef()[7].size());

            for (int i = 0; i < Left.size(); i++)
            {
                for (int k = 0; k < 3; k++)
                {
                    Left[i][k] = disL[i][k];
                }
            }

            for (int i = 0; i < Right.size(); i++)
            {
                for (int k = 0; k < 3; k++)
                {
                    Right[i][k] = disR[i][k];
                }
            }

            for (int i = 0; i < Top.size(); i++)
            {
                for (int k = 0; k < 3; k++)
                {
                    Top[i][k] = disT[i][k];
                }
            }

            for (int i = 0; i < Bot.size(); i++)
            {
                for (int k = 0; k < 3; k++)
                {
                    Bot[i][k] = disB[i][k];
                }
            }

            PointDisplacement.boundaryFieldRef()[1] == Left;
            PointDisplacement.boundaryFieldRef()[3] == Right;
            PointDisplacement.boundaryFieldRef()[5] == Top;
            PointDisplacement.boundaryFieldRef()[7] == Bot;
            volVectorField& cellDisplacement = const_cast<volVectorField&>
                                               (
                                                   mesh.objectRegistry::lookupObject<volVectorField>
                                                   (
                                                           "cellDisplacement"
                                                   )
                                               );
            surfaceScalarField& faceDiffusivity = const_cast<surfaceScalarField&>
                                                  (
                                                          mesh.objectRegistry::lookupObject<surfaceScalarField>
                                                          (
                                                                  "faceDiffusivity"
                                                          )
                                                  );
            volPointInterpolation::New(mesh).interpolate
            (
                cellDisplacement,
                PointDisplacement
            );
            motionDiffusivity& diffu = diffusivityPtr();
            diffu.correct();
            fvVectorMatrix TEqn(DEIM_function_geom::evaluate_expression(cellDisplacement,
                                diffu().operator()()));
            GeoList.append(TEqn);
            solve(TEqn);
            volPointInterpolation::New(mesh).interpolate
            (
                cellDisplacement,
                PointDisplacement
            );
            GeoField.append(cellDisplacement);
            mesh.movePoints(P0 + PointDisplacement.primitiveField());
            ITHACAutilities::meshNonOrtho(mesh, NonOrtho);
        }

        void updateMeshOnline(Eigen::MatrixXd pars)
        {
            fvMesh& mesh = _mesh();
            mesh.movePoints(P0);
            volScalarField& NonOrtho = _NonOrtho();
            pointVectorField& PointDisplacement = const_cast<pointVectorField&>
                                                  (
                                                          mesh.objectRegistry::lookupObject<pointVectorField>
                                                          (
                                                                  "pointDisplacement"
                                                          )
                                                  );
            surfaceScalarField& faceDiffusivity = const_cast<surfaceScalarField&>
                                                  (
                                                          mesh.objectRegistry::lookupObject<surfaceScalarField>
                                                          (
                                                                  "faceDiffusivity"
                                                          )
                                                  );
            PointDisplacement = PointDisplacement * 0;
            List<vector> disL = ITHACAutilities::displacedSegment(x0left, pars(0, 0),
                                pars(0, 0), -pars(0, 1), pars(0, 1));
            List<vector> disR = ITHACAutilities::displacedSegment(x0right, 0, 0, 0, 0);
            List<vector> disT = ITHACAutilities::displacedSegment(x0top, pars(0, 0), 0,
                                pars(0, 1), 0);
            List<vector> disB = ITHACAutilities::displacedSegment(x0bot, pars(0, 0), 0,
                                -pars(0, 1), 0);
            vectorField Left;
            vectorField Right;
            vectorField Top;
            vectorField Bot;
            Left.resize(PointDisplacement.boundaryFieldRef()[1].size());
            Right.resize(PointDisplacement.boundaryFieldRef()[3].size());
            Top.resize(PointDisplacement.boundaryFieldRef()[5].size());
            Bot.resize(PointDisplacement.boundaryFieldRef()[7].size());

            for (int i = 0; i < Left.size(); i++)
            {
                for (int k = 0; k < 3; k++)
                {
                    Left[i][k] = disL[i][k];
                }
            }

            for (int i = 0; i < Right.size(); i++)
            {
                for (int k = 0; k < 3; k++)
                {
                    Right[i][k] = disR[i][k];
                }
            }

            for (int i = 0; i < Top.size(); i++)
            {
                for (int k = 0; k < 3; k++)
                {
                    Top[i][k] = disT[i][k];
                }
            }

            for (int i = 0; i < Bot.size(); i++)
            {
                for (int k = 0; k < 3; k++)
                {
                    Bot[i][k] = disB[i][k];
                }
            }

            PointDisplacement.boundaryFieldRef()[1] == Left;
            PointDisplacement.boundaryFieldRef()[3] == Right;
            PointDisplacement.boundaryFieldRef()[5] == Top;
            PointDisplacement.boundaryFieldRef()[7] == Bot;
            volVectorField& cellDisplacement = const_cast<volVectorField&>
                                               (
                                                   mesh.objectRegistry::lookupObject<volVectorField>
                                                   (
                                                           "cellDisplacement"
                                                   )
                                               );
            volPointInterpolation::New(mesh).interpolate
            (
                cellDisplacement,
                PointDisplacement
            );
            motionDiffusivity& diffu = diffusivityPtr();
            diffu.correct();
            fvVectorMatrix* TEqn = new fvVectorMatrix
            (
                fvm::laplacian
                (
                    dimensionedScalar("viscosity", dimViscosity, 1.0)
                    *faceDiffusivity,
                    cellDisplacement
                )
            );
            TEqn->solve();
            Eigen::MatrixXd a = Eigen::VectorXd::Zero(DisModes.size());
            List<Eigen::MatrixXd> RedLinSysP = DisModes.project(*TEqn);
            Eigen::VectorXd residual;
            a = reducedProblem::solveLinearSys(RedLinSysP, a, residual);
            std::cout << a << std::endl;
            cellDisplacement = DisModes.reconstruct(a, "cellDisplacement");
            volPointInterpolation::New(mesh).interpolate
            (
                cellDisplacement,
                PointDisplacement
            );
            mesh.movePoints(P0 + PointDisplacement.primitiveField());
            ITHACAutilities::meshNonOrtho(mesh, NonOrtho);
            delete (TEqn);
        }


        void PODDEIM()
        {
            PODDEIM(NTmodes, NmodesDEIMA, NmodesDEIMB);
        }

        void PODDEIM(int NmodesT, int NmodesDEIMA, int NmodesDEIMB)
        {
            volScalarField& T = _T();
            DEIMmatrice = new DEIM_function(Mlist, NmodesDEIMA, NmodesDEIMB, "T_matrix");
            fvMesh& mesh  =  const_cast<fvMesh&>(T.mesh());
            DEIMmatrice->generateSubmeshesMatrix(2, mesh, T);
            DEIMmatrice->generateSubmeshesVector(2, mesh, T);
            ModesTEig = Foam2Eigen::PtrList2Eigen(Tmodes);
            ModesTEig.conservativeResize(ModesTEig.rows(), NmodesT);
            //ITHACAPOD::GrammSchmidt(ModesTEig);
            ReducedMatricesA.resize(NmodesDEIMA);

            for (int i = 0; i < NmodesDEIMA; i++)
            {
                ReducedMatricesA[i] = ModesTEig.transpose() * DEIMmatrice->MatrixOnlineA[i] *
                                      ModesTEig;
            }

            ReducedVectorsB = ModesTEig.transpose() * DEIMmatrice->MatrixOnlineB;
        };

        void OnlineSolve(Eigen::MatrixXd par_new, word Folder)
        {
            volScalarField& T = _T();
            dimensionedScalar& DT = _DT();
            fvMesh& mesh  =  const_cast<fvMesh&>(T.mesh());
            std::ofstream myfileON;
            myfileON.open ("timeON" + name(NTmodes) + "_" + name(NmodesDEIMA) + "_" + name(
                               NmodesDEIMB) + "_lapl2nd" + ".txt");

            for (int i = 0; i < par_new.rows(); i++)
            {
                updateMeshOnline(par_new.row(i));
                DEIMmatrice->generateSubmeshesMatrix(2, mesh, T, 1);
                DEIMmatrice->generateSubmeshesVector(2, mesh, T, 1);
                auto start = std::chrono::high_resolution_clock::now();
                Eigen::MatrixXd thetaonA = DEIMmatrice->onlineCoeffsA(DT);
                Eigen::MatrixXd thetaonB = DEIMmatrice->onlineCoeffsB(DT);
                Eigen::MatrixXd A = EigenFunctions::MVproduct(ReducedMatricesA, thetaonA);
                Eigen::VectorXd x = A.householderQr().solve(ReducedVectorsB * thetaonB);
                auto finish = std::chrono::high_resolution_clock::now();
                elapsedON += (finish - start);
                Eigen::VectorXd full = ModesTEig * x;
                volScalarField Tred("Tred", T);
                Tred = Foam2Eigen::Eigen2field(Tred, full);
                ITHACAstream::exportSolution(Tred, name(i + 1), "./ITHACAoutput/" + Folder);
                ITHACAstream::writePoints(mesh.points(), "./ITHACAoutput/" + Folder,
                                          name(i + 1) + "/polyMesh/");
                Tonline.append(Tred);
            }

            myfileON << elapsedON.count() << std::endl;
            myfileON.close();
        }
};


int main(int argc, char* argv[])
{
    ThermalGeo example(argc, argv);
    // Eigen::MatrixXd parx = ITHACAutilities::rand(100, 1, -0.32, 0.32);
    // Eigen::MatrixXd pary = ITHACAutilities::rand(100, 1, -0.32, 0.32);
    Eigen::MatrixXd pars = ITHACAstream::readMatrix("./Matrices/pars_mat.txt");
    // pars << parx, pary;
    // ITHACAstream::exportMatrix(pars, "pars", "eigen");
    example.OfflineSolve(pars, "./ITHACAoutput/Offline/");
    /// Compute the POD modes
    Modes<scalar> Tmodes_weighted;
    Modes<scalar> Tmodes;
    PtrList<volScalarField> vol1 = example.Volumes;
    double a = 1;

    for (int i = 0; i < vol1.size(); i++)
    {
        ITHACAutilities::assignIF(vol1[i], a);
    }

    // don't touch
    ITHACAPOD::getModes(example.Tfield, example.Tmodes, example.Volumes,
                        example.podex);
    // Non weighted modes
    ITHACAPOD::getModes(example.Tfield, Tmodes, vol1,
                        example.podex);
    // Weighted modes
    ITHACAPOD::getModes(example.Tfield, Tmodes_weighted, example.Volumes,
                        example.podex);
    // Displacement modes
    ITHACAPOD::getModes(example.GeoField, example.DisModes, example.podex, 0, 0,
                        10);
    Tmodes.projectSnapshots(example.Tfield);
    exit(0);
    //ITHACAPOD::getModes(example.Tfield, example.Tmodes, example.podex);//, 0, 0, 20);
    /// Compute the offline part of the DEIM procedure
    example.PODDEIM();
    /// Construct a new set of parameters
    // Eigen::MatrixXd par_new1 = ITHACAutilities::rand(100, 2, -0.28, 0.28);
    // ITHACAstream::exportMatrix(par_new1, "par_online", "eigen");
    Eigen::MatrixXd par_new1 =
        ITHACAstream::readMatrix("./Matrices/par_online_mat.txt");
    /// Solve the online problem with the new parameters
    example.OnlineSolve(par_new1, "Online");
    ///
    example.OfflineSolveNew(par_new1, "./ITHACAoutput/OnlineFull/");
    Eigen::MatrixXd error = ITHACAutilities::error_listfields(example.Tfield_new,
                            example.Tonline);
    ITHACAstream::exportMatrix(error, "error_weighted", "eigen");
    //ITHACAstream::exportMatrix(error,
    //                           "error_" + name(example.NTmodes) + "_" + name(example.NmodesDEIMA) + "_" + name(
    //                               example.NmodesDEIMA), "python", ".");
    std::cout << error.mean() << std::endl;
    //Info << "End\n" << endl;
    return 0;
}
