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
    Example of a heat transfer Reduction Problem
SourceFiles
    02thermalBlock.C
\*---------------------------------------------------------------------------*/

#include <iostream>
#include "fvCFD.H"
#include "IOmanip.H"
#include "Time.H"
#include "Burgers.H"
#include "ITHACAPOD.H"
#include "DEIM.H"
#include "ReducedProblem.H"
#include <Eigen/Dense>
#define _USE_MATH_DEFINES
#include <cmath>

class DEIM_function : public DEIM<volVectorField>
{
    public:
        using DEIM::DEIM;
        PtrList<volScalarField> fields;
        autoPtr<volScalarField> subField;
};

/// \brief Class where the tutorial number 2 is implemented.
/// \details It is a child of the laplacianProblem class and some of its
/// functions are overridden to be adapted to the specific case.
class tutorial23: public Burgers
{
    public:
        explicit tutorial23(int argc, char* argv[])
            :
            Burgers(argc, argv),
            U(_U())
        {}

        /// Velocity field
        volVectorField& U;

        autoPtr<DEIM_function> DEIMmatriceU;

        autoPtr<DEIM_function> DEIMmatriceR;

        volVectorModes UmodesRU;
        volVectorModes UmodesRR;

        PtrList<volVectorField> UrGal;
        PtrList<volVectorField> UrCol;

        autoPtr<volVectorField> Ur;
        autoPtr<volVectorField> UrRes;

        void offlineSolve(word folder = "./ITHACAoutput/Offline/")
        {
            if (offline)
            {
                ITHACAstream::readMiddleFields(Ufield, U, "./ITHACAoutput/Offline/");
            }
            else
            {
                truthSolve(folder);
            }
        }

        void residualComputation(label Nmodes = 0,
                                 word folder = "./ITHACAoutput/Offline/")
        {
            restart();
            residual(Nmodes, folder);
        }

        void DEIM(label Nmodes)
        {
            DEIMmatriceU = autoPtr<DEIM_function> (new DEIM_function(Ufield, Nmodes,
                                                   "U", "U"));
            DEIMmatriceR = autoPtr<DEIM_function> (new DEIM_function(resField, Nmodes,
                                                   "Ures", "Res"));
            DEIMmatriceU->generateSubmesh(2, _mesh(), _U());
            DEIMmatriceR->generateSubmesh(2, _mesh(), _U());
        }

        void subFields1(volVectorField Uf, label Nmodes)
        {
            Ur.reset(autoPtr<volVectorField>(new volVectorField(
                                                 DEIMmatriceU->submesh->interpolate(Uf))));
            UmodesRU.resize(0);

            for (int i = 0; i < Nmodes; ++i)
            {
                UmodesRU.append(DEIMmatriceU->submesh->interpolate(Umodes[i])->clone());
            }
        }

        void onlineCollocation1(label Nmodes, word folder = "./ITHACAoutput/test1/")
        {
            Eigen::MatrixXd M = ITHACAutilities::getValues(UmodesRU,
                                DEIMmatriceU->localMagicPoints, DEIMmatriceU->xyz);
            double dt = _runTime().deltaT().value();
            M /= dt;
            Umodes.resize(Nmodes);
            Eigen::MatrixXd a_old = Umodes.project(_U(), Nmodes);
            //Eigen::MatrixXd a_old = UmodesRU.project(Ur(), Nmodes);
            label i = 1;
            Eigen::MatrixXd a(a_old);
            volVectorField Urec(_U().clone());
            Umodes.reconstruct(Urec, a_old, "Urec");
            UrCol.append(Urec.clone());
            ITHACAstream::exportSolution(Urec, name(i + 1), folder);
            i++;
            PtrList<volVectorField> lapl;
            PtrList<volVectorField> div;

            for (int i = 0; i < Nmodes; ++i)
            {
                lapl.append(fvc::laplacian(_nu(), UmodesRU[i]));
            }

            Eigen::MatrixXd B = ITHACAutilities::getValues(lapl,
                                DEIMmatriceU->localMagicPoints, DEIMmatriceU->xyz);
            UmodesRU.reconstruct(Ur(), a_old, "U");
            surfaceScalarField phi = fvc::flux(Ur());
            List<Eigen::MatrixXd> LinSys(2);

            while (_simple().loop())
            {
                Info << "Time = " << _runTime().timeName() << nl << endl;
                div.resize(0);

                for (int i = 0; i < Nmodes; ++i)
                {
                    div.append(fvc::div(phi, UmodesRU[i]));
                }

                Eigen::MatrixXd C = ITHACAutilities::getValues(div,
                                    DEIMmatriceU->localMagicPoints, DEIMmatriceU->xyz);
                LinSys[0] = M + C - B;
                LinSys[1] = M * a_old;
                std::cout << EigenFunctions::condNumber(LinSys[0]) << std::endl;
                Eigen::VectorXd ares(Nmodes);
                a = reducedProblem::solveLinearSys(LinSys, a, ares);
                a_old = a;
                UmodesRU.reconstruct(Ur(), a_old, "U");
                phi = fvc::flux(Ur());
                Umodes.reconstruct(Urec, a, "Urec");
                UrCol.append(Urec.clone());
                ITHACAstream::exportSolution(Urec, name(i + 1), folder);
                i++;
            }
        }

        void subFields2(volVectorField Uf, label Nmodes)
        {
            UrRes.reset(autoPtr<volVectorField>(new volVectorField(
                                                    DEIMmatriceR->submesh->interpolate(Uf))));
            UmodesRR.resize(0);

            for (int i = 0; i < Nmodes; ++i)
            {
                UmodesRR.append(DEIMmatriceR->submesh->interpolate(Umodes[i])->clone());
            }
        }

        void onlineCollocation2(label Nmodes, word folder = "./ITHACAoutput/test2/")
        {
            Eigen::MatrixXd M = ITHACAutilities::getValues(Umodes,
                                DEIMmatriceR->magicPoints(), DEIMmatriceR->xyz);
            double dt = _runTime().deltaT().value();
            M /= dt;
            Umodes.resize(Nmodes);
            Eigen::MatrixXd a_old = Umodes.project(_U(), Nmodes);
            label i = 1;
            Eigen::MatrixXd a(a_old);
            volVectorField Urec(_U().clone());
            Umodes.reconstruct(Urec, a_old, "Urec");
            UrCol.append(Urec.clone());
            ITHACAstream::exportSolution(Urec, name(i + 1), folder);
            i++;
            PtrList<volVectorField> lapl;
            PtrList<volVectorField> div;

            for (int i = 0; i < Nmodes; ++i)
            {
                lapl.append(fvc::laplacian(_nu(), Umodes[i]));
            }

            Eigen::MatrixXd B = ITHACAutilities::getValues(lapl,
                                DEIMmatriceR->magicPoints(), DEIMmatriceR->xyz);
            Umodes.reconstruct(Urec, a_old, "U");
            surfaceScalarField phi = fvc::flux(Urec);
            List<Eigen::MatrixXd> LinSys(2);

            while (_simple().loop())
            {
                Info << "Time = " << _runTime().timeName() << nl << endl;
                div.resize(0);

                for (int i = 0; i < Nmodes; ++i)
                {
                    div.append(fvc::div(phi, Umodes[i]));
                }

                Eigen::MatrixXd C = ITHACAutilities::getValues(div,
                                    DEIMmatriceR->magicPoints(), DEIMmatriceR->xyz);
                LinSys[0] = M + C - B;
                LinSys[1] = M * a_old;
                std::cout << EigenFunctions::condNumber(LinSys[0]) << std::endl;
                Eigen::VectorXd ares(Nmodes);
                a = reducedProblem::solveLinearSys(LinSys, a, ares);
                a_old = a;
                Umodes.reconstruct(Urec, a_old, "U");
                phi = fvc::flux(Urec);
                Umodes.reconstruct(Urec, a, "Urec");
                ITHACAstream::exportSolution(Urec, name(i + 1), folder);
                UrCol.append(Urec.clone());
                Umodes.reconstruct(Urec, a_old, "U");
                i++;
            }
        }

        void onlineGalerkin(label Nmodes, word folder = "./ITHACAoutput/test3/")
        {
            Eigen::MatrixXd M = ITHACAutilities::getMassMatrix(Umodes, Nmodes);
            Eigen::MatrixXd B = M;

            for (int i = 0; i < Nmodes; ++i)
            {
                for (int j = 0; j < Nmodes; ++j)
                {
                    B(i, j) = fvc::domainIntegrate(Umodes[i] & fvc::laplacian(_nu(),
                                                   Umodes[j])).value();
                }
            }

            Eigen::MatrixXd C = M;
            double dt = _runTime().deltaT().value();
            M /= dt;
            Umodes.resize(Nmodes);
            Eigen::MatrixXd a_old = Umodes.project(_U(), Nmodes);
            label i = 1;
            Eigen::MatrixXd a(a_old);
            volVectorField Urec(_U().clone());
            Umodes.reconstruct(Urec, a_old, "Urec");
            UrGal.append(Urec.clone());
            ITHACAstream::exportSolution(Urec, name(i + 1), folder);
            i++;
            surfaceScalarField phi = fvc::flux(Urec);
            List<Eigen::MatrixXd> LinSys(2);

            while (_simple().loop())
            {
                Info << "Time = " << _runTime().timeName() << nl << endl;

                for (int i = 0; i < Nmodes; ++i)
                {
                    for (int j = 0; j < Nmodes; ++j)
                    {
                        C(i, j) = fvc::domainIntegrate(Umodes[i] & fvc::div(phi, Umodes[j])).value();
                    }
                }

                LinSys[0] = M + C - B;
                LinSys[1] = M * a_old;
                Eigen::VectorXd ares(Nmodes);
                a = reducedProblem::solveLinearSys(LinSys, a, ares);
                a_old = a;
                Umodes.reconstruct(Urec, a, "Urec");
                UrGal.append(Urec.clone());
                phi = fvc::flux(Urec);
                ITHACAstream::exportSolution(Urec, name(i + 1), folder);
                i++;
            }
        }


};


int main(int argc, char* argv[])
{
    // Create the train object of the tutorial23 type
    tutorial23 train(argc, argv);
    List<Eigen::MatrixXd> sys(2);
    sys[0] = Eigen::MatrixXd::Random(3, 2);
    sys[1] = Eigen::MatrixXd::Random(3, 1);
    Eigen::MatrixXd a = Eigen::MatrixXd::Random(2, 1);
    Eigen::VectorXd ares(sys[1]);
    a = reducedProblem::solveLinearSys(sys, a, ares);
    ITHACAparameters* para = ITHACAparameters::getInstance(train._mesh(),
                             train._runTime());
    int NmodesUout = para->ITHACAdict->lookupOrDefault<int>("NmodesUout", 20);
    int Nmodes = para->ITHACAdict->lookupOrDefault<int>("NmodesProj", 20);
    train.offlineSolve();
    ITHACAPOD::getModes(train.Ufield, train.Umodes, train._U().name(),
                        train.podex, 0, 0,
                        NmodesUout);
    train.residualComputation(Nmodes);
    volVectorField resAv = ITHACAutilities::computeAverage(train.resField);
    List<label> muMax;
    List<label> xyzMax;

    for (int i = 0; i < 100; ++i)
    {
        label j = findMax(resAv);
        muMax.append(j);
        resAv.ref()[j] = vector(0, 0, 0);
        xyzMax.append(0);
    }

    Info << muMax << endl;
    train.DEIM(Nmodes);
    train.restart();
    // train.subFields1(train._U(), Nmodes);
    // train.onlineCollocation1(Nmodes);
    // train.restart();
    train.onlineGalerkin(Nmodes);
    train.restart();
    List<label> mu = train.DEIMmatriceU().magicPoints();
    List<label> mr = train.DEIMmatriceR().magicPoints();
    List<label> xyz1 = train.DEIMmatriceU().xyz();
    List<label> xyz2 = train.DEIMmatriceR().xyz();
    // mu.append(mr);
    // xyz1.append(xyz2);
    List<label> muTot;
    List<label> xyzTot;

    for (label i = 0; i < train._U().size(); i++)
    {
        muTot.append(i);
        xyzTot.append(0);
    }

    train.DEIMmatriceU().setMagicPoints(mu, xyz1);
    train.DEIMmatriceU().generateSubmesh(2, train._mesh(), train._U(),
                                         false);
    // train.DEIMmatriceU().setMagicPoints(mu,xyz1);
    train.subFields1(train._U(), Nmodes);
    train.onlineCollocation1(Nmodes, "./ITHACAoutput/test4/");
    Eigen::MatrixXd col = ITHACAutilities::errorL2Rel(train.Ufield, train.UrCol,
                          &mu);
    Eigen::MatrixXd gal = ITHACAutilities::errorL2Rel(train.Ufield, train.UrGal,
                          &mu);
    cnpy::save(col, "erCol_U.npy");
    cnpy::save(gal, "erGal.npy");
}
