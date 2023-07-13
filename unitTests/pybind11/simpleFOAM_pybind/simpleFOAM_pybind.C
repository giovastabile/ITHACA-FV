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
\*---------------------------------------------------------------------------*/


#include "fvCFD.H"
#include "IOmanip.H"
#include "Time.H"
#include "steadyNS.H"
#include "ITHACAPOD.H"
#include "ITHACAutilities.H"
#include <cstddef>
#define _USE_MATH_DEFINES
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include "iostream"
#include "Foam2Eigen.H"

#if PY_VERSION_HEX < 0x03000000
#define MyPyText_AsString PyString_AsString
#else
#define MyPyText_AsString PyUnicode_AsUTF8
#endif

namespace py = pybind11;

class simpleFOAM_pybind : public steadyNS
{
public:
    scalar residual = 1;
    scalar uresidual = 1;
    scalar presidual = 1;
    simpleFOAM_pybind(int argc, char* argv[])
    {
        _args = autoPtr<argList>(
                    new argList(argc, argv, true, true, /*initialise=*/false));
        argList& args = _args();
#include "createTime.H"
#include "createMesh.H"
        _simple = autoPtr<simpleControl>
                  (
                      new simpleControl
                      (
                          mesh
                      )
                  );
        simpleControl& simple = _simple();
#include "createFields.H"
#include "createFvOptions.H"
        turbulence->validate();
        ITHACAdict = new IOdictionary
        (
            IOobject
            (
                "ITHACAdict",
                runTime.system(),
                mesh,
                IOobject::MUST_READ,
                IOobject::NO_WRITE
            )
        );
        tolerance = ITHACAdict->lookupOrDefault<scalar>("tolerance", 1e-5);
        maxIter = ITHACAdict->lookupOrDefault<scalar>("maxIter", 1000);
        bcMethod = ITHACAdict->lookupOrDefault<word>("bcMethod", "lift");
        M_Assert(bcMethod == "lift" || bcMethod == "penalty" || bcMethod == "none",
                 "The BC method must be set to lift or penalty or none in ITHACAdict");
        fluxMethod = ITHACAdict->lookupOrDefault<word>("fluxMethod", "inconsistent");
        M_Assert(fluxMethod == "inconsistent" || bcMethod == "consistent",
                 "The flux method must be set to inconsistent or consistent in ITHACAdict");
        para = ITHACAparameters::getInstance(mesh, runTime);
        offline = ITHACAutilities::check_off();
        podex = ITHACAutilities::check_pod();
        supex = ITHACAutilities::check_sup();
    }
    ~simpleFOAM_pybind() {};
    Eigen::Map<Eigen::MatrixXd> getU()
    {
        Eigen::Map<Eigen::MatrixXd> Ueig(Foam2Eigen::field2EigenMap(_U()));
        return std::move(Ueig);
    }
    Eigen::Map<Eigen::MatrixXd> getP()
    {
        Eigen::Map<Eigen::MatrixXd> Peig(Foam2Eigen::field2EigenMap(_p()));
        return std::move(Peig);
    }
    Eigen::Map<Eigen::MatrixXd> getPhi()
    {
        Eigen::Map<Eigen::MatrixXd> Phieig(Foam2Eigen::field2EigenMap(_phi()));
        return std::move(Phieig);
    }

    void printU()
    {
        Info << _U() << endl;
    }

    void printP()
    {
        Info << _p() << endl;
    }
    void printPhi()
    {
        Info << _phi() << endl;
    }
    void setU(Eigen::VectorXd U)
    {
        _U() = Foam2Eigen::Eigen2field(_U(), U);
    }
    void setP(Eigen::VectorXd p)
    {
        _p() = Foam2Eigen::Eigen2field(_p(), p);
    }

    Eigen::VectorXd getResidual()
    {
        Time& runTime = _runTime();
        fvMesh& mesh = _mesh();
        surfaceScalarField& phi = _phi();
        volVectorField& U = _U();
        volScalarField& p = _p();
        fv::options& fvOptions = _fvOptions();
        simpleControl& simple = _simple();
        IOMRFZoneList& MRF = _MRF();
        singlePhaseTransportModel& laminarTransport = _laminarTransport();
        this->turbulence->validate();
        MRF.correctBoundaryVelocity(U);
        tmp<fvVectorMatrix> tUEqn
        (
            fvm::div(phi, U)
            + MRF.DDt(U) //non c'è
            + turbulence->divDevReff(U)
            ==
            fvOptions(U)    // non c'è
            - fvc::grad(p)
        );
        fvVectorMatrix& UEqn = tUEqn.ref();
        Foam::vectorField tres = UEqn.residual();
        Eigen::VectorXd resU = Foam2Eigen::field2Eigen(tres);
        tmp<fvVectorMatrix> tUEqn2(UEqn == fvc::grad(p));
        fvVectorMatrix& UEqn2 = tUEqn2.ref();
        UEqn2.relax();
        dimensionedScalar small("small", dimensionSet(0, 0, -1, 0, 0, 0, 0), 1e-18);
        volScalarField rAU(1.0 / UEqn2.A());
        volVectorField HbyA(constrainHbyA(rAU * UEqn2.H(), U, p));
        surfaceScalarField phiHbyA("phiHbyA", fvc::flux(HbyA));
        MRF.makeRelative(phiHbyA);
        adjustPhi(phiHbyA, U, p);
        tmp<volScalarField> rAtU(rAU);

        if (simple.consistent())
        {
            rAtU = 1.0 / (1.0 / rAU - UEqn2.H1());
            phiHbyA +=
                fvc::interpolate(rAtU() - rAU) * fvc::snGrad(p) * mesh.magSf();
            HbyA -= (rAU - rAtU()) * fvc::grad(p);
        }

        constrainPressure(p, U, phiHbyA, rAtU(), MRF);
        fvScalarMatrix pEqn
        (
            fvm::laplacian(rAtU(), p) == fvc::div(phiHbyA)
        );
        Foam::scalarField tpres = pEqn.residual();
        Eigen::VectorXd resP = Foam2Eigen::field2Eigen(tpres);
        Eigen::VectorXd res(resU.size() + resP.size());
        res << resU, resP;
        return res;
    }
    void solveOneStep()
    {
        _simple().loop();
        Vector<double> uresidual_v(0, 0, 0);
        scalar csolve = 0;

// Variable that can be changed
        turbulence->read();
        std::ofstream res_os;
        res_os.open("./ITHACAoutput/Offline/residuals", std::ios_base::app);
        Info << "Time = " << _runTime().timeName() << nl << endl;
        {
#include "UEqn.H"
#include "pEqn.H"
            scalar C = 0;

            for (label i = 0; i < 3; i++)
            {
                if (C < uresidual_v[i])
                {
                    C = uresidual_v[i];
                }
            }

            uresidual = C;
            residual = max(presidual, uresidual);
            Info << "\nResidual: " << residual << endl << endl;
        }
        _laminarTransport().correct();
        turbulence->correct();
        csolve = csolve + 1;
        Info << "ExecutionTime = " << _runTime().elapsedCpuTime() << " s"
             << "  ClockTime = " << _runTime().elapsedClockTime() << " s"
             << nl << endl;
    }
    scalar getResP()
    {
        return presidual;
    }
    scalar getResU()
    {
        return uresidual;
    }
    scalar getRes()
    {
        return residual;
    }
    void restart()
    {
        steadyNS::restart();
        residual = 1;
        uresidual = 1;
        presidual = 1;
    }
    void exportU(std::string& subFolder, std::string& folder, std::string& fieldname)
    {
        ITHACAstream::exportSolution(_U(), subFolder, folder, fieldname);
    }
    void exportP(std::string& subFolder, std::string& folder, std::string& fieldname)
    {
        ITHACAstream::exportSolution(_p(), subFolder, folder, fieldname);
    }
    void changeViscosity(scalar viscosity)
    {
        change_viscosity(viscosity);
    }
};


PYBIND11_MODULE(simpleFOAM_pybind, m)
{
    // bindings to Matrix class
    py::class_<simpleFOAM_pybind>(m, "simpleFOAM_pybind")
    .def(py::init([](
    std::vector<std::string> args) {
        std::vector<char*> cstrs;
        cstrs.reserve(args.size());
        for (auto& s : args)
            cstrs.push_back(const_cast<char*>(s.c_str()));
        return new simpleFOAM_pybind(cstrs.size(), cstrs.data());
    }),
    py::arg("args") = std::vector<std::string> { "." })
    .def("getU", &simpleFOAM_pybind::getU, py::return_value_policy::reference_internal)
    .def("getP", &simpleFOAM_pybind::getP, py::return_value_policy::reference_internal)
    .def("getPhi", &simpleFOAM_pybind::getP, py::return_value_policy::reference_internal)
    .def("printU", &simpleFOAM_pybind::printU)
    .def("printP", &simpleFOAM_pybind::printP)
    .def("printPhi", &simpleFOAM_pybind::printPhi)
    .def("solveOneStep", &simpleFOAM_pybind::solveOneStep)
    .def("getResidual", &simpleFOAM_pybind::getResidual, py::return_value_policy::reference_internal)
    .def("setU", &simpleFOAM_pybind::setU, py::return_value_policy::reference_internal)
    .def("setP", &simpleFOAM_pybind::setP, py::return_value_policy::reference_internal)
    .def("restart", &simpleFOAM_pybind::restart)
    .def("getResU", &simpleFOAM_pybind::getResU)
    .def("getResP", &simpleFOAM_pybind::getResP)
    .def("getRes", &simpleFOAM_pybind::getRes)
    .def("exportU", &simpleFOAM_pybind::exportU)
    .def("exportP", &simpleFOAM_pybind::exportP)
    .def("changeViscosity", &simpleFOAM_pybind::changeViscosity)
    ;
}
