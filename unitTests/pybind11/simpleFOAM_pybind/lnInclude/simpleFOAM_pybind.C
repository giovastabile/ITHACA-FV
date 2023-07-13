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
    .def("printPhi", &simpleFOAM_pybind::printPhi);
}
