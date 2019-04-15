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
_T = autoPtr<volScalarField>
     (
         new volScalarField
         (
             IOobject
             (
                 "T",
                 runTime.timeName(),
                 mesh,
                 IOobject::MUST_READ,
                 IOobject::AUTO_WRITE
             ),
             mesh
         )
     );

volScalarField& T = _T();

_cv = autoPtr<volScalarField>
      (
          new volScalarField
          (
              IOobject
              (
                  "cv",
                  runTime.timeName(),
                  mesh,
                  IOobject::NO_READ,
                  IOobject::AUTO_WRITE
              ),
              mesh,
              dimensionedScalar("zero", dimVolume, 0.0)
          )
      );

_NonOrtho = autoPtr<volScalarField>
            (
                new volScalarField
                (
                    IOobject
                    (
                        "NonOrtho",
                        runTime.timeName(),
                        mesh,
                        IOobject::NO_READ,
                        IOobject::AUTO_WRITE
                    ),
                    mesh,
                    dimensionedScalar("zero", dimArea, 0.0)
                )
            );

volScalarField& NonOrtho = _NonOrtho();

IOdictionary transportProperties
(
    IOobject
    (
        "transportProperties",
        runTime.constant(),
        mesh,
        IOobject::MUST_READ_IF_MODIFIED,
        IOobject::NO_WRITE
    )
);


Info << "Reading diffusivity DT\n" << endl;

_DT = autoPtr<dimensionedScalar>
      (
          new dimensionedScalar
          (
              transportProperties.lookup("DT")
          )
      );

// std::cerr << "debug point 2" << std::endl;
//            fvMesh& mesh2  =  const_cast<fvMesh&>(T.mesh());
// _PointDisplacement = autoPtr<pointVectorField>
//                      (
//                          new pointVectorField
//                          (
//                              const_cast<pointVectorField&>
//                              (
//                                  mesh2.objectRegistry::lookupObject<pointVectorField>
//                                  (
//                                      "pointDisplacement"
//                                  )
//                              )
//                          )
//                      );
//                      std::cerr << "debug point 4" << std::endl;