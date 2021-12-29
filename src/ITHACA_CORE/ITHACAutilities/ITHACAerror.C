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

#include "ITHACAerror.H"

/// \file
/// Source file of the ITHACAerror file.

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

namespace ITHACAutilities
{

template<class Type, template<class> class PatchField, class GeoMesh>
GeometricField<Type, PatchField, GeoMesh> onSubmesh(
    GeometricField<Type, PatchField, GeoMesh>& field, List<label>* labels,
    autoPtr<fvMeshSubset>& submesh)
{
    autoPtr<GeometricField<Type, PatchField, GeoMesh>> field_S;

    if (labels != NULL)
    {
        submesh = autoPtr<fvMeshSubset>(new fvMeshSubset(field.mesh()));
#if OPENFOAM >= 1812
        submesh->setCellSubset(*labels);
#else
        submesh->setLargeCellSubset(*labels);
#endif
        GeometricField<Type, PatchField, GeoMesh> field_tmp(submesh->interpolate(
                    field));
        field_S = autoPtr<GeometricField<Type, PatchField, GeoMesh>>
                  (new GeometricField<Type, PatchField, GeoMesh>(field_tmp.clone()));
    }
    else
    {
        field_S = autoPtr<GeometricField<Type, PatchField, GeoMesh>>
                  (new GeometricField<Type, PatchField, GeoMesh>(field.clone()));
    }

    return GeometricField<Type, PatchField, GeoMesh>(field_S().clone());
}
template volScalarField onSubmesh(volScalarField&
                                  field1, List<label>* labels, autoPtr<fvMeshSubset>& submesh);
template volVectorField onSubmesh(volVectorField&
                                  field1, List<label>* labels, autoPtr<fvMeshSubset>& submesh);

template<class Type>
double L2Norm(GeometricField<Type, fvPatchField, volMesh>& field,
              List<label>* labels)
{
    autoPtr<fvMeshSubset> submesh;
    GeometricField<Type, fvPatchField, volMesh> fieldtmp = onSubmesh(field,
            labels, submesh);
    return Foam::sqrt(fvc::domainIntegrate(magSqr(fieldtmp)).value());
}

template double L2Norm(volScalarField& field, List<label>* labels);
template double L2Norm(volVectorField& field, List<label>* labels);



template<class Type>
double LinfNorm(GeometricField<Type, fvPatchField, volMesh>& field,
                List<label>* labels)
{
    autoPtr<fvMeshSubset> submesh;
    GeometricField<Type, fvPatchField, volMesh> fieldtmp = onSubmesh(field,
            labels, submesh);
    return Foam::max(Foam::sqrt(magSqr(fieldtmp))).value();;
}

template<class Type>
double H1Seminorm(GeometricField<Type, fvPatchField, volMesh>& field,
                  List<label>* labels)
{
    autoPtr<fvMeshSubset> submesh;
    GeometricField<Type, fvPatchField, volMesh> fieldtmp = onSubmesh(field,
            labels, submesh);
    return Foam::sqrt(fvc::domainIntegrate(magSqr(fvc::grad(fieldtmp))).value());
}

template double H1Seminorm(volScalarField& field, List<label>* labels);
template double H1Seminorm(volVectorField& field, List<label>* labels);


template<class Type, template<class> class PatchField, class GeoMesh>
double frobNorm(GeometricField<Type, PatchField, GeoMesh>& field,
                List<label>* labels)
{
    double norm(0);
    autoPtr<fvMeshSubset> submesh;
    GeometricField<Type, PatchField, GeoMesh> fieldtmp = onSubmesh(field, labels,
            submesh);
    Eigen::VectorXd vF = Foam2Eigen::field2Eigen(fieldtmp);
    return vF.norm();
}

template double frobNorm(volScalarField& field, List<label>* labels);
template double frobNorm(volVectorField& field, List<label>* labels);



template double LinfNorm(volScalarField& field, List<label>* labels);
template double LinfNorm(volVectorField& field, List<label>* labels);

template<class Type, template<class> class PatchField, class GeoMesh>
double errorFrobRel(GeometricField<Type, PatchField, GeoMesh>& field1,
                    GeometricField<Type, PatchField, GeoMesh>& field2, List<label>* labels)
{
    double err;
    GeometricField<Type, PatchField, GeoMesh> errField(field1 - field2);

    if (frobNorm(field1, labels) <= 1e-6)
    {
        err = 0;
    }
    else
    {
        err = frobNorm(errField, labels) / frobNorm(field1, labels);
    }

    return err;
}


template double errorFrobRel(GeometricField<scalar, fvPatchField, volMesh>&
                             field1,
                             GeometricField<scalar, fvPatchField, volMesh>& field2, List<label>* labels);
template double errorFrobRel(GeometricField<vector, fvPatchField, volMesh>&
                             field1,
                             GeometricField<vector, fvPatchField, volMesh>& field2, List<label>* labels);

template double errorFrobRel(GeometricField<scalar, fvsPatchField, surfaceMesh>&
                             field1,
                             GeometricField<scalar, fvsPatchField, surfaceMesh>& field2,
                             List<label>* labels);


template<typename T>
double errorLinfRel(GeometricField<T, fvPatchField, volMesh>& field1,
                    GeometricField<T, fvPatchField, volMesh>& field2, List<label>* labels)
{
    double err;
    GeometricField<T, fvPatchField, volMesh> errField(field1 - field2);

    if (LinfNorm(field1, labels) <= 1e-6)
    {
        err = 0;
    }
    else
    {
        err = LinfNorm(errField, labels) / LinfNorm(field1, labels);
    }

    return err;
}


template double errorLinfRel(GeometricField<scalar, fvPatchField, volMesh>&
                             field1,
                             GeometricField<scalar, fvPatchField, volMesh>& field2, List<label>* labels);
template double errorLinfRel(GeometricField<vector, fvPatchField, volMesh>&
                             field1,
                             GeometricField<vector, fvPatchField, volMesh>& field2, List<label>* labels);

template<>
double errorL2Abs(GeometricField<vector, fvPatchField, volMesh>& field1,
                  GeometricField<vector, fvPatchField, volMesh>& field2, volScalarField& Volumes)

{
    volScalarField diffFields2 = (((field1 - field2) & (field1 - field2)) *
                                  Volumes).ref();
    double err = Foam::sqrt(gSum(diffFields2));
    return err;
}

template<>
double errorL2Abs(GeometricField<scalar, fvPatchField, volMesh>& field1,
                  GeometricField<scalar, fvPatchField, volMesh>& field2, volScalarField& Volumes)

{
    volScalarField diffFields2 = (((field1 - field2) * (field1 - field2)) *
                                  Volumes).ref();
    double err = Foam::sqrt(gSum(diffFields2));
    return err;
}

template<typename T>
double errorL2Abs(GeometricField<T, fvPatchField, volMesh>& field1,
                  GeometricField<T, fvPatchField, volMesh>& field2, List<label>* labels)
{
    GeometricField<T, fvPatchField, volMesh> errField(field1 - field2);
    double err = L2Norm(errField, labels);
    return err;
}

template double errorL2Abs(GeometricField<scalar, fvPatchField, volMesh>&
                           field1,
                           GeometricField<scalar, fvPatchField, volMesh>& field2, List<label>* labels);
template double errorL2Abs(GeometricField<vector, fvPatchField, volMesh>&
                           field1,
                           GeometricField<vector, fvPatchField, volMesh>& field2, List<label>* labels);


template<class T, template<class> class PatchField, class GeoMesh>
Eigen::MatrixXd errorFrobRel(PtrList<GeometricField<T, PatchField, GeoMesh>>&
                             fields1,
                             PtrList<GeometricField<T, PatchField, GeoMesh>>& fields2, List<label>* labels)
{
    Eigen::VectorXd err;

    if (fields1.size() != fields2.size())
    {
        Info << "The two fields do not have the same size, code will abort" << endl;
        exit(0);
    }

    err.resize(fields1.size(), 1);

    for (label k = 0; k < fields1.size(); k++)
    {
        err(k, 0) = errorFrobRel(fields1[k], fields2[k], labels);
        Info << " Error is " << err[k] << endl;
    }

    return err;
}

template Eigen::MatrixXd errorFrobRel(
    PtrList<GeometricField<scalar, fvPatchField, volMesh>>& fields1,
    PtrList<GeometricField<scalar, fvPatchField, volMesh>>& fields2,
    List<label>* labels);
template Eigen::MatrixXd errorFrobRel(
    PtrList<GeometricField<vector, fvPatchField, volMesh>>& fields1,
    PtrList<GeometricField<vector, fvPatchField, volMesh>>& fields2,
    List<label>* labels);
template Eigen::MatrixXd errorFrobRel(
    PtrList<GeometricField<scalar, fvsPatchField, surfaceMesh>>& fields1,
    PtrList<GeometricField<scalar, fvsPatchField, surfaceMesh>>& fields2,
    List<label>* labels);


template<typename T>
Eigen::MatrixXd errorL2Abs(
    PtrList<GeometricField<T, fvPatchField, volMesh>>& fields1,
    PtrList<GeometricField<T, fvPatchField, volMesh>>& fields2,
    PtrList<volScalarField>& Volumes)
{
    M_Assert(fields1.size() == fields2.size(),
             "The two fields do not have the same size, code will abort");
    M_Assert(fields1.size() == Volumes.size(),
             "The volumes field and the two solution fields do not have the same size, code will abort");
    Eigen::VectorXd err;
    err.resize(fields1.size(), 1);

    for (label k = 0; k < fields1.size(); k++)
    {
        err(k, 0) = errorL2Abs(fields1[k], fields2[k], Volumes[k]);
        Info << " Error is " << err[k] << endl;
    }

    return err;
}

template Eigen::MatrixXd errorL2Abs(
    PtrList<GeometricField<scalar, fvPatchField, volMesh>>& fields1,
    PtrList<GeometricField<scalar, fvPatchField, volMesh>>& fields2,
    PtrList<volScalarField>& Volumes);
template Eigen::MatrixXd errorL2Abs(
    PtrList<GeometricField<vector, fvPatchField, volMesh>>& fields1,
    PtrList<GeometricField<vector, fvPatchField, volMesh>>& fields2,
    PtrList<volScalarField>& Volumes);

template<typename T>
Eigen::MatrixXd errorL2Abs(PtrList<GeometricField<T, fvPatchField, volMesh>>&
                           fields1,
                           PtrList<GeometricField<T, fvPatchField, volMesh>>& fields2, List<label>* labels)
{
    Eigen::VectorXd err;

    if (fields1.size() != fields2.size())
    {
        Info << "The two fields do not have the same size, code will abort" << endl;
        exit(0);
    }

    err.resize(fields1.size(), 1);

    for (label k = 0; k < fields1.size(); k++)
    {
        err(k, 0) = errorL2Abs(fields1[k], fields2[k], labels);
        Info << " Error is " << err[k] << endl;
    }

    return err;
}

template Eigen::MatrixXd errorL2Abs(
    PtrList<GeometricField<scalar, fvPatchField, volMesh>>& fields1,
    PtrList<GeometricField<scalar, fvPatchField, volMesh>>& fields2,
    List<label>* labels);
template Eigen::MatrixXd errorL2Abs(
    PtrList<GeometricField<vector, fvPatchField, volMesh>>& fields1,
    PtrList<GeometricField<vector, fvPatchField, volMesh>>& fields2,
    List<label>* labels);

template<typename T>
double errorL2Rel(GeometricField<T, fvPatchField, volMesh>& field1,
                  GeometricField<T, fvPatchField, volMesh>& field2, List<label>* labels)
{
    double err;
    GeometricField<T, fvPatchField, volMesh> errField(field1 - field2);

    if (L2Norm(field1, labels) <= 1e-6)
    {
        err = 0;
    }
    else
    {
        err = L2Norm(errField, labels) / L2Norm(
                  field1, labels);
    }

    return err;
}

template double errorL2Rel(GeometricField<scalar, fvPatchField, volMesh>&
                           field1,
                           GeometricField<scalar, fvPatchField, volMesh>& field2, List<label>* labels);
template double errorL2Rel(GeometricField<vector, fvPatchField, volMesh>&
                           field1,
                           GeometricField<vector, fvPatchField, volMesh>& field2, List<label>* labels);

template<typename T>
Eigen::MatrixXd errorL2Rel(PtrList<GeometricField<T, fvPatchField, volMesh>>&
                           fields1, PtrList<GeometricField<T, fvPatchField, volMesh>>& fields2,
                           List<label>* labels)
{
    Eigen::VectorXd err;

    if (fields1.size() != fields2.size())
    {
        Info << "The two fields do not have the same size, code will abort" << endl;
        exit(0);
    }

    err.resize(fields1.size(), 1);

    for (label k = 0; k < fields1.size(); k++)
    {
        err(k, 0) = errorL2Rel(fields1[k], fields2[k], labels);
        Info << " Error is " << err[k] << endl;
    }

    return err;
}

template Eigen::MatrixXd errorL2Rel(
    PtrList<GeometricField<scalar, fvPatchField, volMesh>>& fields1,
    PtrList<GeometricField<scalar, fvPatchField, volMesh>>& fields2,
    List<label>* labels);
template Eigen::MatrixXd errorL2Rel(
    PtrList<GeometricField<vector, fvPatchField, volMesh>>& fields1,
    PtrList<GeometricField<vector, fvPatchField, volMesh>>& fields2,
    List<label>* labels);


double L2normOnPatch(fvMesh& mesh, volScalarField& field,
                     word patch)
{
    double L2 = 0;
    //Access the mesh information for the boundary
    label patchID = mesh.boundaryMesh().findPatchID(patch);
    const polyPatch& cPatch = mesh.boundaryMesh()[patchID];
    //List of cells close to a boundary
    const labelUList& faceCells = cPatch.faceCells();
    forAll(cPatch, faceI)
    {
        //id of the owner cell having the face
        label faceOwner = faceCells[faceI] ;
        scalar faceArea = mesh.magSf().boundaryField()[patchID][faceI];
        L2 += faceArea * field[faceOwner] * field[faceOwner];
    }
    return Foam::sqrt(L2);
}

double L2productOnPatch(fvMesh& mesh, List<scalar>& field1,
                        List<scalar>& field2, word patch)
{
    M_Assert(field1.size() == field2.size(),
             "The two fields do not have the same size, code will abort");
    double L2 = 0;
    //Access the mesh information for the boundary
    label patchID = mesh.boundaryMesh().findPatchID(patch);
    const polyPatch& cPatch = mesh.boundaryMesh()[patchID];
    M_Assert(field1.size() == cPatch.size(),
             "The filds must have the same size of the patch, code will abort");
    forAll(cPatch, faceI)
    {
        scalar faceArea = mesh.magSf().boundaryField()[patchID][faceI];
        L2 += faceArea * field1[faceI] * field2[faceI];
    }
    return L2;
}

double LinfNormOnPatch(fvMesh& mesh, volScalarField& field,
                       word patch)
{
    double Linf = 0;
    //Access the mesh information for the boundary
    label patchID = mesh.boundaryMesh().findPatchID(patch);
    const polyPatch& cPatch = mesh.boundaryMesh()[patchID];
    //List of cells close to a boundary
    const labelUList& faceCells = cPatch.faceCells();
    forAll(cPatch, faceI)
    {
        label faceOwner = faceCells[faceI] ;

        if (faceI == 0)
        {
            Linf = std::abs(field[faceOwner]);
        }
        else if (std::abs(field[faceOwner]) > Linf)
        {
            Linf = std::abs(field[faceOwner]);
        }
    }
    return Linf;
}

double integralOnPatch(fvMesh& mesh, volScalarField& field,
                       word patch)
{
    double integral = 0;
    //Access the mesh information for the boundary
    label patchID = mesh.boundaryMesh().findPatchID(patch);
    const polyPatch& cPatch = mesh.boundaryMesh()[patchID];
    //List of cells close to a boundary
    const labelUList& faceCells = cPatch.faceCells();
    forAll(cPatch, faceI)
    {
        //id of the owner cell having the face
        label faceOwner = faceCells[faceI] ;
        scalar faceArea = mesh.magSf().boundaryField()[patchID][faceI];
        integral += faceArea * field[faceOwner];
    }
    return integral;
}

double integralOnPatch(fvMesh& mesh, List<scalar> field,
                       word patch)
{
    double integral = 0;
    //Access the mesh information for the boundary
    label patchID = mesh.boundaryMesh().findPatchID(patch);
    const polyPatch& cPatch = mesh.boundaryMesh()[patchID];
    //List of cells close to a boundary
    const labelUList& faceCells = cPatch.faceCells();
    int patchSize = mesh.magSf().boundaryField()[patchID].size();
    std::string message = "The input list (size = " + std::to_string(field.size())
                          + ") must have the same size of the patch mesh (size = "
                          + std::to_string(patchSize) + ")";
    M_Assert( patchSize == field.size(), message.c_str());
    forAll(cPatch, faceI)
    {
        //id of the owner cell having the face
        label faceOwner = faceCells[faceI] ;
        scalar faceArea = mesh.magSf().boundaryField()[patchID][faceI];
        integral += faceArea * field[faceI];
    }
    return integral;
}

}
