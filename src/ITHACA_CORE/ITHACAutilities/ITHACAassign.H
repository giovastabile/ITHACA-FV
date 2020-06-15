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
Namespace
    ITHACAutilites
Description
    Utilities to assign fields and BCs of OF fields
SourceFiles
    ITHACAassign.C
\*---------------------------------------------------------------------------*/

/// \file
/// Header file of the ITHACAassign file.
/// \dir
/// Directory containing the header and source files for the ITHACAassign files.

#ifndef ITHACAassign_H
#define ITHACAassign_H

#include "fvCFD.H"
#include "IOmanip.H"
#include "freestreamFvPatchField.H"
#include <sys/stat.h>
#include <unistd.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#include <Eigen/Eigen>
#pragma GCC diagnostic pop
#include <functional>
#include "./colormod.H"
#include "polyMeshTools.H"
#include <chrono>
#include "mixedFvPatchFields.H"
#include "fvMeshSubset.H"
using namespace std::placeholders;
#include "Foam2Eigen.H"
#include "ITHACAerror.H"

/// Namespace to implement some useful assign operation of OF fields
namespace ITHACAutilities
{
//-----------------------------------------------------------------------------
/// @brief      A function to compute time-averaged fields for a set of different parameter samples
/// and also the fields with the corresponding averaged subtracted
///
/// @param[in]  fields     The fields from which the time-averaged fields have to be computed
/// @param[in]  ind        The indices at which fields for different samples of the parameter start
/// @param      ave        The computed time-averaged fields
///
/// @tparam     TypeField  Type of field
///
/// @return     A list of fields which correspond to the original fields subtracted by
/// the time-averaged part
///
template<class TypeField>
PtrList<TypeField> averageSubtract(PtrList<TypeField>
                                   fields, Eigen::MatrixXd ind, PtrList<TypeField>& ave);

//-----------------------------------------------------------------------------
/// @brief      Calculates the average of a list of fields
///
/// @param      fields     The fields
///
/// @tparam     TypeField  { description }
///
/// @return     The average.
///
template<class TypeField>
TypeField computeAverage(PtrList<TypeField>& fields);

//--------------------------------------------------------------------------
/// Assign internal field
///
/// @param[in,out]  field  Field to which assign the internal field
/// @param          value  The value you want to assign
///
/// @tparam         T      type of field (volVectorField or volScalarField)
///
template<typename T>
void assignIF(GeometricField<T, fvPatchField, volMesh>& field, T& value);

//--------------------------------------------------------------------------
/// Assign internal field
///
/// @param[in,out]  field  Field to which assign the internal field
/// @param          value  The value you want to assign
/// @param          indices  The list of indices where you want assign
///
/// @tparam         T      type of field (volVectorField or volScalarField)
///
template<typename T>
void assignIF(GeometricField<T, fvPatchField, volMesh>& field, T& value,
              List<int>& indices);

//--------------------------------------------------------------------------
/// Assign internal field
///
/// @param[in,out]  field  Field to which assign the internal field
/// @param          value  The value you want to assign
/// @param          index  The index where you want assign
///
/// @tparam         T      type of field (volVectorField or volScalarField)
///
template<typename T>
void assignIF(GeometricField<T, fvPatchField, volMesh>& field, T& value,
              int index);

//--------------------------------------------------------------------------
/// Assign one to volScalarField
///
/// @param[in,out] s     field where you want to assign the internal field
///                      condition
/// @param         L     List of indices
///
void assignONE(volScalarField& s, List<int>& L);


//--------------------------------------------------------------------------
/// Assign uniform Boundary Condition to a volScalarField
///
/// @param[in]  s       field where you want to assign the BC.
/// @param[in]  BC_ind  The BC index.
/// @param[in]  value   The value you want to assign (it must be a double).
///
void assignBC(volScalarField& s, label BC_ind, double value);

//--------------------------------------------------------------------------
/// Assign Boundary Condition to a volScalarField
///
/// @param[in]  s       field where you want to assign the BC.
/// @param[in]  BC_ind  The BC index.
/// @param[in]  valueVec   The value you want to assign (it must be an Eigen MatrixXd).
///
void assignBC(volScalarField& s, label BC_ind, Eigen::MatrixXd valueVec);

//--------------------------------------------------------------------------
/// Assign Boundary Condition to a volScalarField
///
/// @param[in]  s       field where you want to assign the BC.
/// @param[in]  BC_ind  The BC index.
/// @param[in]  valueList   The value you want to assign (it must be a list of doubles).
///
void assignBC(volScalarField& s, label BC_ind, List<double> valueList);

//--------------------------------------------------------------------------
/// Assign Boundary Condition to a volVectorField
///
/// @param[in]  s       field where you want to assign the BC.
/// @param[in]  BC_ind  The BC index.
/// @param[in]  value   The value you want to assign (it must be an OpenFOAM vector).
///
void assignBC(volVectorField& s, label BC_ind, vector value);

//--------------------------------------------------------------------------
/// Assign Boundary Condition to a volVectorField
///
/// @param[in]  s       field where you want to assign the BC.
/// @param[in]  BC_ind  The BC index.
/// @param[in]  valueVec   The value you want to assign (it must be an Eigen MatrixXd).
///
void assignBC(volVectorField& s, label BC_ind, Eigen::MatrixXd valueVec);

//--------------------------------------------------------------------------
/// Assign Boundary Condition to a volVectorField
///
/// @param[in]  s       field where you want to assign the BC.
/// @param[in]  BC_ind  The BC index.
/// @param[in]  valueList   The value you want to assign (it must be a list of doubles).
///
void assignBC(volVectorField& s, label BC_ind, List<vector> valueList);

//------------------------------------------------------------------------------
/// @brief      Assign Boundary Condition to a surfaceField
///
/// @param      field   field where you want to assign the BC.
/// @param[in]  BC_ind  The bc ind
/// @param      value   you want to assign (it must be a list of double or vector).
///
/// @tparam     T type of field (surfaceVectorField or surfaceScalarField)
///
template<typename T>
void assignBC(GeometricField<T, fvsPatchField, surfaceMesh>& field,
              label BC_ind,
              List<T>& value);

//------------------------------------------------------------------------------
/// @brief      Assign Boundary Condition to a surfaceField
///
/// @param      field   field where you want to assign the BC.
/// @param[in]  BC_ind  The bc ind
/// @param      value   you want to assign (it must be column matrix).
///
/// @tparam     T type of field (surfaceVectorField or surfaceScalarField)
///
template<typename T>
void assignBC(GeometricField<T, fvsPatchField, surfaceMesh>& field,
              label BC_ind,
              Eigen::MatrixXd& value);

//------------------------------------------------------------------------------
/// @brief      Assign Boundary Condition to a surfaceField
///
/// @param      field   field where you want to assign the BC.
/// @param[in]  BC_ind  The bc ind
/// @param      value   you want to assign (it must be a double or vector).
///
/// @tparam     T type of field (surfaceVectorField or surfaceScalarField)
///
template<typename T>
void assignBC(GeometricField<T, fvsPatchField, surfaceMesh>& field,
              label BC_ind,
              T& value);

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcomment"
//------------------------------------------------------------------------------
/// Set value of a volScalarField to a constant inside a given box
///
/// @details    the Box must be defined with a 2*3 Eigen::MatrixXd in the
///             following way
///
///             @\[ Box=\\begin{bmatrix} x_1 & y_1 & z_1 @\ @\ \ x_2 & y_2 & z_2
///             @\ @\ @\end{bmatrix} @\]
///
///             where
///             @f$ x_1, y_1, z_1
///             @f$ and
///             @f$ x_2, y_2, z_2
///             @f$ are the coordinates of the two corners defining the box.
///
/// @param[in]  field  The field.
/// @param[in]  Box    The box.
/// @param[in]  value  The value you want to give to the volScalarField
///
/// @tparam     T      type of field can be scalar or vector
///
template<typename T>
void setBoxToValue(GeometricField<T, fvPatchField, volMesh>& field,
                   Eigen::MatrixXd Box, T value);
#pragma GCC diagnostic pop

//--------------------------------------------------------------------------
/// @brief      Change the boundary condition type for a GeometricField
///
/// @param      field   The field
/// @param[in]  BCtype  The The BC type you want to set
/// @param[in]  BC_ind  The BC index
///
/// @tparam     TypeField  The type of field it can be scalar or vector
///
template<class TypeField>
void changeBCtype(GeometricField<TypeField, fvPatchField, volMesh>&
                  field,
                  word BCtype, label BC_ind);


//--------------------------------------------------------------------------
/// @brief      Sets some given Indices of a list of objects to given values
///
/// @param      ind2set       The indices of the list that you want to assign
/// @param      value2set     The values you want to set
/// @param      movingIDS     The index of the list in the original list that you want to assign
/// @param      originalList  The original list
///
/// @tparam     T             Type of list
///
template<typename T>
void setIndices2Value(labelList& ind2set, List<T>& value2set,
                      labelList& movingIDS, List<T>& originalList);

//--------------------------------------------------------------------------
/// @brief      Assign value of a boundary condition of type "mixed"
///
/// @param[in,out]   field      Field
/// @param[in]       BC_ind     Index of the boundary patch
/// @param[in]       value      Reference fixed value
/// @param[in]       grad       Reference normal gradient
/// @param[in]       valueFrac  Weight field
///
/// @tparam          type_f     Type of the field, it can be scalar or vector.
///
template<typename type_f>
void assignMixedBC(GeometricField<type_f, fvPatchField, volMesh>& field,
                   label BC_ind, Eigen::MatrixXd& value,
                   Eigen::MatrixXd& grad, Eigen::MatrixXd& valueFrac);


//--------------------------------------------------------------------------
/// @brief      Assign value of a boundary condition of type "mixed"
///
/// @param[in,out]   field      Field
/// @param[in]       BC_ind     Index of the boundary patch
/// @param[in]       value      Reference fixed value
/// @param[in]       grad       Reference normal gradient
/// @param[in]       valueFrac  Weight field
///
/// @tparam          type_f     Type of the field, it can be scalar or vector.
///
template<typename type_f>
void assignMixedBC(GeometricField<type_f, fvPatchField, volMesh>& field,
                   label BC_ind, List<type_f>& value,
                   List<type_f>& grad, List<scalar>& valueFrac);


//--------------------------------------------------------------------------
/// @brief      Normalize list of Geometric fields
///
/// @param      fields  The fields to be normalized
///
/// @tparam     type_f  type of the Field can be Scalar or Vector
///
template<typename type_f>
void normalizeFields(
    PtrList<GeometricField<type_f, fvPatchField, volMesh>>& fields);

template<typename type_f>
void normalizeFields(
    PtrList<GeometricField<type_f, fvPatchField, volMesh>>& fields)
{
    Eigen::MatrixXd eigenFields = Foam2Eigen::PtrList2Eigen(fields);
    List<Eigen::MatrixXd> eigenFieldsBC = Foam2Eigen::PtrList2EigenBC(fields);

    for (label i = 0; i < fields.size(); i++)
    {
        double norm = L2Norm(fields[i]);
        GeometricField<type_f, fvPatchField, volMesh> tmp(fields[0].name(),
                fields[0] * 0);
        Eigen::VectorXd vec = eigenFields.col(i) / norm;
        tmp = Foam2Eigen::Eigen2field(tmp, vec);

        // Adjusting boundary conditions
        for (int k = 0; k < tmp.boundaryField().size(); k++)
        {
            Eigen::MatrixXd vec = eigenFieldsBC[k].col(i) / norm;
            assignBC(tmp, k, vec);
        }

        fields.set(i, tmp);
    }
}

template void normalizeFields(
    PtrList<GeometricField<scalar, fvPatchField, volMesh>>& fields);
template void normalizeFields(
    PtrList<GeometricField<vector, fvPatchField, volMesh>>& fields);

}

#endif