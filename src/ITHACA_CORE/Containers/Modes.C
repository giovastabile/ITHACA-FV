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


/// \file
/// Source file of the Modes class.

#include "Modes.H"

template<>
PtrList<GeometricField<scalar, fvPatchField, volMesh>>
        Modes<scalar>::projectSnapshots(
            PtrList<GeometricField<scalar, fvPatchField, volMesh>> snapshots,
            int numberOfModes,
            word innerProduct)
{
    if (numberOfModes == 0)
    {
        numberOfModes == this->size();
    }

    M_Assert(numberOfModes <= this->size(),
             "The number of Modes used for the projection cannot be bigger than the number of available modes");
    Eigen::MatrixXd M_vol;
    Eigen::MatrixXd M;
    PtrList<GeometricField<scalar, fvPatchField, volMesh>> projSnap = snapshots;
    Eigen::MatrixXd projSnapI;
    Eigen::MatrixXd projSnapCoeff;

    for (int i = 0; i < snapshots.size(); i++)
    {
        if (innerProduct == "L2")
        {
            Eigen::VectorXd V = Foam2Eigen::field2Eigen(snapshots[i].mesh().V());
            M_vol = V.asDiagonal();
        }

        else if (innerProduct == "Frobenius")
        {
            M_vol =  Eigen::MatrixXd::Identity(snapshots[i].size(), snapshots[i].size());
        }

        else
        {
            std::cout << "Inner product not defined" << endl;
            exit(0);
        }

        //projSnapI(j,0) = modes[j].transpose()*M*(Foam2Eigen::field2Eigen(snapshots[i]));
        M = EigenModes[0].transpose() * M_vol * EigenModes[0];
        projSnapI = EigenModes[0].transpose() * (Foam2Eigen::field2Eigen(snapshots[i]));
        projSnapCoeff = M.fullPivLu().solve(projSnapI);
        projSnap.append(reconstruct(projSnapCoeff, "projSnap"));
    }

    return projSnap;
}

template<>
PtrList<GeometricField<vector, fvPatchField, volMesh>>
        Modes<vector>::projectSnapshots(
            PtrList<GeometricField<vector, fvPatchField, volMesh>> snapshots,
            int numberOfModes, word innerProduct)
{
    if (numberOfModes == 0)
    {
        numberOfModes == this->size();
    }

    M_Assert(numberOfModes <= this->size(),
             "The number of Modes used for the projection cannot be bigger than the number of available modes");
    Eigen::MatrixXd M_vol;
    Eigen::MatrixXd M;
    PtrList<GeometricField<vector, fvPatchField, volMesh>> projSnap = snapshots;
    Eigen::MatrixXd projSnapI;
    Eigen::MatrixXd projSnapCoeff;

    for (int i = 0; i < snapshots.size(); i++)
    {
        if (innerProduct == "L2")
        {
            Eigen::VectorXd V = Foam2Eigen::field2Eigen(snapshots[i].mesh().V());
            Eigen::VectorXd V3d = (V.replicate(3, 1));
            M_vol = V3d.asDiagonal();
        }

        else if (innerProduct == "Frobenius")
        {
            M_vol =  Eigen::MatrixXd::Identity(snapshots[i].size(), snapshots[i].size());
        }

        else
        {
            std::cout << "Inner product not defined" << endl;
            exit(0);
        }

        //projSnapI(j,0) = modes[j].transpose()*M*(Foam2Eigen::field2Eigen(snapshots[i]));
        M = EigenModes[0].transpose() * M_vol * EigenModes[0];
        projSnapI = EigenModes[0].transpose() * (Foam2Eigen::field2Eigen(snapshots[i]));
        projSnapCoeff = M.fullPivLu().solve(projSnapI);
        projSnap.append(reconstruct(projSnapCoeff, "projSnap"));
    }

    return projSnap;
}

