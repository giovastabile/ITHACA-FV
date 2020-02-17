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
    Example of steady NS Reduction Problem
SourceFiles
    03steadyNS.C
\*---------------------------------------------------------------------------*/

#include "SteadyNSSimple.H"
#include "ITHACAstream.H"
#include "ITHACAPOD.H"
#include "ReducedSimpleSteadyNS.H"
#include "forces.H"
#include "IOmanip.H"
#include "DEIM.H"


class DEIM_function : public DEIM<fvVectorMatrix>
{
    public:
        using DEIM::DEIM;

        static fvVectorMatrix divU(volVectorField& U, surfaceScalarField& phi)
        {
            fvVectorMatrix divU
            (
                fvm::div(phi, U)
            );
            return divU;
        }

        PtrList<volVectorField> fieldsA;
        PtrList<volVectorField> fieldsB;
        PtrList<surfaceScalarField> phi;
};

class tutorial18 : public SteadyNSSimple
{
    public:
        /// Constructor
        explicit tutorial18(int argc, char* argv[])
            :
            SteadyNSSimple(argc, argv),
            U(_U()),
            p(_p()),
            phi(_phi())
        {}

        /// Velocity field
        volVectorField& U;
        /// Pressure field
        volScalarField& p;
        ///
        surfaceScalarField& phi;

        /// Perform an Offline solve
        void offlineSolve()
        {
            Vector<double> inl(0, 0, 0);
            List<scalar> mu_now(1);

            // if the offline solution is already performed read the fields
            if (offline)
            {
                ITHACAstream::read_fields(Ufield, U, "./ITHACAoutput/Offline/");
                ITHACAstream::read_fields(Pfield, p, "./ITHACAoutput/Offline/");
                mu_samples =
                    ITHACAstream::readMatrix("./ITHACAoutput/Offline/mu_samples_mat.txt");
            }

            else
            {
                label BCind = 0;

                for (label i = 0; i < mu.rows(); i++)
                {
                    mu_now[0] = mu(i, 0);
                    change_viscosity(mu(i, 0));
                    truthSolve2(mu_now);
                }
            }
        }



};

class tutorial18red : public reducedSimpleSteadyNS
{
    public:

        tutorial18red(SteadyNSSimple& FOMproblem)
            :
            reducedSimpleSteadyNS(FOMproblem)
        {}

        DEIM_function* DEIMmatrice;
        std::vector<Eigen::MatrixXd> ReducedMatricesA;
        std::vector<Eigen::MatrixXd> ReducedVectorsB;

        void PODDEIM(int NmodesT, int NmodesDEIMA, int NmodesDEIMB)
        {
            fvMesh& mesh  =  const_cast<fvMesh&>(problem->_U().mesh());
            DEIMmatrice = new DEIM_function(problem->Ulist, NmodesDEIMA, NmodesDEIMB,
                                            "U_matrix");
            DEIMmatrice->fieldsA = DEIMmatrice->generateSubmeshesMatrix(1, mesh,
                                   problem->_U());
            DEIMmatrice->fieldsB = DEIMmatrice->generateSubmeshesVector(1, mesh,
                                   problem->_U());
            ReducedMatricesA.resize(NmodesDEIMA);
            ReducedVectorsB.resize(NmodesDEIMB);

            for (int i = 0; i < NmodesDEIMA; i++)
            {
                ReducedMatricesA[i] = ULmodes.EigenModes[0].leftCols(NmodesT).transpose() *
                                      DEIMmatrice->MatrixOnlineA[i] *
                                      ULmodes.EigenModes[0].leftCols(NmodesT);
            }

            for (int i = 0; i < NmodesDEIMB; i++)
            {
                ReducedVectorsB[i] = ULmodes.EigenModes[0].leftCols(NmodesT).transpose() *
                                     DEIMmatrice->MatrixOnlineB;
            }
        };

};

int main(int argc, char* argv[])
{
    // Construct the tutorial object
    tutorial18 example(argc, argv);
    // Read some parameters from file
    ITHACAparameters para;
    int NmodesUout = para.ITHACAdict->lookupOrDefault<int>("NmodesUout", 15);
    int NmodesPout = para.ITHACAdict->lookupOrDefault<int>("NmodesPout", 15);
    int NmodesUproj = para.ITHACAdict->lookupOrDefault<int>("NmodesUproj", 10);
    int NmodesPproj = para.ITHACAdict->lookupOrDefault<int>("NmodesPproj", 10);
    // Read the par file where the parameters are stored
    example.mu = Eigen::VectorXd::LinSpaced(100, 0.01, 0.001);
    // Set the inlet boundaries patch 0 directions x and y
    example.inletIndex.resize(1, 2);
    example.inletIndex(0, 0) = 0;
    example.inletIndex(0, 1) = 0;
    // Perform the offline solve
    example.offlineSolve();
    //example.liftSolve();
    example.liftfield.append(ITHACAutilities::computeAverage(example.Ufield));
    ITHACAutilities::normalizeFields(example.liftfield);
    example.liftfield[0].rename("Ulift0");
    ITHACAstream::exportSolution(example.liftfield[0], "0", "./");
    // Homogenize the snapshots
    example.computeLift(example.Ufield, example.liftfield, example.Uomfield);
    // Perform POD on velocity and pressure and store the first 10 modes
    ITHACAPOD::getModes(example.Uomfield, example.Umodes, example.podex, 0, 0,
                        NmodesUout);
    ITHACAPOD::getModes(example.Pfield, example.Pmodes, example.podex, 0, 0,
                        NmodesPout);
    // Create the reduced object
    tutorial18red reduced(example);
    // Compute the offline part of the DEIM procedure
    reduced.PODDEIM(10, 20, 1);
    Info << reduced.DEIMmatrice->submeshListA.size() << endl;
    Info << example.phi.size() << endl;
    surfaceScalarField phi = reduced.DEIMmatrice->submeshListA[0].interpolate(example.phi);
    Info << phi << endl;

    exit(0);
    reduced.project(NmodesUproj, NmodesPproj);
    //reduced.project(NmodesUproj, NmodesPproj);
    PtrList<volVectorField> U_rec_list;
    PtrList<volScalarField> P_rec_list;
    Eigen::MatrixXd vel(1, 1);
    vel(0, 0) = 1.0;

    //Perform the online solutions
    for (label k = 0; k < (example.mu).size(); k++)
    {
        scalar mu_now = example.mu(k);
        example.restart();
        example.change_viscosity(mu_now);
        reduced.onlineViscosity = mu_now;
        reduced.setOnlineVelocity(vel);
        reduced.solveOnline_Simple(mu_now, NmodesUproj, NmodesPproj, 0,
                                   "./ITHACAoutput/Offline/");
    }

    exit(0);
}
