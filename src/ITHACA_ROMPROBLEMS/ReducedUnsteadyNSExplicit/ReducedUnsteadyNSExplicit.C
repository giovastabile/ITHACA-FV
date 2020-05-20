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
/// Source file of the reducedUnsteadyNSExplicit class


#include "ReducedUnsteadyNSExplicit.H"
#include <math.h> 
#include "fvCFD.H"

// * * * * * * * * * * * * * * * Constructors * * * * * * * * * * * * * * * * //

// Constructor initialization
reducedUnsteadyNSExplicit::reducedUnsteadyNSExplicit()
{
}

reducedUnsteadyNSExplicit::reducedUnsteadyNSExplicit(unsteadyNSExplicit& FOMproblem)
    :
    problem(&FOMproblem)
{
    N_BC = problem->inletIndex.rows();
    Nphi_u = problem->B_matrix.rows();
    Nphi_p = problem->K_matrix.cols();
    Nphi_phi = Nphi_u;

    for (label k = 0; k < problem->liftfield.size(); k++)
    {
        Umodes.append(problem->liftfield[k]);
    }

    for (label k = 0; k < problem->NUmodes; k++)
    {
        Umodes.append(problem->Umodes[k]);
    }

    for (label k = 0; k < problem->NPmodes; k++)
    {
        Pmodes.append(problem->Pmodes[k]);
    }

    if (problem->ExplicitMethod == "A")
    {
    	for (label k = 0; k < Nphi_phi; k++)
    	{
            Phimodes.append(problem->Phimodes[k]);
    	}
    }

    newton_object = newton_unsteadyNSExplicit(Nphi_p, Nphi_p,
                        FOMproblem);

}


// * * * * * * * * * * * * * Solve Functions  * * * * * * * * * * * //


void reducedUnsteadyNSExplicit::solveOnline(Eigen::MatrixXd vel,
                                        label startSnap)
{
    M_Assert(exportEvery >= dt,
             "The time step dt must be smaller than exportEvery.");
    M_Assert(storeEvery >= dt,
             "The time step dt must be smaller than storeEvery.");
    M_Assert(ITHACAutilities::isInteger(storeEvery / dt) == true,
             "The variable storeEvery must be an integer multiple of the time step dt.");
    M_Assert(ITHACAutilities::isInteger(exportEvery / dt) == true,
             "The variable exportEvery must be an integer multiple of the time step dt.");
    M_Assert(ITHACAutilities::isInteger(exportEvery / storeEvery) == true,
             "The variable exportEvery must be an integer multiple of the variable storeEvery.");
    int numberOfStores = round(storeEvery / dt);
    vel_now = vel;


if (problem->ExplicitMethod == "Ales")
{
    // Create and resize the solution vectors
    Eigen::VectorXd a_n = Eigen::VectorXd::Zero(Nphi_u);
    Eigen::VectorXd a_o = Eigen::VectorXd::Zero(Nphi_u);
    Eigen::MatrixXd b_n = Eigen::VectorXd::Zero(Nphi_p);
    Eigen::MatrixXd b_p = Eigen::VectorXd::Zero(Nphi_p);
    Eigen::MatrixXd b_q = Eigen::VectorXd::Zero(Nphi_p);
    Eigen::MatrixXd xx = Eigen::VectorXd::Zero(Nphi_p);

    // Counting variable
    int counter = 0;

    // Set the initial time
    time = tstart;

    // Determine number of time steps
    while (time < finalTime - 0.5 * dt)
    {
        time = time + dt;
        counter ++;
    }
    time = tstart;

    a_o = ITHACAutilities::get_coeffs(problem->Ufield[0],
                     Umodes);
    b_n = ITHACAutilities::get_coeffs(problem->Pfield[0],
                     Pmodes);


    if (problem->bcMethod == "lift")
    {
        for (label j = 0; j < N_BC; j++)
        {
            a_n(j) = vel_now(j, 0);
	    a_o(j) = vel_now(j, 0);
        }
    }

    // Set size of online solution
    online_solution.resize(counter+1);
    // Create vector to store temporal solution and save initial condition as first solution
    Eigen::MatrixXd tmp_sol(Nphi_u + Nphi_p + 1, 1);
    tmp_sol(0) = time;
    tmp_sol.col(0).segment(1, Nphi_u) = a_o;
    tmp_sol.col(0).tail(b_n.rows()) = b_n;
    online_solution[0] = tmp_sol;

    /// Pressure field
    volScalarField p = problem->_p;

    for (label i = 1; i < online_solution.size(); i++)
    {

    	time = time + dt;
    	std::cout << " ################## time =   " << time <<
                  " ##################" << std::endl;

	    Eigen::VectorXd presidualOld = Eigen::VectorXd::Zero(Nphi_p);
            Eigen::VectorXd presidual = Eigen::VectorXd::Zero(Nphi_p);
            scalar P_norm_res(1);

	    Eigen::VectorXd RHS  = Eigen::VectorXd::Zero(Nphi_p);
	    // Diffusion Term
    	    Eigen::VectorXd M5 = problem->DF_matrix * a_o *nu ;
	    // Convection Term
	    Eigen::MatrixXd cp(1, 1);
	    // Mom term
	    Eigen::MatrixXd M4 = problem->P_matrix * a_o;

    	    for (label l = 0; l < Nphi_p; l++)
    	    {
        	cp = a_o.transpose() * Eigen::SliceFromTensor(problem->Cf_tensor, 0,
                l) * a_o;
        	RHS(l) = (1/dt) * M4(l,0)- cp(0,0) + M5(l,0); //- cp(0,0)- M501(l,0)- problem->RedLinSysPConv[1]
    	    }
	    
	/*  List<Eigen::MatrixXd> RedLinSysP = problem->RedLinSysP;
	   RedLinSysP[1] =(1/dt)*RedLinSysP[1]+nu * problem->RedLinSysPDiff[1]+ problem->RedLinSysPConv[1];

 	   b_n = reducedProblem::solveLinearSysAxb(RedLinSysP, RHS, xx, presidual);*/
   
	b_n = ITHACAutilities::get_coeffs(problem->Pfield[i], Pmodes);

	dimensionedScalar dt_fake
        (
            "dt_fake",
            dimensionSet(0, 0, 1, 0, 0, 0, 0),
            scalar(1.0)
        );


	volVectorField U_rec("U_rec", Umodes[0]*0);
	for (label j = 0; j < Nphi_u; j++)
        {
            U_rec += Umodes[j] * a_o(j);
        }
    Vector<double> inl(1, 0, 0);
    Info << U_rec.boundaryField() << endl;
    // ITHACAutilities::changeBCtype(U_rec,"fixedValue",0);
    // ITHACAutilities::assignBC(U_rec,0,inl);
	//volVectorField U_rec("U_rec", problem->Ufield[i]);
	ITHACAstream::exportSolution(U_rec , name(i), "./ITHACAoutput/intersol");
	
	volVectorField U1aux("U1aux",problem->_U);
	U1aux =   dt*dt_fake * (-fvc::div(fvc::flux(U_rec),U_rec));
	volScalarField rhs = (1/dt)*(1/dt_fake)*fvc::div(U1aux);

	ITHACAstream::exportSolution(U1aux , name(i), "./ITHACAoutput/intersol");
	//ITHACAstream::exportSolution(rhs , name(i), "./ITHACAoutput/intersol");

	volVectorField U_rec2("U_rec2", problem->_Ub);
	for (label j = 0; j < Nphi_u; j++)
        {
            U_rec2 += Umodes[j] * a_o(j);
        }
    Vector<double> v0(0, 0, 0);
    ITHACAutilities::changeBCtype(U_rec2,"fixedValue",0);
    ITHACAutilities::assignBC(U_rec2,0,v0);
	//volVectorField U_rec2("U_rec", problem->Ufield[i]);
	//ITHACAutilities::changeBCtype( U_rec2,"fixedValue",1);
	//ITHACAutilities::assignBC( U_rec2,1,v0);
	ITHACAstream::exportSolution(U_rec2 , name(i), "./ITHACAoutput/intersol");

	volVectorField U0("U0",problem->_U());
	ITHACAutilities::assignBC(U0,0,inl);
    ITHACAutilities::assignIF(U0,v0);
	ITHACAutilities::changeBCtype( U0,"fixedValue",1);
	ITHACAutilities::assignBC( U0,1,v0);
	ITHACAstream::exportSolution(U0 , name(i), "./ITHACAoutput/intersol");

	volVectorField U_b("U_b", problem->_Ub);

	volVectorField U2aux("U2aux",problem->_U());
	U2aux =   dt*dt_fake * (
				-fvc::div(fvc::flux(U_rec2),U_rec2)
				-fvc::div(fvc::flux(U0),U0)
				-fvc::div(fvc::flux(U0),U_rec2)
				-fvc::div(fvc::flux(U_rec2),U0)
				//+fvc::div(fvc::flux(U_b),U_rec2)
				//+fvc::div(fvc::flux(U_rec2),U_b)
				//+fvc::div(fvc::flux(U_b),U0)
				//+fvc::div(fvc::flux(U0),U_b)
				);

	ITHACAstream::exportSolution(U2aux , name(i), "./ITHACAoutput/intersol");
  	volVectorField Uout("Uout", problem->_U());
	Uout =  U2aux-U1aux;
	ITHACAstream::exportSolution(Uout , name(i), "./ITHACAoutput/intersol");

	volScalarField rhs2 = (1/dt)*(1/dt_fake)*fvc::div(U2aux);
	//ITHACAstream::exportSolution(rhs2 , name(i), "./ITHACAoutput/intersol");
	
	b_p = ITHACAutilities::get_coeffs(rhs,Pmodes);
	//std::cout << "  b_p   " <<  b_p << std::endl;
 	//b_q = ITHACAutilities::get_coeffs(rhs2,Pmodes);
	//std::cout << "  b_q  " <<  b_q << std::endl;

        List<Eigen::MatrixXd> RedLinSysP = problem->RedLinSysP;
	RedLinSysP[1] =(1/dt)*RedLinSysP[1]+ nu * problem->RedLinSysPDiff[1]+ 
			(1/dt) * M4 	   +M5 + b_p;
 	
        b_n = reducedProblem::solveLinearSysAxb(RedLinSysP, xx, xx, presidual);


	    // Convective term
    	    Eigen::MatrixXd cc(1, 1);
    	    // Mom Term
    	    //Eigen::VectorXd M1 = problem->M_matrix * a_o;
    	    // Diff Term
    	    Eigen::VectorXd M2 = problem->B_matrix * a_o * nu ;
	    // Pressure Term
	    Eigen::VectorXd M3 = problem->K_matrix * b_n;

    	    for (label l = 0; l < Nphi_u; l++)
    	    {
        	cc = a_o.transpose() * Eigen::SliceFromTensor(problem->C_tensor, 0,
              		l) * a_o;
        	 a_n(l) =  (M2(l)  - cc(0,0))*dt  -problem->BC_matrix(l,0)*dt+ a_o(l)- dt*M3(l);
    	    }

	 if (problem->bcMethod == "lift")
    	{
            for (label j = 0; j < N_BC; j++)
            {
            	a_n(j) = vel_now(j, 0);
            }
    	}

    	tmp_sol(0) = time;
    	tmp_sol.col(0).segment(1, Nphi_u) = a_n;
    	tmp_sol.col(0).tail(b_n.rows()) = b_n;
    	online_solution[i] = tmp_sol;
 
    	a_o = a_n;
    }
}
else if (problem->ExplicitMethod == "A")
{

    // Create and resize the solution vectors
    Eigen::VectorXd a_n = Eigen::VectorXd::Zero(Nphi_u);
    Eigen::VectorXd a_o = Eigen::VectorXd::Zero(Nphi_u);
    Eigen::MatrixXd b = Eigen::VectorXd::Zero(Nphi_p);
    Eigen::MatrixXd xx = Eigen::VectorXd::Zero(Nphi_p);
    Eigen::VectorXd c_o = Eigen::VectorXd::Zero(Nphi_phi);
    Eigen::VectorXd c_n = Eigen::VectorXd::Zero(Nphi_phi);


    a_o = ITHACAutilities::get_coeffs(problem->Ufield[0],
                   Umodes);
    b = ITHACAutilities::get_coeffs(problem->Pfield[0],
                   Pmodes);
    c_o = ITHACAutilities::get_coeffs(problem->Phifield[0],
                   Phimodes);


     if (problem->bcMethod == "lift")
    	{
            for (label j = 0; j < N_BC; j++)
            {
            	a_n(j) = vel_now(j, 0);
	   	a_o(j) = vel_now(j, 0);
            }
    	}

    // Counting variable
    int counter = 0;

    // Set the initial time
    time = tstart;

    // Determine number of time steps
    while (time < finalTime - 0.5 * dt)
    {
        time = time + dt;
        counter ++;
    }
    time = tstart;

    // Set size of online solution
    online_solution.resize(counter+1);
    // Create vector to store temporal solution and save initial condition as first solution
    Eigen::MatrixXd tmp_sol(Nphi_u + Nphi_p + 1, 1);
    tmp_sol(0) = time;
    tmp_sol.col(0).segment(1, Nphi_u) = a_o;
    tmp_sol.col(0).tail(b.rows()) = b;
    online_solution[0] = tmp_sol;


    for (label i = 1; i < online_solution.size(); i++)
    {
	Eigen::VectorXd presidualOld = Eigen::VectorXd::Zero(Nphi_p);
        Eigen::VectorXd presidual = Eigen::VectorXd::Zero(Nphi_p);
        scalar P_norm_res(1);
        time = time + dt;
  	std::cout << " ################## time =   " << time <<
                  " ##################" << std::endl;

	Eigen::VectorXd RHS  = Eigen::VectorXd::Zero(Nphi_p);
	// Diffusion Term
    	Eigen::VectorXd M5 = problem->DF_matrix * a_o *nu ;
	// Convection Term
	Eigen::MatrixXd cp(1, 1);
	// Mom term
	Eigen::MatrixXd M4 = problem->P_matrix * a_o;

    	for (label l = 0; l < Nphi_p; l++)
    	{
        	cp = c_o.transpose() * Eigen::SliceFromTensor(problem->Cf_tensor, 0,
                l) * a_o;
        	RHS(l) = (1/dt) * M4(l,0) - cp(0,0)+ M5(l,0);

    	}
	List<Eigen::MatrixXd> RedLinSysP = problem->RedLinSysP;
	RedLinSysP[1] =(1/dt)*RedLinSysP[1]+ problem->RedLinSysPConv[1]+ nu * problem->RedLinSysPDiff[1];
	b = reducedProblem::solveLinearSysAxb(RedLinSysP, RHS, xx, presidual);
    	//b = reducedProblem::solveLinearSysAxb(problem->RedLinSysP, RHS, xx, presidual);
	// Convective term
    	Eigen::MatrixXd cc(1, 1);
    	// Diff Term
    	Eigen::VectorXd M2 = problem->B_matrix * a_o * nu ;
	// Pressure Term
	Eigen::VectorXd M3 = problem->K_matrix * b;

    	for (label l = 0; l < Nphi_u; l++)
    	{
        	cc = c_o.transpose() * Eigen::SliceFromTensor(problem->C_tensor, 0,
              		l) * a_o;
        	 a_n(l) = a_o(l) + (M2(l)  - cc(0,0)-problem->BC_matrix(l,0)- M3(l))*dt  ;
    	}


	/*surfaceScalarField Phi_rec("Phi_rec", problem->_phi);

        for (label j = 0; j < Nphi_p; j++)
        {
             Phi_rec += Phimodes[j]* c_o(j) ;
        }

	dimensionedScalar dt_fake
        (
            "dt_fake",
            dimensionSet(0, 0, 1, 0, 0, 0, 0),
            scalar(1.0)
        );

	dimensionedScalar nu_fake
        (
            "nu_fake",
            dimensionSet(0, 2, -1, 0, 0, 0, 0),
            scalar(1.0)
        );*/
	Eigen::VectorXd c_star = Eigen::VectorXd::Zero(Nphi_phi);
	Eigen::VectorXd c_star3 = Eigen::VectorXd::Zero(Nphi_phi);
	Eigen::VectorXd c_c = Eigen::VectorXd::Zero(Nphi_phi);
	surfaceScalarField phi_2("Phi_2", problem->_phi);
	surfaceScalarField phi_3("Phi_3", problem->_phi);
	volVectorField U_rec("U_rec", problem->_U);

     /*   for (label j = 0; j < Nphi_u; j++)
        {
             U_rec += Umodes[j] * a_o(j);
        }

	surfaceScalarField phi_c("Phi_c", problem->_phi);
	phi_c =	dt_fake*fvc::flux(fvc::div(Phi_rec,U_rec))	;		
		c_c = ITHACAutilities::get_coeffs(phi_c,
                   Phimodes);

	c_star = ITHACAutilities::get_coeffs(phi_2,
                   Phimodes);*/

		
	//std::cout << " c_c=   " << c_c<< std::endl;
	Eigen::MatrixXd M6 = problem->I_matrix * a_o;
	Eigen::MatrixXd M7 = problem->ID_matrix * a_o*nu;
	Eigen::MatrixXd M8 = problem->Kf_matrix*b.col(0);
	Eigen::MatrixXd M11(Nphi_phi,Nphi_phi);
	Eigen::MatrixXd M12 = Eigen::VectorXd::Zero(Nphi_phi);
	Eigen::MatrixXd ci(Nphi_phi,1);
	//Eigen::MatrixXd M9 = problem->Mf_matrix.colPivHouseholderQr().solve(M6 + dt*(M7-M8));
	//Eigen::MatrixXd M9 = problem->Mf_matrix.fullPivLu().solve(M6 + dt*(M7-M8));
	Eigen::MatrixXd M9 = problem->Mf_matrix.colPivHouseholderQr().solve(M6+ dt*(-M8+M7+nu*problem->BC_matrix_PPE));
	for (label k = 0; k < Nphi_phi; k++)
    	{	
		M12 = dt*problem->Ci_matrix[k] * a_o*c_o(k) +M12;//
		
	}

	Eigen::MatrixXd M13 = problem->Mf_matrix.colPivHouseholderQr().solve(M12);
		c_n =  M9 -M13;//dt*c_c(k)+dt*nu*c_star(k)+dt*nu*c_star3(k)
	tmp_sol(0) = time;
    	tmp_sol.col(0).segment(1, Nphi_u) = a_n;
    	tmp_sol.col(0).tail(b.rows()) = b;
    	online_solution[i] = tmp_sol;

	 if (problem->bcMethod == "lift")
    	{
            for (label j = 0; j < N_BC; j++)
            {
            	a_n(j) = vel_now(j, 0);
            }
    	}

	a_o = a_n;
	c_o = c_n;


    	}
}



	ITHACAstream::exportMatrix(online_solution, "red_coeff", "eigen",
                               "./ITHACAoutput/red_coeff");

	int exportEveryIndex = round(exportEvery / dt);
	int outputIndex = round(finalTime / exportEvery)+1;	
	
}




void reducedUnsteadyNSExplicit::reconstruct(fileName folder)
{
    mkDir(folder);
    ITHACAutilities::createSymLink(folder);
    int counter = 0;
    int nextwrite = 0;
    int counter2 = 1;
    int exportEveryIndex = round(exportEvery / storeEvery);

    for (label i = 0; i < online_solution.size(); i++)
    {
        if (counter == nextwrite)
        {
            volVectorField U_rec("U_rec", Umodes[0] * 0);

            for (label j = 0; j < Nphi_u; j++)
            {
                U_rec += Umodes[j] * online_solution[i](j + 1, 0);
            }

            ITHACAstream::exportSolution(U_rec,  name(counter2), folder);

            volScalarField P_rec("P_rec", problem->Pmodes[0] * 0);

            for (label j = 0; j < Nphi_p; j++)
            {
                P_rec += Pmodes[j] * online_solution[i](j + Nphi_u + 1, 0);
            }

	   // volVectorField P_grad= fvc::grad(P_rec);
            ITHACAstream::exportSolution(P_rec, name(counter2), folder);
            nextwrite += exportEveryIndex;
            double timenow = online_solution[i](0, 0);
            std::ofstream of(folder + name(counter2) + "/" + name(timenow));
            counter2 ++;
            UREC.append(U_rec);
            PREC.append(P_rec);
        }

        counter++;
    }
}



	//if (problem->Method == "FE")
	//{

//}
	/*else if (problem->Method == "RK3")
	{
	std::cout << " ########## RK3   " <<
                  " ###########" << std::endl;

	Eigen::VectorXd presidualOld = Eigen::VectorXd::Zero(Nphi_p);
        Eigen::VectorXd presidual = Eigen::VectorXd::Zero(Nphi_p);
        scalar P_norm_res(1);

	Eigen::VectorXd a_2 = Eigen::VectorXd::Zero(Nphi_u);
	Eigen::VectorXd a_3 = Eigen::VectorXd::Zero(Nphi_u);
	Eigen::VectorXd RHS  = Eigen::VectorXd::Zero(Nphi_p);

	// Diffusion Term
    	Eigen::VectorXd M5_1 = problem->DF_matrix * a_o *nu ;
	// Convection Term
	Eigen::MatrixXd cp_1(1, 1);
	// Mom term
	Eigen::MatrixXd M4 = problem->P_matrix * a_o;

    	for (label l = 0; l < Nphi_p; l++)
    	{
        	cp_1 = a_o.transpose() * Eigen::SliceFromTensor(problem->Cf_tensor, 0,
                l) * a_o;
        	RHS(l) = (1/(problem->c2*dt)) * (M4(l,0)  - (problem->a21 *dt) * (cp_1(0,0)- M5_1(l,0)));

    	}
	List<Eigen::MatrixXd> RedLinSysProm2 = problem->RedLinSysP;
	RedLinSysProm2[1] = (problem->a21/(problem->c2))*RedLinSysProm2[1];
	
    	b_n = reducedProblem::solveLinearSysAxb(RedLinSysProm2, RHS, xx, presidual);

	// Convective term
    	Eigen::MatrixXd cc_1(1, 1);
    	// Diff Term
    	Eigen::VectorXd M2_1 = problem->B_matrix * a_o * nu ;
	// Pressure Term
	Eigen::VectorXd M3_2 = problem->K_matrix * b_n;

    	for (label l = 0; l < Nphi_u; l++)
    	{
        	cc_1 = a_o.transpose() * Eigen::SliceFromTensor(problem->C_tensor, 0,
              		l) * a_o;
        	 a_2(l) = a_o(l)+  (M2_1(l)  - cc_1(0,0)-problem->BC_matrix(l,0))*(problem->a21 *dt) 
			 	- (problem->c2*dt)*M3_2(l);
    	}

	// Stage 3
        presidual = Eigen::VectorXd::Zero(Nphi_p);

    	Eigen::VectorXd M5_2 = problem->DF_matrix * a_2 *nu ;
	Eigen::MatrixXd cp_2(1, 1);

    	for (label l = 0; l < Nphi_p; l++)
    	{
	        cp_1 = a_o.transpose() * Eigen::SliceFromTensor(problem->Cf_tensor, 0,
                l) * a_o;
        	cp_2 = a_2.transpose() * Eigen::SliceFromTensor(problem->Cf_tensor, 0,
                l) * a_2;
        	RHS(l) = (1/(problem->c3*dt)) * (M4(l,0) 
			 - (problem->a31 *dt) * (cp_1(0,0)- M5_1(l,0))
			 - (problem->a32 *dt) * (cp_2(0,0)- M5_2(l,0)));

    	}
	List<Eigen::MatrixXd> RedLinSysProm3 = problem->RedLinSysP;
	RedLinSysProm3[1] = (1/(problem->c3))*RedLinSysProm3[1];
	
    	b_n = reducedProblem::solveLinearSysAxb(RedLinSysProm3, RHS, xx, presidual);


    	Eigen::MatrixXd cc_2(1, 1);
    	Eigen::VectorXd M2_2 = problem->B_matrix * a_2 * nu ;
	Eigen::VectorXd M3_3 = problem->K_matrix * b_n;

    	for (label l = 0; l < Nphi_u; l++)
    	{	cc_1 = a_o.transpose() * Eigen::SliceFromTensor(problem->C_tensor, 0,
              		l) * a_o;
        	cc_2 = a_2.transpose() * Eigen::SliceFromTensor(problem->C_tensor, 0,
              		l) * a_2;
        	 a_3(l) = a_o(l)+  (M2_1(l)  - cc_1(0,0)-problem->BC_matrix(l,0))*(problem->a31 *dt) 
				+  (M2_2(l)  - cc_2(0,0)-problem->BC_matrix(l,0))*(problem->a32 *dt) 
			 	- (problem->c3*dt)*M3_3(l);
    	}

	// Stage Final
        presidual = Eigen::VectorXd::Zero(Nphi_p);
    	Eigen::VectorXd M5_3 = problem->DF_matrix * a_3 *nu ;
	Eigen::MatrixXd cp_3(1, 1);

    	for (label l = 0; l < Nphi_p; l++)
    	{
	        cp_1 = a_o.transpose() * Eigen::SliceFromTensor(problem->Cf_tensor, 0,
                l) * a_o;
        	cp_2 = a_2.transpose() * Eigen::SliceFromTensor(problem->Cf_tensor, 0,
                l) * a_2;
        	cp_3 = a_3.transpose() * Eigen::SliceFromTensor(problem->Cf_tensor, 0,
                l) * a_3;
        	RHS(l) = (1/dt) * (M4(l,0)  
			- (problem->b1 *dt) * (cp_1(0,0)- M5_1(l,0))
			- (problem->b2 *dt) * (cp_2(0,0)- M5_2(l,0))
			- (problem->b3 *dt) * (cp_3(0,0)- M5_3(l,0)));

    	}

    	b_n = reducedProblem::solveLinearSysAxb(problem->RedLinSysP, RHS, xx, presidual);

    	Eigen::MatrixXd cc_3(1, 1);
    	Eigen::VectorXd M2_3 = problem->B_matrix * a_3 * nu ;
	Eigen::VectorXd M3 = problem->K_matrix * b_n;

    	for (label l = 0; l < Nphi_u; l++)
    	{	
	         cc_1 = a_o.transpose() * Eigen::SliceFromTensor(problem->C_tensor, 0,
              		l) * a_o;
        	 cc_2 = a_2.transpose() * Eigen::SliceFromTensor(problem->C_tensor, 0,
              		l) * a_2;
        	 cc_3 = a_3.transpose() * Eigen::SliceFromTensor(problem->C_tensor, 0,
              		l) * a_3;
        	 a_n(l) = a_o(l)+  (M2_1(l)  - cc_1(0,0)-problem->BC_matrix(l,0))*(problem->b1 *dt) 
				+  (M2_2(l)  - cc_2(0,0)-problem->BC_matrix(l,0))*(problem->b2 *dt) 
				+  (M2_3(l)  - cc_3(0,0)-problem->BC_matrix(l,0))*(problem->b3 *dt) 
			 	-  dt*M3(l);
    	}

	}*/

//************************************************************************* //
