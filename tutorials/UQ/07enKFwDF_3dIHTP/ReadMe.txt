For this simulation: 
    
    Probe, temperature= (1.7, 0.02, 0.24)
    Probe, heat flux= (1.7, 0.0, 0.24)
    Number of seeds (Samples) = 120      # Please see Nsamples in the 06enKFwDF_3dIHTP.C to adjust

    Adding some codes to export the ensemble of heat flux weights at each time step in the forecast step, please see the void stateProjection() inside 06enKFwDF_3dIHTP.C) and plot PDF of this ensemble weights via PDF.py 
    Adding some codes to export the Posterior ensemble of heat flux weights (after the analysis stage), please see the void Fang2017filter_wDF::run inside Fang2017filter_wDF.C) and plot PDF of this ensemble weights via PDFpost1.py and PDFpost.py

    Adding some codes to export trueBC (True heat flux) to make comparison, please see void set_gTrue inside 06enKFwDF_3dIHTP.H

    Adding some codes to export heatFluxSpaceBasis in order to plot the reconstructed heat flux, please see void sequentialIHTP::setSpaceBasis inside sequentialIHTP.C

    For the covariance of the prior weights, we can take, for example, 20 percent of the weight prior mean(parameterPriorMean). parameterPriorMean has some zero elements.
       Therefore, for all the zero diagonal elements of parameterPriorCov, I added a smal number.
       Exporting the covariance of the prior weights and wight prior mean in order to plot the prior PDF for basis function weights.
       Please see 06enKFwDF_3dIHTP.C




PDFpost1.py

    By changing the Weight inside the code, you can plot the prior and Posterior PDF of each weight over the observation TimeInterval
    Plotting posterior mean variation over observation time for the specified ensemble weight number. 
    Plotting posterior standard deviation variation over observation time for the specified ensemble weight number. 

plots.py
    
    Plotting true and reconstructed temperature for specified probe with confidence interval
    Plotting true and reconstructed heat flux for specified probe on the hotSide BC 
    Plotting true and reconstructed heat flux on the hotSide BC 
    Plotting true and reconstructed heat flux on the hotSide BC with confidence interval

PDFpost.py

    By changing only "Weight" and "TimeInterval" variables inside the code, you can plot the corresponding bell curve for the specified weight at specified observation time
