_cv = autoPtr<volScalarField>
      (
          new volScalarField
          (
              IOobject
              (
                  "cv",
                  _runTime().timeName(),
                  mesh,
                  IOobject::NO_READ,
                  IOobject::AUTO_WRITE
              ),
              mesh,
              dimensionedScalar("zero", dimVolume, 0.0)
          )
      );
