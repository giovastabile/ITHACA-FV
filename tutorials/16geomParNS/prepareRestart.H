_nut = autoPtr<volScalarField> (new volScalarField(const_cast<volScalarField&>
                                (mesh.objectRegistry::lookupObject<volScalarField>("nut"))));
_nuTilda = autoPtr<volScalarField> (new volScalarField(
                                        const_cast<volScalarField&>
                                        (mesh.objectRegistry::lookupObject<volScalarField>("nuTilda"))));

volScalarField nut0(_nut());
volScalarField nuTilda0(_nuTilda());

_nut0 = autoPtr<volScalarField>
        (
            new volScalarField(nut0)
        );

_nuTilda0 = autoPtr<volScalarField>
            (
                new volScalarField(nuTilda0)
            );
