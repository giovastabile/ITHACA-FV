IOdictionary thermocouplesPosition
(
    IOobject
    (
        "thermocouplesDict",
        runTime.constant(),
        mesh,
        IOobject::MUST_READ,
        IOobject::NO_WRITE
    )
);

List<vector> TCpos(thermocouplesPosition.lookup("positions"));
thermocouplesNum = TCpos.size();
thermocouplesPosition.lookup("timeSamplesT0") >> timeSamplesT0;
thermocouplesPosition.lookup("timeSamplesDeltaT") >> timeSamplesDeltaT;
timeSamplesNum = (runTime.endTime().value() - timeSamplesT0) / timeSamplesDeltaT + 1;
Info << "First sampling time = " << timeSamplesT0 << endl;
Info << "Delta time between samples = " << timeSamplesDeltaT << endl;
Info << "Number of samplings = " << timeSamplesNum << endl;
