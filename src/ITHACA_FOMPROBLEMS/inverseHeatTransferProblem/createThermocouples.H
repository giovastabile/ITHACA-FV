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
thermocouplesPosition.lookup("timeSamplesNumber") >> timeSamplesNum;
thermocouplesPosition.lookup("timeSamplesT0") >> timeSamplesT0;
thermocouplesPosition.lookup("timeSamplesDeltaT") >> timeSamplesDeltaT;
