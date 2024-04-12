from enum import Enum, IntEnum


class Sensor(Enum):
    ACCELEROMETER = "acc"
    GYROSCOPE = "gyro"


class IgnoreReason(IntEnum):
    DontIgnore = 0
    InitialHandWash = 1
    NoMovement = 2
    TooEarlyInRecording = 3
    RepetitionSame = 4
    RepetitionCompToRoutine = 5
    RepetitionRoutineToComp = 6
    BeforeHandWash = 7
    AfterHandWash = 8


class HandWashingType(Enum):
    NoHandWash = 0
    Routine = 1
    Compulsive = 2


class Label(Enum):
    NoLabel = 0
    Certain = 1
    BeginUncertain = 2
    EndUncertain = 3
    BeginEndUncertain = 4

class LabelMergeParameter(Enum):
    Intersection = 0
    Union = 1
    IgnoreUncertain = 2

label_mapping = {
    "Certain": Label.Certain,
    "Begin uncertain": Label.BeginUncertain,
    "End uncertain": Label.EndUncertain,
    "Begin AND End uncertain": Label.BeginEndUncertain}

parameter_mapping = {
    "Intersection": LabelMergeParameter.Intersection,
    "Union": LabelMergeParameter.Union,
    "IgnoreUncertain": LabelMergeParameter.IgnoreUncertain
}
