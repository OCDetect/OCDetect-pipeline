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

enum_labels = {
    "Certain": Label.Certain.value,
    "Begin uncertain": Label.BeginUncertain.value,
    "End uncertain": Label.EndUncertain.value,
    "Begin AND End uncertain": Label.BeginEndUncertain.value}

def string_to_parameter(parameter_string):
    for parameter in LabelMergeParameter:
        if parameter.name == parameter_string:
            return parameter
    return None
