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


class HandWashingType(Enum):
    NoHandWash = 0
    Routine = 1
    Compulsive = 2
