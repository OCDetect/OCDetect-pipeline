from enum import Enum


class Sensor(Enum):
    ACCELEROMETER = "acc"
    GYROSCOPE = "gyro"


class IgnoreReason(Enum):
    DontIgnore = 0
    InitialHandWash = 1
    NoMovement = 2
    TooEarlyInRecording = 3
    RepetitionSame = 4
    RepetitionCompToRoutine = 5
    RepetitionRoutineToComp = 6

