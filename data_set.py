from collections import namedtuple

Data_set = namedtuple('Data_set',
                      'name classes_num length train_size test_size')

Adiac = Data_set(
    name='Adiac', classes_num=37, train_size=390, test_size=391, length=176)
Beef = Data_set(
    name='Beef', classes_num=5, train_size=30, test_size=30, length=470)
CBF = Data_set(
    name='CBF', classes_num=3, train_size=30, test_size=900, length=128)
ChlorineConcentration = Data_set(
    name='ChlorineConcentration',
    classes_num=3,
    train_size=467,
    test_size=3840,
    length=166)
SyntheticControl = Data_set(
    name='SyntheticControl',
    classes_num=6,
    train_size=300,
    test_size=300,
    length=60)
CinC_ECG_torso = Data_set(
    name='CinC_ECG_torso',
    classes_num=4,
    train_size=40,
    test_size=1380,
    length=1639)

Coffee = Data_set(
    name='Coffee', classes_num=2, train_size=28, test_size=28, length=286)

Cricket_X = Data_set(
    name='Cricket_X',
    classes_num=12,
    train_size=390,
    test_size=390,
    length=300)

Cricket_Y = Data_set(
    name='Cricket_X',
    classes_num=12,
    train_size=390,
    test_size=390,
    length=300)

Cricket_Z = Data_set(
    name='Cricket_X',
    classes_num=12,
    train_size=390,
    test_size=390,
    length=300)

DiatomSizeReduction = Data_set(
    name='CDiatomSizeReduction',
    classes_num=4,
    train_size=16,
    test_size=306,
    length=345)

ECGFiveDays = Data_set(
    name='ECGFiveDays',
    classes_num=2,
    train_size=23,
    test_size=861,
    length=136)

FaceAll = Data_set(
    name='FaceAll', classes_num=14, train_size=560, test_size=1690, length=131)

FaceFour = Data_set(
    name='FaceFour', classes_num=4, train_size=24, test_size=88, length=350)

FacesUCR = Data_set(
    name='FacesUCR',
    classes_num=14,
    train_size=200,
    test_size=2050,
    length=131)

fiftywords = Data_set(
    name='fiftywords',
    classes_num=50,
    train_size=450,
    test_size=455,
    length=270)

FISH = Data_set(
    name='FISH', classes_num=7, train_size=175, test_size=175, length=463)

Haptics = Data_set(
    name='Haptics', classes_num=5, train_size=155, test_size=308, length=1092)

InlineSkate = Data_set(
    name='InlineSkate',
    classes_num=7,
    train_size=100,
    test_size=550,
    length=1882)

Lighting7 = Data_set(
    name='Lighting7', classes_num=7, train_size=70, test_size=73, length=319)

data_set_dict = {
    'Adiac': Adiac,
    'Beef': Beef,
    'CBF': CBF,
    'synthetic_control': SyntheticControl,
    'ChlorineConcentration': ChlorineConcentration,
    'CinC_ECG_torso': CinC_ECG_torso,
    'Coffee': Coffee,
    'Cricket_X': Cricket_X,
    'Cricket_Y': Cricket_Y,
    'Cricket_Z': Cricket_Z,
    'DiatomSizeReduction': DiatomSizeReduction,
    'ECGFiveDays': ECGFiveDays,
    'FaceAll': FaceAll,
    'FaceFour': FaceFour,
    'FacesUCR': FacesUCR,
    '50words': fiftywords,
    'FISH': FISH,
    'Haptics': Haptics,
    'InlineSkate': InlineSkate,
    'Lighting7': Lighting7,
}
