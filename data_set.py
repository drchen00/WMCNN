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
}
