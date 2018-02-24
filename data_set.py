from collections import namedtuple

Data_set = namedtuple('Data_set',
                      'name classes_num length train_size test_size')

Adiac = Data_set(
    name='Adiac', classes_num=37, train_size=390, test_size=391, length=176)
Beef = Data_set(
    name='Beef', classes_num=5, train_size=30, test_size=30, length=470)
CBF = Data_set(
    name='CBF', classes_num=3, train_size=30, test_size=900, length=128)

data_set_dict = {'Adiac': Adiac, 'Beef': Beef, 'CBF': CBF}
