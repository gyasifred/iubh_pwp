from pwp.utils import Preprocessing, IdealFunction
import unittest
import pandas as pd
from pandas.testing import assert_frame_equal  # <-- for testing dataframes


prep = Preprocessing()  # instant of the preprocessing class
train = prep.import_dataset('train .csv')  # reads in train train.csv file
ideal = prep.import_dataset('ideal.csv')  # reads in the ideal.csv file
X_train, Y_train = Preprocessing.values(train)
X_ideal, Y_ideal = Preprocessing.values(ideal)
Y_ideal_col = Y_ideal.iloc[:].values


class DFTests(unittest.TestCase):
    """Test if the CSV files imported was the one as expected."""

    def setUp(self):
        test_file_name = 'test.csv'
        try:
            data = Preprocessing().import_dataset(test_file_name)
        except IOError:
            print(f'{test_file_name} does not exist')
        self.fixture = data

    def test_dataFrame_constructedAsExpected(self):
        foo = pd.read_csv("test.csv")
        assert_frame_equal(self.fixture, foo)


class TestIdealFunction(unittest.TestCase):
    """The class will test to ensure the class Idealfunction can locate the right 
    ideal function for a giving training data.
    """

    def testing_classinstance(self):
        """Test if an object instatiated is an instant of a class."""
        result = IdealFunction()
        message = "object not instance of result"
        self.assertIsInstance(result, Preprocessing, message)

    def test_maxdev(self):
        """Test if the maximum deviation between two numpy array is as expected."""
        result = IdealFunction().max_deviation(Y_train['y1'], Y_ideal["y4"])
        self.assertEqual(result, 0.4974319999999999)

    def testlocateideafun(self):
        """Test if the method is able to locate the right ideal function for a given training data."""
        result = IdealFunction().locate_ideal_fun(
            Y_train['y1'], Y_ideal_col, Y_ideal)
        self.assertEqual(result, 'y4')


unittest.main()
