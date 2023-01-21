# Programming with python Assignment IU
import pandas as pd
import numpy as np
from bokeh.plotting import figure, output_notebook, show
from sqlalchemy import create_engine


class Preprocessing:
    """Import the comma-separated values file with the pandas' library,
    extract the dependent(X) and independent(Y) values from the data frame.
    Args:
    param filename: path to the csv file
    """

    def __init__(self) -> None:
        pass

    def import_dataset(self, filename):
        """Import csv files with the pandas library."""

        try:
            return pd.read_csv(filename)  # read in csv file
        except IOError:
            return f'{filename} does not exist'

    @classmethod
    def values(cls, dataset):
        """Extract X and Y values from the data frame."""
        X = dataset.iloc[:, 0]  # X values(independent variables)
        Y = dataset.iloc[:, 1:]  # Y values (dependent variables)
        return X, Y


class Visualization(Preprocessing):
    """The class will use the plotting module bokeh for visualization.
    """

    def __init__(self) -> None:
        super().__init__()

    def plot_bokeh(self, X, Y, title, data):
        """Plot the X and Y values with the bokeh model.
        Bokeh is an interactive plotting tool.
        Hovertool tool of bokeh is used to visualize individual points in Html
        """
        TOOLTIPS = [
            ("index", "$index"),
            ("(x,y)", "($x, $y)")]
        p = figure(title=title, x_axis_label=X, y_axis_label=Y,
                   width=800, height=200, tooltips=TOOLTIPS)
        p.circle(X, Y, size=5, source=data)
        output_notebook()
        show(p)


class IdealFunction(Preprocessing):
    """
    1.Calculate the minimum Least Square between the training data and the ideal functions.
    2.Assigned an ideal function to each training data
    3.Calculate the deviation between the train data and its assigned ideal function
    4.Extrapolate the maximum deviation and map each test point to the ideal function.
    """

    def __init__(self) -> None:
        super().__init__()

    def least_square(self, y_pred, y_ideal):
        """Calculate the least square by summing all the square deviations."""
        if y_pred.ndim > 1:
            y_pred = y_pred.flatten()  # convert to 1 dimension
        else:
            y_pred = y_pred
        lse = list()  # holds all the least square error
        for col in range(y_ideal.shape[1]):  # illiterate over all the columns
            # Calculte the least squre errors
            ls = sum(np.subtract(y_pred, y_ideal[:, col])**2)
            lse.append(ls)
        return (lse)

    def locate_ideal_fun(self, X, Y, dataset):
        """Calculate least squares,
        determine the minimum least squares and locate it index,
        finally use the index to locate the column of the minimum least squares
        """
        l_se = self.least_square(X, Y)  # calculate all least squares
        y = min(l_se)  # look for the minimum least square
        w = l_se.index(y)  # locate the index of the minimum least square
        # identified the column with the minimum least square
        x = dataset.columns[w]
        return x

    @classmethod
    def max_deviation(cls, data1, data2):
        """Determine the maximum deviation between the train data,
        and its corresponding ideal function
        """
        if data1.ndim > 1:
            data1 = data1.flatten()  # convert to 1 dimension
        else:
            data1 = data1
        # calculate the deviation and locate the maximum deviation
        return np.max(abs(np.subtract(data1, data2)))

    @classmethod
    def deviations(cls, xtest, xideal, ideal_table, y_test):
        """
        This method executes the following purpose:
        1. Look for the ideal coordinates in the ideal functions corresponding to the test points
        2. Return the deviation between every test point and the ideal coordinates
        """
        import numpy as np

        id = []  # Holds all ideal coordinates
        dev = []  # holds all the deviations between the test points and the ideal points
        x_ideal = list(xideal)  # convert the Data Frame to list
        for i in xtest:  # illiterate over the test points
            for j in x_ideal:  # illiterate over the list of ideal functions
                if i == j:  # compares the test points to the elements of the ideal function
                    # if the test point is equal to the ideal point, locate the index of the ideal points
                    ind = x_ideal.index(j)
            # add the ideal point corresponding to the particular tst point
            id.append(ideal_table[ind:ind+1])

        # takes the list of the ideal points corresponding to the test point and transform them to NumPy arrays
        t_id = np.array(id).transpose()
        t_id = t_id[1:]  # drop the first column of the NumPy array
        if y_test.ndim > 1:  # check the dimension of the NumPy array
            # convert to 1 dimension if dimension is not 1d.
            y_test = y_test.flatten()
        for k in t_id:  # iterate over columns of a NumPy array
            if k.ndim > 1:  # check the dimension of the array if it > 1d
                k = k.flatten()  # convert to 1d if the dimension is greater than 1
                # calculate the absolute difference on each row between the test data and the ideal function
                y = abs(np.subtract(y_test, k))
                dev.append(y)  # append the absolute difference to a list
            else:
                y = abs(np.subtract(y_test, k))
                dev.append(y)
        return dev, id   # returns the deviations and the ideal points

    def dev_mapper(self, dev, max_dev):
        """
       1. Illiterate over the element in the deviation.
       2. Compare each element to the maximum deviation obtain between a train data and its ideal
        function times the square root of 2.
       3.If the deviation is less than the resultant, that deviation is added to a list. IF not, none is added
        to the list.
        """
        d = []  # Holds the list of the deviations that satisfied the requirement
        for i in dev:  # illiterate over the deviations
            if i < (max_dev * np.sqrt(2)):  # compare the deviation with the conditional statement
                # append the deviation if the condition is satisfied
                d.append(i)
            else:
                d.append(None)  # append none if the condition is not satisfied
        return d

    def func_map(self, min_dev, dev, fun_no):
        """
        1.This attribute has three inputs the minimum deviation calculated from every row of the
        deviations table,
        a column of the deviation table, and an ideal function number.
        2.The function compares the items of the minimum deviations to each element of the column.
        3. If the minimum deviation is in the column of the deviations,
        4. It will return the ideal function number from which that deviation column was calculated.
        """
        func = []  # Holds the ideal functions
        for i in min_dev:  # illiterate over the element in the minimum deviations.
            if i in dev:  # search for each element of the minimum deviation in the deviations.
                # append the ideal function number if the minimum deviation id  present in a particular deviation
                func.append(fun_no)

            else:
                func.append(None)  # append none if vice versa
        return func


class SqliteSqlalchemy(Preprocessing):
    """The class will the use sqlalchemy engine to import CSV tables into SQLite database."""

    def __init__(self) -> None:
        super().__init__()

    def import_csv_table(self, filename, title, databasename):
        """Import CSV table, create database, save the table into the assignment database."""
        df = self.import_dataset(filename)  # import the csv file
        # Create the assignment database with the SQLite engine
        engine = create_engine(
            'sqlite:///{}.db'.format(databasename), echo=True)
        sqlite_connection = engine.connect()  # set up the connection
        sqlite_table = title
        # append the csv table into the assignment database
        df.to_sql(sqlite_table, sqlite_connection, if_exists='append')
        sqlite_connection.close()  # close the connection

