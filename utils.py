from csv import DictReader


def load_dataset(file, x_column: str, y_column: str) -> list:
    """
    Load a CSV dataset.

    Args:
        file: File object to load the data from.
        x_column (str): Name of the column with X (input) values.
        y_column (str): Name of the column with Y (output) values.

    Returns:
        list: Dataset with X and Y points.
    """

    dataset = []

    # open and read the CSV file
    csv_data = DictReader(file)

    # fill the dataset with datapoints
    # dataset format: [(x, y), (x, y), ...]
    for row in csv_data:
        x = row[x_column]
        y = row[y_column]

        # convert the x and y values to float, as they are string by default
        x = float(x)
        y = float(y)

        datapoint = (x, y)
        dataset.append(datapoint)

    return dataset


def step(dataset: list, a: float, b: float, learning_rate: float) -> tuple[float, float, float]:
    """
    Do one step/update in the training process.

    Args:
        dataset (list): Dataset to train on.
        a (float): Slope parameter of the model.
        b (float): Y-intercept/Bias of the model.
        learning_rate (float): Learning rate for the update formula.

    Returns:
        tuple[float, float, float]: Updated parameters "a" and "b" and the average loss/error.
    """

    # total gradient/derivative of parameters "a" and "b" for all training data
    total_gradient_a = 0
    total_gradient_b = 0

    total_error = 0

    for model_input, label in dataset:
        # calculate the AI's prediction using the formula: ax + b
        prediction = a * model_input + b

        # calculate the error/cost
        error = prediction - label
        squared_error = pow(error, 2)

        # add the squared error to the total error to keep track of the model's performance
        total_error += squared_error

        # calculate the partial derivative of "a" and "b"
        total_gradient_a += error * model_input
        total_gradient_b += error

    # calculate the average gradient/derivative
    average_gradient_a = total_gradient_a / len(dataset)
    average_gradient_b = total_gradient_b / len(dataset)

    # calculate the average error
    average_error = total_error / len(dataset)

    # update the model's parameters
    a -= average_gradient_a * learning_rate
    b -= average_gradient_b * learning_rate

    return a, b, average_error
