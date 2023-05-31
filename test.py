from os import path

from matplotlib import pyplot

from utils import load_dataset

# ----------------
#      CONFIG
# ----------------
CSV_PATH = "datasets/ice-cream.csv"
X_COLUMN = "temp"
Y_COLUMN = "sold"


# -----------------
#      TESTING
# -----------------

# check that the model was trained
if not path.isfile("model.txt"):
    print("ERROR: No save-data found. Please train the model first by running \"main.py\".")
    exit(0)

# get the parameters
with open("model.txt", "r") as f:
    a, b = f.read().split(" ")
    a, b = float(a), float(b)

# load the dataset
with open(CSV_PATH, "r") as f:
    dataset = load_dataset(f, X_COLUMN, Y_COLUMN)

# separate the x-values from the y-values
x_values = [point[0] for point in dataset]
y_values = [point[1] for point in dataset]

# calculate R-squared
dataset_mean = sum(y_values) / len(y_values)
variance = sum([pow(dataset_mean - y, 2) for y in y_values]) / len(y_values)
mse = sum([pow(a * x + b - y, 2) for x, y in zip(x_values, y_values)]) / len(y_values)
r_squared = (variance - mse) / variance

print(f"R²: {r_squared}")
print(f"R² as a percentage: {r_squared * 100}%")
input("Press ENTER to continue")

# figure out the minimum and maximum input value
minimum_x = min(x_values)
maximum_x = max(x_values)

# get the prediction for the min and max input value
y1, y2 = a * minimum_x + b, a * maximum_x + b

# plot the line
pyplot.plot([minimum_x, maximum_x], [y1, y2], color="red")

# plot the training data
pyplot.scatter(x_values, y_values)

pyplot.show()

# let the user predict values using the trained model
while True:
    x = float(input("AI's input: "))
    print(f"Prediction: {a * x + b}")
