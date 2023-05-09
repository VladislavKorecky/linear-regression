from random import random

from matplotlib import pyplot

from utils import load_dataset, step


# ----------------
#      CONFIG
# ----------------
NUMBER_OF_EPOCHS = 100000
LEARNING_RATE = 0.001

CSV_PATH = "datasets/ice-cream.csv"
X_COLUMN = "temp"
Y_COLUMN = "sold"


# -----------------
#       SETUP
# -----------------

# load the dataset
with open(CSV_PATH, "r") as f:
    dataset = load_dataset(f, X_COLUMN, Y_COLUMN)

# randomly initialize model's parameters
a = random()
b = random()


# ------------------
#      TRAINING
# ------------------

# list of all losses
loss_history = []

for epoch in range(NUMBER_OF_EPOCHS):
    # do one step/update of the model
    a, b, loss = step(dataset, a, b, LEARNING_RATE)

    print(loss)
    loss_history.append(loss)

# save the parameters
with open("model.txt", "w") as f:
    f.write(f"{a} {b}")

# graph the loss
pyplot.plot(loss_history)
pyplot.show()
