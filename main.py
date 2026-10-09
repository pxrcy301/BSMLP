import numpy as np
import torch
import torch.nn as nn

from blackScholesDataGenerator import computeOptionsPrices

np.random.seed(0)
torch.manual_seed(0)

# A | Generate data
# Feature vector [sigma, S/K, tau, r]
# Target: C/K
#
# The price is homogeneous in (S, K): C(S, K) = K * C(S/K, 1).
# So sample moneyness S/K directly and set K = 1; then C itself is C/K.
# (Sampling S and K independently on [1, 1000] gives S/K from 0.001 to 1000,
#  with very few samples near the money.)
nDataPointsTrain = 100_000
nDataPointsValidation = 10_000
nDataPointsTest = 10_000

sigmaRange = [0.00, 0.30]
moneynessRange = [0.5, 1.5]      # S/K
tauRange = [0.0, 1.2]
rRange = [0.00, 0.08]
K = 100.0

def generateData(nDataPoints, sigmaRange, moneynessRange, tauRange, rRange):
    sigma = np.random.uniform(sigmaRange[0], sigmaRange[1], nDataPoints)
    moneyness = np.random.uniform(moneynessRange[0], moneynessRange[1], nDataPoints)
    tau = np.random.uniform(tauRange[0], tauRange[1], nDataPoints)
    r = np.random.uniform(rRange[0], rRange[1], nDataPoints)

    C_over_K = computeOptionsPrices(sigma, moneyness*K, K, tau, r)/K   # K = 100

    features = np.column_stack((sigma, moneyness, tau, r))           # shape (N, 4)
    targets = C_over_K.reshape(-1, 1)                                # shape (N, 1)

    # NumPy defaults to float64; the model's weights are float32.
    return (torch.tensor(features, dtype=torch.float32),
            torch.tensor(targets, dtype=torch.float32))


TRAIN_FEATURES, TRAIN_TARGETS = generateData(nDataPointsTrain, sigmaRange, moneynessRange, tauRange, rRange)
VALIDATION_FEATURES, VALIDATION_TARGETS = generateData(nDataPointsValidation, sigmaRange, moneynessRange, tauRange, rRange)
TEST_FEATURES, TEST_TARGETS = generateData(nDataPointsTest, sigmaRange, moneynessRange, tauRange, rRange)

assert torch.isfinite(TRAIN_TARGETS).all()

# B | Define the model
INPUT_SIZE = TRAIN_FEATURES.shape[1]
HIDDEN_SIZES = [32, 32]
OUTPUT_SIZE = 1


class MLP(nn.Module):
    def __init__(self, inputSize, hiddenSizes, outputSize):
        super().__init__()
        layers = []
        prevDim = inputSize
        for hidDim in hiddenSizes:
            layers.append(nn.Linear(prevDim, hidDim))
            layers.append(nn.ReLU())
            prevDim = hidDim
        layers.append(nn.Linear(prevDim, outputSize))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


model = MLP(INPUT_SIZE, HIDDEN_SIZES, OUTPUT_SIZE)

# C | Train the model
batchSize = 256
nEpochs = 200
validationInterval = 10
learningRate = 1E-3

optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
lossFunction = nn.MSELoss()

nTrain = TRAIN_FEATURES.shape[0]

validationLosses = []
for nEpoch in range(nEpochs):
    model.train()

    # New random order of the training examples each epoch
    permutation = torch.randperm(nTrain)

    for start in range(0, nTrain, batchSize):
        batchIdx = permutation[start:start + batchSize]
        xBatch = TRAIN_FEATURES[batchIdx]     # shape (batchSize, 4)
        yBatch = TRAIN_TARGETS[batchIdx]      # shape (batchSize, 1)

        prediction = model(xBatch)
        loss = lossFunction(prediction, yBatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (nEpoch + 1) % validationInterval == 0:
        model.eval()
        with torch.no_grad():
            valPredictions = model(VALIDATION_FEATURES)
            valLoss = lossFunction(valPredictions, VALIDATION_TARGETS).item()
        validationLosses.append(valLoss)
        print(f"epoch {nEpoch + 1:4d} | val MSE {valLoss:.2e} | RMSE {valLoss**0.5:.2e}")

# D | Final evaluation on the test set (once, after all tuning)
model.eval()
with torch.no_grad():
    prediction = model(TEST_FEATURES)
    loss = lossFunction(prediction, TEST_TARGETS).item()
    print(f"Test RMSE: {loss**2*100:.2e}% of the strike price.")
# TODO 7

print(validationLosses)


# D | Plot the validation losses
import matplotlib.pyplot as plt

validationEpochs = range(validationInterval, nEpochs + 1, validationInterval)   # 10, 20, ..., 200

plt.figure(figsize=(7, 4))
plt.plot(validationEpochs, validationLosses, marker="o")
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Validation MSE")
plt.title("Validation loss")
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.show()
