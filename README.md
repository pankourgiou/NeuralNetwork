The program teaches a neural network to learn the XOR function — given two inputs (0 or 1), output 1 if they differ, 0 if they're the same:
0, 0 → 0
0, 1 → 1
1, 0 → 1
1, 1 → 0
XOR is a classic test for neural networks because it can't be solved with a single straight line — you need at least one hidden layer.

The network structure
It's a simple 3-layer network:

Input layer — 2 neurons (the two numbers)
Hidden layer — 4 neurons (does the internal reasoning)
Output layer — 1 neuron (the answer)

Neurons are connected by weights — numbers that get tuned during training. w1 holds the weights between input→hidden, w2 holds hidden→output.

Random weights
Before training, all weights are set to small random numbers between -1 and 1. The LCG is just a simple math formula that produces a stream of pseudo-random numbers without needing any library.

Sigmoid
Every neuron squashes its value through the sigmoid function, which maps any number into the range (0, 1). This is what lets the network output a probability-like value rather than an unbounded number.

Training (the core loop)
It runs 10,000 times. Each iteration does two things:

Forward pass — feed the input through the network, multiply by weights, apply sigmoid, and get a prediction.
Backpropagation — compare the prediction to the correct answer, calculate the error, then nudge every weight slightly in the direction that would have made the prediction more accurate. The learningRate (0.5) controls how big those nudges are.

Over thousands of repetitions, the weights gradually settle into values that give correct answers.

At the end
It prints the final predictions for all four XOR inputs. A well-trained run will show values close to 0, 1, 1, 0 — matching the expected outputs.
