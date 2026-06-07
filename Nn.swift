// NeuralNetwork.swift
// Swift 4.1.3 for Windows - no external imports

// MARK: - Random Number Generator (LCG)

var lcgState: UInt64 = 123456789

func lcgNext() -> Double {
    lcgState = lcgState &* 6364136223846793005 &+ 1442695040888963407
    let value = Double(lcgState >> 33) / Double(0x7FFFFFFF)
    return value * 2.0 - 1.0
}

func randomWeight() -> Double {
    return lcgNext()
}

// MARK: - Activation Functions

func sigmoid(_ x: Double) -> Double {
    return 1.0 / (1.0 + _exp(-x))
}

func sigmoidDerivative(_ y: Double) -> Double {
    return y * (1.0 - y)
}

// MARK: - Neural Network

class NeuralNetwork {

    var inputSize: Int
    var hiddenSize: Int
    var outputSize: Int

    var w1: [[Double]]
    var w2: [[Double]]

    var learningRate: Double

    init(inputSize: Int,
         hiddenSize: Int,
         outputSize: Int,
         learningRate: Double) {

        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.learningRate = learningRate

        w1 = Array(
            repeating: Array(repeating: 0.0, count: hiddenSize),
            count: inputSize
        )

        w2 = Array(
            repeating: Array(repeating: 0.0, count: outputSize),
            count: hiddenSize
        )

        initializeWeights()
    }

    func initializeWeights() {

        for i in 0..<inputSize {
            for h in 0..<hiddenSize {
                w1[i][h] = randomWeight()
            }
        }

        for h in 0..<hiddenSize {
            for o in 0..<outputSize {
                w2[h][o] = randomWeight()
            }
        }
    }

    func predict(_ input: [Double]) -> Double {

        var hidden = Array(repeating: 0.0, count: hiddenSize)

        for h in 0..<hiddenSize {

            var sum = 0.0

            for i in 0..<inputSize {
                sum += input[i] * w1[i][h]
            }

            hidden[h] = sigmoid(sum)
        }

        var output = 0.0

        for h in 0..<hiddenSize {
            output += hidden[h] * w2[h][0]
        }

        return sigmoid(output)
    }

    func train(input: [Double], target: Double) {

        // Forward pass

        var hidden = Array(repeating: 0.0, count: hiddenSize)

        for h in 0..<hiddenSize {

            var sum = 0.0

            for i in 0..<inputSize {
                sum += input[i] * w1[i][h]
            }

            hidden[h] = sigmoid(sum)
        }

        var output = 0.0

        for h in 0..<hiddenSize {
            output += hidden[h] * w2[h][0]
        }

        output = sigmoid(output)

        // Error

        let outputError = target - output
        let outputDelta = outputError * sigmoidDerivative(output)

        // Hidden layer update

        for h in 0..<hiddenSize {

            let hiddenError = outputDelta * w2[h][0]
            let hiddenDelta = hiddenError * sigmoidDerivative(hidden[h])

            // Update output weights

            w2[h][0] += learningRate * outputDelta * hidden[h]

            // Update hidden weights

            for i in 0..<inputSize {
                w1[i][h] += learningRate * hiddenDelta * input[i]
            }
        }
    }
}

// MARK: - Utility

func roundTo4(_ value: Double) -> Double {
    return (value * 10000.0).rounded() / 10000.0
}

// MARK: - XOR Dataset

let trainingData: [([Double], Double)] = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 1.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 0.0)
]

// MARK: - Create Network

let network = NeuralNetwork(
    inputSize: 2,
    hiddenSize: 4,
    outputSize: 1,
    learningRate: 0.5
)

print("Training started...\n")

// MARK: - Training Loop

for epoch in 0..<10000 {

    for sample in trainingData {

        let input = sample.0
        let target = sample.1

        network.train(
            input: input,
            target: target
        )
    }

    if epoch % 1000 == 0 {

        var totalError = 0.0

        for sample in trainingData {

            let prediction = network.predict(sample.0)
            totalError += abs(sample.1 - prediction)
        }

        print(
            "Epoch \(epoch) - Error: \(roundTo4(totalError))"
        )
    }
}

// MARK: - Results

print("\nTraining complete.\n")
print("Predictions:\n")

for sample in trainingData {

    let prediction = network.predict(sample.0)

    print(
        "\(sample.0) -> \(roundTo4(prediction))"
    )
}
