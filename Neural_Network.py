import numpy as np
import pandas as pd
import tkinter as tk

MODEL_FILE = "digit_model.npz"

# Use: Loads the CSV file and prepares X and Y for training.
def load_data(file_path):
    data = pd.read_csv(file_path).to_numpy()
    np.random.shuffle(data)

    # First column is the digit label, remaining columns are pixel values.
    Y = data[:, 0]
    X = data[:, 1:].T

    # Pixel values are from 0 to 255. Dividing by 255 keeps values small,
    # which helps the neural network train faster and more safely.
    X = X.astype(np.float32) / 255.0

    return X, Y


# Use: Creates the starting weights and biases for a 2-hidden-layer network.
def init_params(input_size, hidden1_size, hidden2_size, output_size):
    # He initialization works well with ReLU because it keeps activations stable.
    W1 = (
        np.random.randn(hidden1_size, input_size).astype(np.float32)
        * np.sqrt(2 / input_size)
    )
    b1 = np.zeros((hidden1_size, 1), dtype=np.float32)
    W2 = (
        np.random.randn(hidden2_size, hidden1_size).astype(np.float32)
        * np.sqrt(2 / hidden1_size)
    )
    b2 = np.zeros((hidden2_size, 1), dtype=np.float32)
    W3 = (
        np.random.randn(output_size, hidden2_size).astype(np.float32)
        * np.sqrt(2 / hidden2_size)
    )
    b3 = np.zeros((output_size, 1), dtype=np.float32)

    return W1, b1, W2, b2, W3, b3


# Use: ReLU activation keeps positive values and turns negative values into 0.
def relu(Z):
    return np.maximum(Z, 0)


# Use: Finds the slope of ReLU during backward propagation.
def relu_derivative(Z):
    return Z > 0


# Use: Converts final layer scores into probabilities for each digit class.
def softmax(Z):
    # Subtracting the max improves numerical stability.
    shifted_Z = Z - np.max(Z, axis=0, keepdims=True)
    exp_Z = np.exp(shifted_Z)
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)


# Use: Converts labels like 7 into vectors like [0,0,0,0,0,0,0,1,0,0].
def one_hot(Y, num_classes):
    one_hot_Y = np.zeros((num_classes, Y.size), dtype=np.float32)
    one_hot_Y[Y, np.arange(Y.size)] = 1
    return one_hot_Y


# Use: Runs input data through both hidden layers to produce predictions.
def forward_prop(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1.dot(X) + b1
    A1 = relu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = relu(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)

    return Z1, A1, Z2, A2, Z3, A3


# Use: Calculates gradients for all 3 weight layers and 3 bias layers.
def backward_prop(Z1, A1, Z2, A2, A3, W2, W3, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y, A3.shape[0])

    dZ3 = A3 - one_hot_Y
    dW3 = (1 / m) * dZ3.dot(A2.T)
    db3 = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)

    dZ2 = W3.T.dot(dZ3) * relu_derivative(Z2)
    dW2 = (1 / m) * dZ2.dot(A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = W2.T.dot(dZ2) * relu_derivative(Z1)
    dW1 = (1 / m) * dZ1.dot(X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2, dW3, db3


# Use: Updates weights and biases using the gradients and learning rate.
def update_params(
    W1,
    b1,
    W2,
    b2,
    W3,
    b3,
    dW1,
    db1,
    dW2,
    db2,
    dW3,
    db3,
    learning_rate,
):
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    W3 = W3 - learning_rate * dW3
    b3 = b3 - learning_rate * db3

    return W1, b1, W2, b2, W3, b3


# Use: Picks the digit with the highest probability from the output layer.
def get_predictions(A2):
    return np.argmax(A2, axis=0)


# Use: Measures how many predictions match the correct labels.
def get_accuracy(predictions, Y):
    return np.mean(predictions == Y)


# Use: Calculates cross-entropy loss so we can track training progress.
def get_loss(A2, Y):
    m = Y.size
    correct_class_probs = A2[Y, np.arange(m)]
    return -np.mean(np.log(correct_class_probs + 1e-8))


# Use: Chooses a small random part of the dataset for one training step.
def get_mini_batch(X, Y, batch_size):
    m = Y.size
    indices = np.random.choice(m, batch_size, replace=False)
    return X[:, indices], Y[indices]


# Use: Trains the 2-hidden-layer network by repeating forward/backward/update.
def gradient_descent(
    X,
    Y,
    iterations,
    learning_rate,
    hidden1_size,
    hidden2_size,
    batch_size,
):
    input_size = X.shape[0]
    output_size = 10
    W1, b1, W2, b2, W3, b3 = init_params(
        input_size, hidden1_size, hidden2_size, output_size
    )

    for i in range(iterations + 1):
        X_batch, Y_batch = get_mini_batch(X, Y, batch_size)

        Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X_batch)
        dW1, db1, dW2, db2, dW3, db3 = backward_prop(
            Z1, A1, Z2, A2, A3, W2, W3, X_batch, Y_batch
        )
        W1, b1, W2, b2, W3, b3 = update_params(
            W1,
            b1,
            W2,
            b2,
            W3,
            b3,
            dW1,
            db1,
            dW2,
            db2,
            dW3,
            db3,
            learning_rate,
        )

        if i % 500 == 0:
            predictions = get_predictions(A3)
            accuracy = get_accuracy(predictions, Y_batch)
            loss = get_loss(A3, Y_batch)
            print(f"Iteration: {i}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    return W1, b1, W2, b2, W3, b3


# Use: Runs the trained network on data and returns predicted digit labels.
def make_predictions(X, W1, b1, W2, b2, W3, b3):
    _, _, _, _, _, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
    return get_predictions(A3)


# Use: Returns both the predicted digit and the probability for each digit.
def predict_digit(X, W1, b1, W2, b2, W3, b3):
    _, _, _, _, _, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
    prediction = get_predictions(A3)[0]
    probabilities = A3[:, 0]
    return prediction, probabilities


# Use: Resizes a 2D image using bilinear interpolation.
def resize_image(image, new_height, new_width):
    old_height, old_width = image.shape

    if old_height == new_height and old_width == new_width:
        return image.copy()

    resized = np.zeros((new_height, new_width), dtype=np.float32)
    row_scale = old_height / new_height
    col_scale = old_width / new_width

    for row in range(new_height):
        for col in range(new_width):
            old_row = (row + 0.5) * row_scale - 0.5
            old_col = (col + 0.5) * col_scale - 0.5

            row0 = int(np.floor(old_row))
            col0 = int(np.floor(old_col))
            row1 = min(row0 + 1, old_height - 1)
            col1 = min(col0 + 1, old_width - 1)
            row0 = max(row0, 0)
            col0 = max(col0, 0)

            row_weight = old_row - row0
            col_weight = old_col - col0

            top = (1 - col_weight) * image[row0, col0] + col_weight * image[row0, col1]
            bottom = (1 - col_weight) * image[row1, col0] + col_weight * image[row1, col1]
            resized[row, col] = (1 - row_weight) * top + row_weight * bottom

    return resized


# Use: Shifts a 28x28 image without wrapping pixels around the edges.
def shift_image(image, row_shift, col_shift):
    shifted = np.zeros_like(image)

    source_row_start = max(0, -row_shift)
    source_row_end = min(image.shape[0], image.shape[0] - row_shift)
    source_col_start = max(0, -col_shift)
    source_col_end = min(image.shape[1], image.shape[1] - col_shift)

    target_row_start = max(0, row_shift)
    target_row_end = target_row_start + (source_row_end - source_row_start)
    target_col_start = max(0, col_shift)
    target_col_end = target_col_start + (source_col_end - source_col_start)

    shifted[target_row_start:target_row_end, target_col_start:target_col_end] = image[
        source_row_start:source_row_end, source_col_start:source_col_end
    ]

    return shifted


# Use: Converts a mouse drawing into a centered MNIST-like 28x28 image.
def preprocess_drawing(drawing):
    if np.max(drawing) == 0:
        return np.zeros((28, 28), dtype=np.float32)

    rows, cols = np.where(drawing > 0.05)
    row_min, row_max = rows.min(), rows.max()
    col_min, col_max = cols.min(), cols.max()
    cropped = drawing[row_min : row_max + 1, col_min : col_max + 1]

    height, width = cropped.shape
    scale = 20 / max(height, width)
    new_height = max(1, int(round(height * scale)))
    new_width = max(1, int(round(width * scale)))
    resized = resize_image(cropped, new_height, new_width)

    image = np.zeros((28, 28), dtype=np.float32)
    row_start = (28 - new_height) // 2
    col_start = (28 - new_width) // 2
    image[row_start : row_start + new_height, col_start : col_start + new_width] = resized

    total_brightness = np.sum(image)
    if total_brightness > 0:
        row_indices, col_indices = np.indices(image.shape)
        center_row = np.sum(row_indices * image) / total_brightness
        center_col = np.sum(col_indices * image) / total_brightness
        row_shift = int(round(13.5 - center_row))
        col_shift = int(round(13.5 - center_col))
        image = shift_image(image, row_shift, col_shift)

    return np.clip(image, 0, 1)


# Use: Saves trained weights and biases so you do not need to train every time.
def save_params(file_path, W1, b1, W2, b2, W3, b3):
    np.savez(file_path, W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)


# Use: Loads previously trained weights and biases from your computer.
def load_params(file_path):
    model = np.load(file_path)
    required_keys = {"W1", "b1", "W2", "b2", "W3", "b3"}

    if not required_keys.issubset(model.files):
        raise ValueError("Saved model uses the old 1-hidden-layer format.")

    return model["W1"], model["b1"], model["W2"], model["b2"], model["W3"], model["b3"]


# Use: Creates a small drawing app where you can write a digit and predict it.
class DigitDrawingApp:
    def __init__(self, W1, b1, W2, b2, W3, b3):
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2
        self.W3 = W3
        self.b3 = b3

        self.canvas_size = 280
        self.brush_radius = 12
        self.drawing = np.zeros((self.canvas_size, self.canvas_size), dtype=np.float32)

        self.root = tk.Tk()
        self.root.title("Digit Recognizer")

        self.canvas = tk.Canvas(
            self.root,
            width=self.canvas_size,
            height=self.canvas_size,
            bg="black",
            highlightthickness=0,
        )
        self.canvas.grid(row=0, column=0, columnspan=2, padx=10, pady=10)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<Button-1>", self.draw)

        self.result_label = tk.Label(
            self.root,
            text="Draw a digit",
            font=("Arial", 20),
        )
        self.result_label.grid(row=1, column=0, columnspan=2, pady=5)

        predict_button = tk.Button(
            self.root,
            text="Predict",
            font=("Arial", 14),
            command=self.predict,
        )
        predict_button.grid(row=2, column=0, sticky="ew", padx=10, pady=10)

        clear_button = tk.Button(
            self.root,
            text="Clear",
            font=("Arial", 14),
            command=self.clear,
        )
        clear_button.grid(row=2, column=1, sticky="ew", padx=10, pady=10)

    # Use: Starts the Tkinter window and waits for mouse/button actions.
    def run(self):
        self.root.mainloop()

    # Use: Draws smooth white strokes on a high-resolution input grid.
    def draw(self, event):
        x = event.x
        y = event.y
        radius = self.brush_radius

        self.canvas.create_oval(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            fill="white",
            outline="white",
        )

        row_min = max(0, y - radius)
        row_max = min(self.canvas_size, y + radius + 1)
        col_min = max(0, x - radius)
        col_max = min(self.canvas_size, x + radius + 1)

        for row in range(row_min, row_max):
            for col in range(col_min, col_max):
                distance = np.sqrt((row - y) ** 2 + (col - x) ** 2)
                if distance <= radius:
                    value = 1 - (distance / radius) * 0.35
                    self.drawing[row, col] = max(self.drawing[row, col], value)

    # Use: Clears the drawing area and resets the prediction message.
    def clear(self):
        self.drawing.fill(0)
        self.canvas.delete("all")
        self.result_label.config(text="Draw a digit")

    # Use: Converts your drawing into model input and shows the predicted number.
    def predict(self):
        processed = preprocess_drawing(self.drawing)
        X = processed.reshape(784, 1)
        prediction, probabilities = predict_digit(
            X, self.W1, self.b1, self.W2, self.b2, self.W3, self.b3
        )
        confidence = probabilities[prediction] * 100
        self.result_label.config(text=f"Prediction: {prediction} ({confidence:.1f}%)")


# Use: Starts the program only when this file is run directly.
if __name__ == "__main__":
    try:
        W1, b1, W2, b2, W3, b3 = load_params(MODEL_FILE)
        print("Loaded saved model.")
    except (FileNotFoundError, ValueError):
        print("No compatible saved model found. Training a new 2-hidden-layer model...")
        X_train, Y_train = load_data("train.csv")

        W1, b1, W2, b2, W3, b3 = gradient_descent(
            X_train,
            Y_train,
            iterations=35000,
            learning_rate=0.05,
            hidden1_size=64,
            hidden2_size=32,
            batch_size=128,
        )

        train_predictions = make_predictions(X_train, W1, b1, W2, b2, W3, b3)
        train_accuracy = get_accuracy(train_predictions, Y_train)
        print(f"Final training accuracy: {train_accuracy:.4f}")

        save_params(MODEL_FILE, W1, b1, W2, b2, W3, b3)
        print(f"Saved model to {MODEL_FILE}.")

    app = DigitDrawingApp(W1, b1, W2, b2, W3, b3)
    app.run()
