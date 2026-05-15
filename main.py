# main.py

from utils import load_data
from cnn import CNN
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
x_train, y_train, x_test, y_test = load_data()

# Create CNN
cnn = CNN()

conv = cnn.conv
pool = cnn.pool
fc = cnn.fc

# Fashion MNIST class names
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot"
]

# Training settings
epochs = 4
learning_rate = 0.0001

# Training data
train_images = x_train[:10000]
train_labels = y_train[:10000]

# Lists for graphs
losses = []
accuracies = []

test_losses = []
test_accuracies = []


# =========================
# FORWARD FUNCTION
# =========================

def forward(image, label):

    # Normalize image
    image = image - 0.5

    # Forward pass
    out = conv.forward(image)

    out = pool.forward(out)

    out = fc.forward(out)

    # Cross entropy loss
    loss = -np.log(out[label] + 1e-7)

    # Accuracy
    acc = 1 if np.argmax(out) == label else 0

    return out, loss, acc


# =========================
# TRAINING LOOP
# =========================

for epoch in range(epochs):

    print(f"\n========== Epoch {epoch + 1} ==========")

    loss = 0
    num_correct = 0

    # Epoch totals
    epoch_loss = 0
    epoch_correct = 0

    for i, (image, label) in enumerate(zip(train_images, train_labels)):

        # Forward pass
        out, l, acc = forward(image, label)

        loss += l
        num_correct += acc

        epoch_loss += l
        epoch_correct += acc

        # Initial gradient
        gradient = out.copy()

        gradient[label] -= 1

        # Backpropagation
        gradient = fc.backward(gradient, learning_rate)

        gradient = pool.backward(gradient)

        conv.backward(gradient, learning_rate)

        # Print every 100 images
        if (i + 1) % 100 == 0:

            print(
                f"Step: {i+1} | "
                f"Average Loss: {loss / 100:.3f} | "
                f"Accuracy: {num_correct}%"
            )

            # Reset mini-batch stats
            loss = 0
            num_correct = 0

    # Store epoch metrics
    avg_epoch_loss = epoch_loss / len(train_images)

    avg_epoch_accuracy = (
        epoch_correct / len(train_images)
    ) * 100

    losses.append(avg_epoch_loss)

    accuracies.append(avg_epoch_accuracy)

    print(
        f"\nEpoch Loss: {avg_epoch_loss:.4f}"
    )

    print(
        f"Epoch Accuracy: {avg_epoch_accuracy:.2f}%"
    )

    # =========================
    # TESTING AFTER EACH EPOCH
    # =========================

    test_correct = 0
    test_total = 0
    test_loss = 0

    for test_image, test_label in zip(x_test[:1000], y_test[:1000]):

        out, l, acc = forward(test_image, test_label)

        test_loss += l
        test_correct += acc
        test_total += 1

    testing_accuracy = (
        test_correct / test_total
    ) * 100

    testing_loss = (
        test_loss / test_total
    )

    # Store testing metrics
    test_losses.append(testing_loss)

    test_accuracies.append(testing_accuracy)

    # =========================
    # EPOCH SUMMARY
    # =========================

    print(f"\n----- Epoch {epoch + 1} Summary -----")

    print(
        f"Training Accuracy : "
        f"{avg_epoch_accuracy:.2f}%"
    )

    print(
        f"Training Loss     : "
        f"{avg_epoch_loss:.4f}"
    )

    print(
        f"Testing Accuracy  : "
        f"{testing_accuracy:.2f}%"
    )

    print(
        f"Testing Loss      : "
        f"{testing_loss:.4f}"
    )

    print("------------------------------")


# =========================
# FINAL TRAINING RESULTS
# =========================

final_correct = 0
final_total = 0
final_loss = 0

for image, label in zip(train_images[:1000], train_labels[:1000]):

    out, l, acc = forward(image, label)

    final_loss += l
    final_correct += acc
    final_total += 1

print("\n========== FINAL RESULTS ==========")

print(
    f"Final Accuracy: "
    f"{(final_correct / final_total) * 100:.2f}%"
)

print(
    f"Final Loss: "
    f"{final_loss / final_total:.4f}"
)


# =========================
# TRAINING LOSS GRAPH
# =========================

plt.figure(figsize=(6, 4))

plt.plot(losses)

plt.xlabel("Epoch")

plt.ylabel("Training Loss")

plt.title("Training Loss vs Epoch")

plt.grid(True)

plt.savefig("training_loss_graph.png")

plt.show()


# =========================
# TRAINING ACCURACY GRAPH
# =========================

plt.figure(figsize=(6, 4))

plt.plot(accuracies)

plt.xlabel("Epoch")

plt.ylabel("Training Accuracy")

plt.title("Training Accuracy vs Epoch")

plt.grid(True)

plt.savefig("training_accuracy_graph.png")

plt.show()


# =========================
# TESTING LOSS GRAPH
# =========================

plt.figure(figsize=(6, 4))

plt.plot(test_losses)

plt.xlabel("Epoch")

plt.ylabel("Testing Loss")

plt.title("Testing Loss vs Epoch")

plt.grid(True)

plt.savefig("testing_loss_graph.png")

plt.show()


# =========================
# TESTING ACCURACY GRAPH
# =========================

plt.figure(figsize=(6, 4))

plt.plot(test_accuracies)

plt.xlabel("Epoch")

plt.ylabel("Testing Accuracy")

plt.title("Testing Accuracy vs Epoch")

plt.grid(True)

plt.savefig("testing_accuracy_graph.png")

plt.show()


# =========================
# SINGLE IMAGE PREDICTION
# =========================

test_image = x_test[0]

test_label = y_test[0]

# Prediction
out, _, _ = forward(test_image, test_label)

predicted = np.argmax(out)

# Display image
plt.figure(figsize=(4, 4))

plt.imshow(test_image, cmap='gray')

plt.title(
    f"Predicted: {class_names[predicted]}\n"
    f"Actual: {class_names[test_label]}"
)

plt.axis('off')

plt.savefig("sample_prediction.png")

plt.show()


print("\n========== SINGLE IMAGE PREDICTION ==========")

print(
    "Predicted Class:",
    class_names[predicted]
)

print(
    "Actual Class:",
    class_names[test_label]
)