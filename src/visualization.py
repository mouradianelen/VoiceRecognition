import json
import matplotlib.pyplot as plt


import json
import matplotlib.pyplot as plt


def plot_training_results(plot, json_path, label_prefix="", start_epoch=0):
    """
    Plots the training loss from a JSON file on an existing matplotlib plot.

    Args:
        plot: The matplotlib Axes object to plot on.
        json_path: Path to the JSON file containing training results.
        label_prefix: Prefix for the legend label to distinguish datasets.
        start_epoch: The starting epoch for plotting (default is 0).

    Returns:
        The updated matplotlib Axes object with the new data plotted.
    """
    # Load JSON data
    with open(json_path, "r") as f:
        results = json.load(f)

    # Extract data
    data = results.get("data", {})
    metadata = results.get("metadata", {})

    # Prepare x (epochs) and y (loss) data
    epochs = sorted(int(epoch) for epoch in data.keys())
    losses = [data[str(epoch)]["loss"] for epoch in epochs]

    # Filter epochs and losses to start from the specified epoch
    filtered_epochs = [epoch for epoch in epochs if epoch >= start_epoch]
    filtered_losses = [data[str(epoch)]["loss"] for epoch in filtered_epochs]

    # Extract the learning rate (assuming it's constant or take the first one)
    lr = data[str(epochs[0])]["lr"]

    # Plot the data
    label = f"{label_prefix} (lr={lr})"
    plot.plot(filtered_epochs, filtered_losses, label=label)

    # Add plot details
    plot.set_xlabel("Epochs")
    plot.set_ylabel("Loss")
    plot.legend()
    plot.grid(True)

    return plot
