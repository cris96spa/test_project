from PIL import Image
import polars as pl
from uuid import uuid4
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from scipy.stats import entropy
from matplotlib import pyplot as plt


def saving_sample_wrapper(
    image_generator,
    output_folder: str,
    initial_sample_id: int,
    initial_timestamp: float,
):
    """this wrapper saves generated images and creates target and
    mapping csv files."""
    sample_id = initial_sample_id
    timestamp = initial_timestamp

    # while True:
    elem = next(image_generator)
    image_name_list = []
    label_list = []
    sample_id_list = []
    timestamp_list = []

    for i, image in enumerate(elem[0]):
        name = str(uuid4())
        pil_image = Image.fromarray(np.uint8(image * 255).squeeze())
        pil_image.save(os.path.join(output_folder, 'images', name + '.png'))
        if int(elem[1][i]) == 0:
            pil_image.save(
                os.path.join(
                    output_folder, 'tf_dirs', 'ok_front', name + '.png'
                )
            )
        else:
            pil_image.save(
                os.path.join(
                    output_folder, 'tf_dirs', 'def_front', name + '.png'
                )
            )
        image_name_list.append(name)
        sample_id_list.append(sample_id)
        timestamp_list.append(timestamp)

        # add 10 seconds
        timestamp += 10
        sample_id += 1

    for label in elem[1]:
        label_list.append(int(label))

    target = pl.DataFrame(
        {
            'label': label_list,
            'sample_id': sample_id_list,
            'timestamp': timestamp_list,
        }
    )
    mapping = pl.DataFrame(
        {
            'file_name': list(map(lambda x: x + '.png', image_name_list)),
            'sample_id': sample_id_list,
            'timestamp': timestamp_list,
        }
    )
    with open(os.path.join(output_folder, 'target.csv'), mode='ab') as f:
        target.write_csv(f, include_header=False)
    with open(os.path.join(output_folder, 'mapping.csv'), mode='ab') as f:
        mapping.write_csv(f, include_header=False)

    yield elem


def plot_histogram(image, ax, title='Pixel Intensity Histogram'):
    """Plot histogram of pixel intensities for an image."""
    # Image is assumed to be a tensor in CHW format
    image = image.reshape(-1, 3)  # Flatten the image pixels by channel

    colors = ['r', 'g', 'b']  # Color channels
    for i, color in enumerate(colors):
        ax.hist(
            image[:, i],
            bins=256,
            color=color,
            alpha=0.5,
            label=f'Channel {color.upper()}',
        )
    ax.legend()

    ax.set_title(title)
    ax.set_xlim([0, 1])  # Assuming normalized [0, 1] pixel values
    ax.set_ylim([0, 1000])  # Adjust based on the dataset


def compute_mean_variance(image_tensor):
    """Compute mean and variance for each channel in the image tensor (CHW format)."""
    mean = torch.mean(
        image_tensor, dim=(1, 2)
    )  # Mean for each channel (C, H, W)
    variance = torch.var(image_tensor, dim=(1, 2))  # Variance for each channel
    return mean, variance


def compute_kl_divergence(original_image, transformed_image):
    """Compute KL divergence between pixel intensity histograms of two images."""
    original_hist = np.histogram(
        original_image.numpy().flatten(), bins=256, range=(0, 1), density=True
    )[0]
    transformed_hist = np.histogram(
        transformed_image.numpy().flatten(),
        bins=256,
        range=(0, 1),
        density=True,
    )[0]

    # Add a small value to avoid division by zero
    original_hist = original_hist + 1e-7
    transformed_hist = transformed_hist + 1e-7

    # Compute KL Divergence
    kl_div = entropy(original_hist, transformed_hist)
    return kl_div


def get_features(image, model):
    with torch.no_grad():
        features = model(image)
    return features.squeeze()


def compute_cosine_similarity(original_features, transformed_features):
    """
    Compute cosine similarity between two feature vectors.
    parameters:
    -----------
        original_features (Tensor): Feature vector for the original image
        transformed_features (Tensor): Feature vector for the transformed image
    returns:
    --------
        float: Cosine similarity between the two feature vectors
    """
    cosine_similarity = nn.CosineSimilarity()
    similarity = cosine_similarity(original_features, transformed_features)
    return similarity


def compute_images_similarity(original_image, transformed_image, model):
    """
    Compute the similarity between two images using a pre-trained model.
    parameters:
    -----------
        original_image (Tensor): Original image tensor
        transformed_image (Tensor): Transformed image tensor
        model (nn.Module): Pre-trained model to extract features
    returns:
    --------
        float: Cosine similarity between the two images
    """
    # Extract features for both images
    original_features = get_features(original_image, model)
    transformed_features = get_features(transformed_image, model)

    # Compute cosine similarity
    similarity = compute_cosine_similarity(
        original_features, transformed_features
    )
    return similarity.mean().item()


def plot_similarity_metric(df, metric_name, labels, title=''):
    plt.figure(figsize=(10, 6))
    for row in df.iter_rows(named=True):
        # Check condition for specific mean and sigma
        values = row[metric_name]
        sequence_index = np.arange(len(values))

        # Compute confidence intervals (Here we simulate using standard deviation)
        # You can replace this with actual error bounds if available
        ci = 1.96  # 95% confidence interval
        std_dev = np.std(values) * ci
        lower_bound = np.array(values) - std_dev
        upper_bound = np.array(values) + std_dev

        plt.plot(
            sequence_index,
            values,
            label=f"mean={row['mean']}, sigma={row['sigma']}",
        )
        plt.fill_between(sequence_index, lower_bound, upper_bound, alpha=0.2)

    plt.title(title)
    plt.xlabel('Sequence index (subsampled)')
    plt.ylabel(metric_name)
    plt.legend(title='parameters')
    plt.grid(True)
    plt.show()
