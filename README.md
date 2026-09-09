# Image Inpainting Autoencoder

A convolutional autoencoder that reconstructs masked regions in grayscale images.

## Overview

This project trains a CNN-based encoder-decoder model to perform image inpainting: given a 28×28 grayscale image with an 8×8 pixel patch masked out (zeroed) at its center, the model learns to predict the missing content using the surrounding context.

## How It Works

**Data pipeline** (`get_data`)
- Loads training images from `train_data.npz` and test images from `test_data.npz`
- Creates masked inputs by zeroing out the central 8×8 region (rows/cols 10:18) of each image
- The unmasked original images serve as training labels
- Saves side-by-side input/label visualizations to `train_image_output/`

**Model architecture** (`Model`)
- **Encoder**: Two convolutional blocks (with LeakyReLU + max pooling) extract spatial features, followed by fully connected layers that compress the representation down to a 64-dimensional latent bottleneck
- **Decoder**: A stack of transposed convolutions upsamples the latent vector back to a full 28×28 reconstruction, with a final sigmoid activation to bound pixel outputs
- Input/output pixel values are normalized to [0, 1] internally (divided/multiplied by 255)

**Training** (`train_model`)
- 90/10 train/validation split via `random_split`
- MSE loss between reconstructed and original (unmasked) images
- Adam optimizer with weight decay, batch size 128, 100 epochs
- Tracks and prints per-epoch train/validation loss

**Inference & submission** (`test_model`)
- Runs batched prediction on the test set
- Clips outputs to [0, 255] and casts to `uint8`
- **Only the masked region (10:18, 10:18) is kept** in the final output — everything else is zeroed — since evaluation only scores reconstruction quality in the masked patch
- Saves predictions to `submit_this_test_data_output.npz`
- Saves qualitative before/after visualizations to `test_image_output/`

## Usage

```bash
python main.py
```

Requires `train_data.npz` and `test_data.npz` in the working directory. Produces `submit_this_test_data_output.npz` as the final submission file, plus visualization folders.

## Notes / Potential Improvements

- Training currently computes MSE over the **entire image**, even though only the masked patch is scored — restricting the loss to the masked region could give a more targeted training signal.
- Model depth, dropout rate, and bottleneck size are tunable; deeper decoders or skip connections (U-Net style) could improve reconstruction fidelity.
