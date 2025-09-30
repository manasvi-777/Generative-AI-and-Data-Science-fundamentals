# Transformer Multi-Head Attention (MHA) Implementation Analysis

This repository contains two Jupyter Notebooks that implement and analyze the Encoder-Decoder Multi-Head Attention (MHA) mechanism of the Transformer architecture.

The core objective is to **cross-verify the official PyTorch implementation against a manual, step-by-step tensor calculation**.

---

## Files

The project consists of the following two files:

1.  `Encoder_Decoder_MHA_Masking_InBuilt.ipynb`
    * **Description:** Contains the standard, functional implementation of a single-layer Transformer Encoder-Decoder using the **built-in PyTorch `nn.Transformer`** module. It includes the model definition, mask generation, and a basic forward pass/training step.

2.  `Encoder_Decoder_MHA_Masking_Manual.ipynb`
    * **Description:** Contains the **manual**, low-level implementation of the same single-layer Transformer Encoder-Decoder. This notebook replicates the tensor operations (Embeddings, Positional Encoding, QKV projection, Scaled Dot-Product Attention, Skip Connections, and Layer Normalization) using the exact same initialized weights from the PyTorch model for verification.
