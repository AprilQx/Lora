Model Architecture
================

Overall Architecture
------------------

Our architecture combines a pre-trained LLM with parameter-efficient fine-tuning:

.. figure:: ../_static/model_architecture.png
   :alt: Model Architecture
   :width: 700px
   
   The overall model architecture with LoRA adaptation.

Key Components
------------

1. **Tokenization Layer**: Converts numerical data to text format
2. **Transformer Encoder-Decoder**: Qwen2.5 model with 24 layers
3. **LoRA Adaptation**: Low-rank updates to attention modules
4. **Output Layer**: Language modeling head with trainable bias

LoRA Application
--------------

LoRA is selectively applied to the following components:

.. code-block:: text

    Qwen2.5 Model
    ├── Transformer Blocks (24x)
    │   ├── Self-Attention
    │   │   ├── Query Projection (+ LoRA)
    │   │   ├── Key Projection
    │   │   ├── Value Projection (+ LoRA)
    │   │   └── Output Projection
    │   ├── MLP
    │   └── Layer Norm
    └── LM Head (+ trainable bias)

Training Process
--------------

During training, we:

1. Freeze the base model parameters
2. Apply LoRA to Q and V projections in self-attention
3. Make the LM head bias trainable
4. Train on time series data in text format
5. Monitor validation loss and FLOP usage