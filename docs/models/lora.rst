Low-Rank Adaptation (LoRA)
========================

Overview
--------

Our project implements Low-Rank Adaptation (LoRA) to efficiently fine-tune the Qwen2.5 model 
while minimizing the number of trainable parameters.

LoRA Technique
-------------

LoRA works by adding low-rank updates to the weights of specific layers:

.. math::

   h = W_0 x + \Delta W x = W_0 x + AB x

where:

* :math:`W_0` is the frozen pre-trained weight matrix
* :math:`A` is a low-rank down-projection matrix
* :math:`B` is a low-rank up-projection matrix
* :math:`\Delta W = AB` is the update matrix

Implementation
-------------

We implement LoRA as a custom module that wraps around linear layers:

.. code-block:: python

    class LoRALinear(nn.Module):
        """
        LoRA implementation for Linear layers:
        y = W*x + b + (A*B)*x * (alpha/r)
        """
        def __init__(self, original_linear: nn.Linear, r: int, alpha: int = None, dropout: float = 0.0):
            super().__init__()
            # Store original layer and freeze it
            self.original_linear = original_linear
            self.original_linear.weight.requires_grad = False
            if self.original_linear.bias is not None:
                self.original_linear.bias.requires_grad = False
            
            # LoRA parameters
            self.r = r  # Rank
            self.alpha = alpha if alpha is not None else r  # Scaling factor
            
            # Define LoRA matrices
            self.lora_A = nn.Parameter(torch.empty(r, self.in_features, device=device))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, device=device))
            
            # Initialize A matrix (B is initialized to zeros)
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

Configuration
------------

Our LoRA implementation uses the following parameters:

* **Rank (r)**: 8
* **Alpha (α)**: 16 
* **Target Modules**: Query and Value projection matrices in self-attention
* **Dropout**: 0.5 (during training)

Saving and Loading
----------------

Custom functions handle the saving and loading of LoRA weights:

* ``save_lora_model``: Extracts and saves LoRA weights and configurations
* ``load_lora_weights``: Loads LoRA weights into a model with LoRA layers