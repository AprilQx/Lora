Qwen2.5 Model
============

Overview
--------

We use the Qwen2.5-0.5B-Instruct model as our base model for numerical forecasting. This model 
provides a good balance between performance and efficiency.

Model Details
------------

* **Model Size**: 0.5B parameters
* **Context Length**: 32k tokens
* **Architecture**: Transformer-based with 24 layers
* **Instruction Tuned**: Optimized for following instructions

Implementation
-------------

The model is loaded using Hugging Face's Transformers library:

.. code-block:: python

    def load_qwen():
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        
        # Freeze all parameters except LM head bias
        for param in model.parameters():
            param.requires_grad = False
            
        # Add trainable bias to logits
        assert model.lm_head.bias is None
        model.lm_head.bias = torch.nn.Parameter(
            torch.zeros(model.config.vocab_size, device=model.device)
        )
        model.lm_head.bias.requires_grad = True
        
        return model, tokenizer

Tokenization
-----------

The Qwen tokenizer is used to convert between text and tokens:

* **Vocabulary Size**: ~152k tokens
* **Special Tokens**: Includes system message tokens, user/assistant markers, etc.
* **Numeric Handling**: Efficiently tokenizes individual digits and decimal numbers