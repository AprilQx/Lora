Data Preprocessing
================

Text Formatting for Language Models
----------------------------------

To use numerical time series data with language models, we convert the data into text format:

.. code-block:: python

    def numeric_to_text(trajectory, alpha=10.0, precision=2):
        """
        Convert numeric trajectory to text format.
        
        Args:
            trajectory: Numpy array of shape (timesteps, variables)
            alpha: Scaling parameter
            precision: Decimal precision
            
        Returns:
            Text representation of the trajectory
        """
        scaled_traj = trajectory * alpha
        
        # Format each time step as text
        text_points = []
        for t in range(len(scaled_traj)):
            prey, predator = scaled_traj[t]
            text_points.append(
                f"T{t}: Prey={prey:.{precision}f}, Predator={predator:.{precision}f}"
            )
        
        return " ".join(text_points)

Data Splitting Process
--------------------

We split the data into train, validation, and test sets:

* **Training Set**: 70% (700 trajectories)
* **Validation Set**: 15% (150 trajectories)
* **Test Set**: 15% (150 trajectories)

The split is performed with random sampling (seed=42) without stratification, as our exploratory analysis showed that the data does not have significant clusters requiring stratified sampling.

Data Processing Pipeline
---------------------

Our preprocessing pipeline includes:

1. **Loading raw data** from the HDF5 file
2. **Creating data splits** (train/val/test)
3. **Converting** numerical trajectories to text format
4. **Tokenizing** text data for model input
5. **Saving processed data** in both complete and chunked formats

Processing Parameters
-------------------

* **Alpha (scaling factor)**: 10.0
* **Precision**: 2 decimal places
* **Maximum sequence length**: 512 tokens
* **Stride for sliding window**: 256 tokenslotka_volterra.rst