"""
Post-processing utilities for time series forecasting.
"""
import re
def fix_numeric_text(text):
    """
    Cleans and formats numeric text from model output to ensure it follows
    the expected format of comma-separated values and semicolon-separated timesteps.
    
    Args:
        text: Raw text output from the model
        
    Returns:
        Cleaned and formatted text following the expected pattern
    """
    # Step 1: Extract all numbers with their decimal points
    numbers = re.findall(r'\d+\.\d+|\d+', text)
    
    # Step 2: Determine if we have an even number of values (for prey, predator pairs)
    if len(numbers) % 2 != 0:
        # If odd number, drop the last one as it's likely incomplete
        numbers = numbers[:-1]
    
    # If we have no valid numbers, return empty string
    if not numbers:
        return ""
    
    # Step 3: Pair up the numbers to form timesteps
    timesteps = []
    for i in range(0, len(numbers), 2):
        if i+1 < len(numbers):  # Make sure we have a complete pair
            timestep = f"{numbers[i]},{numbers[i+1]}"
            timesteps.append(timestep)
    
    # Step 4: Join timesteps with semicolons
    return ";".join(timesteps)