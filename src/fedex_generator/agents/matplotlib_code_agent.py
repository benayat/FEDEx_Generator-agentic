from .base_agent import BaseAgent
import re


class MatplotlibCodeAgent(BaseAgent):
    async def generate_matplotlib_code(self, code_instructions, temperature=0, retries=3):
        """
        Generates Python code for creating a Matplotlib visualization based on the given instructions.

        Parameters:
        code_instructions (str): Detailed instructions for creating the Matplotlib visualization.
        temperature (float): The temperature to use for generating code. Default is 0 for deterministic output.
        retries (int): Number of times to retry generating the code in case of an error. Default is 3.

        Returns:
        str: Python code that implements the Matplotlib visualization as per the instructions.
        """
        system_prompt = f"""
    You are a senior Python developer. Write only the Python code based on the given instructions for creating a Matplotlib visualization. The code must be well-formatted, error-free, and executable without additional explanations or markers. The requirements are as follows:

    **Requirements**:

    - Include necessary imports (e.g., `import matplotlib.pyplot as plt`).
    - Implement all visual elements as specified.
    - Ensure the plot is clear, well-labeled, and visually appealing.
    - Ensure the legend is accurate and all components are spacious enough without overlaps.
    - The code must be executable as is, with `plt.show()` called only once at the end.
    """
        user_prompt = f"""
    **Instructions**:
    {code_instructions}

    """
        response_code = None
        for attempt in range(retries):
            response_code = await self.generate_chat_completion_async(system_prompt, user_prompt, temperature=temperature)
            if response_code:
                return response_code.strip()
            else:
                # Log the error or retry attempt (optional)
                print(f"Attempt {attempt + 1} failed. Retrying...")

        # Handle case where response does not contain code in the expected format after all retries
        return "# Error: Code block not found in the response after multiple attempts"