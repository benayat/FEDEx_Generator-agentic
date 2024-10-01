from .base_agent import BaseAgent
import re


class MatplotlibCodeAgent(BaseAgent):
    async def generate_matplotlib_code(self, code_instructions):
        """
        Generates Python code for creating a Matplotlib visualization based on the given instructions.

        Parameters:
        code_instructions (str): Detailed instructions for creating the Matplotlib visualization.

        Returns:
        str: Python code that implements the Matplotlib visualization as per the instructions.
        """
        system_prompt = f"""
    You are a senior, concise and super-organized Python developer. You always comply to the requirements.
    Given instructions, you always follow them to the letter and write a well-formatted and error-free compilable Python code using Matplotlib, to create an impeccable visualization. The requirements are as follows:

     **Requirements**:

    - Include all necessary imports (e.g., `import matplotlib.pyplot as plt, etc.`).
    - Implement all visual elements as specified.
    - Ensure the plot is clear, well-labeled, and visually appealing.
    - Ensure the legend actually includes all relevant information and colors. If a bar should have more than one color, make sure the legend reflects that!
    - Ensure the plot and all its components are spacious enough(labels, title, legents - nothing should overlap with nothing). Overlapping is a no-go! if it's too tight, Use label rotation, wider axis or other techniques to space thing better.
    - Ensure the code is executable as is.
    - make sure the visualization complies with the title and the instructions.
    - Do not include any explanations or anything but the code.
    - After you're done, make sure to go over the code and check for any errors, missing elements or problematic visualizations, and correct them if necessary.
    """
        user_prompt = f"""
    **Instructions**:
    {code_instructions}

    """
        response_code = await self.generate_chat_completion_async(system_prompt, user_prompt, temperature=0)
        return re.search(r"```(.*?)```", response_code, re.DOTALL).group(1).replace("python", "").strip()
