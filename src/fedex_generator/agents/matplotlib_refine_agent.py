from .base_agent import BaseAgent
import re


class MatplotlibRefineAgent(BaseAgent):
    async def refine_matplotlib_code(self, generated_code):
        """
        Refactors given Python Matplotlib code to improve visual appeal and prevent overlaps.

        Parameters:
        generated_code (str): The original Python code for a Matplotlib plot.

        Returns:
        str: Refactored Python code with improved layout and readability.
        """
        system_prompt = f"""
    You are a senior Python developer, code-reviewer, and visualization expert. Your task is:
     1. analyze and review the code, and judge weather the code is well written, displays the data properly and is well written and visually appealing as intended. if it is - just return an empty response.
     2. if it isn't - refactor the code to improve the visual appeal, readability and prevent any overlaps between labels, titles, legends, and plot elements. 
     3. make sure you follow the following requirements to the letter: 

    **Requirements**:

    - Ensure there are no overlaps between labels, titles, legends, and plot elements.
    - Adjust the figure and all visual components(including text, labels, titles, and everything else) size appropriately for clarity.
    - Rotate labels for readability - only if necessary.
    - Apply `tight_layout()` to optimize spacing.
    - Ensure annotations are positioned clearly above or next to the relevant plot elements.
    - Maintain or enhance the overall color scheme to highlight key data.
    - Make sure the intended purpose of the plot is implemented correctly - the plot should loyally visualize the title content. make sure to differentiate(by color, or other relevant techniques) relevant elements clearly for that purpose.
    - Ensure the code remains fully executable as is.
    """
        user_prompt = f"""
    **Generated Code**:
    {generated_code}

    """
        response_code = await self.generate_chat_completion_async(system_prompt, user_prompt, temperature=0.1)
        return re.search(r"```(.*?)```", response_code, re.DOTALL).group(1).replace("python", "").strip()