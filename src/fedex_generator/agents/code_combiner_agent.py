from .base_agent import BaseAgent
import re


class CodeCombinerAgent(BaseAgent):
    async def combine_multiple_code_strings(self, generated_codes: list[str]):
        """
        Combines multiple generated Matplotlib-plot code strings while preserving and improving visual appeal, and preventing overlaps and redundancies.

        Parameters:
        generated_code strings (list[str]): The original Python code strings for Matplotlib plots.

        Returns:
        str: Combined and refined Python code with improved layout and readability.
        """
        system_prompt = """
You are a senior Python developer and visualization expert. Your task is to combine multiple Matplotlib plot code strings into a single, cohesive script, **that will only render once**. Ensure the combined code maintains or improves visual appeal, readability, and prevents any overlaps between labels, titles, legends, and plot elements.

**Requirements**:
- Combine the provided code strings into a single script, removing duplicated imports or anything that shouldn't be there.
- after figuring out each plot's final size, decide on how many plots can fit in a single row and how many rows are needed. Put a slight margin or border between plots if necessary. The most important thing is to make sure that the plots are clearly separated and not cramped together.
- Ensure there are no overlaps between labels, titles, legends, and plot elements, either in the same plot, and even more importantly between elements from different plots.
- Adjust the figure and all visual components (including text, labels, titles, and everything else) size appropriately for clarity.
- Make sure every element is in its appropriate place and the plot is visually appealing(eg. no legend should be inside the plot if there's enough room, otherwise enlarge the plot or put it outside).
- Ensure annotations are positioned clearly above or next to the relevant plot elements.
- Maintain or enhance the overall color scheme to highlight key data.
- Ensure the code remains fully executable as is.
- Your output is a single Python script that can be run to generate all the plots. Nothing else!
"""

        user_prompt = f"""
    **Generated Codes**:
    {generated_codes}
    """
        response_code = await self.generate_chat_completion_async(system_prompt, user_prompt, temperature=0.0)
        return re.search(r"```(.*?)```", response_code, re.DOTALL).group(1).replace("python", "").strip()
