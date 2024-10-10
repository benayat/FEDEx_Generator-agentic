from .base_agent import BaseAgent

class CodeInstructionsAgent(BaseAgent):
    """
    The CodeInstructionsAgent is responsible for generating detailed, step-by-step natural language instructions
    for creating Matplotlib visualizations. It takes an explanation and relevant data to produce concise and clear
    instructions that a developer can use to implement the desired plot without requiring additional context.
    """
    async def generate_code_instructions(self, explanation, data_dict):
        """
        Generates detailed, step-by-step code instructions for creating a Matplotlib visualization based on the given explanation and data.

        Parameters:
        explanation (str): The explanation text to be translated into code instructions.
        data_dict (dict): A dictionary containing all relevant data for generating code instructions.

        Returns:
        str: Detailed code instructions for a Python developer to create a Matplotlib visualization.
        """
        # Build the system prompt with data_dict extraction
        system_prompt = f"""
You are a senior data analyst and QA engineer. Write clear and concise natural language instructions for a developer to create a Matplotlib visualization. Extract necessary details from `data_dict`, such as title, plot type, axis labels, colors, highlights, and annotations.
Determine the plot layout based on `explanation` and `data_dict.action_title`. Choose the appropriate layout (e.g., bar plot, scatter plot) and provide the developer with all necessary details to create the plot without needing additional input.

**Explanation**:
{explanation}

**Data** (Extract and use all relevant data from this dictionary):
{data_dict}
"""
        # User prompt provides code instructions based on the extracted data
        user_prompt = f"""
Convert the system's explanation and extracted data into clear, detailed code instructions for creating a Matplotlib visualization.

**Instructions**:
- Determine the most appropriate plot type based on {data_dict['action_title']}, {explanation}, and the rest of data_dict.
- Derive the visualization title from {explanation}.
- Use {explanation} and {data_dict['action_title']} to clarify what the plot is communicating and guide the choice of layout.
- Ensure the x-axis and y-axis are labeled based on the data and context provided.
- Specify colors, annotations, and highlights based on relevant keys in the data and ensure distinct elements are clearly labeled.
- Ensure no text (e.g., labels, titles, legends) overlaps with other elements. Use `tight_layout()` for adjustments.
- If there's a legend, ensure it includes all necessary information and colors.
- Include necessary legends, ensuring they do not overlap with the plot or labels.
- Ensure the instructions are complete so a developer can implement the plot without further input.
- Prevent rendering of an empty plot by associating each bar with a label in the legend.
- Do not include actual code—just the instructions in natural language.
"""

        return await self.generate_chat_completion_async(system_prompt, user_prompt, temperature=0.4)
