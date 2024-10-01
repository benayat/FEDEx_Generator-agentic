from .base_agent import BaseAgent

class CodeInstructionsAgent(BaseAgent):
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
You are a senior data analyst and QA engineer. You are precise and to the point. You write clear, structured, and concise natural language instructions for a developer to create a Matplotlib visualization. You will extract all necessary details from the provided `data_dict`, including but not limited to the title, plot type, axis labels, colors, highlights, and annotations.
The structure of the plot layout must be determined based on the `explanation and `data_dict.action_title`, and the rest of data_dict. The appropriate layout (e.g., bar plot, x-y plot, scatter plot, or another supported layout) should be chosen based on this information. Your instructions should provide the developer with everything they need to create the plot, without requiring any additional variables.

**Explanation**:
{explanation}

**Data** (Extract and use all relevant data from this dictionary):
{data_dict}
"""
        # User prompt provides code instructions based on the extracted data
        user_prompt = f"""
Your task is to convert the system's explanation and the extracted data from the dictionary into clear, detailed code instructions for creating a Matplotlib visualization.

**Instructions**:
- Determine the most appropriate plot type based on {data_dict['action_title']}, {explanation}, and the rest of data_dict. It could be a bar chart, scatter plot, line plot, or another Matplotlib-supported layout. 
- derive the visualization title from {explanation}.
- Use {explanation} and {data_dict['action_title']} to clarify what the plot is communicating and guide the choice of layout.
- Ensure that the x-axis and y-axis (if applicable) are labeled based on the data and context provided by the dictionary.
- Specify colors, annotations, and highlights based on any relevant keys in the data and ensure distinct elements are clearly differentiated and labeled.
- Make sure that no text (e.g., labels, titles, legends) overlaps with other elements. Use layout adjustment techniques such as `tight_layout()`.
- If there's a legend, Double-check that it includes all necessary information and colors. If a bar should have more than one color, make sure the legend reflects that.
- Include any necessary legends, ensuring they have enough room and do not overlap with the plot or labels. Locate the legends in the appropriate location and position them clearly.
- Ensure the instructions are complete, with all details derived from the data, so that a developer can implement the plot without needing any further context or input.
- Prevent the rendering of an empty plot by associating each bar with a label in the legend.
- Do not include any actual code—just the instructions in natural language.
"""

        return await self.generate_chat_completion_async(system_prompt, user_prompt, temperature=0.4)