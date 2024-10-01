import pandas as pd
from .base_agent import BaseAgent


class ExplanationAgent(BaseAgent):

    async def generate_explanation(self, analysis_result):
        """
        Rephrases the explanation provided in the analysis result into a more coherent and fluent natural language style,
        utilizing the context from the action_title, bin_name, and other data elements.

        Parameters:
        analysis_result (dict): A dictionary containing the following keys:
            - 'explanation' (str): The basic explanation of the analysis.
            - 'action_title' (str): The title of the user action.
            - 'bin_name' (str): The name of the bin type.
            - 'source_column' (pd.Series): The source column data.
            - 'result_column' (pd.Series): The result column data.
            - 'values_to_influence_dict' (dict): A dictionary mapping unique values to their influence.
            - 'column_score' (float): The score of the column.

        Returns:
        str: A rephrased, coherent, and fluent explanation.
        """
        # Extract relevant details from the analysis result
        action_title = analysis_result['action_title']
        bin_name = analysis_result['bin_name']
        explanation = analysis_result['explanation']
        source_column_summary = analysis_result['source_column'].describe() if analysis_result[
                                                                                   'source_column'] is not None else "N/A"
        result_column_summary = analysis_result['result_column'].describe() if analysis_result[
                                                                                   'result_column'] is not None else "N/A"
        values_to_influence = analysis_result['values_to_influence_dict']
        column_score = analysis_result['column_score']

        # Create a summary dictionary to provide context for rephrasing
        analysis_context = {
            'action_title': action_title,
            'bin_name': bin_name,
            'explanation': explanation,
            'source_column_summary': source_column_summary,
            'result_column_summary': result_column_summary,
            'values_to_influence': values_to_influence,
            'column_score': column_score
        }

        # System prompt to guide the rephrasing process
        system_prompt = f"""
        You are an expert in data analysis and natural language processing. Your task is to take the provided explanation 
        and rephrase it into fluent, coherent natural language that conveys the information clearly and concisely.

        **Instructions**:

        - Rephrase the provided explanation based on the given context.
        - The rephrasing should highlight the significance of the action described in `action_title`.
        - Incorporate the `bin_name` and other relevant information in a natural and fluent style.
        - Ensure the explanation is self-contained, concise, and professionally written.
        - Avoid jargon, and make the explanation easy to understand while still professional.
        """

        # User prompt containing the extracted data
        user_prompt = f"""
        **Explanation to Rephrase**:
        {explanation}

        **Context**:
        Action Title: {action_title}
        Bin Name: {bin_name}
        Source Column Summary: {source_column_summary}
        Result Column Summary: {result_column_summary}
        Column Score: {column_score}
        Values to Influence: {values_to_influence}
        """

        # Generate the rephrased explanation
        rephrased_explanation = await self.generate_chat_completion_async(system_prompt, user_prompt, temperature=0.5)
        return rephrased_explanation, analysis_context