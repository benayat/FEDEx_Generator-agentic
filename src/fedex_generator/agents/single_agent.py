import json

from fedex_generator.agents.base_agent import BaseAgent


class SingleAgent(BaseAgent):
    async def generate_explanation_and_code_in_json(self, analysis_result):
        """
        Generates a combined explanation and code snippet in JSON format based on the analysis result.

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
        dict: a json object containing the explanation and code snippet.
        """
        # Extract relevant details from the analysis result
        processed_analysis_result = self.process_analysis_result(analysis_result)
        system_prompt = """
        Given an `analysis_result` parameter containing data + insights from a pd-explain Python package, generate a JSON output with a rephrased explanation and matplotlib code for visualization.

# Steps

1. **Rephrase Explanation**: Simplify the explanation provided in the  latex`analysis_result['explanation']` to make it more accessible and understandable.
2. **Create Visualization Code**: Write static, self-sufficient matplotlib code that visualizes the selected data and its insights in the `analysis_result`. This code does not require any additional context from the analysis.

# Details Needed for Code:
- Use the `analysis_result['action_title']`, `analysis_result['bin_name']`, `analysis_result['result_column_summary']` and the `analysis_result[result_column]` to guide the visualization.
- Consider the relevance of `analysis_result['values_to_influence_dict']` and `analysis_result['column_score']` to highlight in the graph if applicable.
- Ensure the code snippet includes necessary imports and matplotlib setup.

# Output Format

Provide the output in JSON format with the following structure:
```json
{
  "rephrased_explanation": "<Simplified explanation here>",
  "matplotlib_code": "<Matplotlib code snippet here as a valid json string>"
}
```

# Examples

**Example Input:**
```python
analysis_result = {
  'explanation': "$\\bf{relationship}$ value $\\bf{Husband}$ (in green)
appear $\\bf{ 1.4\\ times\\ less }$ than before",
  'action_title': "'Dataframe adults, filtered on attribute label'",
  'bin_name': "'CategoricalBin'",
  'result_column_summary': '{"count":37155,"unique":6,"top":"Not-in-family","freq":11307}',
  'values_to_occurrences_source': {'Husband': 19716, 'Not-in-family': 12583, 'Other-relative': 1506, 'Own-child': 7581, 'Unmarried': 5125, 'Wife': 2331}
  'values_to_occurrences_result': {'Husband': 10870, 'Not-in-family': 11307, 'Other-relative': 1454, 'Own-child': 7470, 'Unmarried': 4816, 'Wife': 1238},
  'values_to_influence_dict': {'Husband': 0.07817829035394597, 'Not-in-family': -0.012108444773741234, 'Other-relative': -0.0009277104646251422, 'Own-child': -0.0005472375947424002, 'Unmarried': -0.00375417558359209, 'Wife': -0.010146748491386026},
  'column_score': 0.11111077129828373
}
```

**Example Output:**
```json
{
  "rephrased_explanation": "The group with a 'fnlwgt' value of 215033.3 stands out because it is significantly higher than the average, by about two standard deviations.",
  "matplotlib_code": "
    import matplotlib.pyplot as plt

    age_groups = ['25', '30', '35']
    spending = [1000, 1500, 1200]

    plt.bar(age_groups, spending)
    plt.title('Age Group vs. Spending')
    plt.xlabel('Age Group')
    plt.ylabel('Average Spending')
    plt.show()
  "
}
```

**(Note: Real examples will include more detailed matplotlib code reflecting actual data insights)**

# Notes

- Handle cases where `source_column` has 70% None values by focusing primarily on the `result_column`.
- Ensure the matplotlib code does not assume access to any external data files; all data should be represented within the code snippet.
- The rephrasing should capture the essence of the explanation while being easier to understand.
        """

        user_prompt = f"""
        ${processed_analysis_result}
        """
        explanation_and_code_result = await self.generate_chat_completion_async(system_prompt, user_prompt, temperature=0.5)
        return self.extract_values_from_json(explanation_and_code_result)

    @staticmethod
    def process_analysis_result(analysis_result):
        """
        Processes the analysis result to extract relevant details for generating the explanation and code snippet.

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
        dict: A dictionary containing the processed analysis details.
        """
        action_title = analysis_result['action_title']
        bin_name = analysis_result['bin_name']
        explanation = analysis_result['explanation']
        source_column_summary = analysis_result['source_column'].describe().to_json() if analysis_result[
                                                                                   'source_column'] is not None else "N/A"
        values_to_occurrences_source = {value: occurrences for value, occurrences in
                                 analysis_result['source_column'].value_counts().items()} if analysis_result[
                                                                                                 'source_column'] is not None else "N/A"
        result_column_summary = analysis_result['result_column'].describe().to_json() if analysis_result[
                                                                                   'result_column'] is not None else "N/A"
        values_to_occurrences_result = {value: occurrences for value, occurrences in
                                 analysis_result['result_column'].value_counts().items()} if analysis_result[
                                                                                                 'result_column'] is not None else "N/A"
        values_to_influence = analysis_result['values_to_influence_dict'] if analysis_result['values_to_influence_dict'] is not None else "N/A"
        column_score = analysis_result['column_score']

        return {
            'action_title': action_title,
            'bin_name': bin_name,
            'explanation': explanation,
            'source_column_summary': source_column_summary,
            'values_to_occurrences_source': values_to_occurrences_source,
            'result_column_summary': result_column_summary,
            'values_to_occurrences_result': values_to_occurrences_result,
            'values_to_influence': values_to_influence,
            'column_score': column_score
        }

    @staticmethod
    def extract_values_from_json(json_str):
        data = json.loads(json_str.strip('```json').strip('```'))
        rephrased_explanation = data.get('rephrased_explanation')
        matplotlib_code = data.get('matplotlib_code')
        return rephrased_explanation, matplotlib_code
