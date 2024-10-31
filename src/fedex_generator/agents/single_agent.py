from fedex_generator.agents import utils
from fedex_generator.agents.base_agent import BaseAgent


class SingleAgent(BaseAgent):
    async def generate_explanation_and_code_in_json(self, raw_analysis_result):
        """
        Generates a combined explanation and code snippet in JSON format based on the analysis result.

        Parameters:
        analysis_result (dict): A dictionary containing the following keys:
            - 'explanation' (str): The basic explanation of the analysis.
            - 'action_title' (str): The title of the user action.
            - 'bin'(MultiIndexBin): The bin object.
            - 'values_to_influence_dict' (dict): A dictionary mapping unique values to their influence.
            - 'column_score' (float): The score of the column.

        Returns:
        dict: a json object containing the explanation and code snippet.
        """
        # Extract relevant details from the analysis result
        analysis_result = utils.process_analysis_result(raw_analysis_result)
        system_prompt = """
        Given an `analysis_result` parameter containing data and insights from a pd-explain Python package, generate a JSON output with a rephrased explanation and matplotlib code for visualization.

# Steps

1. **Rephrase Explanation**: 
   - Simplify the explanation in `analysis_result['explanation']`(without information loss) to make it clear, accessible and easily understandable, use analysis_result data for reference.

2. **Create Visualization Code**: 
   - Write static, self-contained matplotlib code that visualizes the relevant data and insights from `analysis_result`.
   - Ensure no external context is required for comprehension.

# Details Needed for Code
- use a paraphrased combination of your rephrased explanation, analysis_result['explanation'] and `analysis_result['action_title']` for the plot title
- Utilize `analysis_result['action_title']`, `analysis_result['column_name']`, `analysis_result[class_to_value]` and `analysis_result['result_column_summary']` to guide visualization.
- Consider `analysis_result['value_to_influences_dict']` and `analysis_result['column_score']` for graph highlights if relevant.
- Include necessary imports and setup for matplotlib in the code snippet.
- Distinguish visualizations by inferred action type:
  - **Filter**: Focus on exceptionality with a bar chart, including "before-filter"(brown) and "after-filter"(blue) legend and frequency percentage on the y-axis, and exceptional bar in green, annotated with its value.
  - **Groupby**: Illustrate diversity with bar charts showing different group classes and their respective values from analysis_result[class_to_value]. y axis is analysis_result['column_name'], exceptional bar in green annotated with its value, and a red line for the mean value, annotated with it's value.  
  - **Outlier/Other**: Customize visualization appropriately based on insights.
  In both Filter and Groupby cases, build the visualization based on analysis_result['class_to_value'] as a basis, and than other relevant data as you see fit. y-axis values should range from 75% of the minimum value to 105% of the maximum value. if min value is close to zero, start from 0.

# Output Format

Provide the output in JSON format with the following structure:
```json
{
  "rephrased_explanation": "<Simplified explanation here>",
  "matplotlib_code": "<Matplotlib code snippet here as a valid JSON string>"
}
```

# Examples

**Example Input:**
```python
analysis_result = {
  'explanation': "$\\bf{relationship}$ value $\\bf{Husband}$ (in green) appear $\\bf{ 1.4\\ times\\ less }$ than before",
  'action_title': "'Dataframe adults, filtered on attribute label'",
  'bin_type': "'CategoricalBin'",
  column_name': 'relationship',
  'result_column_summary': '{"count":37155,"unique":6,"top":"Not-in-family","freq":11307}',
  'class_to_value': {'Husband': 10870, 'Not-in-family': 11307, 'Other-relative': 1454, 'Own-child': 7470, 'Unmarried': 4816, 'Wife': 1238},
  'value_to_influences_dict': {'Husband': 0.078, 'Not-in-family': -0.012, 'Other-relative': -0.001, 'Own-child': -0.0005, 'Unmarried': -0.0037, 'Wife': -0.010},
  'column_score': 0.111
}
```
**Example Output:**
```json
{
  "rephrased_explanation": "The relationship 'Husband' appeared 1.4 times less frequently than before in the data set.",
  "matplotlib_code": "import matplotlib.pyplot as plt\n\n# Define the data\nlabels = ['Husband', 'Not-in-family', 'Other-relative', 'Own-child', 'Unmarried', 'Wife']\nvalues_before = [40, 26, 3, 16, 10, 5]\nvalues_after = [29, 30, 4, 20, 13, 3]\n\nx = range(len(labels))\nfigsize = (10, 6)\nbar_width = 0.35\n\n# Create the plot\nplt.figure(figsize=figsize)\nplt.bar(x, values_before, width=bar_width, label='Before Filter', color='brown', align='center')\nplt.bar([i + bar_width for i in x], values_after, width=bar_width, label='After Filter', color='blue', align='center')\nhighlighted_bar = plt.bar(x[0] + bar_width, values_after[0], width=bar_width, color='green', align='center')\n\n# Highlight the 'Husband' bar after filtering\nplt.bar(0 + bar_width, values_after[0], width=bar_width, color='green', align='center')\n\n# Add labels and title\nplt.xlabel('Relationship')\nplt.ylabel('Frequency(%)')\nplt.title('Occurrence of the \"Husband\" relationship is now 1.4 times less frequent in the dataset compared to before.')\n\n# Determine if rotation is necessary\nmax_label_length = max(len(label) for label in labels)\nrotation = 45 if max_label_length > 10 else 0\n\nplt.xticks([i + bar_width / 2 for i in x], labels, rotation=rotation, ha='right' if rotation else 'center')\nplt.legend()\nplt.tight_layout(pad=3.0)\n# Set the y-axis limits to the maximum value in the data\nall_values_max = max(max(values_before), max(values_after))\nall_values_min = min(min(values_before), min(values_after))\n# Set the y-axis limits properly.\nplt.ylim(0 if all_values_max-all_values_min>all_values_min else all_values_min*0.85, all_values_max * 1.05)\n\n# Annotate the green bar with its value\nfor bar in highlighted_bar:\n    height = bar.get_height()\n    plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}', ha='center', va='bottom')"
}
```

**Example Input:**
```python
analysis_result = {
    'action_title': 'script_name.groupby(workclass).agg(mean)',
    'column_name': 'fnlwgt',
    'bin_type': 'CategoricalBin', 
    'explanation': "Groups with $\\bf{'fnlwgt'='215033.3'}$(in green)\nhave a relatively $\\bf{high\\ 'fnlwgt'}$ value:\n2 standard deviations higher than the mean\n(185902)", 
    'source_column_summary': 'N/A', 
    'class_to_value': {'?': np.float64(187254.77349053233), 'Federal-gov': np.float64(183590.02863128492), 'Local-gov': np.float64(190161.13488520408), 'Never-worked': np.float64(215033.3), 'Private': np.float64(192669.21249926268), 'Self-emp-inc': np.float64(178990.2005899705), 'Self-emp-not-inc': np.float64(175579.0054375971), 'State-gov': np.float64(181933.46491670873), 'Without-pay': np.float64(167902.66666666666)}, 
    'result_column_summary': '{"count":9.0,"mean":185901.5319019141,"std":13293.8223406428,"min":167902.6666666667,"25%":178990.2005899705,"50%":183590.0286312849,"75%":190161.1348852041,"max":215033.3}', 
    'values_to_influence': {215033.3: np.float64(99706609.92881998), 192669.21249926268: np.float64(-13195354.23799768), 190161.13488520408: np.float64(-17084659.672877073), 187254.77349053233: np.float64(-19378668.938545525), 183590.02863128492: np.float64(-18884824.232972533), 181933.46491670873: np.float64(-17421971.513276815), 178990.2005899705: np.float64(-12919026.13609904), 175579.0054375971: np.float64(-4651956.309104413), 167902.66666666666: np.float64(25920565.165124804)}, 
    'column_score': 157089522.15517598}
```
**Example Output:**
```json
{
  "rephrased_explanation": "The group 'Never-worked' has a significantly high 'fnlwgt' value, which is 2 standard deviations above the mean of 185902.",
  "matplotlib_code": "import matplotlib.pyplot as plt\nimport numpy as np\n\nfrom main import highlighted_bar\n\n# Define the data\ncategories = ['?', 'Federal-gov', 'Local-gov', 'Never-worked', 'Private', 'Self-emp-inc', 'Self-emp-not-inc', 'State-gov', 'Without-pay']\nvalues = [187254.77, 183590.03, 190161.13, 215033.3, 192669.21, 178990.2, 175579.01, 181933.46, 167902.67]\nmean_value = 185902\n\n# Create the plot\nplt.figure(figsize=(10, 6))\nbars = plt.bar(categories, values, color=['blue' if category != 'Never-worked' else 'green' for category in categories])\n# Add a horizontal line for the mean\nplt.axhline(y=mean_value, color='red', linestyle='--', linewidth=1, label='Mean Value')\n\n# Add labels and title\nplt.xlabel('Workclass')\nplt.ylabel('fnlwgt')\nplt.title('The group \"Never-worked\" has a significantly high fnlwgt value')\n\n# Rotate x-axis labels if necessary\nplt.xticks(rotation=45, ha='right')\nplt.legend()\nplt.tight_layout(pad=3.0)\nplt.ylim(0 if max(values)-min(values)>min(values) else min(values)*0.85, max(values) * 1.05)\nfor bar in bars:\n    if bar.get_height() == values[3]:  # Check if the bar's height matches the \"Never-worked\" value\n        height = bar.get_height()\n        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}', ha='center', va='bottom')\n\n\n# Annotate the mean value red line with its value\nplt.text(len(categories) - 1, mean_value, f'{mean_value:.2f}', color='red', ha='center', va='bottom')\n\nplt.show()"
}
```

**Example Input:**
```python
analysis_result = {
    'action_title': 'Dataframe houses, filtered on attribute SalePrice', 
    'column_name': 'ExterQual', 
    'bin_type': 'CategoricalBin', 
    'explanation': '$\\bf{ExterQual}$ value $\\bf{TA}$ (in green)\nappear $\\bf{ 4\\ times\\ less }$ than before', 
    'source_column_summary': '{"count":1460,"unique":4,"top":"TA","freq":906}', 
    'class_to_value': {2: {'before': 'Ex (3.56)', 'after': 'Ex (12.98)'}, 4: {'before': 'Fa (0.96)', 'after': 'Fa (0.00)'}, 6: {'before': 'Gd (33.42)', 'after': 'Gd (71.55)'}, 11: {'before': 'TA (62.05)', 'after': 'TA (15.47)'}}, 
    'result_column_summary': '{"count":362,"unique":3,"top":"Gd","freq":259}', 
    'values_to_influence': {'Gd': np.float64(0.06982895386725407), 'TA': np.float64(0.4145614090552915), 'Ex': np.float64(-0.0002906022099191663)}, 
    'column_score': 0.4658518126087943
    }
```
**Example Output:**
```json
{
  "rephrased_explanation": "The 'ExterQual' value 'TA' is now 4 times less frequent than it was previously in the dataset.",
  "matplotlib_code": "import matplotlib.pyplot as plt\nimport numpy as np\n\n# Define the data\ncategories = ['Ex', 'Fa', 'Gd', 'TA']\nvalues_before = [3.56, 0.96, 33.42, 62.05]\nvalues_after = [12.98, 0.00, 71.55, 15.47]\n\nx = np.arange(len(categories))\nfigsize = (10, 6)\nbar_width = 0.35\n\n# Create the plot\nplt.figure(figsize=figsize)\nplt.bar(x, values_before, width=bar_width, label='Before Filter', color='brown', align='center')\nbars_after = plt.bar(x + bar_width, values_after, width=bar_width, label='After Filter', color='blue', align='center')\n# Highlight the 'TA' bar after filtering\nhighlighted_bar = plt.bar(x[3] + bar_width, values_after[3], width=bar_width, color='green', align='center')\n\n\n# Add labels and title\nplt.xlabel('ExterQual')\nplt.ylabel('Frequency (%)')\nplt.title('ExterQual value \"TA\" appears 4 times less frequently after filtering')\n\n# Determine if rotation is necessary\nmax_label_length = max(len(label) for label in categories)\nrotation = 45 if max_label_length > 10 else 0\n\nplt.xticks(x + bar_width / 2, categories, rotation=rotation, ha='right' if rotation else 'center')\nplt.legend()\nplt.tight_layout(pad=3.0)\nplt.ylim(0, max(max(values_before), max(values_after)) * 1.05)\n# Annotate the green bar with its value\nfor bar in highlighted_bar:\n    height = bar.get_height()\n    plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}', ha='center', va='bottom')"
}
```

**(Note: Real examples will include detailed matplotlib code reflecting actual data insights)**

# Notes

- Ensure the matplotlib code is self-contained and does not rely on external data files.
- Maintain the essence of the explanation when rephrasing to improve clarity and accessibility.
- Dynamically adjust the figure size, bar width, and other parameters to ensure the plot dimensions correspond to the data length and content.
        """

        user_prompt = f"""
        ${analysis_result}
        """
        explanation_and_code_result = await self.generate_chat_completion_async(system_prompt, user_prompt,
                                                                                temperature=0.5)
        return utils.extract_values_from_json(explanation_and_code_result)






