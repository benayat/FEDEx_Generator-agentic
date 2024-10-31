import pandas as pd
import json


def summarize_series(series: pd.Series) -> str:
    if series.dtype == 'float' or series.dtype == 'int':
        summary = f"mean={series.mean():.2f}, min={series.min()}, max={series.max()}"
    else:
        value_counts = series.value_counts().head(5).to_dict()
        summary = ', '.join([f"{k}: {v}" for k, v in value_counts.items()])
    return summary


def get_value_counts_for_filter(bin_item):
    if bin_item is not None:
        src_value_counts = bin_item.get_binned_source_column().value_counts(normalize=True) if bin_item.get_binned_source_column() is not None else None
        res_value_counts = bin_item.get_binned_result_column().value_counts(normalize=True)
        return src_value_counts, res_value_counts
    else:
        return "N/A", "N/A", "N/A"


def extract_classes_from_bin(bin_item):
    if bin_item is not None:
        if bin_item.name == 'CategoricalBin':
            return [key for key in bin_item.get_binned_result_column().keys()]
        elif bin_item.name == 'NumericBin':
            src_value_counts, res_value_counts = get_value_counts_for_filter(bin_item)
            if src_value_counts is None:
                src_value_counts = pd.Series(dtype='float64')
            if res_value_counts is None:
                res_value_counts = pd.Series(dtype='float64')
            raw_classes = sorted(set(src_value_counts.keys()).union(res_value_counts.keys()))
            raw_classes = [bin_item.get_bin_representation(i) for i in raw_classes]
            return [s.replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace(',','-') for s in raw_classes]
    else:
        return "N/A"
def extract_classes_for_groupby(bin_item):
    if bin_item.name == 'CategoricalBin':
        return [key for key in bin_item.get_binned_result_column().keys()]
    elif bin_item.name == 'NumericBin':
        return bin_item.get_binned_result_column().keys()

def extract_classes_for_filter(bin_item):
    src_value_counts, res_value_counts = get_value_counts_for_filter(bin_item)
    raw_classes = sorted(set(src_value_counts.keys()).union(res_value_counts.keys()))
    return [bin_item.get_bin_representation(i) for i in raw_classes]

def extract_actual_values_for_groupby(bin_item):
    if bin_item is not None:
        # return bin_item.get_binned_result_column().to_json()
        return bin_item.get_binned_result_column().values

def extract_frequency_values_for_filter(bin_item):
    src_value_counts, res_value_counts = get_value_counts_for_filter(bin_item)
    raw_classes = sorted(set(src_value_counts.keys()).union(res_value_counts.keys()))
    legend_before_value = [f"{c} ({src_value_counts.get(c, 0)*100:.2f})" for c in raw_classes]
    legend_after_value = [f"{c} ({res_value_counts.get(c, 0)*100:.2f})" for c in raw_classes]
    return legend_before_value, legend_after_value


def extract_classes_and_values_by_bin_and_type(bin_item, operation_type):
    if operation_type == 'Filter':
        classes = extract_classes_from_bin(bin_item)
        before_values, after_values = extract_frequency_values_for_filter(bin_item)
        return {cls: {"before": before, "after": after} for cls, before, after in zip(classes, before_values, after_values)}
    elif operation_type == 'Groupby':
        # classes = extract_classes_from_bin(bin_item)
        classes = extract_classes_for_groupby(bin_item)
        actual_values = extract_actual_values_for_groupby(bin_item)
        return {cls: actual for cls, actual in zip(classes, actual_values)}
    else:
        return "N/A"


def extract_values_from_json(json_str):
    data = json.loads(json_str.strip('```json').strip('```'))
    rephrased_explanation = data.get('rephrased_explanation')
    matplotlib_code = data.get('matplotlib_code')
    return rephrased_explanation, matplotlib_code


def process_analysis_result(analysis_result):
    """
    Processes the analysis result to extract relevant details for generating the explanation and code snippet.

    Parameters:
    analysis_result (dict): A dictionary containing the following keys:
        - 'explanation' (str): The basic explanation of the analysis.
        - 'action_title' (str): The title of the user action.
        - 'bin' (MultiIndexBin): The bin object.
        - 'values_to_influence_dict' (dict): A dictionary mapping unique values to their influence.
        - 'column_score' (float): The score of the column.

    Returns:
    dict: A dictionary containing the processed analysis details with the following keys:
        - 'action_title' (str): The title of the user action.
        - 'column_name' (str): The name of the column.
        - 'bin_type' (str): The type of the bin.
        - 'explanation' (str): The basic explanation of the analysis.
        - 'source_column_summary' (str): Summary statistics of the source column in JSON format.
        - 'class_to_value' (dict): A dictionary mapping classes to their values before and after filtering or grouping.
        - 'result_column_summary' (str): Summary statistics of the result column in JSON format.
        - 'values_to_influence' (dict): A dictionary mapping unique values to their influence.
        - 'column_score' (float): The score of the column.
    """
    action_title = analysis_result['action_title']
    column_name = analysis_result['bin'].get_bin_name()
    bin_type = analysis_result['bin'].name
    explanation = analysis_result['explanation']
    source_column = analysis_result['bin'].get_binned_source_column()
    source_column_summary = source_column.describe().to_json() if source_column is not None else "N/A"

    # class_to_value_source = extract_relevant_data_from_column_by_bin(source_column, bin_type)
    result_column = analysis_result['bin'].get_binned_result_column()
    result_column_summary = result_column.describe().to_json() if result_column is not None else "N/A"
    class_to_value = extract_classes_and_values_by_bin_and_type(analysis_result['bin'], "Filter" if "filtered" in action_title else "Groupby")
    values_to_influence = analysis_result['value_to_influences_dict'] if analysis_result[
                                                                             'value_to_influences_dict'] is not None else "N/A"
    column_score = analysis_result['column_score']
    return {
        'action_title': action_title,
        'column_name': column_name,
        'bin_type': bin_type,
        'explanation': explanation,
        'source_column_summary': source_column_summary,
        'class_to_value': class_to_value,
        'result_column_summary': result_column_summary,
        'values_to_influence': values_to_influence,
        'column_score': column_score
    }


def execute_generated_code(code_str, debug=False) -> None:
    """
    Executes the generated Python code for creating a Matplotlib visualization.

    Parameters:
    code_str (str): The Python code to be executed.

    Returns:
    None
    """
    # Save the code to a file (usefull sometimes for debuging)
    if debug:
        print("code_str:", code_str)
        with open('generated_visualization.py', 'w') as file:
            file.write(code_str)

    # Execute the code with source_column and target_column in the context
    exec(code_str)
