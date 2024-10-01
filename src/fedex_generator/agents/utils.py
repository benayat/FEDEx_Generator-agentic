import pandas as pd

def summarize_series(series: pd.Series) -> str:
    if series.dtype == 'float' or series.dtype == 'int':
        summary = f"mean={series.mean():.2f}, min={series.min()}, max={series.max()}"
    else:
        value_counts = series.value_counts().head(5).to_dict()
        summary = ', '.join([f"{k}: {v}" for k, v in value_counts.items()])
    return summary

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
        with open('generated_visualization.py', 'w') as file:
            file.write(code_str)

    # Execute the code with source_column and target_column in the context
    exec(code_str)