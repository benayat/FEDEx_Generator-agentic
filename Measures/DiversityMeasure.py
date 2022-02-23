from Measures.BaseMeasure import BaseMeasure, draw_histogram
from Measures.Bins import Bin, MultiIndexBin
import numpy as np
import utils
import matplotlib.pyplot as plt
import pandas as pd


def draw_bar(x: list, y: list, avg_line=None, items_to_bold=None, head_values=None, xname=None, yname=None, alpha=1.):
    width = 0.5
    ind = np.arange(len(x))
    x = x if utils.is_numeric(x) else [utils.to_valid_latex(i) for i in x]
    y = y if utils.is_numeric(y) else [utils.to_valid_latex(i) for i in y]
    if items_to_bold is not None:
        items_to_bold = items_to_bold if utils.is_numeric(items_to_bold) else [utils.to_valid_latex(i) for i in items_to_bold]

    bar = plt.bar(ind, y, width, alpha=alpha)
    plt.xticks(ind, tuple([str(i) for i in x]), rotation='vertical')

    if avg_line is not None:
        plt.axhline(avg_line, color='red', linewidth=1)
    if items_to_bold is not None:
        for item in items_to_bold:
            bar[x.index(item)].set_color('green')
    if head_values is not None:
        for i, col in enumerate(bar):
            yval = col.get_height()
            plt.text(col.get_x(), yval + .05, head_values[i])

    if xname is not None:
        plt.xlabel(utils.to_valid_latex(xname), fontsize=16)

    if yname is not None:
        plt.ylabel(utils.to_valid_latex(yname), fontsize=16)


def get_agg_func_from_name(name):
    operation = name.split("_")[-1]
    OP_TO_FUNC = {
        "count": np.sum,
        "sum": np.sum,
        "max": np.max,
        "min": np.min,
        "mean": np.mean,
    }

    return OP_TO_FUNC[operation]


def flatten_other_indexes(series, main_index):
    df = pd.DataFrame(series)
    df = df.reset_index(main_index)
    index_name = "_".join([ind for ind in df.index.names])
    df.index = df.index.to_flat_index()
    df.index.names = [index_name]
    df = df.set_index(main_index, append=True)
    return df[df.columns[0]]


class DiversityMeasure(BaseMeasure):
    def __init__(self):
        super().__init__()

    def draw_bar(self, bin_item: MultiIndexBin, influence_vals: dict = None, title=None):
        try:
            res_col = bin_item.get_actual_result_column()
            average = res_col.mean()
            aggregated_result = res_col.groupby(bin_item.get_bin_name()).agg(get_agg_func_from_name(res_col.name))
            max_values, max_influence = self.get_max_k(influence_vals, 1)
            max_value = max_values[0]
            bin_result = bin_item.result_column.copy()
            bin_result = flatten_other_indexes(bin_result, bin_item.get_bin_name())
            smallest_multi_bin = MultiIndexBin(bin_item.source_column, bin_result, 0)
            influences = self.get_influence_col(res_col, smallest_multi_bin, True)
            rc = res_col.reset_index([bin_item.get_bin_name()])

            relevant_items = rc[rc[bin_item.get_bin_name()] == max_value]
            relevant_influences = dict([(k, influences[k]) for k in relevant_items.index])
            max_values, max_influence = self.get_max_k(relevant_influences, 10)
            max_values = sorted(max_values)
            labels = set(aggregated_result.keys())

            MAX_BARS = 25
            if len(labels) > MAX_BARS:
                top_items, _ = self.get_max_k(influence_vals, MAX_BARS)
                labels = sorted(top_items)
            else:
                labels = sorted(labels)

            aggregate_column = [aggregated_result.get(item, 0) for item in labels]
            fig = plt.gcf()
            x, y = fig.get_size_inches()
            fig.set_size_inches(2*x, y)
            plt.subplot(1, 2, 1)
            plt.title(utils.to_valid_latex(title))
            draw_bar(labels, aggregate_column, average, [max_value],
                     xname=bin_item.get_bin_name() + " values", yname=bin_item.get_value_name())

            if len(max_values) > 1:
                plt.subplot(1, 2, 2)
                plt.title(utils.to_valid_latex(f"Zoom in on {bin_item.get_bin_name()} = {max_value}"))
                draw_bar(max_values, [bin_result[i].mean() for i in max_values],
                         average, max_values,
                         xname=f"{bin_item.get_base_name()} where {bin_item.get_bin_name()} = {max_value}", yname=bin_item.get_value_name(), alpha=0.5)

            plt.show()
        except Exception as e:
            plt.title(utils.to_valid_latex(title))
            draw_bar(list(bin_item.get_actual_result_column().index),
                     list(bin_item.get_actual_result_column()),
                     yname=bin_item.get_bin_name())
            plt.show()

    def interestingness_only_explanation(self,  source_col, result_col, col_name):
        return f"After employing the GroupBy operation we can see highly diverse set of values in the column '{col_name}'\n" \
               f"The variance" + \
               (f" was {self.calc_var(source_col)} and now it " if source_col is not None else "") + \
               f" is {self.calc_var(result_col)}"

    def calc_influence_col(self, current_bin: Bin):
        bin_values = current_bin.get_bin_values()
        source_col = current_bin.get_source_by_values(bin_values)
        res_col = current_bin.get_result_by_values(bin_values)
        score_all = self.calc_diversity(source_col, res_col)

        influence = []
        for value in bin_values:
            source_col_only_list = current_bin.get_source_by_values([b for b in bin_values if b != value])
            res_col_only_list = current_bin.get_result_by_values([b for b in bin_values if b != value])

            score_without_bin = self.calc_diversity(source_col_only_list, res_col_only_list)
            influence.append(score_all - score_without_bin)

        return influence

    def calc_var(self, pd_array):
        if utils.is_numeric(pd_array):
            return np.var(pd_array)

        appearances = (pd_array.value_counts()).to_numpy()
        mean = np.mean(appearances)
        return np.sum(np.power(appearances - mean, 2)) / len(appearances)

    def calc_diversity(self, source_col, res_col):
        var_res = self.calc_var(res_col)

        if source_col is not None:
            var_rel = self.calc_var(source_col)
        else:
            var_rel = 1.

        res = (var_res / var_rel) if var_rel != 0 else (0. if var_res == 0 else var_res)
        return 0 if np.isnan(res) else res

    def calc_measure_internal(self, bin: Bin):
        return self.calc_diversity(None if bin.source_column is None else bin.source_column.dropna(),
                                        bin.result_column.dropna())

    def build_explanation(self, current_bin: Bin, max_col_name, max_value, source_name):
        res_col = current_bin.get_actual_result_column()
        if utils.is_categorical(res_col):
            return ""
        var = self.calc_var(res_col)
        if current_bin.name == "NumericBin":
            max_value_numeric = max_value
        elif current_bin.name == "CategoricalBin":
            max_value_numeric = res_col.value_counts()[max_value]
        elif current_bin.name == "MultiIndexBin":
            bin_values = current_bin.get_result_by_values([max_value])
            operation = get_agg_func_from_name(max_col_name)

            max_value_numeric = operation(bin_values)
            max_col_name = current_bin.get_bin_name()
        elif current_bin.name == "NoBin":
            result_column = current_bin.get_actual_result_column()
            max_value_numeric = max_value
            max_value = result_column.index[result_column == max_value].tolist()
            max_col_name = result_column.index.name
        elif type(current_bin) == Bin:
            raise Exception("Bin is not supported")
        else:
            raise Exception(f"unknown bin type {current_bin.name}")

        x = (max_value_numeric - np.mean(res_col)) / np.sqrt(var)
        expl = f"\\textbf{{Explanation:}} See that the column \\textbf{{{max_col_name}}}, presents a significant diversity.\n " \
               f"In particular, groups with \\textbf{{'{max_col_name}'='{max_value}'}} (in green) have a relatively \\textbf{{{'low' if x < 0 else 'high'}}} '{res_col.name}' value:\n" \
               f'{utils.smart_round(np.abs(x))} standard deviation {"lower" if x < 0 else "higher"} than the mean ({utils.smart_round(np.mean(res_col))})'

        return expl