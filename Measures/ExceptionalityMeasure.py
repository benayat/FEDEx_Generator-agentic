import numpy as np
import matplotlib.pyplot as plt
import utils
from Measures.BaseMeasure import BaseMeasure
from Measures.Bins import Bin


class ExceptionalityMeasure(BaseMeasure):
    def __init__(self):
        super().__init__()

    def draw_bar(self, bin_item: Bin, influence_vals: dict = None, title=None):
        res_col = bin_item.get_actual_result_column()
        src_col = bin_item.get_actual_source_column()

        res_probs = res_col.value_counts(normalize=True)
        src_probs = None if src_col is None else src_col.value_counts(normalize=True)
        labels = set(src_probs.keys()).union(res_probs.keys())

        MAX_BARS = 25
        if len(labels) > MAX_BARS:
            labels, _ = self.get_max_k(influence_vals, MAX_BARS)

        labels = sorted(labels)
        probabilities = [100. * src_probs.get(item, 0) for item in labels]
        probabilities2 = [100. * res_probs.get(item, 0) for item in labels]

        width = 0.35
        ind = np.arange(len(labels))

        fig, ax = plt.subplots()
        result_bar = ax.bar(ind + width, probabilities2, width, label="After")
        ax.bar(ind, probabilities, width, label="Before")
        if influence_vals:
            max_label, _ = self.get_max_k(influence_vals, 1)
            max_label = max_label[0]
            result_bar[labels.index(max_label)].set_color('green')

        ax.set_xticks(ind + width / 2)
        label_tags = tuple([utils.to_valid_latex(bin_item.get_bin_representation(i)) for i in labels])
        tags_max_length = max([len(tag) for tag in label_tags])
        ax.set_xticklabels(label_tags, rotation='vertical' if tags_max_length >= 4 else 'horizontal')
        plt.legend(loc='best')
        plt.xlabel(utils.to_valid_latex(bin_item.get_bin_name() + " values"), fontsize=16)
        plt.ylabel("frequency(\\%)", fontsize=16)

        if title is not None:
            ax.set_title(utils.to_valid_latex(title), loc='center', wrap=True)

        plt.show()

    def interestingness_only_explanation(self,  source_col, result_col, col_name):
        if utils.is_categorical(source_col):
            vc = source_col.value_counts()
            source_max = utils.max_key(vc)
            vc = result_col.value_counts()
            result_max = utils.max_key(vc)
            return f"The distribution of column '{col_name}' changed significantly.\n" \
                   f"The most common value was {source_max} and now it is {result_max}."

        std_source = np.sqrt(np.var(source_col))
        mean_source = np.mean(source_col)
        std = np.sqrt(np.var(result_col))
        mean = np.mean(result_col)

        return f"The distribution of column '{col_name}' changed significantly.\n" \
               f" The mean was {mean_source:.2f} and the standard " \
               f"deviation was {std_source:.2f}, and now the mean is {mean:.2f} and the standard deviation is {std:.2f}."

    def calc_measure_internal(self, bin: Bin):
        return ExceptionalityMeasure.kstest(bin.source_column.dropna(), bin.result_column.dropna()) #/ len(source_col.dropna().value_counts())

    @staticmethod
    def kstest(s, r):
        s = np.array(s)
        s = s[s == s]
        r = np.array(r)
        r = r[r == r]
        return 0 if len(r) == 0 else utils.ks_2samp(s, r).statistic

    def calc_influence_col(self, current_bin: Bin):
        bin_values = current_bin.get_bin_values()
        source_col = current_bin.get_source_by_values(bin_values)
        res_col = current_bin.get_result_by_values(bin_values)
        score_all = ExceptionalityMeasure.kstest(source_col, res_col)
        influence = []
        for value in bin_values:
            source_col_only_list = current_bin.get_source_by_values([b for b in bin_values if b != value])
            res_col_only_list = current_bin.get_result_by_values([b for b in bin_values if b != value])

            score_without_bin = ExceptionalityMeasure.kstest(source_col_only_list, res_col_only_list)
            influence.append(score_all - score_without_bin)

        return influence

    def build_explanation(self, current_bin: Bin, col_name, max_value, source_name):
        source_col = current_bin.get_actual_source_column()
        res_col = current_bin.get_actual_result_column()

        res_probs = res_col.value_counts(normalize=True)
        source_probs = source_col.value_counts(normalize=True)
        for bin_value in current_bin.get_bin_values():
            res_probs[bin_value] = res_probs.get(bin_value, 0)
            source_probs[bin_value] = source_probs.get(bin_value, 0)

        additional_explanation = []
        if current_bin.name == "NumericBin":
            values = current_bin.get_bin_values()
            index = values.index(max_value)

            values_range_str = "below {}".format(utils.format_bin_item(values[1])) if max_value == 0 else \
                "above {}".format(utils.format_bin_item(max_value)) if index == len(values) - 1 else \
                "between {} and {}".format(utils.format_bin_item(values[index]), utils.format_bin_item(values[index + 1]))
            factor = res_probs.get(max_value, 0) / source_probs[max_value]
            if factor < 1:
                factor = 1 / factor
                proportion = "less"
            else:
                proportion = "more"
            additional_explanation.append(f"See that the column \\textbf{{\"{col_name}\"}} presents a significant change in distribution.\n"
                                          f"In particular, values \\textbf{{{values_range_str}}} (in green) appears {utils.smart_round(factor)} times {proportion} "
                                          f"than before (was {100 * source_probs[max_value]:.1f}\\% now {100 * res_probs.get(max_value, 0):.1f}\\%)")
        else:
            factor = res_probs.get(max_value, 0) / source_probs[max_value]
            if factor < 1:
                proportion = "less"
                factor = 1 / factor
            else:
                proportion = "more"

            source_prob = 100 * source_probs[max_value]
            res_prob = 100 * res_probs.get(max_value, 0)
            additional_explanation.append(f"See that the column \\textbf{{\"{col_name}\"}} presents a significant change in distribution.\n"
                                          f"In particular, \\textbf{{\"{utils.to_valid_latex(max_value)}\"}} (in green)"
                                          f" appears {utils.smart_round(factor)} times {proportion} "
                                          f"than before (was {source_prob:.1f}\\% now {res_prob:.1f}\\%)")

        influence_top_example = ", ".join(additional_explanation)

        return "\\textbf{Explanation:} " + influence_top_example




