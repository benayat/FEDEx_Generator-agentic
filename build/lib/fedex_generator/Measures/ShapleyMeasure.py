import math

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from itertools import combinations
import math
import inspect

from fedex_generator.commons import utils
from fedex_generator.Measures.BaseMeasure import BaseMeasure, START_BOLD, END_BOLD
from fedex_generator.Measures.Bins import Bin


def is_name_informative(name):
    import string
    return any([char in string.ascii_letters for char in name])

def get_calling_params_name(item):
    frames = inspect.stack()
    highest_var = None
    for frame in frames[2:]:
        prev_locals = frame[0].f_locals
        for var_name in prev_locals:
            if id(item) == id(prev_locals[var_name]) and is_name_informative(var_name):
                highest_var = var_name
    return highest_var


class ShapleyMeasure(BaseMeasure):
    def __init__(self):
        super().__init__()




    def draw_bar(self, bin_item: Bin, influence_vals: dict = None, title=None, ax=None, score=None,
                 show_scores: bool = False):
        res_col = bin_item.get_binned_result_column()
        src_col = bin_item.get_binned_source_column()

        res_probs = res_col.value_counts(normalize=True)
        src_probs = None if src_col is None else src_col.value_counts(normalize=True)
        labels = set(src_probs.keys()).union(res_probs.keys())

        MAX_BARS = 25
        if len(labels) > MAX_BARS:
            labels, _ = self.get_max_k(influence_vals, MAX_BARS)

        labels = sorted(labels)
        probabilities = [100. * src_probs.get(item, 0) for item in labels]
        probabilities2 = [100 * res_probs.get(item, 0) for item in labels]

        width = 0.35
        ind = np.arange(len(labels))

        result_bar = ax.bar(ind + width, probabilities2, width, label="After")

        ax.bar(ind, probabilities, width, label="Before")
        ax.legend(loc='best')
        if influence_vals:
            max_label, _ = self.get_max_k(influence_vals, 1)
            max_label = max_label[0]
            result_bar[labels.index(max_label)].set_color('green')

        ax.set_xticks(ind + width / 2)
        label_tags = tuple([utils.to_valid_latex(bin_item.get_bin_representation(i)) for i in labels])
        tags_max_length = max([len(tag) for tag in label_tags])
        ax.set_xticklabels(label_tags, rotation='vertical' if tags_max_length >= 4 else 'horizontal')

        ax.set_xlabel(utils.to_valid_latex(bin_item.get_bin_name() + " values"), fontsize=20)
        ax.set_ylabel("frequency(\\%)", fontsize=16)


        if title is not None:
            if show_scores:
                ax.set_title(f'score: {score}\n {utils.to_valid_latex(title)}', fontdict={'fontsize': 14})
            else:
                ax.set_title(utils.to_valid_latex(title), fontdict={'fontsize': 14})

        ax.set_axis_on()
        return bin_item.get_bin_name() ####

    def interestingness_only_explanation(self, source_col, result_col, col_name):
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
        return ExceptionalityMeasure.kstest(bin.source_column.dropna(),
                                            bin.result_column.dropna())  # / len(source_col.dropna().value_counts())

    def sat(self, d1, d2, attr, cond, k, cont, att, rem_attr=None, n=1):
        total = 0
        max_val = 0
        # if n == 1:
            # mem = self.mem1
        # else:
            # mem = self.mem2
        ret = 0

        for i in range(2, k):
            # if k in mem.keys():
                # total += mem[k]
                # if mem[k] > max_val:
                    # max_val = mem[k]
                # continue
            subsets_1 = list(combinations(d1.index, i))
            subset_dfs_1 = [d1.loc[list(subset)] for subset in subsets_1]

            subsets_2 = list(combinations(d2.index, k-i))
        # print(len(subsets_2))
            subset_dfs_2 = [d2.loc[list(subset)] for subset in subsets_2]
            for sdf1 in subset_dfs_1:
                # if str(sdf1) in self.mem_comb.keys():
                    # ret += self.mem_comb[str(sdf1)]
                    # continue
                num = 0
                sum = 0
                # print(total)
                for sdf2 in subset_dfs_2:
                    df_join = pd.merge(sdf1, sdf2, on=attr, how='inner')
                    if df_join.shape[0]:
                        
                    # print(f'--------------------------------\n{sdf1}\n**************\n{sdf2}')
                        # total += 1
                        val = df_join[att].agg(cont)
                        if val > max_val:
                            max_val = val
                        # mem[k] = val
                        sum += val
                        total += 1
                        num += 1
                ret += sum / (num)
                # ret = self.mem_comb[str(sdf1)]
                
        # if n == 1:
            # self.mem1 = mem 
        # else:
            # self.mem2 = mem 
        if total == 0:
            return 0
        return ret# sum / num# + np.log2(total)# / num)
    
    def shapley(self, d1, d1_exc, d2, attr, cond, sat,cont, att):
        # m = min(d1.shape[0] + d2.shape[0], 3*d1.shape[0])
        m = 5
        including = 0
        excluding = 0
        self.mem1 = {}
        self.mem2 = {}
        for k in range(m):
            if k not in self.including.keys():
                self.including[k] =  ((math.factorial(k)*math.factorial(m - k - 1))/math.factorial(m)) * sat(d1, d2, attr, cond, k, n=1, cont=cont, att=att)
            including += self.including[k]
            # including += ((math.factorial(k)*math.factorial(m - k - 1))/math.factorial(m)) * sat(d1, d2, attr, cond, k, n=1, cont=cont, att=att)
            excluding += ((math.factorial(k)*math.factorial(m - k - 1))/math.factorial(m)) * sat(d1_exc, d2, attr, cond, k, n=2, cont=cont, att=att)
        return including - excluding
    
    def get_most_contributing_fact(self, op, d1, d2, attr, cond, consider='right', top_k=1, cont=None, att=None):
        if att == None:
            cont = 'count'
            att = list(list(facts.values())[0].reset_index().to_dict().items())[0]
        self.consider = consider
        self.including = {}
        self.mem_comb = {}
        top_shapley = 0
        top_fact = None
        facts = {}
        if consider == 'right':
            d1_consider = d2
            d2_consider = d1
        else:
            d1_consider = d1
            d2_consider = d2
        d2_consider = d2_consider[d2_consider[attr].isin(pd.merge(d1_consider, d2_consider, on=attr, how='inner')[attr])]
        for i in (d1_consider.index):
            # print(f'{i}')#, {d1_consider.loc[i][attr]}, {pd.merge(d1_consider, d2_consider, on=attr, how='inner')[attr].values}')
            if d1_consider.loc[i][attr] not in pd.merge(d1_consider, d2_consider, on=attr, how='inner')[attr].values:
                v_shapley = 0
            else:
                d_tmp = d1_consider.drop(index=i, inplace=False)
                v_shapley = self.shapley(d1_consider, d_tmp, d2_consider, attr, cond, self.sat, cont, att)
            facts[str(d1_consider.loc[i])] = [v_shapley, d1_consider[d1_consider.index == i]]
            if v_shapley > top_shapley:
                top_shapley = v_shapley
                top_fact = d1_consider[d1_consider.index == i]
        pass
        facts = {k: v[1] for k, v in sorted(facts.items(), key=lambda item: -(item[1][0]))}
    # print(f'The top fact here is:\n{str(top_fact)}')
        fig, axes = plt.subplots(1, figsize=(14, 2))

        axes.axis('off')
        # axes[1].axis('off')
        d1_name = r'$\bf{{{}}}$'.format(utils.to_valid_latex(d1.df_name, True))
        d2_name = r'$\bf{{{}}}$'.format(utils.to_valid_latex(d2.df_name, True))
        if consider == 'left':
            consider = d1_name
        else:
            consider = d2_name
        bold_attr = r'$\bf{{{}}}$'.format(utils.to_valid_latex(attr), True)
        
        if top_k == 1:
            explanation = f'When joining the DataFrames\nThe most significant fact from DataFrame {consider} is:\n'
        else:
            explanation = f'When joining the DataFrames\nThe {top_k} most significant facts from DataFrame {consider} in descending order are:'
        # explanation = f'The tuple '
        fact_exp = ''
        n = 0
        while n < top_k:
            i = 0
            explanation += f'\n'
            for k, v in list(facts.values())[n].reset_index().to_dict().items():
                if i != 0:
                    explanation += ', '
                i += 1
            # v = r'$\bf{{{}}}$'.format(utils.to_valid_latex(v[0], True))
                # r'$\bf{{{}}}$'.format(utils.to_valid_latex(f'{k}={v[0]}', True))
                
                # explanation += f'{r'$\bf{{{}}}$'.format(utils.to_valid_latex(k, True))}={r'$\bf{{{}}}$'.format(utils.to_valid_latex(v[0], True))}'
            n += 1
        # explanation += f'\nfrom DataFrame {consider} is {r'$\bf{very}$'} {r'$\bf{significant}$'} for this result.'
        props1 = dict(boxstyle='round', facecolor='white', alpha=0.5)
        axes.text(0.5, 0.5, explanation, transform=axes.transAxes, fontsize=20,
        verticalalignment='center', horizontalalignment='center', bbox=props1)

        props2 = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # axes[1].text(0.5, 0.5, fact_exp, transform=axes[1].transAxes, fontsize=14,
        # verticalalignment='center', horizontalalignment='center', bbox=props2)
        fig.show()
        # txt = ax.text(x, y, n, fontsize=10)
        return top_fact, facts

    def calc_influence_col(self, current_bin: Bin):
        self.including = None
        bin_values = current_bin.get_bin_values()
        source_col = current_bin.get_source_by_values(bin_values)
        res_col = current_bin.get_result_by_values(bin_values)
        if len(bin_values) > 15:
            bin_values = list(source_col.value_counts().nlargest(15).keys())
        score_all = ExceptionalityMeasure.kstest(source_col, res_col)
        influence = []
        for value in bin_values:
            source_col_only_list = current_bin.get_source_by_values([b for b in bin_values if b != value])
            res_col_only_list = current_bin.get_result_by_values([b for b in bin_values if b != value])

            score_without_bin = ExceptionalityMeasure.kstest(source_col_only_list, res_col_only_list)
            influence.append(score_all - score_without_bin)

        return influence

    def build_operation_expression(self, source_name):
        from fedex_generator.Operations.Filter import Filter
        from fedex_generator.Operations.Join import Join

        if isinstance(self.operation_object, Filter):
            return f'Dataframe {self.operation_object.source_name}, ' \
                   f'filtered on attribute {self.operation_object.attribute}'
        elif isinstance(self.operation_object, Join):
            return f'{self.operation_object.right_name} joined with {self.operation_object.left_name} by {self.operation_object.attribute}'

    def build_explanation(self, current_bin: Bin, col_name, max_value, source_name):
        source_col = current_bin.get_binned_source_column()
        res_col = current_bin.get_binned_result_column()

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
                    "between {} and {}".format(utils.format_bin_item(values[index]),
                                               utils.format_bin_item(values[index + 1]))
            factor = res_probs.get(max_value, 0) / source_probs[max_value]   
            proportion = "less" if factor < 1 else "more"  
            if (factor < 1 and factor > 0):
                factor = 1 / factor
            if factor == 0:
                additional_explanation.append(
                    f"{START_BOLD}{utils.to_valid_latex(col_name, True)}{END_BOLD} values "
                    f"{START_BOLD}{utils.to_valid_latex(values_range_str, True)}{END_BOLD}\n"
                    f"are {START_BOLD}no{END_BOLD} {START_BOLD}longer{END_BOLD} {START_BOLD}exist{END_BOLD} (was {round(source_probs[max_value],3)*100}%)")
            else:
                appear_test = f'{utils.smart_round(factor)} times {proportion}'
                additional_explanation.append(
                    f"{START_BOLD}{utils.to_valid_latex(col_name, True)}{END_BOLD} values "
                    f"{START_BOLD}{utils.to_valid_latex(values_range_str, True)}{END_BOLD} (in green)\n"
                    f"appear {START_BOLD}"
                    f"{utils.to_valid_latex(appear_test, True)}"
                    f"{END_BOLD} than before")
        else:
            factor = res_probs.get(max_value, 0) / source_probs[max_value]

            source_prob = 100 * source_probs[max_value]
            res_prob = 100 * res_probs.get(max_value, 0)
            max_value_rep = current_bin.get_bin_representation(max_value)

            if factor == 0:
                proportion_sentes = f' frequency was {source_prob:.1f}% now {res_prob:.1f}%'
            elif factor < 1:
                proportion = "less"
                factor = 1 / factor
                proportion_sentes = f"appear {START_BOLD} " \
                                    f"{utils.to_valid_latex(f'{utils.smart_round(factor)} times {proportion}', True)}" \
                                    f" {END_BOLD} than before"
            else:
                proportion = "more"
                proportion_sentes = f"appear {START_BOLD} " \
                                    f"{utils.to_valid_latex(f'{utils.smart_round(factor)} times {proportion}', True)}" \
                                    f" {END_BOLD} than before"

            additional_explanation.append(
                f"{START_BOLD}{utils.to_valid_latex(col_name, True)}{END_BOLD} value "
                f"{START_BOLD}{utils.to_valid_latex(max_value_rep, True)}"
                f"{END_BOLD} (in green)\n{proportion_sentes}")

        influence_top_example = ", ".join(additional_explanation)
        
        return influence_top_example
