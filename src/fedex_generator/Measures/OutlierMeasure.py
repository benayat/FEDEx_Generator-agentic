from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from fedex_generator.Measures.BaseMeasure import BaseMeasure, START_BOLD, END_BOLD
from fedex_generator.Measures.Bins import Bin, MultiIndexBin
from fedex_generator.commons import utils
ALPHA = 0.9
START_BOLD = r'$\bf{'
END_BOLD = '}$'
# BETA = 0.1
def proportion(series):
    c = series.count()
    return (series.count())/(series.count().sum())

class OutlierMeasure(BaseMeasure):
    def __init__(self):
        super().__init__()
    


    def calc_influence_pred(self, df_before, df_after, target, dir):
        try:
            target_inf = ((df_before[target] - df_after[target]) * dir)/(df_before[target] + df_after[target])
        except:
            return -1
        holdout_inf = 0
        for i in df_before.index:
            if i != target:
                try:
                    holdout_inf += np.sqrt(abs(df_after[i] - df_before[i]))
                except:
                    return -1
        return ALPHA * (target_inf) - (1-ALPHA) * (holdout_inf/(len(df_before.index)))
    def merge_preds(self, df_agg, df_in, df_in_consider, preds, g_att, g_agg, agg_method, target, dir):
        final_pred = []
        final_inf = 0.001
        df_final = df_in.copy()
        df_final_consider = df_in_consider.copy()
        final_agg_df = None
        final_filter = False
        final_filter_df = False

        prev_attrs = []

        for p in preds:
            attr, i, score, kind, rank = p
            if attr in prev_attrs:
                continue
            prev_attrs.append(attr)
            if(kind == 'bin'):
                bin = i
                final_filter_test = final_filter |  ((df_in_consider[attr] < bin[0]) | (df_in_consider[attr] >= bin[1]))
                final_filter_df_test = final_filter_df |  ((df_in[attr] < bin[0]) | (df_in[attr] >= bin[1]))
                # df_exc_consider = df_final_consider[((df_final_consider[attr] < bin[0]) | (df_final_consider[attr] >= bin[1]))]
                # df_exc_final = df_final[((df_final[attr] < bin[0]) | (df_final[attr] >= bin[1]))]
            else:
                final_filter_test = final_filter |  (df_in_consider[attr] != i)
                final_filter_df_test = final_filter_df |  (df_in[attr] != i)
                # df_exc_consider = df_final_consider[df_final_consider[attr] != i]
                # df_exc_final = df_final[df_final[attr] != i]
            df_exc_consider = df_in_consider[final_filter_test]
            df_exc_final = df_in[final_filter_df_test]
            agged_val_consider = df_exc_consider.groupby(g_att)[g_agg].agg(agg_method)
            agged_val = df_exc_final.groupby(g_att)[g_agg].agg(agg_method)
            if(agg_method == 'count'):
                agged_val_consider = agged_val_consider/agged_val_consider.sum()
                agged_val = agged_val/agged_val.sum()
            # inf = self.calc_influence_pred(df_agg_consider, agged_val, target, dir)/(np.sqrt(df_in_consider.shape[0]/df_in_exc.shape[0]) + 0.01)
            inf = self.calc_influence_pred(df_agg, agged_val_consider, target, dir)/pow((df_in_consider.shape[0]/(df_exc_consider.shape[0] + 1)), 2)
            # print(f'inf: {inf/final_inf}')
            if inf/final_inf > 1.3:
                final_pred.append((attr, i, rank))
                final_inf = inf
                df_final = df_exc_final
                df_final_consider = df_exc_consider
                final_agg_df = agged_val
                final_filter = final_filter_test
                final_filter_df = final_filter_df_test
            else:
                break
        return final_pred, final_inf, final_agg_df


    def explain_outlier(self, df_agg, df_in, g_att, g_agg, agg_method, target, dir, control=None, hold_out = [],k=1):
        # attrs = df_in.select_dtypes(include='number').columns.tolist()#[:10]
        attrs = df_in.columns
        attrs = [a for a in attrs if a not in hold_out + [g_att, g_agg] ]
        exps = {}
        agg_title = agg_method
        if(agg_method == 'count'):
                df_agg = df_agg/df_agg.sum()


        worst = 0
        # top_bin_all = None
        top_inf_all = -1
        # top_attr = None
        df = None
        predicates = []
        if control == None:
            control=list(df_agg.index)
        df_in_consider = df_in#[(df_in[g_att].isin(control))|(df_in[g_att] == target)]
        # if dir == -1:
        #     df_in_consider = df_in_consider[df_in_consider[g_agg] < df_agg[target]]
        # else:
        #     df_in_consider = df_in_consider[df_in_consider[g_agg] > df_agg[target]]
        df_agg_consider = df_agg#[control]
        # df_agg_consider[target] = df_agg[target]
        for attr in attrs:

            dtype = df_in[attr].dtype.name
            if dtype in ['int64', 'float64']:
                if(df_in[g_att].dtype.name in ['int64', 'float64'] and df_in[g_att].corr(df_in[attr]) > 0.7) or (df_in[g_agg].dtype.name in ['int64', 'float64'] and df_in[g_agg].corr(df_in[attr]) > 0.7):
                    continue
            # if attr != 'relationship':
            #     continue
            series = df_in[attr]
            dtype = df_in[attr].dtype.name
            flag = False
            df_in_consider_attr = df_in_consider[[g_att, g_agg, attr]]
            if dtype not in ['float64']:
                # if attr == 'explicit':
                    # pass
                vals = series.value_counts()
                if dtype != 'int64' or len(vals) < 20:
                    if len(vals) > 50:
                        continue
                    flag = True
                    top_df = None
                    top_inf = 0
                    top_bin = None
                    for i in vals.index:
                        # df_in_target_exc = df_in[((df_in[g_att].isin(target))&(df_in[attr] != i))]
                        
                        df_in_target_exc = df_in_consider_attr[(df_in_consider_attr[attr] != i)]
                        agged_val = df_in_target_exc.groupby(g_att)[g_agg].agg(agg_method)
                        if(agg_method == 'count'):
                            agged_val = agged_val/agged_val.sum()
                        # agged_df = df_agg.copy()
                        # agged_df[target]=agged_val
                        inf = self.calc_influence_pred(df_agg_consider, agged_val,target, dir)/((df_in_consider.shape[0]/(df_in_target_exc.shape[0] + 0.01)))
                       
                        exps[(attr,i)]=inf
                        # if inf > top_inf:
                        #     top_inf = inf
                        #     top_bin = i
                        #     top_df = agged_val
                        predicates.append((attr, i, inf, 'cat', None))
                    # if top_inf > top_inf_all:
                    #     top_inf_all = top_inf
                    #     top_bin_all = top_bin
                    #     top_attr = attr
                    #     df = top_df
                    
            n_bins = 20
            if not flag:
                _, bins = pd.cut(series, n_bins, retbins=True, duplicates='drop')
                df_bins_in = pd.cut(df_in_consider_attr[attr], bins=bins).value_counts(normalize=True).sort_index()#.rename('idx')
                top_df = None
                top_inf = 0
                top_bin = None
                i = 1
                for bin in df_bins_in.keys():
                    # df_in_exc = df_in[((df_in[g_att].isin(target)) & ((df_in[attr] < bin.left) | (df_in[attr] >= bin.right)))]
                    new_bin = (float("{:.2f}".format(bin.left)), float("{:.2f}".format(bin.right)))
                    df_in_exc = df_in_consider_attr[((df_in_consider_attr[attr] < new_bin[0]) | (df_in_consider_attr[attr] >= new_bin[1]))]
                    agged_val = df_in_exc.groupby(g_att)[g_agg].agg(agg_method)
                    if(agg_method == 'count'):
                        agged_val = agged_val/agged_val.sum()

                    inf = self.calc_influence_pred(df_agg_consider, agged_val, target, dir)/((df_in_consider_attr.shape[0]/df_in_exc.shape[0]) + 1)
                    exps[(attr,(new_bin[0],new_bin[1]))]=inf

                    predicates.append((attr, new_bin, inf, 'bin', i))
                    i += 1
                    
                    

        predicates.sort(key=lambda x:-x[2])
        final_pred, final_inf, final_df = self.merge_preds(df_agg_consider,df_in, df_in_consider, predicates, g_att, g_agg, agg_method, target, dir)
        if final_df is None:
            return "There was no explanation."
        new_df_agg = df_agg.copy()
        new_df_agg[control] = final_df[control]
        new_df_agg[target] = final_df[target]
        final_pred_by_attr = {}
        for a, i, rank in final_pred:
            if a not in final_pred_by_attr.keys():
                final_pred_by_attr[a] = []    
            final_pred_by_attr[a].append((i, rank))
        fig, ax = plt.subplots(layout='constrained', figsize=(7, 7))
        x1 = list(df_agg.index)
        ind1 = np.arange(len(x1))
        y1 = df_agg.values
        

        x2 = list(final_df.index)
        ind2 = np.arange(len(x2))
        y2 = final_df.values
        # if agg_method == 'count':
        #     agg_method ='proportion'
        #     y1 = y1/y1.sum()
        #     y2 = y2/y2.sum()
        if dir == 1:
            highlow = r'$\bf{{{}}}$'.format(utils.to_valid_latex('high'))
            incdec = r'$\bf{{{}}}$'.format(utils.to_valid_latex('decreases'))
        else:
            highlow = r'$\bf{{{}}}$'.format(utils.to_valid_latex('low'))
            incdec = r'$\bf{{{}}}$'.format(utils.to_valid_latex('increases'))

        # explanation = f'The highlighted outlier might have been\ncaused by rows that follow this predicate:\n\n'
        bold_agg_method = r'$\bf{{{}}}$'.format(utils.to_valid_latex(agg_method))
        bold_g_agg = r'$\bf{{{}}}$'.format(utils.to_valid_latex(g_agg))
        bold_g_att_target = r'$\bf{{{}}}$'.format(utils.to_valid_latex(f'{g_att}={target}'))

        explanation = f'This outlier might be caused by rows with:\n'
        for a, bins in final_pred_by_attr.items():
            first = bins[0]
            t = type(first[0])
            for b in bins:
                if type(b[0]) is tuple:
                    pred = f"{b[0][0]} < {a} < {b[0][1]}"
                    inter_exp = r'$\bf{{{}}}$'.format(utils.to_valid_latex(pred))
                else:
                    pred = f"{a}={b[0]}"
                    inter_exp = r'$\bf{{{}}}$'.format(utils.to_valid_latex(pred))
                if b[1] is not None:
                    if b[1] <= 5:
                        inter_exp = inter_exp + '-' + r'$\bf{low}$'
                    elif b[1] >= 25:
                        inter_exp = inter_exp + '-' + r'$\bf{high}$'
                # inter_exp += f', '
            # inter_exp += '\n'
            explanation += inter_exp
        # explanation += f'which removal {incdec} this value.'
        # explanation += f'\nBefore- {float("{:.5f}".format(df_agg[target]))}, After- {float("{:.5f}".format(final_df[target]))}'

        bar1 = ax.bar(ind1-0.2, y1, 0.4, alpha=1., label='All')
        bar2 = ax.bar(ind2+0.2, y2, 0.4,alpha=1., label=f'without {inter_exp}')
        ax.set_ylabel(f'{g_agg} {agg_title}')
        ax.set_xlabel(f'{g_att}')
        ax.set_xticks(ind1)
        ax.set_xticklabels(tuple([str(i) for i in x1]), rotation=45)
        ax.legend(loc='best')
        ax.set_title(explanation)
    # items_to_bold=[target]
        # for t in target:
        bar1[x1.index(target)].set_edgecolor('tab:green')
        bar1[x1.index(target)].set_linewidth(2)
        bar2[x2.index(target)].set_edgecolor('tab:green')
        bar2[x2.index(target)].set_linewidth(2)
        ax.get_xticklabels()[x1.index(target)].set_color('tab:green')
        return #explanation