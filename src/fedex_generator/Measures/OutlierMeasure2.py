from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from fedex_generator.Measures.BaseMeasure import BaseMeasure, START_BOLD, END_BOLD
from fedex_generator.Measures.Bins import Bin, MultiIndexBin
from fedex_generator.commons import utils
ALPHA = 0.5
# BETA = 0.1
class OutlierMeasure(BaseMeasure):
    def __init__(self):
        super().__init__()

    def calc_influence_pred(self, df_before, df_after, target, dir):
        try:
            target_inf = (df_before[target] - df_after[target]) * dir
        except:
            return -1
        holdout_inf = 0
        for i in df_before.index:
            if i != target:
                try:
                    holdout_inf += abs(df_after[i] - df_before[i])
                except:
                    return -1
        norm = target_inf + holdout_inf
        return ALPHA * (target_inf) - (1-ALPHA) * (holdout_inf/(len(df_before.index)))
    def merge_preds(self, df_agg, df_in, preds, g_att, g_agg, agg_method, target, dir):
        final_pred = []
        final_inf = 0.01
        df_final = df_in.copy()
        final_agg_df = None
        for p in preds:
            attr, i, score, kind, rank = p
            if(kind == 'bin'):
                bin = i
                df_exc = df_final[((df_final[attr] < bin[0]) | (df_final[attr] >= bin[1]))]
            else:
                df_exc = df_final[df_final[attr] != i]
            agged_val = df_exc.groupby(g_att)[g_agg].agg(agg_method)
            inf = self.calc_influence_pred(df_agg, agged_val, target, dir)/np.sqrt(df_in.shape[0]/df_exc.shape[0])
            if inf/final_inf>1.1:
                final_pred.append((attr, i, rank))
                final_inf = inf
                df_final = df_exc
                final_agg_df = agged_val
            else:
                break
        return final_pred, final_inf, final_agg_df


    def explain_outlier(self, df_agg, df_in, g_att, g_agg, agg_method, target, dir, control=None,k=1):
        attrs = df_in.select_dtypes(include='number').columns.tolist()#[:10]
        attrs = [a for a in attrs if a not in [g_att, g_agg]]
        exps = {}
        worst = 0
        # top_bin_all = None
        top_inf_all = -1
        # top_attr = None
        df = None
        predicates = []
        if control == None:
            control=list(df_agg.index)
        df_in_consider = df_in[(df_in[g_att].isin(control))|(df_in[g_att] == target)]
        df_agg_consider = df_agg[control]
        df_agg_consider[target] = df_agg[target]
        for attr in attrs:
            if df_in[g_agg].corr(df_in[attr]) > 0.8:
                continue
            # if attr=='duration_minutes':
            #     pass
            series = df_in[attr]
            dtype = df_in[attr].dtype.name
            flag = False
            if dtype in ['int64', 'object']:
                if attr == 'explicit':
                    pass
                vals = series.value_counts()
                if len(vals) < 15:
                    flag = True
                    top_df = None
                    top_inf = 0
                    top_bin = None
                    for i in vals.index:
                        # df_in_target_exc = df_in[((df_in[g_att].isin(target))&(df_in[attr] != i))]
                        df_in_target_exc = df_in_consider[(df_in_consider[attr] != i)]
                        agged_val = df_in_target_exc.groupby(g_att)[g_agg].agg(agg_method)
                        # agged_df = df_agg.copy()
                        # agged_df[target]=agged_val
                        inf = self.calc_influence_pred(df_agg_consider, agged_val,target, dir)/np.sqrt(df_in_consider.shape[0]/df_in_target_exc.shape[0])
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
                    
                            
            if not flag:
                for n in [20]:
                    _, bins = pd.cut(series, n, retbins=True, duplicates='drop')
                    df_bins_in = pd.cut(df_in_consider[attr], bins=bins).value_counts(normalize=True).sort_index()#.rename('idx')
                    top_df = None
                    top_inf = 0
                    top_bin = None
                    i = 1
                    for bin in df_bins_in.keys():
                        # df_in_exc = df_in[((df_in[g_att].isin(target)) & ((df_in[attr] < bin.left) | (df_in[attr] >= bin.right)))]
                        new_bin = (float("{:.2f}".format(bin.left)), float("{:.2f}".format(bin.right)))
                        df_in_exc = df_in_consider[((df_in_consider[attr] < new_bin[0]) | (df_in_consider[attr] >= new_bin[1]))]
                        agged_val = df_in_exc.groupby(g_att)[g_agg].agg(agg_method)
                        # agged_df = df_agg.copy()
                        # agged_df[target]=agged_val
                        inf = self.calc_influence_pred(df_agg_consider, agged_val, target, dir)/np.sqrt(df_in_consider.shape[0]/df_in_exc.shape[0])
                        exps[(attr,(new_bin[0],new_bin[1]))]=inf
                        # if inf > top_inf:
                        #     top_inf = inf
                        #     top_bin = new_bin
                        #     top_df = agged_val
                        predicates.append((attr, new_bin, inf, 'bin', i))
                        i += 1
                    # if top_inf > top_inf_all:
                    #     top_inf_all = top_inf
                    #     top_bin_all = top_bin
                    #     top_attr = attr
                    #     df = top_df
            # if top_inf_all >= 1:
            #             break
        predicates.sort(key=lambda x:-x[2])
        final_pred, final_inf, final_df = self.merge_preds(df_agg, df_in, predicates, g_att, g_agg, agg_method, target, dir)
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
        if agg_method == 'count':
            agg_method ='proportion'
            y1 = y1/y1.sum()
            y2 = y2/y2.sum()

        # explanation = f'The predicate (\'{top_attr}\' = {top_bin_all}) has high influence on this outlier({top_inf_all}).'
        explanation = f'The predicate\n'
        for a, bins in final_pred_by_attr.items():
            first = bins[0]
            t = type(first[0])
            if t is tuple:
                inter_exp = f'{a} between '
            else:
                inter_exp = f'{a} is '
            for b in bins:
                inter_exp += f'{b[0]}'
                if b[1] is not None:
                    if b[1] <= 5:
                        inter_exp += f'- low'
                    elif b[1] >= 15:
                        inter_exp += f'- high'
                inter_exp += f', '
            inter_exp += '\n'
            explanation += inter_exp
        explanation += 'has high influence on this outlier.'

        bar1 = ax.bar(ind1-0.2, y1, 0.4, alpha=1., label='All')
        bar2 = ax.bar(ind2+0.2, y2, 0.4,alpha=1., label=f'Without predicate')
        ax.set_ylabel(f'{g_agg} {agg_method}')
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
        return explanation