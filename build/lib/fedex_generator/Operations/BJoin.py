import pandas as pd

from fedex_generator.commons.consts import TOP_K_DEFAULT, DEFAULT_FIGS_IN_ROW
from fedex_generator.commons import utils
from fedex_generator.commons.DatasetRelation import DatasetRelation
from fedex_generator.Operations import Operation
from fedex_generator.Measures.ExceptionalityMeasure import ExceptionalityMeasure
from fedex_generator.Measures.ShapleyMeasure import ShapleyMeasure

def get_df_name(df, scope):
    name =[x for x in scope if scope[x] is df][0]
    return name
class BJoin(Operation.Operation):
    def __init__(self, left_df, right_df, source_scheme, attribute, result_df=None, left_name=None, right_name=None):
        super().__init__(source_scheme)
        self.source_scheme = source_scheme
        self.attribute = attribute

        df_join = pd.merge(left_df, right_df, on=attribute, how='inner')
            # for a, r, v in cond:
            #     if r == '=':
            #         df_join = df_join[df_join[a] == v]
            #     elif r == '>':
            #         df_join = df_join[df_join[a] > v]
            #     elif r == '>':
            #         df_join = df_join[df_join[a] < v]
        if df_join.shape[0]:
            self.result = True
        else: self.result = False
        self.left_name = utils.get_calling_params_name(left_df)
        self.right_name = utils.get_calling_params_name(right_df)
        self.left_df = left_df
        self.right_df = right_df
        # self.result_df = result_df

    def iterate_attributes(self):
        for attr in self.left_df.columns:
            if attr.lower() == "index":
                continue
            yield attr, DatasetRelation(self.left_df, self.result_df, self.left_name)

        for attr in set(self.right_df.columns) - set(self.left_df.columns):
            if attr.lower() == "index":
                continue
            yield attr, DatasetRelation(self.right_df, self.result_df, self.right_name)

    def explain(self, schema=None, attributes=None, top_k=TOP_K_DEFAULT,
                figs_in_row: int = DEFAULT_FIGS_IN_ROW, show_scores: bool = False, title: str = None, corr_TH: float = 0.7, consider='left'):

        if attributes is None:
            attributes = []

        if schema is None:
            schema = {}
        measure = ShapleyMeasure()
        if self.result:
            score = 1
        else: score = 0
        top_fact, facts = measure.get_most_contributing_fact(self, self.left_df, self.right_df, self.attribute,None, consider=consider)
        exp = f'The result of joining dataframes \'{self.left_name}\' and \'{self.right_name}\' on attribute \'{self.attribute}\' is not empty.\nThe following fact from dataframe \'{self.left_name}\' has significantly contributed to this result:\n'
        facts = list({k: v for k, v in sorted(facts.items(), key=lambda item: -item[1])}.items())
        fact_idx = 0
        exp += str(facts[fact_idx][0])
        top_k -= 1
        while top_k > 0:
            fact_idx += 1
            exp += '\n and then\n'
            exp += str(facts[fact_idx][0])
            top_k -= 1
        return 
