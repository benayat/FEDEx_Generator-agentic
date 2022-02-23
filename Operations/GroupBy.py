import pandas as pd
from Operations import Operation
from DatasetRelation import DatasetRelation
import utils


class GroupBy(Operation.Operation):
    def __init__(self, source_df, source_scheme, group_attributes, agg_dict):
        super().__init__(source_scheme)
        self.source_df, self.source_scheme, self.group_attributes, self.agg_dict = \
            source_df.reset_index(), source_scheme, group_attributes, agg_dict
        self.source_name = utils.get_calling_params_name(source_df)
        group_attributes = self.get_one_to_many_attributes(self.source_df, group_attributes)
        self.result_df = self.source_df.groupby(group_attributes).agg(agg_dict)
        self.result_df.columns = ["_".join(x) for x in list(self.result_df.columns.ravel())]

    def iterate_attributes(self):
        for attr in self.result_df.columns:
            if attr.lower() == "index" or attr in self.agg_dict:
                continue
            yield attr, DatasetRelation(self.source_df, self.result_df, self.source_name)

    def get_results_cols(self, filter_attr, filter_values, col, bins):
        if filter_attr not in self.source_df:
            return pd.cut(self.result_df[col], bins=bins, labels=False, include_lowest=True, duplicates='drop')

        binned_col = pd.cut(self.source_df[filter_attr], bins=bins, labels=False, include_lowest=True, duplicates='drop')
        source_df = self.source_df.copy()
        source_df[filter_attr] = binned_col
        source_df = source_df[source_df[filter_attr].isin(filter_values)]
        result_df = source_df.groupby(self.group_attributes).agg(self.agg_dict)

        return result_df[self.agg_dict.keys()]

    def get_source_col(self, filter_attr, filter_values, bins):
        return None

    @staticmethod
    def is_one_to_many(df, col1, col2):
        first_max_cheap_check = df[[col1, col2]].head(1000).groupby(col1).nunique()[col2].max()
        if first_max_cheap_check != 1:
            return False

        first_max = df[[col1, col2]].groupby(col1).nunique()[col2].max()
        return first_max == 1

    @staticmethod
    def get_one_to_many_attributes(df, group_attributes):
        for col in group_attributes:
            for candidate_col in df:
                if candidate_col in group_attributes:
                    continue

                if GroupBy.is_one_to_many(df, col, candidate_col):
                    group_attributes.append(candidate_col)

        return group_attributes
