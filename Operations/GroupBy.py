import pandas as pd
from Operations import Operation
from DatasetRelation import DatasetRelation
import utils


class GroupBy(Operation.Operation):
    def __init__(self, source_df, source_scheme, group_attributes, agg_dict):
        super().__init__(source_scheme)
        self.source_scheme, self.group_attributes, self.agg_dict = \
            source_scheme, group_attributes, agg_dict
        self.source_name = utils.get_calling_params_name(source_df)
        source_df = source_df.reset_index()
        group_attributes = self.get_one_to_many_attributes(source_df, group_attributes)
        self.result_df = source_df.groupby(group_attributes).agg(agg_dict)
        self.result_df.columns = ["_".join(x) for x in list(self.result_df.columns)]

    def iterate_attributes(self):
        for attr in self.result_df.columns:
            if attr.lower() == "index" or attr in self.agg_dict:
                continue
            yield attr, DatasetRelation(None, self.result_df, self.source_name)

    def get_source_col(self, filter_attr, filter_values, bins):
        return None

    @staticmethod
    def is_one_to_many(df, col1, col2):
        first_max_cheap_check = df[[col1, col2]].head(1000).groupby(col1).nunique()[col2].max()
        if first_max_cheap_check != 1:
            return False

        group_col1 = df[[col1, col2]].groupby(col1)
        col1_unique = group_col1.nunique()
        if col1_unique[col2].max() != 1:
            return False    # not one-to-many

        group_col2 = df[[col1, col2]].groupby(col2)
        col2_unique = group_col2.nunique()

        if len(col1_unique[col1_unique[col2] != 0]) == len(col2_unique[col2_unique[col1] != 0]):
            return False    # one-to-one

        return True

    @staticmethod
    def get_one_to_many_attributes(df, group_attributes):
        for col in group_attributes:
            for candidate_col in df:
                if candidate_col in group_attributes:
                    continue

                if GroupBy.is_one_to_many(df, col, candidate_col):
                    group_attributes.append(candidate_col)

        return group_attributes
