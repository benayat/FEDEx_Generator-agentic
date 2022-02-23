import pandas as pd
from Operations import Operation
from DatasetRelation import DatasetRelation
import utils


class Join(Operation.Operation):
    def __init__(self, left_df, right_df, source_scheme, attribute, right_name=None, left_name=None):
        super().__init__(source_scheme)
        self.left_df, self.source_scheme, self.attribute, self.right_df = \
            left_df.copy().reset_index(), source_scheme, attribute, right_df.copy()

        if left_name is not None:
            self.left_name = left_name
        else:
            self.left_name = utils.get_calling_params_name(left_df)
        self.left_df.columns = [col if col in ["index", attribute] else self.left_name + "_" + col for col in self.left_df]
        if right_name is not None:
            self.right_name = right_name
        else:
            self.right_name = utils.get_calling_params_name(right_df)
        self.right_df.columns = [col if col in ["index", attribute] else self.right_name + "_" + col for col in self.right_df]
        self.result_df = pd.merge(self.left_df, self.right_df, on=[attribute])

    def iterate_attributes(self):
        for attr in self.left_df.columns:
            if attr.lower() == "index":
                continue
            yield attr, DatasetRelation(self.left_df, self.result_df, self.left_name)

        for attr in set(self.right_df.columns) - set(self.left_df.columns):
            if attr.lower() == "index":
                continue
            yield attr, DatasetRelation(self.right_df, self.result_df, self.right_name)
