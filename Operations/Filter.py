import pandas as pd
from Operations import Operation
import operator
from DatasetRelation import DatasetRelation
import utils


operators = {
    "==": operator.eq,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "!=": operator.ne,
    "between": lambda x, tup: x.apply(lambda item: tup[0] <= item < tup[1])
}


def do_operation(a, b, op_str):
    return operators[op_str](a, b)


class Filter(Operation.Operation):
    def __init__(self, source_df, source_scheme, attribute=None, operation_str=None, value=None, result_df=None):
        super().__init__(source_scheme)
        self.source_df = source_df.reset_index()
        if result_df is None:
            self.source_scheme, self.attribute, self.operation_str, self.value = \
                source_scheme, attribute, operation_str, value
            self.result_df = self.source_df[do_operation(self.source_df[attribute], value, operation_str)]
        else:
            self.result_df = result_df
        self.source_name = utils.get_calling_params_name(source_df)

    def get_correlated_attributes(self):
        numeric_df = self.source_df.head(10000)     # for performance we take part of the DB
        for column in numeric_df:
            try:
                if utils.is_numeric(numeric_df[column]):
                    continue

                items = sorted(numeric_df[column].dropna().unique())
                items_map = dict(zip(items, range(len(items))))
                numeric_df[column] = numeric_df[column].map(items_map)
            except Exception as e:
                print(e)

        # for performance, we use the first 50000 rows
        # first_rows = self.source_df.head(50000)
        # corr = first_rows.corr('spearman')
        corr = numeric_df.corr()
        high_correlated_columns = []
        if self.attribute in corr:
            df = corr[self.attribute]

            df = df[df > 0.85].dropna()
            high_correlated_columns = list(df.index)

        return high_correlated_columns

    def iterate_attributes(self):
        high_correlated_columns = self.get_correlated_attributes()

        for attr in self.result_df.columns:
            if attr.lower() == "index" or attr.lower() == self.attribute.lower() or \
                    self.source_scheme.get(attr, None) == 'i' or attr in high_correlated_columns:
                continue
            yield attr, DatasetRelation(self.source_df, self.result_df, self.source_name)

    def get_source_col(self, filter_attr, filter_values, bins):
        if filter_attr not in self.source_df:
            return None

        binned_col = pd.cut(self.source_df[filter_attr], bins=bins, labels=False, include_lowest=True,
                            duplicates='drop')

        return binned_col[binned_col.isin(filter_values)]

