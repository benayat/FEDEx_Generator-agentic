# Method to handle data and setup
def prepare_bar_data(self, bin_item: MultiIndexBin, influence_vals: dict = None):
    max_values, max_influence = self.get_max_k(influence_vals, 1)
    max_value = max_values[0]

    res_col = bin_item.get_binned_result_column()
    average = float(utils.smart_round(res_col.mean()))

    agger_function = self.get_agg_func_from_name(res_col.name)
    aggregated_result = res_col.groupby(bin_item.get_bin_name()).agg(agger_function)

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

    return labels, aggregate_column, average, max_value, res_col