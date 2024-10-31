# Function to handle data and setup
def prepare_bar_data(bin_item: Bin, influence_vals: dict = None):
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

    return labels, probabilities, probabilities2

# Function to handle the actual drawing part
def draw_bar(ax, labels, probabilities, probabilities2, bin_item: Bin, influence_vals: dict = None, title=None, score=None, show_scores: bool = False):
    width = 0.35
    ind = np.arange(len(labels))

    result_bar = ax.bar(ind + width, probabilities2, width, label="After Filter")
    ax.bar(ind, probabilities, width, label="Before Filter")
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
    return bin_item.get_bin_name()