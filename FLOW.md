#### Flow
- action-filter. measure: exceptionality.
legend: before-after.
values -
either type, just get values from the source and result columns.
The green values mark the exceptionals, *always* the after values.(why?)

- action-groupby, Join. measure: diversity.
red-line: mean. 
? blue? mean +- some std.
mostly - done.
todo: fix numerical representation for numerical bins - the classes especially. 


- main differences and common parts in plots: 
in both, the classes are exactly the same. 
different values: 
- in diversity(groupby), the values derive directly and easily from the columns.
- in exceptionality: the values are the *frequency*(!) and calculated from value_counts(for source and for result).
