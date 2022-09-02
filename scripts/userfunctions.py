from collections import Counter
def ids_from_colvals(df_name, col, matchstring):
    ids = eval('{0}[{0}["{1}"]{2}]'.format(df_name, col, matchstring))["id"].tolist()
    return ids

def merge_data_from_ids(ids, datadict):
    c = Counter()
    for id in ids:
        d = datadict[id]
        c.update(d)
    return c
