def ids_from_colvals(col, matchstring):
    ids = eval('jstor_df[jstor_df["{0}"]{1}]'.format(col, matchstring))["id"].tolist()
    return ids

def merge_data_from_ids(ids, datadict):
    c = Counter()
    for id in ids:
        d = datadict[id]
        c.update(d)
    return c