import pandas as pd

def joincsvs(csvspathstomerge=['set1.csv', 'set2.csv', 'set3.csv'],
             resultpathcsv='total.csv',
             id_columns=['Benchmark', 'Threads']
            ):
    df = pd.read_csv(csvspathstomerge[0])
    df.columns = id_columns + ['percentage-increase-0']
    for i,_path in enumerate(csvspathstomerge[1:]):
        mrgdf = pd.read_csv(_path)
        mrgdf.columns = id_columns + [f'percentage-increase-{i+1}']
        df = df.merge(mrgdf)
        df.to_csv(resultpathcsv, index=False)

