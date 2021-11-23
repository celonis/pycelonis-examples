import pandas               as pd
import numpy                as np # Vorinstalliert?

from pycelonis              import get_celonis
from eventCollection_check  import check_eventCollection
from analysis_check         import check_analysis

def get_columns(login, pool_id, to_excel=True):

    ### Connect to Celonis #### 
    celonis = get_celonis(**login)
    pool = celonis.pools.find(pool_id)
    liste = []
    url = 'http://{}/integration/ui/pools/{}/data-configuration/data-jobs?jobId={}&tab=tasks'
    api_token = login.get('api_token')

    ### Dataframe from Event Collection ### 
    print('Getting Event Collection Columns')
    for job in pool.data_jobs:
        print('Getting Event Collection Columns: ID - ' + job.data['id'])
        listOfColumns = check_eventCollection(url.format(login.get('celonis_url'), job.data['dataPoolId'], job.data['id']), r'SAVEDIR PLACEHOLDER', celonis)
        liste.append(pd.concat(listOfColumns))
    columns = pd.concat(liste).replace(r'^\s*$', np.nan, regex=True).ffill(axis = 0)

    ### Dataframe from Analysis ### 
    print('Getting Analysis Columns')
    res = check_analysis(login.get('celonis_url'), celonis).drop(columns=['Analysis']).applymap(lambda x: x.replace('"', ''))

    ### Merge Dataframes ####
    all_columns = pd.concat([columns, res]).drop_duplicates(subset=['Table', 'Required Columns']).sort_values(by=['Table'])

    if to_excel:
        print('Saving to Excel File')
        all_columns.to_excel('Used_Columns.xlsx', index=False)

    return all_columns