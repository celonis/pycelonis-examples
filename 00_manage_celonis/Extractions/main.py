import pandas               as pd
import numpy                as np 
import requests
import urllib

from pycelonis              import get_celonis
from eventCollection_check  import check_eventCollection
from analysis_check         import check_analysis
from column_check           import get_columns

def identify_columns(login, pool_id, hybrid, excel_only):
    all_columns = get_columns(login, pool_id)

    if excel_only is False:

        ## Create Lists and Dictionaries for Input ###
        columns = {}

        for index, row in all_columns.iterrows():
            if (row['Table'] not in columns):
                columns[row['Table']] = []
            columns[row['Table']].append(row['Required Columns'])

        tables = list(columns.keys())
        celonis = get_celonis(**login)
        pool = celonis.pools.find(pool_id)

        if hybrid:
            integration_url = 'integration-hybrid'
        else:
            integration_url = 'integration'

        for connection in pool.data_connections:
            new_job = pool.create_data_job('Automated Job: {}'.format(connection.data['name']), data_source_id=connection.data['id'])
            new_extraction = new_job.create_extraction('Automated Extraction: {}'.format(connection.data['name']))

            for table in tables:
                print('Trying to add {} to {}'.format(table, connection.data['name']))
                url_meta_data = 'http://{}/{}//api/pools/{}/data-sources/{}/meta-data?tableName={}&extractionId={}' \
                    .format(login.get('celonis_url'), integration_url, pool_id, connection.data['id'], table, new_extraction.data['id'])

                try:
                    meta_data = celonis.api_request(url_meta_data, timeout=320)
                except requests.exceptions.HTTPError as err:
                    continue

                res = new_extraction.add_table(table)
                res_id = res.data['id']
                push_columns = [] # Liste aller zu pushenden Columns je ExtractionTable

                for column in meta_data['columns']:
                    if column['pkField'] == True:
                        push_columns.append({'columnName': column['columnName'], 'fromJoin': False, 'anonymized': False, 'primaryKey': True})
                        push_columns.append({'columnName': column['columnName'], 'fromJoin': False, 'anonymized': False, 'primaryKey': False})

                for column in columns.get(table):
                    push_columns.append({'columnName': column, 'fromJoin': False, 'anonymized': False, 'primaryKey': False})

                payload = [{
                    "id":res.data['id'],
                    "taskId":res.data['taskId'],
                    "tableExecutionItemId":None,
                    "tableName":table,
                    "renameTargetTable":False,
                    "targetTableName":None,
                    "columns":push_columns,
                    "joins":[],
                    "dependentTables":[],
                    "filterDefinition":None,
                    "deltaFilterDefinition":None,
                    "useManualPKs":False,
                    "schemaName":None,
                    "creationDateColumn":None,
                    "creationDateValueStart":None,
                    "creationDateValueEnd":None,
                    "creationDateValueToday":False,
                    "changeDateColumn":None,
                    "changeDateOffset":0,
                    "creationDateParameterStart":None,
                    "creationDateParameterEnd":None,
                    "changeDateOffsetParameter":None,
                    "jobId":new_job.data['id'],
                    "parentTable":None,
                    "dependsOn":None,
                    "tableExtractionType":"PARENT_TABLE",
                    "parent":True,
                    "columnValueTable":None,
                    "columnValueColumn":None,
                    "columnValueTargetColumn":None,
                    "columnValuesAtATime":0,
                    "joinType":"NONE",
                    "disabled":False,
                    "connectorSpecificConfiguration":[],
                    "calculatedColumns":[],
                    "endDateDisabled":False}
                ]

                url_change = 'http://{}/{}//api/pools/{}/jobs/{}/extractions/{}/tables/' \
                    .format(login.get('celonis_url'), integration_url, pool_id, new_job.data['id'], new_extraction.data['id'])

                r = celonis.api_request(url_change, message=payload)

                for ex_table in new_extraction.tables:
                    if ex_table.data['id'] == res_id:
                        celonis.api_request('http://{}/{}//api/pools/{}/jobs/{}/extractions/{}/tables/{}' \
                            .format(login.get('celonis_url'), integration_url, pool_id, new_job.data['id'] ,new_extraction.data['id'], ex_table.data['id']), message='DELETE')
            