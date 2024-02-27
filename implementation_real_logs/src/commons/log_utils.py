import pandas as pd
from enum import Enum
from pathlib import Path
import pm4py

import src.commons.shared_variables as shared

from pm4py.utils import get_properties
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.statistics.variants.log.get import get_variants

class LogName(Enum):
    SEPSIS1 = 'sepsis_cases_1'
    UBE = "Synthetic"
    HELPDESK = 'helpdesk'
    BPIC12 = "BPIC12"
    BPIC13I = "BPIC13_I"
    BPIC13CP = "BPIC13_CP"
    ROAD = "Road_Traffic"

class LogExt(Enum):
    CSV = '.csv'
    XES = '.xes'
    XES_GZ = '.xes.gz'


class LogData:
    log: pd.DataFrame
    log_name: LogName
    log_ext: LogExt
    training_trace_ids = [str]
    evaluation_trace_ids = [str]

    # Gathered from encoding
    act_enc_mapping: {str, str}
    res_enc_mapping: {str, str}

    # Gathered from manual log analisys
    case_name_key: str
    act_name_key: str
    res_name_key: str
    timestamp_key: str
    timestamp_key2: str
    timestamp_key3: str
    timestamp_key4: str
    label_name_key: str
    label_pos_val: str
    label_neg_val: str
    compliance_th: float
    evaluation_th: float
    evaluation_prefix_start: int
    evaluation_prefix_end: int

    def __init__(self, log_path: Path):
        file_name = log_path.name
        if file_name.endswith('.xes') or file_name.endswith('.xes.gz'):
            if file_name.endswith('.xes'):
                self.log_name = LogName(log_path.stem)
                self.log_ext = LogExt.XES
            else:  # endswith '.xes.gz'
                self.log_name = LogName(log_path.with_suffix("").stem)
                self.log_ext = LogExt.XES_GZ

            self._set_log_keys_and_ths()
            self.log = pm4py.read_xes(str(log_path))[
                [self.case_name_key, self.label_name_key, self.act_name_key, self.res_name_key, self.timestamp_key]
            ]
            self.log[self.timestamp_key] = pd.to_datetime(self.log[self.timestamp_key])

        elif file_name.endswith('_1.csv')  or file_name.endswith('_2.csv')  or file_name.endswith('_3.csv')   : # for sepsis_1~3, hospital_billing_2~3
            self.log_name = LogName(log_path.stem)
            self.log_ext = LogExt.CSV
            self._set_log_keys_and_ths()
            self.log = pd.read_csv(
                log_path, sep= ";",
                usecols=[self.case_name_key, self.label_name_key, self.act_name_key, self.res_name_key, self.timestamp_key]
            )
            self.log[self.timestamp_key] = pd.to_datetime(self.log[self.timestamp_key])

        elif file_name.endswith('.csv') :
            self.log_name = LogName(log_path.stem)
            self.log_ext = LogExt.CSV
            self._set_log_keys_and_ths()
            self.log = pd.read_csv(
                log_path, 
                usecols=[self.case_name_key, self.act_name_key,  self.timestamp_key]
            )
            self.log[self.case_name_key] = self.log[self.case_name_key].astype(str)
            self.log[self.timestamp_key] = pd.to_datetime(self.log[self.timestamp_key])

        else:
            raise RuntimeError(f"Extension of {file_name} must be in ['.xes', '.xes.gz', '.csv'].")


        trace_ids = self.log[self.case_name_key].unique().tolist()


        # Variant extraction
        parameters = get_properties(self.log, case_id_key =self.case_name_key, activity_key = self.act_name_key, timestamp_key = self.timestamp_key)
        log = log_converter.apply(self.log, variant=log_converter.Variants.TO_EVENT_LOG, parameters=parameters)
        variants = get_variants(log, parameters=parameters)
        print("Size of varients : ",  len(variants))

        v_id = 0
        dict_cv = {}
        df_clusters = pd.DataFrame( columns= ['prefix', 'variant',  "variant_ID", 'case', 'supp' ])
        prefix_lenght = self.evaluation_prefix_start
        
        for variant in variants:
            c = []
            for trace in variants[variant]:
                dict_cv[trace.attributes['concept:name']] = "variant_" + str(v_id)
                c.append(trace.attributes['concept:name'])
                
            prefix = str(variant[:prefix_lenght])
            row = [prefix, variant, v_id, str(c) , len(c)]
            df_clusters = pd.concat([df_clusters, pd.Series(row, index=['prefix', 'variant', "variant_ID", "case", 'supp']).to_frame().T], ignore_index=True)
            v_id = v_id +1
        
        prefix_count = df_clusters['prefix'].value_counts()
        list_prefix = prefix_count[prefix_count > 1].index
        df_clusters = df_clusters.loc[df_clusters['prefix'].isin(list_prefix)].reset_index(drop=True)
        
        variant_top2 = df_clusters.groupby('prefix', as_index = False).apply(lambda x: x['case'].tolist()[sorted(range(len(x['supp'])), reverse =True,  key=lambda k: x['supp'].tolist()[k] )[1]])
        variant_top2.columns = ['prefix', 'case']
        variant_top2['case'] = variant_top2['case'].apply(lambda x: eval(x))
        variant_top2['freq'] = variant_top2['case'].apply(lambda x: len(x))
        
        list_variant_top2 =variant_top2['case'].tolist()
        test_ids = []
        for i in list_variant_top2:
            test_ids = test_ids + i
        
        filterByKey = lambda keys: {x: dict_cv[x] for x in keys}
        dict_cv_train = filterByKey(trace_ids)
        dict_cv_test = filterByKey(test_ids)
        
        # save the test traces in the target cluster if you want
        # test = self.log[self.log[self.case_name_key].isin(test_ids)].sort_values(self.timestamp_key, ascending=True, kind='mergesort').reset_index(drop=True)
        # test.to_csv(str(log_path.stem)+ '_cprefix' + str(self.evaluation_prefix_start) + '_test.csv', index=False)
        
        # Outputs
        self.training_trace_ids = trace_ids  
        self.case_to_variant = dict_cv
        self.case_to_variant_train = dict_cv_train
        self.case_to_variant_test = dict_cv_test
        self.evaluation_trace_ids =  test_ids


    def encode_log(self, resource: bool, timestamp: bool, outcome: bool):
        act_set = list(self.log[self.act_name_key].unique())
        self.act_enc_mapping = dict((chr(idx + shared.ascii_offset), elem) for idx, elem in enumerate(act_set))
        self.log.replace(to_replace={self.act_name_key: {v: k for k, v in self.act_enc_mapping.items()}}, inplace=True)

        if resource:
            res_set = list(self.log[self.res_name_key].unique())
            self.res_enc_mapping = dict((chr(idx + shared.ascii_offset), elem) for idx, elem in enumerate(res_set))
            self.log.replace(to_replace={self.res_name_key: {v: k for k, v in self.res_enc_mapping.items()}}, inplace=True)

        if timestamp:
            temp_time1 = self.log[[self.case_name_key, self.timestamp_key]]
            temp_time1['diff'] = temp_time1.groupby(self.case_name_key)[self.timestamp_key].diff().dt.seconds
            temp_time1['diff'].fillna(0, inplace=True)
            temp_time1['diff'] = temp_time1['diff']/max(temp_time1['diff'])  
            temp_time1['diff_cum'] = temp_time1['diff'].cumsum()
            temp_time1['diff_cum'] = temp_time1['diff_cum'] /max(temp_time1['diff_cum'])
            temp_time1['midnight'] = temp_time1[self.timestamp_key].apply(lambda x:  x.replace(hour=0, minute=0, second=0, microsecond=0))
            temp_time1['times3'] = (temp_time1[self.timestamp_key] - temp_time1['midnight']).dt.seconds / 86400
            temp_time1['times4'] = temp_time1[self.timestamp_key].apply(lambda x:  x.weekday() / 7)
            self.log[self.timestamp_key] = temp_time1['diff']
            del temp_time1

        if outcome:
            self.log.replace(to_replace={self.label_name_key: {self.label_pos_val: '1', self.label_neg_val: '0'}}, inplace=True)
            

    def _set_log_keys_and_ths(self):
        addit = '' if self.log_ext == LogExt.CSV else 'case:'

        if self.log_name == LogName.UBE:
             self.case_name_key = 'case:concept:name'
             self.act_name_key = 'concept:name'
             self.timestamp_key = 'time:timestamp'
             self.evaluation_prefix_start = 3
             self.evaluation_prefix_end = 7
             self.compliance_th = 1.00

        elif self.log_name == LogName.HELPDESK:
             self.case_name_key = 'Case ID'
             self.act_name_key = 'Activity'
             self.timestamp_key = 'Complete Timestamp'
             self.evaluation_prefix_start = 7 # 3, 5, 7
             self.evaluation_prefix_end = 7
             self.compliance_th = 1.00 

        elif self.log_name == LogName.BPIC12:
             self.case_name_key = 'Case ID'
             self.act_name_key = 'Activity'
             self.timestamp_key = 'Complete Timestamp'
             self.evaluation_prefix_start = 5 # 8~12
             self.evaluation_prefix_end = 5
             self.compliance_th = 1.00 

        elif self.log_name == LogName.BPIC13I:
             self.case_name_key = 'Case ID'
             self.act_name_key = 'Activity'
             self.timestamp_key = 'Complete Timestamp'
             self.evaluation_prefix_start = 3
             self.evaluation_prefix_end = 7
             self.compliance_th = 1.00 

        elif self.log_name == LogName.BPIC13CP:
             self.case_name_key = 'Case ID'
             self.act_name_key = 'Activity'
             self.timestamp_key = 'Complete Timestamp'
             self.evaluation_prefix_start = 3
             self.evaluation_prefix_end = 7
             self.compliance_th = 1.00 

        elif self.log_name == LogName.ROAD:
             self.case_name_key = 'Case'
             self.act_name_key = 'Activity'
             self.timestamp_key = 'Timestamp'
             self.evaluation_prefix_start = 3 # 3
             self.evaluation_prefix_end = 7 # 7
             self.compliance_th = 1.00 
          
        elif self.log_name == LogName.SEPSIS1:
            self.case_name_key = addit+'Case ID'
            self.label_name_key = addit+'label'
            self.label_pos_val = 'deviant'
            self.label_neg_val = 'regular'
            self.act_name_key = 'Activity'
            self.res_name_key = 'org:group'
            self.timestamp_key = 'time:timestamp'
            self.compliance_th = 0.77   # 0.62 for complete petrinet, 0.77 for reduced petrinet
            self.evaluation_th = self.compliance_th * shared.th_reduction_factor
            self.evaluation_prefix_start = 8
            self.evaluation_prefix_end = 12

        else:
            raise RuntimeError(f"No settings defined for log: {self.log_name.value}.")
