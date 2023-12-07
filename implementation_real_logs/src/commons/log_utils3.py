import pandas as pd
from enum import Enum
from pathlib import Path
import pm4py
import math

import src.commons.shared_variables as shared

from pm4py.statistics.variants.pandas import get as variants_get
from pm4py.utils import get_properties
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.util import exec_utils
from pm4py.statistics.variants.log.get import get_variants
from pm4py.objects.log.obj import EventLog

class LogName(Enum):
    SYNTH = 'Synthetic log labelled'
    SEPSIS1 = 'sepsis_cases_1'
    SEPSIS2 = 'sepsis_cases_2'
    SEPSIS4 = 'sepsis_cases_4'
    SEPSIS1_10 = 'sepsis_cases_1_sampled10'
    SEPSIS1_20 = 'sepsis_cases_1_sampled20'
    SEPSIS1_30 = 'sepsis_cases_1_sampled30'
    SEPSIS1_40 = 'sepsis_cases_1_sampled40'
    SEPSIS1_50 = 'sepsis_cases_1_sampled50'
    PROD = 'Production'
    UBE = "Synthetic_UBE"
    MC = "mccloud"
    CREDIT = "credit_card"
    HOSPITAL2 = "hospital_billing_2"
    HOSPITAL3 = "hospital_billing_3"
    BPIC12_1 = "bpic2012_O_ACCEPTED-COMPLETE"
    BPIC12_2 = "bpic2012_O_CANCELLED-COMPLETE"
    BPIC12_3 = "bpic2012_O_DECLINED-COMPLETE"
    HELPDESK = 'helpdesk'

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

        elif file_name.endswith('1.csv')  or file_name.endswith('2.csv')  or file_name.endswith('3.csv')   : # for sepsis_1~3, hospital_billing_2~3
            self.log_name = LogName(log_path.stem)
            self.log_ext = LogExt.CSV
            self._set_log_keys_and_ths()
            self.log = pd.read_csv(
                log_path, sep= ";",
                usecols=[self.case_name_key, self.label_name_key, self.act_name_key, self.res_name_key, self.timestamp_key]
            )
            self.log[self.timestamp_key] = pd.to_datetime(self.log[self.timestamp_key])

        elif file_name.endswith('.csv') :
            
            if file_name.endswith('UBE.csv') or file_name.endswith('mccloud.csv') \
                or file_name.endswith('card.csv') or file_name.endswith('desk.csv'):
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
                self.log_name = LogName(log_path.stem)
                self.log_ext = LogExt.CSV
                self._set_log_keys_and_ths()
                self.log = pd.read_csv(
                    log_path, 
                    usecols=[self.case_name_key, self.label_name_key, self.act_name_key, self.res_name_key, self.timestamp_key]
                )
                self.log[self.timestamp_key] = pd.to_datetime(self.log[self.timestamp_key])

        else:
            raise RuntimeError(f"Extension of {file_name} must be in ['.xes', '.xes.gz', '.csv'].")

        trace_ids = self.log[self.case_name_key].unique().tolist()
        parameters = get_properties(self.log, case_id_key =self.case_name_key, activity_key = self.act_name_key, timestamp_key = self.timestamp_key)
        
        # "get_variants_count": this gives wrong count information with hospital billing data. 
        # variants = variants_get.get_variants_count(self.log, parameters=parameters) 
        # variant_count = []
        # for variant in variants:
        #     variant_count.append([variant, variants[variant]])
        # variant_count = sorted(variant_count, key=lambda x: (x[1], x[0]), reverse=True)
        # variant_count = variant_count[round(len(variant_count)*shared.variant_split):]  #math.ceil
        # variants_to_filter = [x[0] for x in variant_count]
        
        
        log = log_converter.apply(self.log, variant=log_converter.Variants.TO_EVENT_LOG, parameters=parameters)
        variants = get_variants(log, parameters=parameters)
        print("Size of varients : ",  len(variants))

        v_id = 0
        dict_cv = {}
        for variant in variants:
            for trace in variants[variant]:
                dict_cv[trace.attributes['concept:name']] = "variant_" + str(v_id)
            v_id = v_id +1

        
        grouped = self.log.groupby(self.case_name_key)
        start_timestamps = grouped[self.timestamp_key].min().reset_index()

        start_timestamps = start_timestamps.sort_values(self.timestamp_key, ascending=True, kind="mergesort")

        train_ids = list(start_timestamps[self.case_name_key])[:int(shared.train_ratio*len(start_timestamps))]
        train = self.log[self.log[self.case_name_key].isin(train_ids)].sort_values(self.timestamp_key, ascending=True, kind='mergesort').reset_index(drop=True)
        test = self.log[~self.log[self.case_name_key].isin(train_ids)].sort_values(self.timestamp_key, ascending=True, kind='mergesort').reset_index(drop=True)
        test_ids = set(test[self.case_name_key].tolist())

        # train.to_csv('helpdesk_train.csv', index=False)
        # test.to_csv('helpdesk_test.csv', index=False)
        
        test_log = pm4py.convert_to_event_log(test, case_id_key=self.case_name_key)
        # pm4py.write_xes(test_log, 'sepsis_test.xes')
        # test.to_csv('sepsis_test.csv', index= False)
        
        # net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(test_log, noise_threshold = 0.04,  activity_key=self.act_name_key,
        #                                                     case_id_key=self.case_name_key,
        #                                                     timestamp_key= self.timestamp_key)

        # net, initial_marking, final_marking = pm4py.discover_petri_net_alpha(test_log, activity_key=self.act_name_key,
        #                                                     case_id_key=self.case_name_key,
        #                                                     timestamp_key= self.timestamp_key)      

        # net, initial_marking, final_marking = pm4py.discover_petri_net_heuristics(test_log, activity_key=self.act_name_key,
        #                                                     case_id_key=self.case_name_key,
        #                                                     timestamp_key= self.timestamp_key) 
                
        # pm4py.write_pnml(net, initial_marking, final_marking, file_name.split(".")[0] +  ".pnml")
        
        # elements_per_fold = round(len(trace_ids1) / shared.folds)
        self.training_trace_ids = train_ids  #[:-elements_per_fold]
        self.case_to_variant = dict_cv
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
            # self.log[self.timestamp_key2] = temp_time1['diff_cum']
            # self.log[self.timestamp_key3] = temp_time1['times3']
            # self.log[self.timestamp_key4] = temp_time1['times4']
            del temp_time1

        if outcome:
            self.log.replace(to_replace={self.label_name_key: {self.label_pos_val: '1', self.label_neg_val: '0'}}, inplace=True)
            

    def _set_log_keys_and_ths(self):
        # In case of log saved with XES format, case attributes must have the 'case:' prefix
        addit = '' if self.log_ext == LogExt.CSV else 'case:'


        if self.log_name == LogName.SYNTH:
            self.case_name_key = addit+'concept:name'
            self.label_name_key = addit+'label'
            self.label_pos_val = 'positive'
            self.label_neg_val = 'negative'
            self.act_name_key = 'concept:name'
            self.res_name_key = 'org:group'
            self.timestamp_key = 'time:timestamp'
            # self.timestamp_key2 = 'time:timestamp2'
            # self.timestamp_key3 = 'time:timestamp3'
            # self.timestamp_key4 = 'time:timestamp4'
            self.compliance_th = 1.0
            self.evaluation_th = self.compliance_th * shared.th_reduction_factor
            self.evaluation_prefix_start = 7
            self.evaluation_prefix_end = 7

        elif self.log_name == LogName.UBE:
             self.case_name_key = 'case:concept:name'
             self.act_name_key = 'concept:name'
             self.timestamp_key = 'time:timestamp'
             self.evaluation_prefix_start = 6
             self.evaluation_prefix_end = 7
             self.compliance_th = 1.00

        elif self.log_name == LogName.HELPDESK:
             self.case_name_key = 'Case ID'
             self.act_name_key = 'Activity'
             self.timestamp_key = 'Complete Timestamp'
             self.evaluation_prefix_start = 3
             self.evaluation_prefix_end = 3
             self.compliance_th = 1.00 

        elif self.log_name == LogName.MC:
             self.case_name_key = 'Case ID'
             self.act_name_key = 'Activity'
             self.timestamp_key = 'Start Timestamp'
             self.evaluation_prefix_start = 3
             self.evaluation_prefix_end = 7
             self.compliance_th = 1.00 

        elif self.log_name == LogName.CREDIT:
             self.case_name_key = 'Case ID'
             self.act_name_key = 'Activity'
             self.timestamp_key = 'Start Timestamp'
             self.evaluation_prefix_start = 3
             self.evaluation_prefix_end = 7
             self.compliance_th = 1.00 
          
        elif self.log_name == LogName.SEPSIS1 \
                or self.log_name == LogName.SEPSIS2 \
                or self.log_name == LogName.SEPSIS4:
            self.case_name_key = addit+'Case ID'
            self.label_name_key = addit+'label'
            self.label_pos_val = 'deviant'
            self.label_neg_val = 'regular'
            self.act_name_key = 'Activity'
            self.res_name_key = 'org:group'
            self.timestamp_key = 'time:timestamp'
            # self.timestamp_key2 = 'time:timestamp2'
            # self.timestamp_key3 = 'time:timestamp3'
            # self.timestamp_key4 = 'time:timestamp4'

            if self.log_name == LogName.SEPSIS1:
                self.compliance_th = 0.77   # 0.62 for complete petrinet, 0.77 for reduced petrinet
            elif self.log_name == LogName.SEPSIS2:
                self.compliance_th = 0.55
            else:   # log_name == LogName.SEPSIS4
                self.compliance_th = 0.77

            self.evaluation_th = self.compliance_th * shared.th_reduction_factor
            self.evaluation_prefix_start = 10
            self.evaluation_prefix_end = 10


        elif self.log_name in [LogName.SEPSIS1_10, LogName.SEPSIS1_20, LogName.SEPSIS1_30, LogName.SEPSIS1_40, LogName.SEPSIS1_50]:
            self.case_name_key = addit+'Case ID'
            self.label_name_key = addit+'label'
            self.label_pos_val = 'deviant'
            self.label_neg_val = 'regular'
            self.act_name_key = 'Activity'
            self.res_name_key = 'org:group'
            self.timestamp_key = 'time:timestamp'
            self.compliance_th = 0.0

            self.evaluation_th = self.compliance_th * shared.th_reduction_factor
            self.evaluation_prefix_start = 10
            self.evaluation_prefix_end = 10

        elif self.log_name in [LogName.HOSPITAL2, LogName.HOSPITAL3]:
            self.case_name_key = addit+'Case ID'
            self.label_name_key = addit+'label'
            self.label_pos_val = 'deviant'
            self.label_neg_val = 'regular'
            self.act_name_key = 'Activity'
            self.res_name_key = 'Resource'
            self.timestamp_key = 'Complete Timestamp'
            self.compliance_th = 0.77

            self.evaluation_th = self.compliance_th * shared.th_reduction_factor
            self.evaluation_prefix_start = 3
            self.evaluation_prefix_end = 3


        elif self.log_name == LogName.PROD:
            self.case_name_key = addit+'Case ID'
            self.label_name_key = addit + 'label'
            self.label_pos_val = 'deviant'
            self.label_neg_val = 'regular'
            self.act_name_key = 'Activity'
            self.res_name_key = 'Resource'
            self.timestamp_key = 'Complete Timestamp'
            self.compliance_th = 0.86
            self.evaluation_th = self.compliance_th * shared.th_reduction_factor
            self.evaluation_prefix_start = 7
            self.evaluation_prefix_end = 7

        else:
            raise RuntimeError(f"No settings defined for log: {self.log_name.value}.")
