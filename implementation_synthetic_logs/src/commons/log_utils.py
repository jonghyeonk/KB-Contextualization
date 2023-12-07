import pandas as pd
from enum import Enum
from pathlib import Path
import pm4py

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
    PROD = 'Production'
    SYNTH2 = 'Synthetic'


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
                [self.case_name_key, self.act_name_key, self.timestamp_key]
                # [self.case_name_key, self.label_name_key, self.act_name_key, self.res_name_key, self.timestamp_key]
            ]
            self.log[self.timestamp_key] = pd.to_datetime(self.log[self.timestamp_key])

        elif file_name.endswith('.csv'):
            self.log_name = LogName(log_path.stem)
            self.log_ext = LogExt.CSV
            self._set_log_keys_and_ths()

            self.log = pd.read_csv(
                log_path, 
                usecols=[self.case_name_key, self.act_name_key,  self.timestamp_key]
            )
            self.log[self.timestamp_key] = pd.to_datetime(self.log[self.timestamp_key])


        else:
            raise RuntimeError(f"Extension of {file_name} must be in ['.xes', '.xes.gz', '.csv'].")

        # Use last fold for evaluation, remaining ones for training (JH: updated)
        if file_name.endswith('Synthetic.xes'):
            self.test_log = pm4py.read_xes(str(log_path.parent) + "\Synthetic_test.xes")[
                [self.case_name_key, self.act_name_key, self.timestamp_key]
            ]
            self.test_log[self.timestamp_key] = pd.to_datetime(self.log[self.timestamp_key])
            
        trace_ids = self.log[self.case_name_key].unique().tolist()
        # elements_per_fold = round(len(trace_ids) / shared.folds)
        
        self.test_log[self.case_name_key] = self.test_log[self.case_name_key].astype(str) + '_test'
        merged = pd.concat([self.log, self.test_log], axis= 0)
        merged = merged.reset_index(drop=True)
        self.log = merged
        # merged.to_csv('Synthetic_merged.csv', index=False)
        test_ids = self.test_log[self.case_name_key].unique().tolist()
        
        self.training_trace_ids = trace_ids
        self.evaluation_trace_ids = test_ids
        
        
        # parameters = get_properties(self.log, case_id_key =self.case_name_key, activity_key = self.act_name_key, timestamp_key = self.timestamp_key)
        # variants = variants_get.get_variants_count(self.log, parameters=parameters)
        # variant_count = []
        # for variant in variants:
        #     variant_count.append([variant, variants[variant]])
        # variant_count = sorted(variant_count, key=lambda x: (x[1], x[0]), reverse=True)

        # variant_count = variant_count[:round(len(variant_count)*0.1)]  # JH 0.1, 0.2, 0.3
        # variants_to_filter = [x[0] for x in variant_count]

        # log = log_converter.apply(self.log, variant=log_converter.Variants.TO_EVENT_LOG, parameters=parameters)

        # positive = exec_utils.get_param_value("positive", parameters, True)
        # variants = get_variants(log, parameters=parameters)

        # print("Size of varients : ",  len(variants))
        # log = EventLog(list(), attributes=log.attributes, extensions=log.extensions, classifiers=log.classifiers,
        #                 omni_present=log.omni_present, properties=log.properties)
        # for variant in variants:
        #     if (positive and variant in variants_to_filter) or (not positive and variant not in variants_to_filter):
        #         for trace in variants[variant]:
        #             log.append(trace)
                    
        # log_data1 = pm4py.convert_to_dataframe(log)
        # trace_ids1 = log_data1[self.case_name_key].unique().tolist() 

        # self.training_trace_ids = trace_ids1
        # self.evaluation_trace_ids = trace_ids#[0:round(len(trace_ids)*1/6)]     # test 

    def encode_log(self):
        act_set = list(self.log[self.act_name_key].unique())
        self.act_enc_mapping = dict((chr(idx + shared.ascii_offset), elem) for idx, elem in enumerate(act_set))
        self.log.replace(to_replace={self.act_name_key: {v: k for k, v in self.act_enc_mapping.items()}}, inplace=True)


        # jh
        temp_time1 = self.log[[self.case_name_key, self.timestamp_key]]
        temp_time1['diff'] = temp_time1.groupby(self.case_name_key)[self.timestamp_key].diff().dt.seconds
        temp_time1['diff'].fillna(0, inplace=True)
        temp_time1['diff'] = temp_time1['diff']/max(temp_time1['diff'])  
        temp_time1['diff_cum'] = temp_time1['diff'].cumsum()
        temp_time1['diff_cum'] = temp_time1['diff_cum'] /max(temp_time1['diff_cum'])
        temp_time1['midnight'] = temp_time1[self.timestamp_key].apply(lambda x:  x.replace(hour=0, minute=0, second=0, microsecond=0))
        temp_time1['times3'] = (temp_time1[self.timestamp_key] - temp_time1['midnight']).dt.seconds / 86400
        temp_time1['times4'] = temp_time1[self.timestamp_key].apply(lambda x:  x.weekday() / 7)
        print(temp_time1.head(3))
        
        
        self.log[self.timestamp_key] = temp_time1['diff']
        # self.log[self.timestamp_key2] = temp_time1['diff_cum']
        # self.log[self.timestamp_key3] = temp_time1['times3']
        # self.log[self.timestamp_key4] = temp_time1['times4']
        del temp_time1
        
        return self


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

        elif self.log_name == LogName.SYNTH2:
             self.case_name_key = 'case:concept:name'
             self.act_name_key = 'concept:name'
             self.timestamp_key = 'time:timestamp'
             self.evaluation_prefix_start = 3
             self.evaluation_prefix_end = 7
             self.compliance_th = 1.0
        
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
