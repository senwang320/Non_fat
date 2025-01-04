## need to run using total segmentation environment which is set with conda environment
# for this workstation, using 'conda activate totalseg-v2'

import os
import pandas as pd

# dose segmentation
nii_out_dir = '/data1/AEC_SharedFiles/Stanford_Data/Exams_niis/dose_niis'
seg_out_dir = '/data1/AEC_SharedFiles/Stanford_Data/Segmentations/TS_dose_segs'

# # noise segmenation
# nii_out_dir = '/data1/AEC_SharedFiles/Stanford_Data/Exams_niis/noise_niis/'
# seg_out_dir = '/data1/AEC_SharedFiles/Stanford_Data/Segmentations/TS_noise_segs/'

# df = pd.read_excel('./scan_records/scan_data_Corhort_ReaderStudy.xlsx')

# source records from GEHC
src_excel_fname = '/data1/AEC_SharedFiles/Stanford_Data/ScanDataSummary.xlsx'

# read data frame from source records and select effective cases with some filters
df     = pd.read_excel(src_excel_fname, sheet_name='Data')
eff_df = df[(df['Images Needed']=='Normal, Dose, Noise') & df['Dose Images Uploaded']]
df     = eff_df
df     = df.reset_index(drop=True)

# basenames for dose images
basenames = ['SHC2.{}.{}.{}_doseims'.format(int(df['Exam Number'][ii]), int(df['Series Number'][ii]), int(df['Scan Number'][ii])) 
                for ii in range(len(df['Exam Number']))] 

# basenames = ['SHC2.{}.{}.{}'.format(int(df['Exam Number'][ii]), int(df['Series Number'][ii]), int(df['Scan Number'][ii])) 
#                 for ii in range(len(df['Exam Number']))] 

# basenames = basenames[0:2]

for ii, this_name in enumerate(basenames):

    print('Processing {}/{}: {}'.format(ii, len(basenames), this_name))
    # # default segmentations
    # seg_cmd = 'TotalSegmentator -i {}/{}.nii.gz -o {}/{}'.format(nii_out_dir, this_name, seg_out_dir, this_name)
    # print(seg_cmd)
    # os.system(seg_cmd)
    
    # # segmentation for body (skin included)
    # seg_cmd = 'TotalSegmentator -i {}/{}.nii.gz -o {}/{} -ta body'.format(nii_out_dir, this_name, seg_out_dir, this_name)
    # os.system(seg_cmd)``
    
    seg_cmd = 'TotalSegmentator -i {}/{}.nii.gz -o {}/{} -ta tissue_types'.format(nii_out_dir, this_name, seg_out_dir, this_name)
    os.system(seg_cmd)
