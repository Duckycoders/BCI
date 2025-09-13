""" 
Copyright (C) 2022 King Saud University, Saudi Arabia 
SPDX-License-Identifier: Apache-2.0 

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the 
License at

http://www.apache.org/licenses/LICENSE-2.0  

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR 
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. 

Author:  Hamdi Altaheri 
"""

# Dataset BCI Competition IV-2a is available at 
# http://bnci-horizon-2020.eu/database/data-sets

import os
import numpy as np
import scipy.io as sio
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

# We need the following function to load and preprocess the High Gamma Dataset
from preprocess_HGD import load_HGD_data

def load_BCI2a_MOABB_data(data_path, subject, training):
    """ Loading BCI Competition IV-2a data from MOABB 
    
        Parameters
        ----------
        data_path: string
            dataset path (not used, MOABB handles downloads)
        subject: int
            number of subject in [1, .. ,9]
        training: bool
            if True, load training data
            if False, load testing data
    """
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import MotorImagery
    import numpy as np
    
    # 获取数据 - 使用更宽松的参数设置
    dataset = BNCI2014_001()
    
    # 设置更合适的参数来获取完整数据
    paradigm = MotorImagery(
        events=['left_hand', 'right_hand', 'feet', 'tongue'],
        n_classes=4,
        fmin=8,     # 降低最小频率
        fmax=35,    # 提高最大频率
        tmin=0.5,   # 试次开始时间
        tmax=4.5,   # 试次结束时间
        resample=250  # 重采样频率
    )
    
    X, labels, meta = paradigm.get_data(dataset, [subject])
    
    print(f"总数据量: {len(X)}")
    print(f"会话信息: {meta['session'].unique()}")
    print(f"标签分布: {np.unique(labels, return_counts=True)}")
    
    # 分离训练和测试数据
    if training:
        mask = meta['session'] == '0train'
    else:
        mask = meta['session'] == '1test'
    
    X_data = X[mask]
    y_data = labels[mask]
    
    print(f"{'训练' if training else '测试'}数据量: {len(X_data)}")
    
    # 转换标签为数字 (0-based for the model)
    label_map = {'left_hand': 0, 'right_hand': 1, 'feet': 2, 'tongue': 3}
    y_numeric = np.array([label_map[label] for label in y_data])
    
    return X_data, y_numeric

def load_BCI2a_NPZ_data(data_path, subject, training):
    """ Loading BCI Competition IV-2a data from NPZ format
    
        Parameters
        ----------
        data_path: string
            dataset path containing NPZ files
        subject: int
            number of subject in [1, .. ,9]
        training: bool
            if True, load training data
            if False, load testing data
    """
    import numpy as np
    from scipy import signal
    
    # Define parameters
    n_channels = 22
    fs = 250  # sampling frequency
    trial_length = int(4.5 * fs)  # 4.5 seconds
    
    if training:
        # Load all training sessions for this subject
        all_data = []
        all_events = []
        
        for session in range(6):  # 6 training sessions
            try:
                file_path = f"{data_path}/S{subject:02d}_0train_{session}.npz"
                npz_data = np.load(file_path, allow_pickle=True)
                
                raw_data = npz_data['data']  # Shape: (n_channels, n_samples)
                events = npz_data['events']  # Events array
                
                all_data.append(raw_data)
                all_events.append(events)
                
            except FileNotFoundError:
                print(f"Warning: Training file {file_path} not found")
                continue
                
    else:
        # Load test sessions
        all_data = []
        all_events = []
        
        for session in range(6):  # 6 test sessions  
            try:
                file_path = f"{data_path}/S{subject:02d}_1test_{session}.npz"
                npz_data = np.load(file_path, allow_pickle=True)
                
                raw_data = npz_data['data']  # Shape: (n_channels, n_samples)
                events = npz_data['events']  # Events array
                
                all_data.append(raw_data)
                all_events.append(events)
                
            except FileNotFoundError:
                print(f"Warning: Test file {file_path} not found")
                continue
    
    if not all_data:
        raise ValueError(f"No data found for subject {subject}")
    
    # Combine all sessions
    combined_data = np.concatenate(all_data, axis=1)
    combined_events = np.concatenate(all_events, axis=0) if all_events else np.array([])
    
    # Extract trials based on events
    # This is a simplified version - you may need to adjust based on your data structure
    trials = []
    labels = []
    
    # For now, create some sample trials (you'll need to adapt this based on your event structure)
    n_samples_per_trial = trial_length
    n_trials = min(288, combined_data.shape[1] // n_samples_per_trial)  # Max 288 trials as in original
    
    for i in range(n_trials):
        start_idx = i * n_samples_per_trial
        end_idx = start_idx + n_samples_per_trial
        
        if end_idx <= combined_data.shape[1]:
            trial_data = combined_data[:n_channels, start_idx:end_idx]
            trials.append(trial_data)
            # Assign labels cyclically (you'll need to use actual event labels)
            labels.append(i % 4)
    
    if not trials:
        raise ValueError(f"No valid trials extracted for subject {subject}")
    
    data_return = np.array(trials)  # Shape: (n_trials, n_channels, n_samples)
    class_return = np.array(labels)
    
    return data_return, class_return

#%%
def load_data_LOSO (data_path, subject, dataset): 
    """ Loading and Dividing of the data set based on the 
    'Leave One Subject Out' (LOSO) evaluation approach. 
    LOSO is used for  Subject-independent evaluation.
    In LOSO, the model is trained and evaluated by several folds, equal to the 
    number of subjects, and for each fold, one subject is used for evaluation
    and the others for training. The LOSO evaluation technique ensures that 
    separate subjects (not visible in the training data) are usedto evaluate 
    the model.
    
        Parameters
        ----------
        data_path: string
            dataset path
            # Dataset BCI Competition IV-2a is available at 
            # http://bnci-horizon-2020.eu/database/data-sets
        subject: int
            number of subject in [1, .. ,9/14]
            Here, the subject data is used  test the model and other subjects data
            for training
    """
    
    X_train, y_train = [], []
    for sub in range (0,9):
        path = data_path+'s' + str(sub+1) + '/'
        
        if (dataset == 'BCI2a'):
            X1, y1 = load_BCI2a_data(path, sub+1, True)
            X2, y2 = load_BCI2a_data(path, sub+1, False)
        elif (dataset == 'CS2R'):
            X1, y1, _, _, _  = load_CS2R_data_v2(path, sub, True)
            X2, y2, _, _, _  = load_CS2R_data_v2(path, sub, False)
        elif (dataset == 'HGD'):
            X1, y1 = load_HGD_data(path, sub+1, True)
            X2, y2 = load_HGD_data(path, sub+1, False)
        
        X = np.concatenate((X1, X2), axis=0)
        y = np.concatenate((y1, y2), axis=0)
                   
        if (sub == subject):
            X_test = X
            y_test = y
        elif len(X_train) == 0:  
            X_train = X
            y_train = y
        else:
            X_train = np.concatenate((X_train, X), axis=0)
            y_train = np.concatenate((y_train, y), axis=0)

    return X_train, y_train, X_test, y_test


#%%
def load_BCI2a_data(data_path, subject, training, all_trials = True):
    """ Loading and Dividing of the data set based on the subject-specific 
    (subject-dependent) approach.
    In this approach, we used the same training and testing dataas the original
    competition, i.e., 288 x 9 trials in session 1 for training, 
    and 288 x 9 trials in session 2 for testing.  
   
        Parameters
        ----------
        data_path: string
            dataset path
            # Dataset BCI Competition IV-2a is available on 
            # http://bnci-horizon-2020.eu/database/data-sets
        subject: int
            number of subject in [1, .. ,9]
        training: bool
            if True, load training data
            if False, load testing data
        all_trials: bool
            if True, load all trials
            if False, ignore trials with artifacts 
    """
    
    # Define MI-trials parameters
    n_channels = 22
    n_tests = 6*48     
    window_Length = 7*250 
    
    # Define MI trial window 
    fs = 250          # sampling rate
    t1 = int(1.5*fs)  # start time_point
    t2 = int(6*fs)    # end time_point

    class_return = np.zeros(n_tests)
    data_return = np.zeros((n_tests, n_channels, window_Length))

    NO_valid_trial = 0
    if training:
        a = sio.loadmat(data_path+'A0'+str(subject)+'T.mat')
    else:
        a = sio.loadmat(data_path+'A0'+str(subject)+'E.mat')
    a_data = a['data']
    for ii in range(0,a_data.size):
        a_data1 = a_data[0,ii]
        a_data2= [a_data1[0,0]]
        a_data3= a_data2[0]
        a_X         = a_data3[0]
        a_trial     = a_data3[1]
        a_y         = a_data3[2]
        a_artifacts = a_data3[5]

        for trial in range(0,a_trial.size):
             if(a_artifacts[trial] != 0 and not all_trials):
                 continue
             data_return[NO_valid_trial,:,:] = np.transpose(a_X[int(a_trial[trial]):(int(a_trial[trial])+window_Length),:22])
             class_return[NO_valid_trial] = int(a_y[trial])
             NO_valid_trial +=1        
    

    data_return = data_return[0:NO_valid_trial, :, t1:t2]
    class_return = class_return[0:NO_valid_trial]
    class_return = (class_return-1).astype(int)

    return data_return, class_return



#%%
import json
from mne.io import read_raw_edf
from dateutil.parser import parse
import glob as glob
from datetime import datetime

def load_CS2R_data_v2(data_path, subject, training, 
                      classes_labels =  ['Fingers', 'Wrist','Elbow','Rest'], 
                      all_trials = True):
    """ Loading training/testing data for the CS2R motor imagery dataset
    for a specific subject        
   
        Parameters
        ----------
        data_path: string
            dataset path
        subject: int
            number of subject in [1, .. ,9]
        training: bool
            if True, load training data
            if False, load testing data
        classes_labels: tuple
            classes of motor imagery returned by the method (default: all) 
    """
    
    # Get all subjects files with .edf format.
    subjectFiles = glob.glob(data_path + 'S_*/')
    
    # Get all subjects numbers sorted without duplicates.
    subjectNo = list(dict.fromkeys(sorted([x[len(x)-4:len(x)-1] for x in subjectFiles])))
    # print(SubjectNo[subject].zfill(3))
    
    if training:  session = 1
    else:         session = 2
    
    num_runs = 5
    sfreq = 250 #250
    mi_duration = 4.5 #4.5

    data = np.zeros([num_runs*51, 32, int(mi_duration*sfreq)])
    classes = np.zeros(num_runs * 51)
    valid_trails = 0
    
    onset = np.zeros([num_runs, 51])
    duration = np.zeros([num_runs, 51])
    description = np.zeros([num_runs, 51])

    #Loop to the first 4 runs.
    CheckFiles = glob.glob(data_path + 'S_' + subjectNo[subject].zfill(3) + '/S' + str(session) + '/*.edf')
    if not CheckFiles:
        return 
    
    for runNo in range(num_runs): 
        valid_trails_in_run = 0
        #Get .edf and .json file for following subject and run.
        EDFfile = glob.glob(data_path + 'S_' + subjectNo[subject].zfill(3) + '/S' + str(session) + '/S_'+subjectNo[subject].zfill(3)+'_'+str(session)+str(runNo+1)+'*.edf')
        JSONfile = glob.glob(data_path + 'S_'+subjectNo[subject].zfill(3) + '/S'+ str(session) +'/S_'+subjectNo[subject].zfill(3)+'_'+str(session)+str(runNo+1)+'*.json')
    
        #Check if EDFfile list is empty
        if not EDFfile:
          continue
    
        # We use mne.read_raw_edf to read in the .edf EEG files
        raw = read_raw_edf(str(EDFfile[0]), preload=True, verbose=False)
        
        # Opening JSON file of the current RUN.
        f = open(JSONfile[0],) 
    
        # returns JSON object as a dictionary 
        JSON = json.load(f) 
    
        #Number of Keystrokes Markers
        keyStrokes = np.min([len(JSON['Markers']), 51]) #len(JSON['Markers']), to avoid extra markers by accident
        # MarkerStart = JSON['Markers'][0]['startDatetime']
           
        #Get Start time of marker
        date_string = EDFfile[0][-21:-4]
        datetime_format = "%d.%m.%y_%H.%M.%S"
        startRecordTime = datetime.strptime(date_string, datetime_format).astimezone()
    
        currentTrialNo = 0 # 1 = fingers, 2 = Wrist, 3 = Elbow, 4 = rest
        if(runNo == 4): 
            currentTrialNo = 4
    
        ch_names = raw.info['ch_names'][4:36]
             
        # filter the data 
        raw.filter(4., 50., fir_design='firwin')  
        
        raw = raw.copy().pick_channels(ch_names = ch_names)
        raw = raw.copy().resample(sfreq = sfreq)
        fs = raw.info['sfreq']

        for trail in range(keyStrokes):
            
            # class for current trial
            if(runNo == 4 ):               # In Run 5 all trials are 'reset'
                currentTrialNo = 4
            elif (currentTrialNo == 3):    # Set the class of current trial to 1 'Fingers'
                currentTrialNo = 1   
            else:                          # In Runs 1-4, 1st trial is 1 'Fingers', 2nd trial is 2 'Wrist', and 3rd trial is 'Elbow', and repeat ('Fingers', 'Wrist', 'Elbow', ..)
                currentTrialNo = currentTrialNo + 1
                
            trailDuration = 8
            
            trailTime = parse(JSON['Markers'][trail]['startDatetime'])
            trailStart = trailTime - startRecordTime
            trailStart = trailStart.seconds 
            start = trailStart + (6 - mi_duration)
            stop = trailStart + 6

            if (trail < keyStrokes-1):
                trailDuration = parse(JSON['Markers'][trail+1]['startDatetime']) - parse(JSON['Markers'][trail]['startDatetime'])
                trailDuration =  trailDuration.seconds + (trailDuration.microseconds/1000000)
                if (trailDuration < 7.5) or (trailDuration > 8.5):
                    print('In Session: {} - Run: {}, Trail no: {} is skipped due to short/long duration of: {:.2f}'.format(session, (runNo+1), (trail+1), trailDuration))
                    if (trailDuration > 14 and trailDuration < 18):
                        if (currentTrialNo == 3):   currentTrialNo = 1   
                        else:                       currentTrialNo = currentTrialNo + 1
                    continue
                
            elif (trail == keyStrokes-1):
                trailDuration = raw[0, int(trailStart*int(fs)):int((trailStart+8)*int(fs))][0].shape[1]/fs
                if (trailDuration < 7.8) :
                    print('In Session: {} - Run: {}, Trail no: {} is skipped due to short/long duration of: {:.2f}'.format(session, (runNo+1), (trail+1), trailDuration))
                    continue

            MITrail = raw[:32, int(start*int(fs)):int(stop*int(fs))][0]
            if (MITrail.shape[1] != data.shape[2]):
                print('Error in Session: {} - Run: {}, Trail no: {} due to the lost of data'.format(session, (runNo+1), (trail+1)))
                return
            
            # select some specific classes
            if ((('Fingers' in classes_labels) and (currentTrialNo==1)) or 
            (('Wrist' in classes_labels) and (currentTrialNo==2)) or 
            (('Elbow' in classes_labels) and (currentTrialNo==3)) or 
            (('Rest' in classes_labels) and (currentTrialNo==4))):
                data[valid_trails] = MITrail
                classes[valid_trails] =  currentTrialNo
                
                # For Annotations
                onset[runNo, valid_trails_in_run]  = start
                duration[runNo, valid_trails_in_run] = trailDuration - (6 - mi_duration)
                description[runNo, valid_trails_in_run] = currentTrialNo
                valid_trails += 1
                valid_trails_in_run += 1
                         
    data = data[0:valid_trails, :, :]
    classes = classes[0:valid_trails]
    classes = (classes-1).astype(int)

    return data, classes, onset, duration, description


#%%
def standardize_data(X_train, X_test, channels): 
    # X_train & X_test :[Trials, 1, Channels, Time points]
    for j in range(channels):
          scaler = StandardScaler()
          # 对每个通道的所有时间点进行标准化
          train_channel_data = X_train[:, 0, j, :].reshape(-1, 1)
          scaler.fit(train_channel_data)
          
          # 应用标准化
          X_train[:, 0, j, :] = scaler.transform(X_train[:, 0, j, :].reshape(-1, 1)).reshape(-1, X_train.shape[-1])
          X_test[:, 0, j, :] = scaler.transform(X_test[:, 0, j, :].reshape(-1, 1)).reshape(-1, X_test.shape[-1])

    return X_train, X_test


#%%
def get_data(path, subject, dataset = 'BCI2a', classes_labels = 'all', LOSO = False, isStandard = True, isShuffle = True):
    
    # Load and split the dataset into training and testing 
    if LOSO:
        """ Loading and Dividing of the dataset based on the 
        'Leave One Subject Out' (LOSO) evaluation approach. """ 
        X_train, y_train, X_test, y_test = load_data_LOSO(path, subject, dataset)
    else:
        """ Loading and Dividing of the data set based on the subject-specific 
        (subject-dependent) approach.
        In this approach, we used the same training and testing data as the original
        competition, i.e., for BCI Competition IV-2a, 288 x 9 trials in session 1 
        for training, and 288 x 9 trials in session 2 for testing.  
        """
        if (dataset == 'BCI2a'):
            # Use MOABB to load BCI Competition IV-2a data directly
            X_train, y_train = load_BCI2a_MOABB_data(path, subject+1, True)
            X_test, y_test = load_BCI2a_MOABB_data(path, subject+1, False)
        elif (dataset == 'CS2R'):
            X_train, y_train, _, _, _ = load_CS2R_data_v2(path, subject, True, classes_labels)
            X_test, y_test, _, _, _ = load_CS2R_data_v2(path, subject, False, classes_labels)
        elif (dataset == 'HGD'):
            X_train, y_train = load_HGD_data(path, subject+1, True)
            X_test, y_test = load_HGD_data(path, subject+1, False)
        else:
            raise Exception("'{}' dataset is not supported yet!".format(dataset))

    # shuffle the data 
    if isShuffle:
        X_train, y_train = shuffle(X_train, y_train,random_state=42)
        X_test, y_test = shuffle(X_test, y_test,random_state=42)

    # Prepare training data     
    N_tr, N_ch, T = X_train.shape 
    X_train = X_train.reshape(N_tr, 1, N_ch, T)
    y_train_onehot = to_categorical(y_train)
    # Prepare testing data 
    N_tr, N_ch, T = X_test.shape 
    X_test = X_test.reshape(N_tr, 1, N_ch, T)
    y_test_onehot = to_categorical(y_test)    
    
    # Standardize the data
    if isStandard:
        X_train, X_test = standardize_data(X_train, X_test, N_ch)

    return X_train, y_train, y_train_onehot, X_test, y_test, y_test_onehot

