# This experiment was created by Kirsten Ziman (Kirsten.K.Ziman.gr@Dartmouth.edu)
# for more information, visit: www.context-lab.com

# Imports ############################################################################################

from experiment_helpers import * # main functions in experiment_helpers.fpy
import pandas as pd
import csv
import random

# Set Up #############################################################################################

# Parameters #
experiment_title = 'Attention and Memory'
practice = True  # instructions & practice

params = {'runs':8, 'presentations_per_run':10, 'invalid_cue_percentage':10, 'mem_to_pres':4, 'mem_pres_split':2}
categories = {'cat1':'Faces', 'cat2':'Places', 'composites':'composites'}
paths = {'data_path':'../../data/', 'stim_path':'../../stim/'}

# Obtain participant info (pop-up) and make subdir #
info = subject_info(experiment_title)
paths['subject'] = subject_directory(info, paths['data_path'])

# Initiate clock #
global_clock = core.Clock()
logging.setDefaultClock(global_clock)

# initiate log file
logDat = logging.LogFile(paths['subject']+'-'+info['run']+'.log', filemode='w', level = logging.DEBUG)
logging.log(level=logging.WARN, msg='warning')
logging.log(level=logging.EXP , msg='experiment')
logging.log(level=logging.DATA, msg='data')
logging.log(level=logging.INFO, msg='info')

# timestamp for reference
logging.info('current time: '+str(time.time()))

# Pre questionnaire #
if int(info['run'])==-1:
    pre_info = pre_questionnaire(info, save_path=paths['subject'])
else:
    practice = False

# Window and Stimulus timing #
win = visual.Window([1024,768], fullscr = True, monitor = 'testMonitor', units='deg', color = 'black', allowGUI=False)
win.setMouseVisible = False

rate = win.getActualFrameRate()
timing = {'cue':int(round(1.5 * rate)), 'probe':int(round(3.0 * rate)), 'mem':int(round(2 * rate)), 'pause':int(round(1 *rate))}


# Run Experiment #####################################################################################

# Hide mouse
event.Mouse(visible=False)

# Instructions & Practice #
if practice:
    
    cats = random.sample(['Face','Place'], k=2)
    sids = random.sample(['<','>'], k=2)
    
    for x in range(11):
        if x ==8:
            practice_instructions(win, paths, pract_text(x), x, timing, acceptedKeys = [], practice=True, 
            cat = cats[0], sid = sids[0])
            
        
        elif x == 9:
            practice_instructions(win, paths, pract_text(x), x, timing, acceptedKeys = [], practice=True, 
            cat = cats[1], sid = sids[1])
        
        else:
            practice_instructions(win, paths, pract_text(x), x, timing, acceptedKeys = [], practice=True)
     
    logging.info('PRACTICE END: '+str(time.time()))
    
    # experiment stops for recalibration) #
    text_present(win, 'Next, we will pause to set up the eye tracker. You can take a break and hit any key when you are ready.',
                     close=True, timing = timing)

# Initialize dataframe #
if int(info['run'])==-1:
    df = initialize_df(info, categories, paths, params)
else:
    df = pd.DataFrame.from_csv(paths['subject']+'intial_df.csv')

# Create df masks #
mask1 = df['Trial Type']=='Presentation'
mask2 = df['Trial Type']=='Memory'

# Pres & Mem runs #
for run in range(int(info['run']),params['runs']):

    # chunk dataframe #
    mask3 = df['Run']==run

    # presentation run #
    text_present(win, pres_text(run))
    presentation_run(win, run, df.loc[mask1][mask3], params, timing, paths)

    # memory run #
    text_present(win, mem_text(run))
    memory_run(win, run, df.loc[mask2][mask3], params, timing, paths)

    # experiment stops for recalibration) #
    if run != params['runs']-1:
        text_present(win, 'Next, we will pause to recalibrate the eye tracker. You can take a break and hit any key when you are ready for recalibration.',
                     close=True, timing = timing)

    # after last memory run #
    else:
        # closing message and post-questionnaire #
        text_present(win, 'Thank you for your participation!', timing=timing, close=True)
        post_info = post_questionnaire(info, save_path=paths['subject'])

