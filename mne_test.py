import mne.io
import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame(columns=['File', 'Segment', 'Time_start', 'Time_end', 'Index_start', 'Index_end'])
time_segment = 180  # seconds - 3 min

for file in range(1, 32):
    raw_fname = '/home/marina/Documents/DTU/Advanced machine learning/Project/Data/real_EEG_data/' + str(file) + '.edf'
    raw = mne.io.read_raw_edf(raw_fname, preload=True)
    data, times = raw[-1, -1:]
    total_time = float(times)
    interval_plot = time_segment * 10 / total_time  # this is because plots work in a decimal scale

    t_start = 0
    t_end = t_start + time_segment
    count_plot = 0
    segment = 1

    while t_end < total_time:
        start, stop = raw.time_as_index([t_start, t_end])
        df = df.append(pd.Series([file, segment, t_start, t_end, start, stop],
                                 index=['File', 'Segment', 'Time_start', 'Time_end', 'Index_start', 'Index_end']),
                       ignore_index=True)

        fig = raw.plot(start=count_plot, duration=interval_plot, show=False);
        path = '/home/marina/Documents/DTU/Advanced machine learning/Project/Data/eeg_plots/' + str(file) + ' segment ' + str(segment) + '.png'
        fig.savefig(path, quality=1)  # save the figure to file
        print(str(file) + ' segment ' + str(segment) + ' from ' + str(t_start) + ' to ' + str(t_end))

        count_plot += interval_plot
        t_start = t_end
        t_end += time_segment
        segment += 1

df.to_csv('labeled_eeg.csv')
