import mne.io
import pandas as pd
from progress.bar import Bar

if __name__ == '__main__':
    path_to_egg_data = "../real_EEG_data/"
    time_segment = 180  # seconds - 3 min
    time_list, list_start, list_stop, shape = [], [], [], []

    bar = Bar("Processing the files", max=31)

    # read all the .edf files
    for file in range(1, 32):
        raw_fname = path_to_egg_data + str(file) + '.edf'
        raw = mne.io.read_raw_edf(raw_fname, preload=True)
        data, times = raw[-1, -1:]

        total_time = float(times)
        time_list.append(total_time)

        # Splitting the data into chunks of 3 minutes
        t_start = 0
        t_end = t_start + time_segment
        segment = 0
        while t_end < total_time:
            start, stop = raw.time_as_index([t_start, t_end])
            list_start.append(start)
            list_stop.append(stop)

            data = raw.get_data(start=start, stop=stop)
            df = pd.DataFrame(data,
                              index=['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8', 'ch9', 'ch10', 'ch11',
                                     'ch12',
                                     'ch13', 'ch14', 'ch15'])
            shape.append(df.shape)
            df.to_csv("../real_EEG_data/csv_files/" + str(file) + '_' + str(segment) + '.csv')
            t_start = t_end
            t_end += time_segment
            segment += 1
        bar.next()
        print("\n")
    bar.finish()
