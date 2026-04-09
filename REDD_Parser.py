import numpy        as np
import pandas       as pd
from   pathlib      import Path
from   collections  import defaultdict
from   NILM_Dataset import *
from   Pretrain_Dataset import *


class Redd_Parser:

    SUPPORTED_APPLIANCES = ['dishwasher', 'refrigerator', 'microwave', 'washer_dryer']
    CLEANED_SAMPLING     = '6s'
    CLEANED_NAME_MAP     = {
        'main'         : 'aggregate',
        'fridge'       : 'refrigerator',
        'dish washer'  : 'dishwasher',
        'washer dryer' : 'washer_dryer',
    }

    def __init__(self,args,stats = None):
        self.data_location   = Path(args.redd_location)
        self.house_indicies  = args.house_indicies
        self.appliance_names = args.appliance_names
        self.sampling        = args.sampling
        self.normalize       = args.normalize


        self.cutoff          =  [args.cutoff[appl]    for appl in ['aggregate']+args.appliance_names]
        self.threshold       =  [args.threshold[appl] for appl in args.appliance_names]
        self.min_on          =  [args.min_on[appl]    for appl in args.appliance_names]
        self.min_off         =  [args.min_off[appl]   for appl in args.appliance_names]

        self.val_size        =  args.validation_size
        self.window_size     =  args.window_size
        self.window_stride   =  args.window_stride

        self.x, self.y       = self.load_data()

        if self.normalize == 'mean':
            if stats is None:
                self.x_mean = np.mean(self.x)
                self.x_std  = np.std(self.x)
            else:
                self.x_mean,self.x_std = stats
            self.x = (self.x - self.x_mean) / self.x_std
        elif self.normalize == 'minmax':
            self.x_min = min(self.x)
            self.x_max = max(self.x)
            self.x = (self.x - self.x_min)/(self.x_max-self.x_min)
        self.status          = self.compute_status(self.y)

        
    def load_data(self):
        for appliance in self.appliance_names:
            assert appliance in self.SUPPORTED_APPLIANCES

        for house_id in self.house_indicies:
            assert house_id in [1, 2, 3, 4, 5, 6]

        if self._has_cleaned_layout():
            entire_data = self._load_cleaned_data()
        elif self._has_raw_layout():
            entire_data = self._load_raw_data()
        else:
            raise FileNotFoundError(
                f'No REDD data found under {self.data_location}. '
                'Expected either house_*/channel_*.dat files or cleaned redd_house*_*.csv files.'
            )

        selected_columns = ['aggregate'] + self.appliance_names
        entire_data      = entire_data[selected_columns].apply(pd.to_numeric, errors='coerce')
        entire_data      = entire_data.dropna().copy()
        entire_data      = entire_data[entire_data['aggregate'] > 0]
        entire_data      = entire_data.mask(entire_data < 5, 0)
        entire_data      = entire_data.clip(lower=0)
        entire_data      = entire_data.clip(upper=pd.Series(self.cutoff, index=selected_columns), axis=1)

        return entire_data['aggregate'].to_numpy(), entire_data[self.appliance_names[0]].to_numpy()

    def _has_cleaned_layout(self):
        return any(self.data_location.glob('redd_house*_*.csv'))

    def _has_raw_layout(self):
        return any(self.data_location.glob('house_*'))

    def _load_cleaned_data(self):
        house_frames = []

        for house_id in self.house_indicies:
            segment_paths = sorted(
                self.data_location.glob(f'redd_house{house_id}_*.csv'),
                key=self._cleaned_segment_key,
            )
            if not segment_paths:
                continue

            segment_frames     = []
            house_has_appliance = False

            for segment_path in segment_paths:
                segment_data = pd.read_csv(segment_path, index_col=0)
                segment_data = segment_data.rename(columns=self._normalize_cleaned_column_name)

                if 'aggregate' not in segment_data.columns:
                    raise ValueError(f'Cleaned REDD file {segment_path} is missing the aggregate/main column.')

                available_appliances = [appliance for appliance in self.appliance_names if appliance in segment_data.columns]
                house_has_appliance  = house_has_appliance or bool(available_appliances)

                for appliance in self.appliance_names:
                    if appliance not in segment_data.columns:
                        segment_data[appliance] = 0.0

                segment_data = segment_data[['aggregate'] + self.appliance_names].copy()
                segment_data = self._resample_cleaned_segment(segment_data)
                segment_frames.append(segment_data.reset_index(drop=True))

            if house_has_appliance and segment_frames:
                house_frames.append(pd.concat(segment_frames, ignore_index=True))

        if not house_frames:
            raise ValueError(
                f'None of the requested appliances {self.appliance_names} were found in cleaned REDD files at {self.data_location}.'
            )

        return pd.concat(house_frames, ignore_index=True)

    def _load_raw_data(self):
        house_frames = []

        for house_id in self.house_indicies:
            house_folder = self.data_location.joinpath('house_' + str(house_id))
            if not house_folder.exists():
                continue

            house_label = pd.read_csv(house_folder.joinpath('labels.dat'), sep=' ', header=None)
            main_1      = pd.read_csv(house_folder.joinpath('channel_1.dat'), sep=' ', header=None)
            main_2      = pd.read_csv(house_folder.joinpath('channel_2.dat'), sep=' ', header=None)

            house_data            = pd.merge(main_1, main_2, how='inner', on=0)
            house_data.iloc[:, 1] = house_data.iloc[:, 1] + house_data.iloc[:, 2]
            house_data            = house_data.iloc[:, 0: 2]
            house_data.columns    = ['time', 'aggregate']
            house_data['time']    = pd.to_datetime(house_data['time'], unit='s')
            house_data            = house_data.set_index('time').resample(self.sampling).mean().ffill(limit=30)

            app_index_dict = self._get_raw_appliance_channels(house_label)
            if all(channels == [-1] for channels in app_index_dict.values()):
                continue

            for appliance in self.appliance_names:
                channel_indices = app_index_dict[appliance]
                if channel_indices == [-1]:
                    house_data[appliance] = 0.0
                    continue

                appliance_frames = []
                for channel_idx in channel_indices:
                    channel_path = house_folder.joinpath('channel_' + str(channel_idx) + '.dat')
                    if not channel_path.exists():
                        continue

                    appl_data         = pd.read_csv(channel_path, sep=' ', header=None)
                    appl_data.columns = ['time', appliance]
                    appl_data['time'] = pd.to_datetime(appl_data['time'], unit='s')
                    appl_data         = appl_data.set_index('time').resample(self.sampling).mean().ffill(limit=30)
                    appliance_frames.append(appl_data[appliance])

                if not appliance_frames:
                    house_data[appliance] = 0.0
                    continue

                summed_appliance = pd.concat(appliance_frames, axis=1, join='inner').sum(axis=1).to_frame(name=appliance)
                house_data       = house_data.join(summed_appliance, how='inner')

            house_frames.append(house_data[['aggregate'] + self.appliance_names].reset_index(drop=True))

        if not house_frames:
            raise ValueError(
                f'None of the requested appliances {self.appliance_names} were found in raw REDD data at {self.data_location}.'
            )

        return pd.concat(house_frames, ignore_index=True)

    def _get_raw_appliance_channels(self, house_label):
        labels         = house_label.iloc[:, 1].astype(str).str.strip()
        channel_ids    = house_label.iloc[:, 0].astype(int)
        app_index_dict = defaultdict(list)

        for appliance in self.appliance_names:
            matches = channel_ids[labels == appliance].tolist()
            app_index_dict[appliance] = matches if matches else [-1]

        return app_index_dict

    def _normalize_cleaned_column_name(self, column_name):
        normalized = str(column_name).strip()
        return self.CLEANED_NAME_MAP.get(normalized, normalized)

    def _cleaned_segment_key(self, segment_path):
        return int(segment_path.stem.split('_')[-1])

    def _resample_cleaned_segment(self, segment_data):
        if self.sampling == self.CLEANED_SAMPLING:
            return segment_data

        # Cleaned REDD CSVs are already sampled; synthesize a time index only to support coarser re-binning.
        synthetic_time       = pd.date_range('1970-01-01', periods=len(segment_data), freq=self.CLEANED_SAMPLING)
        segment_data         = segment_data.copy()
        segment_data.index   = synthetic_time
        return segment_data.resample(self.sampling).mean().ffill(limit=30)
    
    
    def compute_status(self, data):

        initial_status = data >= self.threshold[0]
        status_diff    = np.diff(initial_status)
        events_idx     = status_diff.nonzero()

        events_idx  = np.array(events_idx).squeeze()
        events_idx += 1

        if initial_status[0]:
            events_idx = np.insert(events_idx, 0, 0)

        if initial_status[-1]:
            events_idx = np.insert(events_idx, events_idx.size, initial_status.size)

        events_idx     = events_idx.reshape((-1, 2))
        on_events      = events_idx[:, 0].copy()
        off_events     = events_idx[:, 1].copy()
        assert len(on_events) == len(off_events)

        if len(on_events) > 0:
            off_duration = on_events[1:] - off_events[:-1]
            off_duration = np.insert(off_duration, 0, 1000)
            on_events    = on_events[off_duration > self.min_off[0]]
            off_events   = off_events[np.roll(off_duration, -1) > self.min_off[0]]

            on_duration  = off_events - on_events
            on_events    = on_events[on_duration  >= self.min_on[0]]
            off_events   = off_events[on_duration >= self.min_on[0]]
            assert len(on_events) == len(off_events)

        temp_status = data.copy()
        temp_status[:] = 0
        for on, off in zip(on_events, off_events):
            temp_status[on: off] = 1
        status = temp_status

        return status    
    
    def get_train_datasets(self):
        val_end = int(self.val_size * len(self.x))
        
        train = NILMDataset(self.x[val_end:],
                            self.y[val_end:],
                            self.status[val_end:],
                            self.window_size,
                            self.window_stride)
        
        val   = NILMDataset(self.x[:val_end],
                            self.y[:val_end],
                            self.status[:val_end],
                            self.window_size,
                            self.window_size) #non-overlapping windows

        return train, val

    def get_pretrain_datasets(self, mask_prob=0.25):
        val_end = int(self.val_size * len(self.x))

        val     = NILMDataset(self.x[:val_end],
                               self.y[:val_end],
                               self.status[:val_end],
                               self.window_size,
                               self.window_size
                             )
        train   = Pretrain_Dataset(self.x[val_end:],
                                   self.y[val_end:],
                                   self.status[val_end:],
                                   self.window_size,
                                   self.window_stride,
                                   mask_prob=mask_prob
                                   )
        return train, val        
