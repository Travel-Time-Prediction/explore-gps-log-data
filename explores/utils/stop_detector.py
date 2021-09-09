import pandas as pd
import math

from tqdm.auto import tqdm

tqdm.pandas(desc='working : ')

class StopDetector:
    def __init__(self, df):
        self.df = df

    def get_stop_time_range(self, max_diameter, min_duration):
        uid_lists = self._process_uid()
        all_detected_stops = []
        for uid in tqdm(uid_lists[:]):
            # print(f"uid : {uid}")
            df_uid = (self.df[self.df['unit_id'] == uid].sort_values(by=['time_stamp'])).reset_index(drop=True)
            all_detected_stops = all_detected_stops + self._process_stop(uid, df_uid, max_diameter, min_duration) 
        return pd.DataFrame(all_detected_stops)
    
    def _process_stop(self, uid, df_uid, max_diameter, min_duration):
        detected_stops = []
        segment_geoms = []
        segment_times = []
        is_stopped = False
        previously_stopped = False

        for _, row in df_uid.iterrows():
            segment_geoms.append(row[['lat', 'lon']].values)
            segment_times.append(row['time_stamp'])

            if not is_stopped:
                while len(segment_geoms) > 2 and (segment_times[-1] - segment_times[0]).total_seconds() >= min_duration:
                    # print(f"duration : {(segment_times[-1] - segment_times[0]).total_seconds()}")
                    segment_geoms.pop(0)
                    segment_times.pop(0)

            # print(f"segment geom : {segment_geoms}")
            # print(f"segment time : {segment_times}")
            # print(f"distance : {self._process_distance(segment_geoms)}")

            if len(segment_geoms) > 1 and self._process_distance(segment_geoms) < max_diameter:
                is_stopped = True
            else:
                is_stopped = False

            # print(f"stopped : {is_stopped}")

            if len(segment_geoms) > 1:
                segment_time_end = segment_times[-2]
                segment_time_begin = segment_times[0]
                segment_geom_begin = segment_geoms[0]
                if not is_stopped and previously_stopped:
                    # print(f"duration 2 : {(segment_time_end - segment_time_begin).total_seconds()}")
                    if (segment_time_end - segment_time_begin).total_seconds() >= min_duration:
                        detected_stops.append({
                            'unit_id': uid,
                            'lat': segment_geom_begin[0],
                            'lon': segment_geom_begin[1],
                            'time_start': segment_time_begin,
                            'time_stop': segment_time_end
                        })
                        segment_geoms = []
                        segment_times = []

            previously_stopped = is_stopped

        if is_stopped and (segment_times[-1] - segment_times[0]).total_seconds() >= min_duration:
            detected_stops.append({
                'unit_id': uid,
                'lat': segment_geoms[0][0],
                'lon': segment_geoms[0][1],
                'time_start': segment_times[0],
                'time_stop': segment_times[-1]
            })

        return detected_stops
            
    def _process_uid(self):
        uid_list = self.df['unit_id'].to_list()
        uid_list = list(set(uid_list))
        return uid_list

    def _process_distance(self, geom):
        if len(geom) == 1:
            return 0
        if len(geom) == 2:
            return self._process_meansure_distance(geom[0], geom[1])
        return self._process_meansure_distance(geom[0], geom[-1])

    def _process_meansure_distance(self, point1, point2):
        lon1 = float(point1[1])
        lon2 = float(point2[1])
        lat1 = float(point1[0])
        lat2 = float(point2[0])
        distance = math.sqrt(((lat2 - lat1) ** 2) + ((lon2 - lon1) ** 2))
        return distance * 100000