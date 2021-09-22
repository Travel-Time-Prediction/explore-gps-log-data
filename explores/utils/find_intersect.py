import math
import pandas as pd
import matplotlib.pyplot as plt

class FindIntersect:
    def __init__(self, df_gps):
        self.df_gps = df_gps
        self.df_road = None
        self.coor_road_lists = None
        self.intersect_lists = []
        self.road_data_dir = 'E:/data/road'

        # init dataframe gps for find intersect
        self.df_gps['coor'] = self.df_gps.apply(lambda row: f"{row['lat']:.3f}, {row['lon']:.3f}", axis=1)

    def set_raod_stop_point(self, rd, df_stop):
        self.df_stop = df_stop
        self.coor_road_lists = self._process_df_road(rd)

    def _process_df_road(self, rd):
        # get road dataframe
        self.df_road = self._get_road_data(rd)

        # clean dataframe
        self.df_road['coor'] = self.df_road.apply(lambda row: f"{row['latx']:.3f}, {row['lonx']:.3f}", axis=1)
        df_road = self.df_road.drop_duplicates(subset=['coor']).reset_index(drop=True)

        coor_lists = df_road['coor'].to_list()

        return coor_lists

    def _get_road_data(self, rd):
        df_road = pd.read_csv(self.road_data_dir + '/roaddb.csv')
        df_road = df_road[df_road['rd'] == rd].reset_index(drop=True)
        return df_road

    def find_intersect(self, cluster_start, cluster_stop):
        df_gps_tmp = self.df_gps[self.df_gps['coor'].isin(self.coor_road_lists)].reset_index(drop=True) 

        
        # init min max coor start value
        start_lat_lon, stop_lat_lon = self._process_lat_lon(cluster_start, cluster_stop)

        # start
        # condition filter start
        cond_lat_start = (df_gps_tmp['lat'] >= start_lat_lon[0]) & (df_gps_tmp['lat'] <= start_lat_lon[1])
        cond_lon_start = (df_gps_tmp['lon'] >= start_lat_lon[2]) & (df_gps_tmp['lon'] <= start_lat_lon[3])

        df_start_tmp = df_gps_tmp[cond_lat_start & cond_lon_start].reset_index(drop=True)
        uid_lists = list(set((df_start_tmp['unit_id'].to_list())))
        
        # stop
        # condition filter stop
        cond_lat_stop = (df_gps_tmp['lat'] >= stop_lat_lon[0]) & (df_gps_tmp['lat'] <= stop_lat_lon[1])
        cond_lon_stop = (df_gps_tmp['lon'] >= stop_lat_lon[2]) & (df_gps_tmp['lon'] <= stop_lat_lon[3])
        cond_uid_stop = df_gps_tmp['unit_id'].isin(uid_lists)

        df_intersect_tmp = df_gps_tmp[cond_lat_stop & cond_lon_stop & cond_uid_stop].reset_index(drop=True)
        uid_intersect_lists = list(set(df_intersect_tmp['unit_id'].to_list()))

        # clear data in memory
        del df_gps_tmp
        del df_start_tmp

        return df_intersect_tmp, uid_intersect_lists

    def _process_lat_lon(self, cluster_start, cluster_stop):
        lat_start, lon_start, lat_stop, lon_stop = self._process_centroid(cluster_start, cluster_stop)

        start_lat_lon = (lat_start - (1 * 0.01), lat_start + (1 * 0.01), lon_start - (1 * 0.01), lon_start + (1 * 0.01))
        stop_lat_lon = (lat_stop - (1 * 0.01), lat_stop + (1 * 0.01), lon_stop - (1 * 0.01), lon_stop + (1 * 0.01))

        return start_lat_lon, stop_lat_lon

    def find_intersect_all(self):
        self.intersect_lists = []
        label_lists = list(set(self.df_stop['label'].to_list()))
        for label_start in label_lists:
            for label_stop in label_lists:
                if label_start != label_stop:
                    _, uid_list = self.find_intersect(label_start, label_stop)
                    if len(uid_list) > 50:
                        self.intersect_lists.append(
                            {
                                'start_point': label_start,
                                'stop_point': label_stop,
                                'amount': len(uid_list),
                                'distance': self._process_distance(label_start, label_stop)
                            }
                        )

        self.intersect_lists = sorted(self.intersect_lists, key=lambda row: (row['distance'], row['amount']), reverse=True)

    def _process_distance(self, cluster_start, cluster_stop):
        lat_start, lon_start = self.df_stop.loc[self.df_stop['label'] == cluster_start, ['lat', 'lon']].mean()
        lat_stop, lon_stop = self.df_stop.loc[self.df_stop['label'] == cluster_stop, ['lat', 'lon']].mean()
        distance = math.sqrt(((lat_stop - lat_start) ** 2) + ((lon_stop - lon_start) ** 2))
        return distance

    def get_intersect_point(self):
        return self.intersect_lists

    def get_centroid_order(self, order):
        return self._process_centroid(self.intersect_lists[order - 1]['start_point'], self.intersect_lists[order - 1]['stop_point'])

    def _process_centroid(self, cluster_start, cluster_stop):
        lat_start, lon_start = self.df_stop.loc[self.df_stop['label'] == cluster_start, ['lat', 'lon']].mean()
        lat_stop, lon_stop = self.df_stop.loc[self.df_stop['label'] == cluster_stop, ['lat', 'lon']].mean()
        return lat_start, lon_start, lat_stop, lon_stop

    def show_start_stop(self, order):
        try:
            label_lists = list(set(self.df_stop['label'].to_list()))
            self._process_plot(self.intersect_lists[order - 1]['start_point'], self.intersect_lists[order - 1]['stop_point'], self.intersect_lists[order - 1]['amount'], self.intersect_lists[order - 1]['distance'] * 100, order, label_lists)
        except:
            print(f"Don't have intersect lists")
            
    def _process_plot(self, cluster_start, cluster_stop, amount, distance, order, labels):
        start_lat_lon, stop_lat_lon = self._process_lat_lon(cluster_start, cluster_stop)

        start = {
            'coor': (start_lat_lon[2], start_lat_lon[0]),
            'width': start_lat_lon[3] - start_lat_lon[2],
            'height': start_lat_lon[1] - start_lat_lon[0]
        }

        stop = {
            'coor': (stop_lat_lon[2], stop_lat_lon[0]),
            'width': stop_lat_lon[3] - stop_lat_lon[2],
            'height': stop_lat_lon[1] - stop_lat_lon[0]
        }

        fig, ax = plt.subplots(figsize=(20, 15))

        customPalette = ['#36382E', '#FF206E', '#E5C687', '#41EAD4', '#5C80BC', '#FB8B24', '#04A777', '#7E8D85', '#B3BFB8', '#6F8695']

        plt.scatter(
            x=self.df_road['lon'],
            y=self.df_road['lat'],
            s=0.5,
            alpha=0.5
        )

        for i, label in enumerate(labels):
            plt.scatter(
                x=self.df_stop[self.df_stop['label'] == label]['lon'],
                y=self.df_stop[self.df_stop['label'] == label]['lat'],
                color=customPalette[i % len(customPalette)],
                alpha=1
            )

            if label != cluster_start and label != cluster_stop:
                plt.annotate(
                    label,
                    self.df_stop.loc[self.df_stop['label'] == label, ['lon', 'lat']].mean(),
                    horizontalalignment='center',
                    verticalalignment='center',
                    size=10,
                    weight='bold',
                    color='white',
                    backgroundcolor=customPalette[i % len(customPalette)]
                )

        ax.add_patch(Rectangle(start['coor'], start['width'], start['height'], fc='none', color='red', linewidth=1))
        ax.text(start['coor'][0], start['coor'][1] + start['height'], f"start : {cluster_start}", color='red')
        ax.add_patch(Rectangle(stop['coor'], stop['width'], stop['height'], fc='none', color='blue', linewidth=1))
        ax.text(stop['coor'][0], stop['coor'][1] + stop['height'], f"stop : {cluster_stop}", color='blue')
        plt.title(f"order: {order} || cluster start: {cluster_start} || cluster stop: {cluster_stop} || distance: {distance} || amount: {amount}")
        plt.show()