import numpy as np
from numpy.lib.function_base import quantile
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.cluster

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN, KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import estimate_bandwidth



class GPS_DBScan:
    def __init__(self, df, df_road):
        self.df = df
        self.df_road = df_road
        self.df_dbscan = None
        self.rd = 0

    def set_rd(self, rd):
        self.rd = rd

    def get_eps(self, n_neighbors):
        # init dataframe for find epsilon
        df_neighbors = self._process_data()

        # train NearestNeighbors
        neighbors = NearestNeighbors(n_neighbors=n_neighbors)
        neighbors_fit = neighbors.fit(df_neighbors)
        distances, _ = neighbors_fit.kneighbors(df_neighbors)

        # sort value for plot
        distances = np.sort(distances, axis=0)
        distances = distances[:, 1]

        # plot
        plt.plot(distances[:])
        plt.show()

        # clear data in memory
        del df_neighbors
        del distances

    def dbscan(self, eps, min_samples):
        # clear data in memory
        del self.df_dbscan

        # init dataframe for dbscan
        self.df_dbscan = self._process_data()

        # dbscan clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(self.df_dbscan)
        self.df_dbscan['label'] = clustering.labels_

        # clear noise in dataframe
        idx_noise = self.df_dbscan[self.df_dbscan['label'] == -1].index
        self.df_dbscan = self.df_dbscan.drop(idx_noise).reset_index(drop=True)

        # get list labels
        labels = list(set(self.df_dbscan['label'].to_list()))

        # plot
        self._plot_dbscan(labels)

    def get_top_cluster(self):
        return self.df_dbscan['label'].value_counts()[:10]
    
    def get_df(self):
        return self.df_dbscan
    
    def _plot_dbscan(self, labels):
         # init color for plot
        customPalette = ['#36382E', '#FF206E', '#E5C687', '#41EAD4', '#5C80BC', '#FB8B24', '#04A777', '#7E8D85', '#B3BFB8', '#6F8695']

        plt.figure(figsize=(20, 15))

        plt.scatter(
            x=self.df_road['lon'],
            y=self.df_road['lat'],
            s=0.5,
            alpha=0.5
        )

        for i, label in enumerate(labels):
            plt.scatter(
                x=self.df_dbscan[self.df_dbscan['label'] == label]['lon'],
                y=self.df_dbscan[self.df_dbscan['label'] == label]['lat'],
                color=customPalette[i % len(customPalette)],
                alpha=1
            )

            plt.annotate(
                label,
                self.df_dbscan.loc[self.df_dbscan['label'] == label, ['lon', 'lat']].mean(),
                horizontalalignment='center',
                verticalalignment='center',
                size=10,
                weight='bold',
                color='white',
                backgroundcolor=customPalette[i % len(customPalette)]
            )

        plt.show()

    def _process_data(self):
        df_tmp = self.df.copy()
        if self.rd > 0:
            df_tmp = df_tmp[df_tmp['rd'] == self.rd].reset_index(drop=True)
        df_tmp = df_tmp[['lat', 'lon']]
        return df_tmp


class GPS_Kmean:
    def __init__(self, df, df_road):
        self.df = df
        self.df_road = df_road
        self.df_kmean = None
        self.rd = 0
        self.k = 0
    
    def set_rd(self, rd):
        self.rd = rd

    def get_k_elbow_method(self, max_k):
        # init dataframe for find elbow method
        df_elbow = self._process_data()

        distortions = []
        for i in range(1, max_k + 1):
            if len(df_elbow) >= i:
                model = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
                model.fit(df_elbow)
                distortions.append(model.inertia_)

        self.k = [i for i in np.diff(distortions, 2)].index(min([i for i in np.diff(distortions, 2)]))
        self._plot_k(distortions)

        # clear data in memory
        del df_elbow

    def kmean(self):
        # clear data in memory
        del self.df_kmean

        if self.k < 1:
            self.get_k_elbow_method(10)

        # init dataframe for kmean
        self.df_kmean = self._process_data()

        # kmean clustering
        model = KMeans(n_clusters=self.k, init='k-means++')
        self.df_kmean['cluster'] = model.fit_predict(self.df_kmean)

        closest, _ = scipy.cluster.vq.vq(model.cluster_centers_, self.df_kmean.drop(['cluster'], axis=1).values)

        self.df_kmean['centroid'] = 0
        for i in closest:
            self.df_kmean['centroid'][i] = 1

        # plot
        self._plot_kmean(model.cluster_centers_)

    def get_top_cluster(self):
        if self.df_kmean is None:
            return f"not dataframe from kmean"
        return self.df_kmean['cluster'].value_counts()[:10]
    
    def get_df(self):
        return self.df_kmean
    
    def _plot_kmean(self, m_centroids):
        fig, ax = plt.subplots()
        ax.scatter(x=self.df_road['lon'], y=self.df_road['lat'], s=0.5, alpha=0.5, c='lightskyblue')
        sns.scatterplot(x='lon', y='lat', data=self.df_kmean, palette=sns.color_palette('bright', self.k), hue='cluster', size='centroid', size_order=[1, 0], legend='brief', ax=ax).set_title('Clustering (k='+str(self.k)+')')
        ax.scatter(m_centroids[:, 1], m_centroids[:, 0], s=50, c='black', marker='x')
        plt.show()

    def _plot_k(self, distortions):
        fig, ax = plt.subplots()
        ax.plot(range(1, len(distortions) + 1), distortions)
        ax.axvline(self.k, ls='--', color="red", label="k = "+str(self.k))
        ax.set(title='The Elbow Method', xlabel='Number of clusters', ylabel="Distortion")
        ax.legend()
        ax.grid(True)
        plt.show()

    def _process_data(self):
        df_tmp = self.df.copy()
        if self.rd > 0:
            df_tmp = df_tmp[df_tmp['rd'] == self.rd].reset_index(drop=True)
        df_tmp = df_tmp[['lat', 'lon']]
        return df_tmp


class GPS_MeanShift :

    def __init__(self, df, df_road):
        self.df = df
        self.df_road = df_road
        self.rd = 0
        self.df_meanshift = None

    def set_rd(self, rd):
        self.rd = rd

    def _process_data(self):
        df_tmp = self.df.copy()
        if self.rd > 0:
            df_tmp = df_tmp[df_tmp['rd'] == self.rd].reset_index(drop=True)
        df_tmp = df_tmp[['lat', 'lon']]
        return df_tmp

    def meanshift(self ,quantile ,n_sample=None):
        # clear data in memory
        del self.df_meanshift

        # init dataframe for dbscan
        self.df_meanshift = self._process_data()

        # meanshift clustering 
        bandwidth = estimate_bandwidth(self.df_meanshift, quantile= quantile, n_samples= n_sample,random_state = 0)
        clustering = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(self.df_meanshift)
        self.df_meanshift["label"] = clustering.labels_

        # get list labels
        labels = list(set(self.df_meanshift['label'].to_list()))

        # plot
        self._plot_meanshift(labels)

    def _plot_meanshift(self, labels):
         # init color for plot
        customPalette = ['#36382E', '#FF206E', '#E5C687', '#41EAD4', '#5C80BC', '#FB8B24', '#04A777', '#7E8D85', '#B3BFB8', '#6F8695']

        plt.figure(figsize=(20, 15))

        plt.scatter(
            x=self.df_road['lon'],
            y=self.df_road['lat'],
            s=0.5,
            alpha=0.5
        )

        for i, label in enumerate(labels):
            plt.scatter(
                x=self.df_meanshift[self.df_meanshift['label'] == label]['lon'],
                y=self.df_meanshift[self.df_meanshift['label'] == label]['lat'],
                color=customPalette[i % len(customPalette)],
                alpha=1
            )

            plt.annotate(
                label,
                self.df_meanshift.loc[self.df_meanshift['label'] == label, ['lon', 'lat']].mean(),
                horizontalalignment='center',
                verticalalignment='center',
                size=10,
                weight='bold',
                color='white',
                backgroundcolor=customPalette[i % len(customPalette)]
            )

        plt.show()

    def get_top_cluster(self):
        return self.df_meanshift['label'].value_counts()[:10]

