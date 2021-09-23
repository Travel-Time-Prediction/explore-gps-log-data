import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class FindTravelTime():


    def __init__(self,df) :
        self.df = df
        self.road_centroid = [  {"rd" : 1 , "start":(15.561914417289183, 100.12577439385959),"stop": (16.85901078855188, 99.12904080671557)},
                                {"rd" : 2 , "start":(14.850897312327184, 101.65609584505698),"stop": (16.27010669391355, 102.78036219710563)},
                                {"rd" : 4 , "start":(11.495837799354899, 99.59964040070624,),"stop": (13.314148102062967, 99.82484325754068)},
                                {"rd" : 7 , "start":(13.113253464736975, 100.95531837250864),"stop": (13.73277648540468, 100.76177909399911)},
                                {"rd" : 9 , "start":(13.649629122634497, 100.68650542608228,),"stop": (13.642726998935187,100.41451712861084)},
                                {"rd" : 32 ,"start":(14.198731291192727, 100.61185751693245),"stop": (15.275452778629077,100.19545773779554)},
                                {"rd" : 41 ,"start":(9.946246687710023, 99.06392436080117),"stop": (8.167021308334858, 99.66352696811437)},
                                {"rd" : 304 ,"start":(13.879089398416676,101.56885167383179),"stop":(14.388486307650279,101.86460689848363) },
                                {"rd" : 35 ,"start":(13.640325795495965, 100.40820722072358),"stop":(13.34104092128205, 99.83952892671479) },
                                {"rd" : 331 ,"start":(13.59731304067158, 101.29603419258949),"stop":(13.11827478284012, 101.02046471680859) }, ]
        


    def _init_minmaxlatlon(self,start_lat,start_lon,stop_lat,stop_lon) : 
        self.min_start_lat = start_lat - 0.01
        self.max_start_lat = start_lat + 0.01
        self.min_start_lon = start_lon - 0.01
        self.max_start_lon = start_lon + 0.01
        self.min_stop_lat = stop_lat - 0.01
        self.max_stop_lat = stop_lat + 0.01
        self.min_stop_lon = stop_lon - 0.01
        self.max_stop_lon = stop_lon + 0.01
    
    def set_road(self,num):  #insert index 0 - 9
        mapping = {"1": 0, "2":1 ,"4":2 , "7":3, "9":4, "32":5 , "41":6 , "304":7 ,"35":8, "331":9  }
        
        self._init_minmaxlatlon(self.road_centroid[mapping[str(num)]]["start"][0],self.road_centroid[mapping[str(num)]]
        ["start"][1],self.road_centroid[mapping[str(num)]]["stop"][0],self.road_centroid[mapping[str(num)]]["stop"][1])


    def _preprocess(self):
        start_lat_range = (self.min_start_lat <= self.df["lat"] ) & (self.df["lat"] <= self.max_start_lat)
        start_lon_range = (self.min_start_lon <= self.df["lon"] ) & (self.df["lon"] <= self.max_start_lon)

        stop_lat_range = (self.min_stop_lat <= self.df["lat"] ) & (self.df["lat"] <= self.max_stop_lat)
        stop_lon_range = (self.min_stop_lon <= self.df["lon"] ) & (self.df["lon"] <= self.max_stop_lon)

        df_start = self.df[start_lat_range & start_lon_range].reset_index(drop=True)
        df_stop = self.df[stop_lat_range & stop_lon_range].reset_index(drop=True)

    
        df_stop = df_stop[df_stop["unit_id"].isin(list(set(df_start["unit_id"])))].reset_index(drop=True) 
        df_stop["position"] = "stop"

        df_start = df_start[df_start["unit_id"].isin(list(set(df_stop["unit_id"])))].reset_index(drop=True)
        df_start["position"] = "start"

        df_tmp = df_start.append(df_stop).sort_values(by=["time_stamp"]).reset_index(drop=True)


        return df_tmp

    def find_travel_time(self):
        df = self._preprocess()
        list_uid = list(df["unit_id"].unique())
         
        start_stop = pd.DataFrame(columns=["unit_id","lat_0","lon_0","lat_1","lon_1","t0","t1","delta_t"])
        stop_start = pd.DataFrame(columns=["unit_id","lat_0","lon_0","lat_1","lon_1","t0","t1","delta_t"])

        for uid in list_uid :

            temp = df[df["unit_id"] == uid].reset_index(drop=True)
    
            for i in range(len(temp)-1):
                if (temp.loc[i,"position"] == "start") & (temp.loc[i+1,"position"]== "stop") :
                    start_stop = start_stop.append({"unit_id":temp.loc[i,"unit_id"],"lat_0":temp.loc[i,"lat"],
                    "lon_0":temp.loc[i,"lon"],"lat_1":temp.loc[i+1,"lat"],"lon_1":temp.loc[i+1,"lon"],"t0":temp.loc[i,"time_stamp"], 
                    "t1":temp.loc[i+1,"time_stamp"],"delta_t": pd.Timedelta(temp.loc[i+1,"time_stamp"] - temp.loc[i,"time_stamp"]).seconds},ignore_index=True)
                elif (temp.loc[i,"position"] == "stop") & (temp.loc[i+1,"position"]== "start") :
                    stop_start = stop_start.append({"unit_id":temp.loc[i,"unit_id"],"lat_0":temp.loc[i,"lat"],
                    "lon_0":temp.loc[i,"lon"],"lat_1":temp.loc[i+1,"lat"],"lon_1":temp.loc[i+1,"lon"],"t0":temp.loc[i,"time_stamp"],
                     "t1":temp.loc[i+1,"time_stamp"],"delta_t": pd.Timedelta(temp.loc[i+1,"time_stamp"] - temp.loc[i,"time_stamp"]).seconds},ignore_index=True)
        self.df_start_stop = start_stop
        self.df_stop_start = stop_start

        return start_stop ,stop_start

    def find_travel_time_select_hour(self,hour=1):
        df = self._preprocess()
        list_uid = list(df["unit_id"].unique())
         
        start_stop = pd.DataFrame(columns=["time_range","unit_id","lat_0","lon_0","lat_1","lon_1","delta_t"])
        stop_start = pd.DataFrame(columns=["time_range","unit_id","lat_0","lon_0","lat_1","lon_1","delta_t"])

        for uid in list_uid :

            temp = df[df["unit_id"] == uid].reset_index(drop=True)
    
            for i in range(len(temp)-1):
                if (temp.loc[i,"position"] == "start") & (temp.loc[i+1,"position"]== "stop") :
                    start_stop = start_stop.append({"unit_id":temp.loc[i,"unit_id"],"lat_0":temp.loc[i,"lat"],
                    "lon_0":temp.loc[i,"lon"],"lat_1":temp.loc[i+1,"lat"],"lon_1":temp.loc[i+1,"lon"],"time_range":(temp.loc[i,"time_stamp"].hour//hour)*hour, 
                    "delta_t": pd.Timedelta(temp.loc[i+1,"time_stamp"] - temp.loc[i,"time_stamp"]).seconds},ignore_index=True)
                elif (temp.loc[i,"position"] == "stop") & (temp.loc[i+1,"position"]== "start") :
                    stop_start = stop_start.append({"unit_id":temp.loc[i,"unit_id"],"lat_0":temp.loc[i,"lat"],
                    "lon_0":temp.loc[i,"lon"],"lat_1":temp.loc[i+1,"lat"],"lon_1":temp.loc[i+1,"lon"],"time_range":(temp.loc[i,"time_stamp"].hour//hour)*hour,
                    "delta_t": pd.Timedelta(temp.loc[i+1,"time_stamp"] - temp.loc[i,"time_stamp"]).seconds},ignore_index=True)

        start_stop = start_stop.sort_values(by=["time_range"]).reset_index(drop=True)
        stop_start = stop_start.sort_values(by=["time_range"]).reset_index(drop=True)

        start_stop = self._clean_outlier(start_stop)
        stop_start = self._clean_outlier(stop_start)

        self.df_start_stop = start_stop
        self.df_stop_start = stop_start


        return start_stop ,stop_start

    def plot_filter(self,df_road):

        plt.figure(figsize=(20,15))
        
        plt.scatter(x=df_road['lon'],y=df_road['lat'],s=0.5,alpha=0.5)

        df = self._preprocess()
        plt.scatter(x = df["lon"],y=df["lat"],c="red"  )

        plt.show

    def _clean_outlier(self,df):
        hour = list(set(df["time_range"]))
        all_clean = pd.DataFrame()

        for hr in hour :
            temp = df[df["time_range"] == hr]
            q1 = temp["delta_t"].quantile(0.25)
            q3 = temp["delta_t"].quantile(0.75)
            iqr = q3-q1  
            #print("hr = {} lower = {}  upper = {}".format(hr,(q1 - 1.5 * iqr),(q3 + 1.5 * iqr)))
            all_clean = all_clean.append(temp[(temp["delta_t"] >= (q1 - 1.5 * iqr))  & (temp["delta_t"] <= (q3 + 1.5 * iqr)) ])

        return all_clean

        
    