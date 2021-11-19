import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn import preprocessing, mixture 

from .trader import Trader

class Company:
    def __init__(self, stock_names, num_factors_max, delay_time_max, activation_funcs, binary_operators, num_traders, Q=0.2, time_window=None, how_recruit="random"):
        """ time_window shold be larger than num_factors_max
            1-Qがbad_tradersの割合を表す，つまりQが大きいほど解雇される割合が少なくなる．Qは生存率ともいえる．
            とりあえず全員をbad_traderにしてみた，意味はない
        """
        self.stock_names = stock_names
        self.num_stock = len(stock_names)
        self.num_factors_max = num_factors_max
        self.delay_time_max = delay_time_max
        self.activation_funcs = activation_funcs
        self.binary_operators = binary_operators
        self.num_traders = num_traders
        self.Q = Q
        # if time_window==None, train by using all data
        self.time_window = time_window
        self.how_recruit = how_recruit
        
        self.df_y_train = None

        self.initialize_traders()

    def fit(self, df_y_train):
        if self.df_y_train is not None:
            raise Exception("Error")
        self.df_y_train = df_y_train
        y_train = self.df_y_train.T.values

        for t in tqdm(range(self.delay_time_max+1, len(self.df_y_train))):
            data_to_stack = y_train[:, t-self.delay_time_max-1:t]
            self.observe(data_to_stack)

            if self.delay_time_max + 1 + self.time_window < t <= len(self.df_y_train):
                y_true = y_train[:, t-self.time_window+1:t+1]
                self.educate(y_true)
                self.fire_and_recruit(t, y_true)
        
        # 最後の時刻tのデータはeducateするためのt+1のデータがないためstockだけする
        # stockしないとself.aggregate()で予測する時にt-1までのデータで時刻tを予測することになる
        data_to_stack = y_train[:, t-self.delay_time_max:t+1]
        self.observe(data_to_stack) 

    def fit_new_data(self, dict_y, tuning=False):
        # 最初にdf_y_trainにデータを追加。この時点では計算されていない。
        self.add_new_data(dict_y)

        # tuning=Trueであれば、新しいデータを使用して再度educate&fire_recruitを実施する
        if tuning:
            self.educate_fire_recruit_by_new_data()

        # 新しいデータを用いてfactorを計算する
        self.observe_new_data()

    def add_new_data(self, dict_y):
        self.df_y_train = self.df_y_train.append(dict_y, ignore_index=True)

    def educate_fire_recruit_by_new_data(self):
        # 一番最新のデータから教育
        t = len(self.df_y_train)
        y_train = self.df_y_train.T.values
        y_true = y_train[:, t-self.time_window:t]
        self.educate(y_true)
        self.fire_and_recruit(t-1, y_true)

    def observe_new_data(self):
        # 一番最新のデータをobserve
        t = len(self.df_y_train)
        y_train = self.df_y_train.T.values
        data_to_stack = y_train[:, t-self.delay_time_max-1:t]
        self.observe(data_to_stack)

    def initialize_traders(self):
        self.traders = [[] for _ in range(self.num_traders)]
        for i_trader in range(self.num_traders):
            self.traders[i_trader] = Trader(self.num_stock, self.num_factors_max, self.delay_time_max, self.activation_funcs, self.binary_operators, self.time_window)

    def generate_trader_without_singular_initialize_params(self, i_trader, i_stock, t):
        y_train = self.df_y_train.T.values
        for i_stock in range(self.num_stock):
            self.traders[i_trader].reset_params(i_stock)
            for idx in reversed(range(self.time_window)):
                self.traders[i_trader].stack_factors(y_train[:,t-self.delay_time_max-1-idx:t-idx], i_stock)
            while self.traders[i_trader]._check_rank_deficient(i_stock):
                self.traders[i_trader].reset_params(i_stock)
                for idx in reversed(range(self.time_window)):
                    self.traders[i_trader].stack_factors(y_train[:,t-self.delay_time_max-1-idx:t-idx], i_stock)

    def generate_trader_without_singular(self, i_trader, i_stock, t, vbgmm_num_factor, vbgmm_factor_params):
        y_train = self.df_y_train.T.values
        for i_stock in range(self.num_stock):
            list_params = self.sample_params(vbgmm_num_factor, vbgmm_factor_params)
            self.traders[i_trader].set_params(i_stock, list_params)
            for idx in reversed(range(self.time_window)):
                self.traders[i_trader].stack_factors(y_train[:,t-self.delay_time_max-1-idx:t-idx], i_stock)
            while self.traders[i_trader]._check_rank_deficient(i_stock):
                list_params = self.sample_params(vbgmm_num_factor, vbgmm_factor_params)
                self.traders[i_trader].set_params(i_stock, list_params)
                for idx in reversed(range(self.time_window)):
                    self.traders[i_trader].stack_factors(y_train[:,t-self.delay_time_max-1-idx:t-idx], i_stock)

    def observe(self, data):
        """ 全てのTraderのfactorsを計算して保存
        """
        for i_trader in range(self.num_traders):
            for i_stock in range(self.num_stock):
                self.traders[i_trader].stack_factors(data, i_stock)

    def educate(self, y_true):
        """ 下位Q%よりエラー率が悪いトレーダーが教育される
        """
        bad_traders = self.find_bad_traders(y_true, 1-self.Q)
        for i_trader in range(self.num_traders):
            for i_stock in range(self.num_stock):
                if bad_traders[i_stock][i_trader]:
                    self.traders[i_trader].learn(y_true[i_stock], i_stock)

    def find_bad_traders(self, y_true, Q):
        """ 上位Q%よりエラー率が悪いトレーダーを見つける
        """
        cumulative_errors = np.zeros((self.num_stock, self.num_traders))
        for i_trader in range(self.num_traders):
            for i_stock in range(self.num_stock):
                self.traders[i_trader].calc_cumulative_error(y_true)
                cumulative_errors[i_stock][i_trader] = self.traders[i_trader].cumulative_error[i_stock]

        bad_traders = np.ones((self.num_stock, self.num_traders)) > 0.0
        for i_stock in range(self.num_stock):
            bad_traders[i_stock] = cumulative_errors[i_stock] > np.percentile(cumulative_errors[i_stock], 100.*Q)
        return bad_traders

    def fire_and_recruit(self, t, y_true):
        """ 上位Q%よりエラー率が悪いトレーダーを解雇、補充する 
        """
        if self.how_recruit == "gmm":
            list_vbgmm = []
            good_traders = ~self.find_bad_traders(y_true, 1-self.Q)
            for i_stock in range(self.num_stock):
                list_vbgmm.append(self.fit_vbgmm(i_stock, good_traders))
        
        bad_traders = self.find_bad_traders(y_true, self.Q)
        for i_trader in range(self.num_traders):
            for i_stock in range(self.num_stock):
                if bad_traders[i_stock][i_trader]:
                    if self.how_recruit == "gmm":
                        self.generate_trader_without_singular(i_trader, i_stock, t, list_vbgmm[i_stock][0], list_vbgmm[i_stock][1])
                    else: 
                        self.generate_trader_without_singular_initialize_params(i_trader, i_stock, t)
                    self.traders[i_trader].calc_cumulative_error(y_true)

    def sample_params(self, vbgmm_num_factor, vbgmm_factor_params):
        list_params = []
        num_factor = np.vectorize(round)(vbgmm_num_factor.sample(1)[0][0])[0]
        num_factor = self.check_param(num_factor, 1, self.num_factors_max)
        list_params.append({"num_factor": num_factor})
        for i_factor in range(num_factor):
            factor_params = np.vectorize(round)(vbgmm_factor_params.sample(1)[0])[0]
            dict_params = {}
            for key, value in zip(["delay_P", "delay_Q", "stock_P", "stock_Q", "activation_func", "binary_operator"], factor_params):
                if key in ["delay_P", "delay_Q"]:
                    dict_params[key] = self.check_param(value, 0, self.delay_time_max-1)
                elif key in ["stock_P", "stock_Q"]:
                    dict_params[key] = self.check_param(value, 0, self.num_stock-1)
                elif key == "activation_func":
                    dict_params[key] = self.check_param(value, 0, len(self.activation_funcs)-1)      
                elif key == "binary_operator":
                    dict_params[key] = self.check_param(value, 0, len(self.binary_operators)-1)      
            list_params.append(dict_params)
        return list_params

    def check_param(self, param_, min_, max_):
        if param_ < min_:
            param_ = min_
        elif param_ > max_:
            param_ = max_
        else:
            param_ = param_
        return param_

    def fit_vbgmm(self, i_stock, good_traders):
        """ good_traders[i_stock]のTraderのパラメータのみでFittingする
        """
        df_num_factor = pd.DataFrame(columns=["num_factor"])
        df_factor_params = pd.DataFrame(columns=["delay_P", "delay_Q", "stock_P", "stock_Q", "activation_func", "binary_operator"])
        for i_trader, trader in enumerate(self.traders):
            if good_traders[i_stock][i_trader]:
                list_params = trader.get_params(i_stock)
                dict_num_factor = list_params[0]
                df_num_factor = df_num_factor.append(dict_num_factor, ignore_index=True)
                
                for i_factor in range(dict_num_factor["num_factor"]):
                    dict_factor_params = list_params[i_factor+1]
                    df_factor_params = df_factor_params.append(dict_factor_params, ignore_index=True)
        print(df_factor_params)
        vbgmm_num_factor = self.VBGMM(df_num_factor, n_components=self.num_stock)
        vbgmm_factor_params = self.VBGMM(df_factor_params, n_components=3)

        return vbgmm_num_factor, vbgmm_factor_params

    def VBGMM(self, X, n_components=10):
        model = mixture.BayesianGaussianMixture(n_components=n_components)
        return model.fit(X)
        
    def aggregate(self):
        predictions = np.zeros((self.num_stock, self.num_traders))
        weights = np.zeros((self.num_stock, self.num_traders))

        bad_traders = np.ones((self.num_stock, self.num_traders)) > 0.0
        for i_stock in range(self.num_stock):
            cumulative_errors = np.array([trader.cumulative_error[i_stock] for trader in self.traders])
            bad_traders[i_stock] = cumulative_errors > np.percentile(cumulative_errors, 100.*self.Q)

        for i_trader in range(self.num_traders):
            for i_stock in range(self.num_stock):
                if not bad_traders[i_stock][i_trader]: # Q-パーセンタイル以上の成績の良いTraderのみを対象
                    weights[i_stock][i_trader] = (1.0 / self.traders[i_trader].cumulative_error[i_stock]) 
                    predictions[i_stock][i_trader] = self.traders[i_trader].predict()[i_stock]
        
        predictions_weighted = np.zeros(self.num_stock)
        for i_stock in range(self.num_stock):
            predictions_weighted[i_stock] = (weights[i_stock]*predictions[i_stock]).sum() / (weights[i_stock].sum())
        return predictions_weighted