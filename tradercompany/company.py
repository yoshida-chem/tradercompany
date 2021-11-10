import numpy as np

from .trader import Trader

class Company:
    def __init__(self, y, num_stock, num_factors_max, delay_time_max, activation_funcs, binary_operators, num_traders, Q=0.2, time_window=None):
        """ time_window shold be larger than num_factors_max
            1-Qがbad_tradersの割合を表す，つまりQが大きいほど解雇される割合が少なくなる．Qは生存率ともいえる．
            とりあえず全員をbad_traderにしてみた，意味はない
        """
        self.y = y
        self.num_stock = num_stock
        self.num_factors_max = num_factors_max
        self.delay_time_max = delay_time_max
        self.activation_funcs = activation_funcs
        self.binary_operators = binary_operators
        self.num_traders = num_traders
        self.Q = Q
        # if time_window==None, train by using all data
        self.time_window = time_window
        
        self.traders = [[] for _ in range(num_traders)]
        for i_trader in range(num_traders):
            self.traders[i_trader] = Trader(num_stock, num_factors_max, delay_time_max, activation_funcs, binary_operators, time_window)
            for i_stock in range(num_stock):
                self.generate_trader_without_singular(i_trader, i_stock, len(y[0]), y)
        self.bad_traders = np.ones((num_stock, num_traders)) > 0.0
        
    def generate_trader_without_singular(self, i_trader, i_stock, t, y):
        for i_stock in range(self.num_stock):
            for idx in reversed(range(self.time_window)):
                self.traders[i_trader].stack_factors(y[:,t-self.delay_time_max-1-idx:t-idx], i_stock)
            while self.traders[i_trader]._check_rank_deficient(i_stock):
                self.traders[i_trader].reset_params(i_stock)
                for idx in reversed(range(self.time_window)):
                    self.traders[i_trader].stack_factors(y[:,t-self.delay_time_max-1-idx:t-idx], i_stock)
                    
    def observe(self, data):
        """ 全てのTraderのfactorsを計算して保存
        """
        for i_trader in range(self.num_traders):
            for i_stock in range(self.num_stock):
                self.traders[i_trader].stack_factors(data, i_stock)

    def educate(self, t):
        """ とりあえず、全トレーダーが教育されている
            tを引数にとっているが、y_trueを引数にとったほうが良さそう
        """
        for i_trader in range(self.num_traders):
            for i_stock in range(self.num_stock):
                y_true = self.y[i_stock][t-self.time_window+1:t+1]
                self.traders[i_trader].learn(y_true, i_stock)

    def find_bad_traders(self, y_true):
        """ 
        """
        cumulative_errors = np.zeros((self.num_stock, self.num_traders))
        for i_trader in range(self.num_traders):
            for i_stock in range(self.num_stock):
                self.traders[i_trader].calc_cumulative_error(y_true)
                cumulative_errors[i_stock][i_trader] = self.traders[i_trader].cumulative_error[i_stock]

        for i_stock in range(self.num_stock):
            self.bad_traders[i_stock] = cumulative_errors[i_stock] > np.percentile(cumulative_errors[i_stock], 100.*self.Q)

    def fire_and_recruit(self, t, y):
        y_true = y[:,t-self.time_window+1:t+1]
        self.find_bad_traders(y_true)
        for i_trader in range(self.num_traders):
            for i_stock in range(self.num_stock):
                if self.bad_traders[i_stock][i_trader]:
                    self.generate_trader_without_singular(i_trader, i_stock, t, y)
                    self.traders[i_trader].calc_cumulative_error(y_true) # これいるか？
                    
    def aggregate(self):
        predictions = np.zeros((self.num_stock, self.num_traders))
        weights = np.zeros((self.num_stock, self.num_traders))

        for i_trader in range(self.num_traders):
            for i_stock in range(self.num_stock):
                if not self.bad_traders[i_stock][i_trader]: # Q-パーセンタイル以上の成績の良いTraderのみを対象
                    weights[i_stock][i_trader] = (1.0 / self.traders[i_trader].cumulative_error[i_stock]) # 重みは累積誤差の逆数，つまり誤差が少ないTraderほど重く見られる
                    predictions[i_stock][i_trader] = self.traders[i_trader].predict()[i_stock]
        
        predictions_weighted = np.zeros(self.num_stock)
        for i_stock in range(self.num_stock):
            predictions_weighted[i_stock] = (weights[i_stock]*predictions[i_stock]).sum() / (weights[i_stock].sum())
        return predictions_weighted