import numpy as np

class Trader():
    def __init__(self, num_stock, num_factors_max, delay_time_max, activation_funcs, binary_operators, time_window=None):
        # set hyperparameters
        self.num_stock = num_stock
        self.num_factors_max = num_factors_max
        self.delay_time_max = delay_time_max
        self.activation_funcs = activation_funcs
        self.binary_operators = binary_operators
        # if time_window==None, train by using all data
        self.time_window = time_window
        
        # initialize by stock
        self.num_factors = [np.nan for _ in range(num_stock)]
        self.delay_P = [np.nan for _ in range(num_stock)]
        self.delay_Q = [np.nan for _ in range(num_stock)]
        self.stock_P = [np.nan for _ in range(num_stock)]
        self.stock_Q = [np.nan for _ in range(num_stock)]
        self.activation_func = [np.nan for _ in range(num_stock)]
        self.binary_operator = [np.nan for _ in range(num_stock)]
        self.w = [np.nan for _ in range(num_stock)]
        
        self.X_factors = [[] for _ in range(num_stock)]
        self.cumulative_error = [np.nan for _ in range(num_stock)]
        
        # initialized by uniform distribution
        for i_stock in range(num_stock):
            self.num_factors[i_stock] = np.random.choice(range(1, self.num_factors_max+1))
            self.delay_P[i_stock] = np.random.choice(delay_time_max, self.num_factors[i_stock])
            self.delay_Q[i_stock] = np.random.choice(delay_time_max, self.num_factors[i_stock])
            self.stock_P[i_stock] = np.random.choice(num_stock, self.num_factors[i_stock])
            self.stock_Q[i_stock] = np.random.choice(num_stock, self.num_factors[i_stock])
            self.activation_func[i_stock] = np.random.choice(activation_funcs, self.num_factors[i_stock])
            self.binary_operator[i_stock] = np.random.choice(binary_operators, self.num_factors[i_stock])
            self.w[i_stock] = np.random.randn(self.num_factors[i_stock])
        
            self.X_factors[i_stock] = np.zeros((0, self.num_factors[i_stock]))
            self.cumulative_error[i_stock] = 0.0
        
    def reset_params(self, i_stock):
        # initialized by uniform distribution
        self.num_factors[i_stock] = np.random.choice(range(1, self.num_factors_max+1))
        self.delay_P[i_stock] = np.random.choice(self.delay_time_max, self.num_factors[i_stock])
        self.delay_Q[i_stock] = np.random.choice(self.delay_time_max, self.num_factors[i_stock])
        self.stock_P[i_stock] = np.random.choice(self.num_stock, self.num_factors[i_stock])
        self.stock_Q[i_stock] = np.random.choice(self.num_stock, self.num_factors[i_stock])
        self.activation_func[i_stock] = np.random.choice(self.activation_funcs, self.num_factors[i_stock])
        self.binary_operator[i_stock] = np.random.choice(self.binary_operators, self.num_factors[i_stock])
        self.w[i_stock] = np.random.randn(self.num_factors[i_stock])

        self.X_factors[i_stock] = np.zeros((0, self.num_factors[i_stock]))
        self.cumulative_error[i_stock] = 0.0
    
    def calc_factor(self, data, i_stock, j):
        """ i_stock番目の株に関してj番目の項を計算
        """
        Aj = self.activation_func[i_stock][j]
        Oj = self.binary_operator[i_stock][j]
        Pj = self.stock_P[i_stock][j]
        Qj = self.stock_Q[i_stock][j]
        Dj = self.delay_P[i_stock][j]
        Fj = self.delay_Q[i_stock][j]
        return Aj(Oj(data[Pj][self.delay_time_max-Dj], data[Qj][self.delay_time_max-Fj]))
    
    def calc_factors(self, data, i_stock):
        num_factors = self.num_factors[i_stock]
        factors = np.zeros(num_factors)
        for j in range(num_factors): # 各ファクターごとに
            factors[j] = self.calc_factor(data, i_stock, j)
        return factors
    
    def stack_factors(self, data, i_stock):
        """
        time_windowの数だけ計算した項(factors)をX_factorsに保存する
        time_windowを超えた場合、古いfactorsを削除し、最新のfactorsを一番下に保存する
        """
        factors = self.calc_factors(data, i_stock)
        if self.time_window is None:
            self.X_factors[i_stock] = np.vstack([self.X_factors[i_stock], factors])
        elif len(self.X_factors[i_stock]) < self.time_window:
            self.X_factors[i_stock] = np.vstack([self.X_factors[i_stock], factors])
        else:
            # 
            self.X_factors[i_stock] = np.roll(self.X_factors[i_stock], -1, axis = 0)
            self.X_factors[i_stock][-1] = factors
        return None
    
    def learn(self, y, i_stock):
        epsilon = 0.0001
        X = self.X_factors[i_stock]
        if self._check_rank_deficient(i_stock):
            self.w[i_stock] = np.zeros(len(self.w[i_stock]))
        else:
            self.w[i_stock] = np.linalg.inv(X.T.dot(X) + epsilon).dot(X.T).dot(y)
    
    def _check_rank_deficient(self, i_stock):
        """ ランク落ちを確認
        """
        X = self.X_factors[i_stock]
        return np.linalg.matrix_rank(X.T.dot(X)) < self.num_factors[i_stock]
        
    def predict(self): # test dataに対するprediction, これに基づいてモデルの評価を行う．
        """ 最新のfactorsを使用して次の時刻の予測を行う
        """
        y_pred = np.zeros(self.num_stock)
        for i_stock in range(self.num_stock):
            y_pred[i_stock] = self.X_factors[i_stock][-1].dot(self.w[i_stock])
        return y_pred
    
    def predict_for_X_factors(self):
        """ len(X_factors) < time_windowの時エラー
        """
        y_preds = np.zeros((self.num_stock, self.time_window))
        for i_stock in range(self.num_stock):
            y_preds[i_stock] = self.X_factors[i_stock].dot(self.w[i_stock])
        return y_preds
    
    def calc_cumulative_error(self, y_true):
        y_preds = self.predict_for_X_factors()
        errors = (y_preds - y_true)**2.0
        errors = np.sqrt(errors.mean(1))
        for i_stock in range(self.num_stock):
            self.cumulative_error[i_stock] = errors[i_stock]
        return self.cumulative_error
