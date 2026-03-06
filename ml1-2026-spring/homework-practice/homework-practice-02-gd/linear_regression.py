import numpy as np 
from interfaces import LossFunction, LossFunctionClosedFormMixin, LinearRegressionInterface, AbstractOptimizer
from descents import AnalyticSolutionOptimizer
from typing import Dict, Type, Optional, Callable
from abc import abstractmethod, ABC
from scipy.sparse.linalg import svds


class MSELoss(LossFunction, LossFunctionClosedFormMixin):

    def __init__(self, analytic_solution_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = None):

        if analytic_solution_func is None:
            self.analytic_solution_func = self._plain_analytic_solution
        else:
            self.analytic_solution_func = analytic_solution_func

        

    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        """
        X: np.ndarray, матрица регрессоров 
        y: np.ndarray, вектор таргета
        w: np.ndarray, вектор весов

        returns: float, значение MSE на данных X,y для весов w
        """
        return np.mean((X @ w - y) ** 2)

    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        X: np.ndarray, матрица регрессоров 
        y: np.ndarray, вектор таргета
        w: np.ndarray, вектор весов

        returns: np.ndarray, численный градиент MSE в точке w
        """
        return (2 / X.shape[0]) * (X.T @ (X @ w - y))

    def analytic_solution(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Возвращает решение по явной формуле (closed-form solution)

        X: np.ndarray, матрица регрессоров 
        y: np.ndarray, вектор таргета

        returns: np.ndarray, оптимальный по MSE вектор весов, вычисленный при помощи аналитического решения для данных X, y
        """
        # Функция-диспатчер в одну из истинных функций для вычисления решения по явной формуле (closed-form)
        # Необходима в связи c наличием интерфейса analytic_solution у любого лосса; 
        # self-injection даёт возможность выбирать, какое именно closed-form решение использовать
        return self.analytic_solution_func(X, y)
        
    
    @classmethod
    def _plain_analytic_solution(cls, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        X: np.ndarray, матрица регрессоров 
        y: np.ndarray, вектор таргета

        returns: np.ndarray, вектор весов, вычисленный при помощи классического аналитического решения
        """
        return np.linalg.inv(X.T @ X) @ X.T @ y

    
    @classmethod
    def _svd_analytic_solution(cls, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        X: np.ndarray, матрица регрессоров 
        y: np.ndarray, вектор таргета

        returns: np.ndarray, вектор весов, вычисленный при помощи аналитического решения на SVD
        """
        U, S, Vt = svds(X, k=min(X.shape) - 1, solver="arpack", tol=0.0)
        S_inv = np.zeros_like(S)
        S_inv[S > 2e-10] = 1.0 / S[S > 2e-10]
        return Vt.T @ (S_inv * (U.T @ y))

class L2Regularization(LossFunction):

    def __init__(self, core_loss: LossFunction, mu_rate: float = 1.0,
                 analytic_solution_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = None):
        self.core_loss = core_loss
        self.mu_rate = mu_rate

        # analytic_solution_func is meant to be passed separately, 
        # as it is not linear to core solution
        
    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        core_part = self.core_loss.loss(X, y, w)
        w_reg = w.copy()
        w_reg[-1] = 0.0
        penalty_part = (self.mu_rate / 2) * np.sum(w_reg ** 2)
        return core_part + penalty_part
    
    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        core_part = self.core_loss.gradient(X, y, w)
        w_reg = w.copy()
        w_reg[-1] = 0.0
        penalty_part = self.mu_rate * w_reg
        return core_part + penalty_part



class CustomLinearRegression(LinearRegressionInterface):
    def __init__(
        self,
        optimizer: AbstractOptimizer,
        # l2_coef: float = 0.0,
        loss_function: LossFunction = MSELoss()
    ):
        self.optimizer = optimizer
        self.optimizer.set_model(self)

        # self.l2_coef = l2_coef
        self.loss_function = loss_function
        self.loss_history = []
        self.w = None
        self.X_train = None
        self.y_train = None
        

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        returns: np.ndarray, вектор \hat{y}
        """
        if self.w is None:
            raise ValueError("weights are not initialized")
        return X @ self.w

    def compute_gradients(self, X_batch: np.ndarray | None = None, y_batch: np.ndarray | None = None) -> np.ndarray:
        """
        returns: np.ndarray, градиент функции потерь при текущих весах (self.w)
        Если переданы аргументы, то градиент вычисляется по ним, иначе - по self.X_train и self.y_train
        """
        if self.w is None:
            raise ValueError("weights are not initialized")
        if X_batch is None and y_batch is None:
            X_batch, y_batch = self.X_train, self.y_train
        elif (X_batch is None) != (y_batch is None):
            raise ValueError("X_batch and y_batch must be provided together or not at all")
        return self.loss_function.gradient(X_batch, y_batch, self.w)


    def compute_loss(self, X_batch: np.ndarray | None = None, y_batch: np.ndarray | None = None) -> float:
        """
        returns: np.ndarray, значение функции потерь при текущих весах (self.w) по self.X_train, self.y_train
        Если переданы аргументы, то градиент вычисляется по ним, иначе - по self.X_train и self.y_train
        """
        if self.w is None:
            raise ValueError("weights are not initialized")
        if X_batch is None and y_batch is None:
            X_batch, y_batch = self.X_train, self.y_train
        elif (X_batch is None) != (y_batch is None):
            raise ValueError("X_batch and y_batch must be provided together or not at all")
        return self.loss_function.loss(X_batch, y_batch, self.w)



    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Инициирует обучение модели заданным функцией потерь и оптимизатором способом.
        
        X: np.ndarray, 
        y: np.ndarray
        """
        self.X_train, self.y_train = X, np.asarray(y)
        self.w = np.zeros(X.shape[1])
        self.loss_history = []
        self.optimizer.optimize()
        
        
class LogCoshLoss(LossFunction):
    def loss(self, X, y, w):
        r = X @ w - y
        return np.mean(np.log(np.cosh(r)))

    def gradient(self, X, y, w):
        r = X @ w - y
        return (X.T @ np.tanh(r)) / X.shape[0]


class HuberLoss(LossFunction):
    def __init__(self, delta=1.0):
        self.delta = delta

    def loss(self, X, y, w):
        abs_r = np.abs(X @ w - y)
        return np.mean(np.where(
            abs_r < self.delta,
            0.5 * abs_r**2,
            self.delta * abs_r - 0.5 * self.delta**2,
        ))

    def gradient(self, X, y, w):
        r = X @ w - y
        g = np.where(np.abs(r) < self.delta, r, self.delta * np.sign(r))
        return (X.T @ g) / X.shape[0]
