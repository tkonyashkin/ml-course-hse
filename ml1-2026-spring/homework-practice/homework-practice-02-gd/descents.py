import numpy as np
from abc import ABC, abstractmethod
from interfaces import LearningRateSchedule, AbstractOptimizer, LinearRegressionInterface


# ===== Learning Rate Schedules =====
class ConstantLR(LearningRateSchedule):
    def __init__(self, lr: float):
        self.lr = lr

    def get_lr(self, iteration: int) -> float:
        return self.lr


class TimeDecayLR(LearningRateSchedule):
    def __init__(self, lambda_: float = 1.0):
        self.s0 = 1
        self.p = 0.5
        self.lambda_ = lambda_

    def get_lr(self, iteration: int) -> float:
        """
        returns: float, learning rate для iteration шага обучения
        """
        return self.lambda_ * (self.s0 / (self.s0 + iteration)) ** self.p


# ===== Base Optimizer =====
class BaseDescent(AbstractOptimizer, ABC):
    """
    Оптимизатор, имплементирующий градиентный спуск.
    Ответственен только за имплементацию общего алгоритма спуска.
    Все его составные части (learning rate, loss function+regularization) находятся вне зоны ответственности этого класса (см. Single Responsibility Principle).
    """
    def __init__(self, 
                 lr_schedule: LearningRateSchedule = TimeDecayLR(), 
                 tolerance: float = 1e-6,
                 max_iter: int = 1000
                ):
        self.lr_schedule = lr_schedule
        self.tolerance = tolerance
        self.max_iter = max_iter

        self.iteration = 0
        self.model: LinearRegressionInterface = None

    @abstractmethod
    def _update_weights(self) -> np.ndarray:
        """
        Вычисляет обновление согласно конкретному алгоритму и обновляет веса модели, перезаписывая её атрибут.
        Не имеет прямого доступа к вычислению градиента в точке, для подсчета вызывает model.compute_gradients.

        returns: np.ndarray, w_{k+1} - w_k
        """
        pass

    def _step(self) -> np.ndarray:
        """
        Проводит один полный шаг интеративного алгоритма градиентного спуска

        returns: np.ndarray, w_{k+1} - w_k
        """
        delta = self._update_weights()
        self.iteration += 1
        return delta

    def optimize(self) -> None:
        """
        Оркестрирует весь алгоритм градиентного спуска.
        """
        self.iteration = 0
        loss_history = [self.model.compute_loss()]

        for _ in range(self.max_iter):
            delta = self._step()
            loss_history.append(self.model.compute_loss())

            if np.isnan(delta).any() or np.linalg.norm(delta) ** 2 < self.tolerance:
                break

        self.model.loss_history = loss_history


# ===== Specific Optimizers =====
class VanillaGradientDescent(BaseDescent):
    def _update_weights(self) -> np.ndarray:
        grad = self.model.compute_gradients()
        lr = self.lr_schedule.get_lr(self.iteration)
        d = -lr * grad
        self.model.w += d
        return d


class StochasticGradientDescent(BaseDescent):
    def __init__(self, *args, batch_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size

    def _update_weights(self) -> np.ndarray:
        idx = np.random.randint(0, self.model.X_train.shape[0], size=self.batch_size)
        grad = self.model.compute_gradients(self.model.X_train[idx], self.model.y_train[idx])
        lr = self.lr_schedule.get_lr(self.iteration)
        d = -lr * grad
        self.model.w += d
        return d


class SAGDescent(BaseDescent):
    def __init__(self, *args, batch_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_memory = None
        self.avg_grad = None
        self.batch_size = batch_size

    def _update_weights(self) -> np.ndarray:
        X_train = self.model.X_train
        y_train = self.model.y_train
        num_objects, num_features = X_train.shape

        if self.grad_memory is None:
            self.grad_memory = np.zeros((num_objects, num_features))
            self.avg_grad = np.zeros(num_features)

        idx = np.unique(np.random.randint(0, num_objects, size=self.batch_size))

        for j in idx:
            old_grad = self.grad_memory[j].copy()
            new_grad = self.model.compute_gradients(X_train[j:j + 1], y_train[j:j + 1])
            self.grad_memory[j] = new_grad
            self.avg_grad += (new_grad - old_grad) / num_objects

        lr = self.lr_schedule.get_lr(self.iteration)
        d = -lr * self.avg_grad
        self.model.w += d
        return d


class MomentumDescent(BaseDescent):
    def __init__(self,  *args, beta=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.velocity = None

    def _update_weights(self) -> np.ndarray:
        grad = self.model.compute_gradients()
        lr = self.lr_schedule.get_lr(self.iteration)

        if self.velocity is None:
            self.velocity = np.zeros_like(self.model.w)

        self.velocity = self.beta * self.velocity + lr * grad
        d = -self.velocity
        self.model.w += d
        return d
        


class Adam(BaseDescent):
    def __init__(self, *args, beta1=0.9, beta2=0.999, eps=1e-8, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None

    def _update_weights(self) -> np.ndarray:
        grad = self.model.compute_gradients()
        lr = self.lr_schedule.get_lr(self.iteration)

        if self.m is None:
            self.m = np.zeros_like(self.model.w)
            self.v = np.zeros_like(self.model.w)

        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad ** 2)
        m_unbiased = self.m / (1.0 - self.beta1 ** (self.iteration + 1))
        v_unbiased = self.v / (1.0 - self.beta2 ** (self.iteration + 1))
        d = -lr * m_unbiased / (np.sqrt(v_unbiased) + self.eps)
        self.model.w += d
        return d



# ===== Non-iterative Algorithms ====
class AnalyticSolutionOptimizer(AbstractOptimizer):
    """
    Универсальный дамми-класс для вызова аналитических решений 
    """
    def __init__(self):
        self.model = None
    

    def optimize(self) -> None:
        """
        Определяет аналитическое решение и назначает его весам модели.
        """
        if self.model is None:
            raise ValueError("Model is not set for the optimizer.")
        
        self.model.w = self.model.loss_function.analytic_solution(self.model.X_train, self.model.y_train)
        self.model.loss_history.append(self.model.compute_loss())
