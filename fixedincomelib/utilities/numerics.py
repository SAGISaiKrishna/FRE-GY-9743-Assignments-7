import copy
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import List


class InterpMethod(Enum):

    PIECEWISE_CONSTANT_LEFT_CONTINUOUS = 'PIECEWISE_CONSTANT_LEFT_CONTINUOUS'
    LINEAR = 'LINEAR'
    MONOTONE_CONVEX = 'MONOTONE_CONVEX'

    @classmethod
    def from_string(cls, value: str) -> 'InterpMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class ExtrapMethod(Enum):

    FLAT = 'FLAT'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'ExtrapMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class Interpolator1D(ABC):
    """Abstract interface for a 1-D interpolator."""

    def __init__(self,
                 axis1: np.ndarray,
                 values: np.ndarray,
                 interpolation_method: InterpMethod,
                 extrapolation_method: ExtrapMethod) -> None:

        self.axis1_ = axis1
        self.values_ = values
        self.interp_method_ = interpolation_method
        self.extrap_method_ = extrapolation_method
        self.length_ = len(self.axis1_)

    @abstractmethod
    def interpolate(self, x: float) -> float:
        pass

    @abstractmethod
    def integrate(self, start_x: float, end_x: float) -> float:
        pass

    @abstractmethod
    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        pass

    @abstractmethod
    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        pass

    @property
    def axis1(self) -> np.ndarray:
        return self.axis1_

    @property
    def values(self) -> np.ndarray:
        return self.values_

    @property
    def length(self) -> int:
        return self.length_

    @property
    def interp_method(self) -> str:
        return self.interp_method_.to_string()

    @property
    def extrap_method(self) -> str:
        return self.extrap_method_.to_string()


class Interpolator1DPCP(Interpolator1D):
    """Piecewise-constant left-continuous interpolator with FLAT extrapolation.

    With axis1 = [1, 3, 5, 7], values = [3, 4, 5, 6]:
        f(0.5) = 3, f(1) = 3, f(1.5) = 4, f(3) = 4, f(5.5) = 6, f(8) = 6
    """

    def __init__(self, axis1: np.ndarray, values: np.ndarray,
                 extrapolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values,
                         InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS,
                         extrapolation_method)
        assert self.extrap_method_ == ExtrapMethod.FLAT

    def interpolate(self, x: float) -> float:
        idx = min(int(np.searchsorted(self.axis1_, x, side='left')), self.length_ - 1)
        return float(self.values_[idx])

    def integrate(self, start_x: float, end_x: float) -> float:
        edges = np.concatenate(([-np.inf], np.asarray(self.axis1_, dtype=float)[:-1], [np.inf]))
        lo = np.maximum(start_x, edges[:-1])
        hi = np.minimum(end_x, edges[1:])
        overlaps = np.maximum(hi - lo, 0.0)
        return float(np.dot(np.asarray(self.values_, dtype=float), overlaps))

    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        idx = min(int(np.searchsorted(self.axis1_, x, side='left')), self.length_ - 1)
        grad = np.zeros(self.length_)
        grad[idx] = 1.0
        return grad

    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        edges = np.concatenate(([-np.inf], np.asarray(self.axis1_, dtype=float)[:-1], [np.inf]))
        lo = np.maximum(start_x, edges[:-1])
        hi = np.minimum(end_x, edges[1:])
        return np.maximum(hi - lo, 0.0)



class Interpolator1DMonotoneConvex(Interpolator1D):
    """HW7: implement your own monotone, convex IFR interpolator here."""

    def __init__(self, axis1: np.ndarray, values: np.ndarray,
                 extrapolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values, InterpMethod.MONOTONE_CONVEX,
                         extrapolation_method)
        assert self.extrap_method_ == ExtrapMethod.FLAT
        x = np.asarray(axis1, dtype=float)
        y = np.asarray(values, dtype=float)
        n = len(x)

        # Each piece is f(t) = c0 + c1 (t - start) + c2 (t - start)^2.
        self.starts_, self.ends_, self.coefs_ = [], [], []
        if n < 2:
            return

        h = np.diff(x)
        d = np.diff(y) / h
        m = np.full(n, d[0])
        for i in range(1, n - 1):
            if d[i - 1] * d[i] > 0:
                m[i] = (h[i] * d[i - 1] + h[i - 1] * d[i]) / (h[i - 1] + h[i])
            else:
                m[i] = 0.0
        # A zero end slope joins the flat extrapolation smoothly; it is only
        # allowed at the start of a rising curve or the end of a falling one.
        m[0] = 0.0 if d[0] >= 0 else 2 * d[0] - m[1]
        m[-1] = 0.0 if d[-1] <= 0 else 2 * d[-1] - m[-2]

        for i in range(n - 1):
            # Slope must rise across the interval (convex) and keep one sign
            # (monotone). Only non-convex data gets clipped, leaving a kink.
            if d[i] >= 0:
                s0, s1 = min(max(m[i], 0.0), d[i]), max(m[i + 1], d[i])
            else:
                s0, s1 = min(m[i], d[i]), min(max(m[i + 1], d[i]), 0.0)
            a, b = d[i] - s0, s1 - d[i]
            # Slope goes s0 -> d[i] -> s1; this knot makes the piece end at y[i + 1].
            knot = h[i] * b / (a + b) if a + b > 0 else h[i]
            if knot > 0:
                self._add_piece(x[i], x[i] + knot, y[i], s0, a / (2 * knot))
            if knot < h[i]:
                y_knot = y[i] + knot * (s0 + d[i]) / 2
                self._add_piece(x[i] + knot, x[i + 1], y_knot, d[i],
                                b / (2 * (h[i] - knot)))
        self.starts_ = np.array(self.starts_)
        self.ends_ = np.array(self.ends_)
        self.coefs_ = np.array(self.coefs_)

    def _add_piece(self, start, end, c0, c1, c2):
        self.starts_.append(start)
        self.ends_.append(end)
        self.coefs_.append((c0, c1, c2))

    def interpolate(self, x: float) -> float:
        if x <= self.axis1_[0]:
            return float(self.values_[0])
        if x >= self.axis1_[-1]:
            return float(self.values_[-1])
        k = int(np.searchsorted(self.starts_, x, side='right')) - 1
        c0, c1, c2 = self.coefs_[k]
        u = x - self.starts_[k]
        return float(c0 + c1 * u + c2 * u * u)

    def integrate(self, start_x: float, end_x: float) -> float:
        if end_x < start_x:
            return -self.integrate(end_x, start_x)
        first, last = float(self.axis1_[0]), float(self.axis1_[-1])
        total = self.values_[0] * max(min(end_x, first) - start_x, 0.0)
        total += self.values_[-1] * max(end_x - max(start_x, last), 0.0)
        if len(self.starts_) == 0:
            return float(total)
        lo = np.clip(start_x, self.starts_, self.ends_) - self.starts_
        hi = np.clip(end_x, self.starts_, self.ends_) - self.starts_
        c0, c1, c2 = self.coefs_.T
        total += np.sum(c0 * (hi - lo) + c1 * (hi**2 - lo**2) / 2
                        + c2 * (hi**3 - lo**3) / 3)
        return float(total)

    # Optional risk interface
    # The test cells for risk are not prepared but you can do it by your own.
    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        raise NotImplementedError("Optional risk: ordinate gradient is not implemented")

    def gradient_of_integrated_value_wrt_ordinate(
        self, start_x: float, end_x: float,
    ) -> np.ndarray:
        raise NotImplementedError("Optional risk: integrated ordinate gradient is not implemented")


class InterpolatorFactory:

    @staticmethod
    def create_1d_interpolator(axis1: np.ndarray | List,
                               values: np.ndarray | List,
                               interpolation_method: InterpMethod,
                               extrapolation_method: ExtrapMethod):

        axis1_ = copy.deepcopy(axis1)
        values_ = copy.deepcopy(values)
        if isinstance(axis1_, list):
            axis1_ = np.array(axis1_)
        if isinstance(values_, list):
            values_ = np.array(values_)
        assert len(axis1_.shape) == 1 and len(values_.shape) == 1
        assert len(axis1_) == len(values_)
        assert np.all(np.diff(axis1_) >= 0)

        if interpolation_method == InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS:
            return Interpolator1DPCP(axis1_, values_, extrapolation_method)
        elif interpolation_method == InterpMethod.MONOTONE_CONVEX:
            return Interpolator1DMonotoneConvex(axis1_, values_, extrapolation_method)
        else:
            raise ValueError(f'Unsupported interpolation method: {interpolation_method}')
