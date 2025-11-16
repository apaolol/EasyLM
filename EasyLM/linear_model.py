"""
LinearModel: Ordinary Least Squares implementation similar to R's lm,
refactored to avoid warnings under all conditions.
"""

import numpy as np
import pandas as pd
from scipy import stats

from .base_model import BaseModel
from .exceptions import FitError, PredictError
from .utils import check_array, add_constant
from .summary_formatter import SummaryFormatter


class LinearModel(BaseModel):
    """
    OLS linear model using numpy.linalg.lstsq with safe statistical computations.
    Zero-RSS and singular cases are handled without warnings.
    """

    def __init__(self, add_intercept=True):
        super().__init__()
        self.add_intercept = add_intercept
        self.cov_params_ = None
        self.sigma2_ = None
        self.residuals_ = None
        self.fittedvalues_ = None
        self.summary_formatter = SummaryFormatter()

    # ---------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------

    def _prepare(self, X):
        X = check_array(X, "X")
        if self.add_intercept:
            X = add_constant(X, prepend=True)
        return X

    def _safe_div(self, num, den):
        """
        Safe divide: returns 0 if division invalid.
        Prevents runtime warnings.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.divide(num, den, where=(den != 0))
            out = np.where(den == 0, 0.0, out)
        return out

    # ---------------------------------------------------------
    # Fit model
    # ---------------------------------------------------------

    def fit(self, X, y):
        try:
            Xp = self._prepare(X)
            y = check_array(y, "y").ravel()

            if Xp.shape[0] != y.shape[0]:
                raise FitError("Number of rows in X and length of y do not match.")

            beta, residuals_sum, rank, s = np.linalg.lstsq(Xp, y, rcond=None)

            self.params_ = beta
            self.n_obs_ = Xp.shape[0]
            self.n_features_ = Xp.shape[1]

            # Compute fitted/residuals
            fitted = Xp @ beta
            resid = y - fitted
            self.fittedvalues_ = fitted
            self.residuals_ = resid

            # Compute RSS safely
            if residuals_sum.size:
                rss = float(residuals_sum[0])
            else:
                rss = float(np.sum(resid ** 2))

            df_resid = self.n_obs_ - self.n_features_
            df_resid = max(df_resid, 0)

            # sigma² safe: if df_resid==0 → NaN
            self.sigma2_ = rss / df_resid if df_resid > 0 else np.nan

            # Covariance matrix safe inverse
            try:
                xtx_inv = np.linalg.inv(Xp.T @ Xp)
                self.cov_params_ = xtx_inv * self.sigma2_
            except np.linalg.LinAlgError:
                self.cov_params_ = np.full((self.n_features_, self.n_features_), np.nan)

            self.is_fitted = True
            return self

        except Exception as e:
            raise FitError(f"Failed to fit LinearModel: {e}") from e

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------

    def predict(self, X):
        if not self.is_fitted:
            raise PredictError("Model is not fitted yet.")
        Xp = self._prepare(X)

        if Xp.shape[1] != self.params_.shape[0]:
            raise PredictError("Design matrix has incompatible number of columns.")

        return Xp @ self.params_

    # ---------------------------------------------------------
    # Coefficient table
    # ---------------------------------------------------------

    def _coef_table(self):
        if not self.is_fitted:
            raise FitError("Model not fitted.")

        coef = self.params_

        if self.cov_params_ is None:
            se = np.full_like(coef, np.nan)
        else:
            se = np.sqrt(np.clip(np.diag(self.cov_params_), a_min=0, a_max=None))

        # compute t-values safely
        tvals = self._safe_div(coef, se)

        df_resid = max(0, self.n_obs_ - self.n_features_)
        if df_resid == 0:
            pvals = np.ones_like(tvals)
        else:
            pvals = 2 * stats.t.sf(np.abs(tvals), df=df_resid)

        return {
            "coef": coef,
            "std_err": se,
            "t": tvals,
            "p": pvals,
        }

    # ---------------------------------------------------------
    # Information criteria (warning-free)
    # ---------------------------------------------------------

    def aic(self):
        if not self.is_fitted:
            raise FitError("Model not fitted.")

        rss = float(np.sum(self.residuals_ ** 2))
        n = self.n_obs_
        k = self.n_features_

        if rss <= 0:
            return float("-inf")  # perfect fit → no uncertainty

        return 2 * k + n * np.log(rss / n)

    def bic(self):
        if not self.is_fitted:
            raise FitError("Model not fitted.")

        rss = float(np.sum(self.residuals_ ** 2))
        n = self.n_obs_
        k = self.n_features_

        if rss <= 0:
            return float("-inf")

        return np.log(n) * k + n * np.log(rss / n)

    # ---------------------------------------------------------
    # R^2
    # ---------------------------------------------------------

    def r_squared(self):
        if not self.is_fitted:
            raise FitError("Model not fitted.")

        y = self.fittedvalues_ + self.residuals_
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        ss_res = float(np.sum(self.residuals_ ** 2))

        if ss_tot == 0:
            return 1.0 if ss_res == 0 else np.nan

        return 1 - ss_res / ss_tot

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(self):
        if not self.is_fitted:
            raise FitError("Model not fitted.")

        coef_table = self._coef_table()
        info = {
            "n_obs": int(self.n_obs_),
            "n_params": int(self.n_features_),
            "df_resid": int(max(0, self.n_obs_ - self.n_features_)),
            "sigma2": float(self.sigma2_),
            "aic": float(self.aic()),
            "bic": float(self.bic()),
            "r_squared": float(self.r_squared()),
        }

        return self.summary_formatter.format(coef_table, info)
