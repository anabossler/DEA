### Bias correction DEA

import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import truncnorm
import random


def rDEA(inputs, outputs, covariates, RTS, L1, L2, alpha):
    X   = inputs
    Y   = outputs
    Z   = covariates
    out = {}

    # number of firms
    N   = X.shape[1]

    # 1. Compute the original DEA (input oriented)
    delta_hat = DEA(input=X, output=Y, RTS=RTS)
    out['delta_hat'] = np.array(delta_hat)

    # 2. Maximum likelihood for estimating beta and sigma_{\epsilon}
    # - only use delta_hat>1
    # from regression we get beta_hat and sigma_hat
    ind_not1 = ( 1/out['delta_hat'] ) > 1
    ty = out['delta_hat'][ind_not1]
    tx = Z[:,ind_not1]
    if sum(ind_not1) <= (tx.shape[0] + 1):
        ## too few non-frontier samples
        raise Exception("Too few firms to build the model for environmental variables.")

    # Run regression
    regr = LinearRegression()
    regr.fit(tx.reshape(-1,Z.shape[0]), ty.reshape(-1,1))
    beta_hat = np.array(list(regr.intercept_) + list(regr.coef_[0]))
    prediction = regr.predict(tx.reshape(-1,Z.shape[0]))
    residual = (ty.reshape(-1,1) - prediction)
    sigma_hat = (residual.std() / np.sqrt(len(beta_hat))) * 2
    out['beta_hat'] = beta_hat
    out['sigma_hat'] = sigma_hat

    # calculate weighted predictors
    Z1  = np.vstack((np.array([1] * len(Z[0])),Z))
    Zbeta = np.dot(beta_hat, Z1)

    # 3. Loop of L1 times (100) to get bootstrapped values of delta_hat for each firm
    delta_hat_star = []
    for i in range(0,L1):
        random.seed(a=i)
        # 3.1 Draw \epsilon_i for each observation from trunc. normal
        epsilon = truncnorm.rvs(a=1-Zbeta, b=float('inf'), scale=sigma_hat, size=N)

        # 3.2 Compute \delta^*_i
        delta_star = Zbeta + epsilon

        # 3.3 Compute y^*, rescaling
        rescaleFactor = delta_star / delta_hat
        X_star = X * rescaleFactor

        # 3.4 Compute DEA for y^* and x to get \widehat{\delta}^*_i
        delta_hat_star = delta_hat_star + [DEA(input=X_star, output=Y, RTS=RTS)]

    # 4. for each firm compute bias-corrected estimator \doublehat{\delta}_i.
    delta_hat_hat = 2 * np.array(delta_hat) - np.mean(delta_hat_star, axis=0)

    # list for output variables:
    out['bias'] = np.mean(delta_hat_star, axis=0) - np.array(delta_hat)
    out['delta_hat_hat'] = delta_hat_hat

    # estimate CI for delta
    delta_ci = np.quantile(delta_hat_star, q=( (alpha / 2), (1-(alpha/2)) ), axis=0)
    out['delta_ci_low'] = delta_ci[0] - 2 * out['bias']
    out['delta_ci_high'] = delta_ci[1] - 2 * out['bias']

    ## from Kneip et al (2008), page 1676-1677:
    ratio_ci_low = delta_ci[0] / out['delta_hat'] - 1
    ratio_ci_high = delta_ci[1] / out['delta_hat'] - 1
    out['delta_ci_kneip_low']  = 1 / ( (1 + ratio_ci_high) / out["delta_hat"] )
    out['delta_ci_kneip_high'] = 1 / ( (1 + ratio_ci_low)  / out["delta_hat"] )

    # 5. Maximum likelihood to estimate \doublehat{\beta} and \doublehat{\sigma}
    # - only use delta_hat>1
    # - Z should scaled by mean(Z)
    # from regression we get beta_hat_hat and sigma_hat_hat
    ind_not2 = (1 / delta_hat_hat) > 1
    out['delta_hat_hat_used'] = np.mean(ind_not2)

    ty = delta_hat_hat[ind_not2]
    tx = Z[:,ind_not2]
    # Run regression
    regr = LinearRegression()
    regr.fit(tx.reshape(-1, Z.shape[0]), ty.reshape(-1, 1))
    beta_hat_hat = np.array(list(regr.intercept_) + list(regr.coef_[0]))
    prediction = regr.predict(tx.reshape(-1, Z.shape[0]))
    residual = (ty.reshape(-1, 1) - prediction)
    sigma_hat_hat = (residual.std() / np.sqrt(len(beta_hat))) * 2
    out['beta_hat_hat'] = beta_hat_hat
    out['sigma_hat_hat'] = sigma_hat_hat

    # 6. Bootstrap loop for L2 (2000) steps.
    beta_hat_hat_boot = []
    sigma_hat_hat_boot = []

    Zbeta_hat_hat = np.dot(beta_hat_hat, Z1)
    out['Zbeta_hat_hat'] = Zbeta_hat_hat

    for i in range(0,L2):
        random.seed(a=i)
        # 6.1 Draw \epsilon_i for each observation from trunc. normal
        epsilon = truncnorm.rvs(a=1 - Zbeta_hat_hat, b=float('inf'), scale=sigma_hat_hat, size=N)

        # 6.2 Compute \delta^{**}_i
        delta_star_star = Zbeta_hat_hat + epsilon

        # 6.3. Maximum likelihood to estimate \doublehat{\beta} and \doublehat{\sigma}
        ind_not3 = (1 / delta_star_star) > 1.0

        # for package truncreg:
        ty = delta_star_star[ind_not3]
        tx = Z[:, ind_not3]
        # Run regression
        regr = LinearRegression()
        regr.fit(tx.reshape(-1, Z.shape[0]), ty.reshape(-1, 1))
        beta_hhb = np.array(list(regr.intercept_) + list(regr.coef_[0]))
        prediction = regr.predict(tx.reshape(-1, Z.shape[0]))
        residual = (ty.reshape(-1, 1) - prediction)
        sigma_hhb = (residual.std() / np.sqrt(len(beta_hat))) * 2

        beta_hat_hat_boot  = beta_hat_hat_boot + [list(beta_hhb)]
        sigma_hat_hat_boot = sigma_hat_hat_boot + [sigma_hhb]

    # calculating the mean:
    beta_hat_hat_star = np.mean(beta_hat_hat_boot, axis=0)
    sigma_hat_hat_star = np.mean(sigma_hat_hat_boot)

    out['beta_hat_hat_star'] = beta_hat_hat_star
    out['sigma_hat_hat_star'] = sigma_hat_hat_star

    # 7. Calculate confidence interval for \doublehat{\beta}^* and \doublehat{\sigma}^*
    out['beta_ci'] = np.quantile(beta_hat_hat_boot, q=( (alpha / 2), (1-(alpha/2)) ), axis=0)
    out['sigma_ci'] = np.quantile(sigma_hat_hat_boot, q=( (alpha / 2), (1-(alpha/2)) ), axis=0)

    ## centering CI to the original estimates \doublehat{\beta}^* and \doublehat{\sigma}^*
    out['beta_ci'] = out['beta_ci'] + (beta_hat_hat - beta_hat_hat_star)
    out['sigma_ci'] = out['sigma_ci'] + (sigma_hat_hat - sigma_hat_hat_star)

    return out


# Test - Compare with R
x = np.array([(1.1, 2.2, 6.6, 4.4, 5.5)])
y = np.array([(10.01, 100.1, 75.07, 100.1, 100.1)])
z = np.array([(-3.3, -2.2, 0.0, 1.0, 1.5)])
result_crs = []
result_crs = rDEA(inputs=x, outputs=y, covariates=z, RTS='crs', L1=100, L2=2000, alpha=.05)
print(result_crs)
result_vrs = []
result_vrs = rDEA(inputs=x, outputs=y, covariates=z, RTS='vrs', L1=100, L2=2000, alpha=.05)
print(result_vrs)