# Confidence intervals

## Introduction

-   TODO


## Cochran's theorem

$$ \frac{n \hat{\sigma}^2}{\sigma^2} \sim \chi^{2}_{n-1} $$

where the MLEs for a normal distribution are

$$ \hat{\mu} = \frac{1}{n} \sum_{i=1}^{n} x_i $$

and

$$ \hat{\sigma}^2 = \frac{1}{n} \sum_{i=1}^{n} ( x_i - \hat{\mu} )^2  $$

See Wikipedia: [Cochran's theorem](https://en.wikipedia.org/wiki/Cochran%27s_theorem#Estimation_of_variance).

Note the unbiased sample variance is

$$ s^2 = \frac{1}{(n-1)} \sum_{i=1}^{n} ( x_i - \hat{\mu} )^2  $$

So

$$ s^2 = \frac{n}{(n-1)} \hat{\sigma}^2 $$

and

$$ \frac{(n-1) s^2}{\sigma^2} \sim \chi^{2}_{n-1} $$


## Quantiles to p-values

Cumulative distribution function:

$$ F(y) = \int_{-\infty}^{y} f(x) dx $$

$$ \bar{F}(y) = 1 - F(y) = \int_{y}^{\infty} f(x) dx $$

$p$-value from test statistic $q$:

$$ p = 1 - \alpha = \int_{-\infty}^{q_{\alpha}} f(q) dq = F(q(\alpha))$$

Critical value of test statistic for a given $p$-value:

$$ q_{\alpha} = F^{-1}(1 - \alpha) = \mathrm{ppf}(1 - \alpha) $$

Two sided:

$$ 1 - \alpha = \int_{q_{\alpha}^\mathrm{lower}}^{q_{\alpha}^\mathrm{upper}} f(q) dq $$

$$ 1 - \frac{\alpha}{2} = \int_{-\infty}^{q_{\alpha}^\mathrm{upper}} f(q) dq = F(q_{\alpha}^\mathrm{upper}) $$

$$ 1 - \frac{\alpha}{2} = \int_{q_{\alpha}^\mathrm{lower}}^{\infty} f(q) dq = \bar{F}(q_{\alpha}^\mathrm{lower}) = 1 - F(q_{\alpha}^\mathrm{lower}) $$

$$ q_{\alpha}^\mathrm{upper} = F^{-1}(1 - \frac{\alpha}{2}) = \mathrm{ppf}(1 - \frac{\alpha}{2}) $$

$$ q_{\alpha}^\mathrm{lower} = F^{-1}(\frac{\alpha}{2}) = \mathrm{ppf}(\frac{\alpha}{2}) $$

See also:

-   Quantiles of $\chi^2 \Rightarrow$ $p$-values [Table](https://math.arizona.edu/~jwatkins/chi-square-table.pdf)


## Wishart distribution

Scatter matrix:

$$ S = \sum_{i=1}^{n} ( x_i - \bar{x} ) ( x_i - \bar{x} )^\intercal  $$

If $X \sim N_{p}(0, V)$ then $S \sim W_{p}(V, n)$.

If $X \sim N_{p}(\mu, V)$ then $S \sim W_{p}(V, n-1)$.

If $p=1$ and $V=1$, then $W_{1}(1, n) = \chi^{2}_{n}$.

Variance of Wishart:

$$ n \cdot ( V_{ii} V_{jj} + V_{ij}^{2} ) $$


## Confidence intervals for sample covariance

Unbiased estimator of variance of scatter matrix:

$$ \mathrm{Var}(\hat{S}) = (n-1) ( V_{ii} V_{jj} + V_{ij}^{2} ) $$

Sample covariance matrix:

$$ V = \frac{1}{n-1} S $$

Variance of sample covariance matrix:

```math
\mathrm{Var}(\hat{V}) = \frac{1}{(n-1)^2} \mathrm{Var}(\hat{S}) = \frac{1}{n-1} ( \hat{V}_{ii} \hat{V}_{jj} + \hat{V}_{ij}^{2} )
```

Asymptotically assuming the errors are normally distributed:

```math
\hat{\sigma}_{ij} = \sqrt{\frac{1}{n-1} ( \hat{V}_{ii} \hat{V}_{jj} + \hat{V}_{ij}^{2} )}
```

and we have a frequentist confidence interval:

```math
V_{ij} = \hat{V}_{ij} \pm z_{\alpha} \hat{\sigma}_{ij}
```

at a confidence level picked by the two-sided $z$ score:

```math
z_{\alpha} = \Phi^{-1}\left(1 - \frac{\alpha}{2}\right)
```

because

```math
\Phi(z) = \int_{-\infty}^{z} \phi(x) dx  = 1 - \frac{\alpha}{2}
```

where

```math
\phi(x) = \frac{1}{\sqrt{2\pi}} e^{x^2/2}
```

Instead of using quantiles of the normal distribution we could use the quantiles of the Wishart distribution more directly.

```math
V_{ij} = \hat{V}_{ij} \pm \Delta_{ij}^{\alpha}
```

where $\Delta_{ij}^{\alpha}$ are determined by the quantiles of the Wishart distribution, $Q_{ij}$.

```math
1 - \frac{\alpha}{2} = F_{W}(Q_{ij}^\mathrm{upper}; \hat{V}_{ij})
```

```math
Q_{ij}^\mathrm{upper} = F_{W}^{-1}(1 - \frac{\alpha}{2}; \hat{V}_{ij})
```

```math
Q_{ij}^\mathrm{lower} = F_{W}^{-1}(\frac{\alpha}{2}; \hat{V}_{ij})
```

```math
Q_{ij}^\mathrm{lower} < V_{ij} < Q_{ij}^\mathrm{upper} \qquad \mathrm{at}~(1-\alpha)~\mathrm{CL}
```


## Notes about covariance and Wishart

-   Cowan (1998), p. 20-22.
-   Cowan (1998), p. 68.  V[s^2] 
-   James (2006), p. 205.
-   <https://adamheins.com/blog/wishart-confidence-intervals>
-   <https://github.com/adamheins/wishart-confidence-intervals>
-   <https://en.wikipedia.org/wiki/Cochran%27s_theorem#Sample_mean_and_sample_variance>
-   <https://www.maxturgeon.ca/w20-stat7200/slides/wishart-distribution.pdf>
-   <https://stats.stackexchange.com/questions/354443/how-do-i-find-the-elliptical-confidence-region-from-columns-of-a-matrix-that-f>
-   <https://math.stackexchange.com/questions/3729438/confidence-interval-for-generalized-variance-determinant-of-covariance-matrix>
-   <https://www.sciencedirect.com/science/article/pii/S0047259X15002353>
-   <https://arxiv.org/abs/0704.2278>
-   <https://www.statlect.com/fundamentals-of-statistics/set-estimation-variance>

