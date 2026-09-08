# Linear Algebra

## Identities

```math
\vec{x}^\intercal A \vec{x} = \mathrm{tr}(A \vec{x} \vec{x}^\intercal)
```

## Positive semi-definite matrices

A matrix, $A$, is positive semi-definite (PSD) if for every vector $\vec{x}$, $\vec{x}^\intercal A \vec{x} \geq 0$

Equivalently, all its eigenvalues are $\geq 0$.


## Gaussian random vectors

If $`\vec{x}`$ is a random vector with

```math
\mathrm{E}(\vec{x}) = \mu \qquad \mathrm{and} \qquad \mathrm{Cov}(\vec{x}) = \Sigma
```

and if $A$ is symmetric, then

```math
\mathrm{E}(A \vec{x}) = A \vec{\mu}
```

```math
\mathrm{Cov}(A \vec{x}) = A \Sigma A^\intercal
```

```math
\mathrm{E}(\vec{x}^\intercal A \vec{x}) = \vec{\mu}^\intercal A \vec{\mu} + \mathrm{tr}(A\Sigma)
```

If $`\vec{x}`$ is a random vector $\sim N(\vec{\mu}, \Sigma)$

```math
\mathrm{Var}(\vec{x}^\intercal A \vec{x}) = 2 \mathrm{tr}(A\Sigma{}A\Sigma) + 4 \vec{\mu}^\intercal A \Sigma A \vec{\mu} 
```

