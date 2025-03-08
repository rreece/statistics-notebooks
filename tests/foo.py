from covariance_calculators.estimators import OnlineCovariance, EMACovariance

c = OnlineCovariance(1)
#c = EMACovariance(1, alpha=0.1)

print(c.cov)
c.add([2])
print(c.cov)
c.add([1])
print(c.cov)
c.add([0.5])
print(c.cov)

#c = EMACovariance(2, alpha=0.1)
#
#print(c.cov)
#c.add([2, 1])
#print(c.cov)
#c.add([1, 0])
#print(c.cov)
#c.add([0.5, 0.2])
#print(c.cov)

