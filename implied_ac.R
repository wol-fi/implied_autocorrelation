# read in
df <- read_csv("mfiv_smfiv_spx.csv")
df <- df[,c("smfiv30", "smfiv91", "smfiv182", "smfiv273", "smfiv365")]
tau <- c(30, 91, 182, 273, 365)/365

# 1. de-annualize the variance swaps:
F2 <- as.matrix(df) %*% diag(tau)

# 2. take logs
X <- log(tau)
Y <- log(F2)

# 3. run the regression for each day
X <- cbind(1, X) # adds intercept
mdl <- lm.fit(X, t(Y))

# 4. get the implied Hurst exponents & transform to Pearson's auto-corr.
H <- 0.5 * mdl$coefficients[2,] 

k <- 1 # lag
ac1 <- 0.5*(abs(k+1)^(2*H) - 2*abs(k)^(2*H) + abs(k-1)^(2*H)) # Pearson's A.C.

