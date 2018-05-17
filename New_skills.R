#Binomial distribution
dbinom(10, 100, .3)
x <- 1:100
plot(dbinom(x, 100, .3))
abline(v=30)

# Comparing distributions
xvA <- 1:10000
par(pch=16)
plot(dbinom(xvA, 10000, .01), col="darkgreen", xlim=c(0,200))
abline(v=100, col="darkgreen")
p <- .9973002
qbinom(c(1-p,p), 10000, .01)
binom.test(120, 10000, .01, alternative = "greater")
points(dbinom(xvA, 10000, .012), col="chocolate1")
abline(v=120, col="chocolate1")
qbinom(c(1-p,p), 10000, .012)

# From the blog
n=10000; p=0.01; q=1-p; mean=100
paste(mean - 3 * sqrt(n*p*q), "," ,mean + 3 * sqrt(n*p*q))

n=10000; p=0.012; q=1-p; mean=120
paste(mean - 3 * sqrt(n*p*q), ",", mean + 3 * sqrt(n*p*q))

# Testing several percentages points
mean = 0.01
sigma = sqrt((mean * 0.99)/10000)
p_a_values = c(
        qnorm(0.01, mean = mean, sd = sigma),
        qnorm(0.25, mean = mean, sd = sigma),
        qnorm(0.50, mean = mean, sd = sigma),
        qnorm(0.75, mean = mean, sd = sigma), 
        qnorm(0.99, mean = mean, sd = sigma))
p_a_values

# parametric Type II
count = 50000; start = 1000
data = data.frame(x= numeric(0), error= numeric(0), parametric_mean = character(0))
p_a_values = factor(p_a_values)

for(p_a in p_a_values) {
        n = start:(start+count)
        x = rep(0, count); error = rep(0, count); parametric_mean = rep('0', count);
        for(i in n) {
                p_a_numeric = as.numeric(as.character(p_a))
                critical = qbinom(0.95, i, p_a_numeric)
                t2_error = pbinom(critical, i, 0.012)
                
                index = i - start + 1
                x[index] = i
                error[index] = t2_error
                parametric_mean[index] = p_a
        }
        data = rbind(data, data.frame(x = x, error = error, parametric_mean=parametric_mean))
}

options(repr.plot.width=7, repr.plot.height=3)
ggplot(data=data, aes(x=x, y=error, color=parametric_mean, group=parametric_mean)) +
        geom_line()

plot(data$x, data$error, col=data$parametric_mean)

# BayesAB
A <- rbinom(40, 1, .5)
B <- rbinom(40, 1, .3)
test1 <- bayesTest(A, B, , distribution = "bernoulli", priors = c("alpha" = 10, "beta" = 10))
summary(test1)
plot(test1)




