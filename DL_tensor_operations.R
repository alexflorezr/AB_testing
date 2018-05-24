# Tensor dot
naive_vector_dot <- function(x, y) {
        z <- 0
        for (i in 1:length(x))
                z <- z + x[[i]] * y[[i]]
        z
}
tx <- 1:10
ty <- 101:110
naive_vector_dot(tx, ty)
tx %*% ty
sum(tx*ty)
