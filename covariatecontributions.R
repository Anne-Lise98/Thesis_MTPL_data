covariatecontributions <- function(kk){
  dat_plt <- data.frame(var = test_smp[, col_names_reduced[kk]],
                      bx = beta_smp[, kk + length(col_names_reduced)] * beta_smp[, kk],
                      col = rep("green", nsample))
plt <- ggplot(dat_plt, aes(x = var, y = bx)) + geom_point() +
  geom_smooth(size = line_size) +
  geom_hline(yintercept = 0, colour = "red", size = line_size) +
  geom_hline(yintercept = c(-1, 1) / 4, colour = "orange", size = line_size, linetype = "dashed") +
  lims(y = c(-1.25, 1.25)) +
  labs(title = paste0("Covariate contribution: ", col_names_reduced[kk]),
       x = paste0(col_names_reduced[kk], " x"),
       y = "covariate contribution beta(x) * x")
suppressMessages(print(plt))}
