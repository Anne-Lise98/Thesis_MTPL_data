regressionattention <- function(ll) {
  dat_plt <- data.frame(var = test_smp[, col_names[ll]],
                        bx = beta_smp[, ll + length(col_features)],
                        col = rep("green", nsample))
  plt <- ggplot(dat_plt, aes(x = var, y = bx)) + geom_point() + 
    geom_hline(yintercept = 0, colour = "red", size = line_size) + 
    geom_hline(yintercept = c(-quant_rand, quant_rand), colour = "green", size = line_size) +
    geom_hline(yintercept = c(-1,1)/4, colour = "orange", size = line_size, linetype = "dashed") +
    geom_rect(
      mapping = aes(xmin = min(var), xmax = max(var), ymin = -quant_rand, ymax = quant_rand),
      fill = dat_plt$col, alpha = 0.002
    ) + lims(y = c(-0.75,0.75)) +
    labs(title = paste0("Regression attention: ", col_names[ll]),
         subtitle = paste0("Coverage Ratio: ", paste0(round(II[, col_names[ll]] * 100, 2)), "%"),
         x = paste0(col_names[ll], " x"), y = "regression attention beta(x)")
}
