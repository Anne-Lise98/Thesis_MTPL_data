ax_limit <- c(0,50000)
line_size <- 1.1

keras_plot_loss_min <- function(x, seed) {
  x <- x[[2]]
  ylim <- range(x)
  vmin <- which.min(x$val_loss)
  df_val <- data.frame(epoch = 1:length(x$loss), train_loss = x$loss, val_loss = x$val_loss)
  df_val <- gather(df_val, variable, loss, -epoch)
  plt <- ggplot(df_val, aes(x = epoch, y = loss, group = variable, color = variable)) +
    geom_line(size = line_size) + geom_vline(xintercept = vmin, color = "green", size = line_size) +
    labs(title = paste("Train and validation loss for seed", seed),
         subtitle = paste("Green line: Smallest validation loss for epoch", vmin))
  suppressMessages(print(plt))
}
