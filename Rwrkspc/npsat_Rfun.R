npsat.ReadWaterTableCloudPoints <- function(prefix, suffix = 'top', nproc = 1, iter = 1, print_file = F){
  iter <- iter - 1
  df <- data.frame(matrix(data = NA, nrow = 0, ncol = 4))
  for (i in 0:(nproc-1)) {
    fname <- paste0(prefix,suffix,'_',sprintf("%03d",iter),'_', sprintf("%04d",i), '.xyz')
    npoints <- read.table(file = fname,
                          header = FALSE, sep = "", skip = 0, nrows = 1,
                          quote = "",fill = TRUE,
                          col.names = c("N"))
    datapnts <- read.table(file = fname,
                           header = FALSE, sep = "", skip = 1, nrows = npoints$N,
                           quote = "",fill = TRUE)
    df <- rbind(df, datapnts)
  }
  x <- c("X", "Y", "Zold", "Znew")
  colnames(df) <- x
  return(df)
}

npsat.writeWells <- function(filename, wells){
  write(dim(well_df)[1], file = filename, append = FALSE)
  write.table(wells, file = filename, sep = " ", row.names = FALSE, col.names = FALSE, quote = FALSE, append = TRUE)
}