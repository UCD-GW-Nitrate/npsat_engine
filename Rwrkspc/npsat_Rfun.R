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

#' npsat.input.WriteScattered
#'
#' @param filename The name of the file
#' @param PDIM is the number of columns that correspond to coordinates.  This is 1 for 1D points of 2 for 2D points.
#' @param TYPE Valid options for type are FULL, HOR or VERT
#' @param MODE Valid options for mode are SIMPLE or STRATIFIED
#' @param DATA the data to be printed. Data should have as many columns as needed.
#' For example it can be :
#' [x v]
#' [x v1 z1 v2 z2 ... vn-1 zn-1 vn]
#' [x y v]
#' [x y v1 z1 v2 z2 ... vn-1 zn-1 vn]
#'
#' @return
#' @export
#'
#' @examples
#' For 2D interpolation such as top, bottom elevation or recharge
#' npsat.input.WriteScattered(filename, 2, "HOR", "SIMPLE", data)
npsat.input.WriteScattered <- function(filename, PDIM, TYPE, MODE, DATA){
  write("SCATTERED", file = filename, append = FALSE)
  write(TYPE, file = filename, append = TRUE)
  write(MODE, file = filename, append = TRUE)
  Ndata <- dim(DATA)[2] - PDIM
  write(paste(dim(DATA)[1], Ndata), file = filename, append = TRUE)
  write.table(DATA, file = filename, sep = " ", row.names = FALSE, col.names = FALSE, quote = FALSE, append = TRUE)
}


#' npsat.Input.readScattered reads the scattered format files
#'
#' @param filename is the name of the file
#' @param PDIM is the dimention of the point. Valid options are 2 for 2D points or 1 for 1D points
#'
#' @return a data frame with the X Y Values
#' @export
#'
#' @examples
npsat.Input.readScattered <- function(filename, PDIM = 2){
  # Read header
  tmp <- readLines(filename, n = 4)
  N <- as.numeric(strsplit(tmp[4], " ")[[1]])
  if (PDIM == 1){
    cnames <- c("X")
  }
  else if (PDIM == 2){
    cnames <- c("X", "Y", "V")
  }
  if(N[2] == 3){
    cnames <- c(cnames, "V")
  }
  else if (N[2] > 3){
    cnames <- c(cnames, "V1")
    for (i in seq(1, (N[2]-1)/2, 1)) {
      cnames <- c(cnames, paste0("Z", i), paste0("V", i+1))
    }
    
  }
  DATA <- read.table(file = filename,
                     header = FALSE, sep = "", skip = 4, nrows = N[1],
                     quote = "",fill = TRUE,
                     col.names = cnames)
  
  return(DATA)
}