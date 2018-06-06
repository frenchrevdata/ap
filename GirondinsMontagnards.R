install.packages("readxl", repos = "http://cran.us.r-project.org")
install.packages("corrplot", repos = "http://cran.us.r-project.org")
install.packages("openxlsx", repos = "http://cran.us.r-project.org")
install.packages("data.table", repos = "http://cran.us.r-project.org")
install.packages("plyr", repos = "http://cran.us.r-project.org")
install.packages("tibble", repos = "http://cran.us.r-project.org")
library(readxl)
library(openxlsx)
library(gplots)
library(corrplot)
library(data.table)
library(plyr)
library(tibble)

data <- read_excel("combined_frequency_morethan1speaker.xlsx")
data <- as.data.frame(data)
row.names(data) <- data[,1]
data[,1] <- NULL
data[is.na(data)] <- 0

#### Chi-squared calculations

df1 <- cbind(data, t(apply(data, 1, function(x) {
  ch <- chisq.test(x)
  c(unname(ch$statistic), ch$p.value)})))

colnames(df1)[3:4] <- c('x-squared', 'p-value')
df1$`x-squared-relative` <- ifelse(df1$Girondins > df1$Montagnards, df1$`x-squared`, df1$`x-squared`*(-1))
write.xlsx(df1, "chisq_Girondins_Montagnards.xlsx", row.names = TRUE)

#### Log post odds calculations

logpostodds <- function(dataframe){
  fulldata <- dataframe
  rows <- nrow(fulldata)
  empty_vector <- rep(0, rows)
  fulldata$prior <- fulldata$Girondins + fulldata$Montagnards
  fulldata$sigmasquared <- empty_vector
  fulldata$sigma <- empty_vector
  fulldata$delta <- empty_vector
  n1 = sum(fulldata$Girondins)
  n2 = sum(fulldata$Montagnards)
  nprior = sum(fulldata$prior)
  nsum = sum(fulldata$sigmasquared)
  for (bigram in 1:nrow(fulldata)) {
    l1 <- as.double(fulldata[bigram, "Girondins"] + fulldata[bigram, "prior"]) / 
      ((n1 + nprior) - (fulldata[bigram, "Girondins"] +fulldata[bigram, "prior"]))
    l2 <- as.double(fulldata[bigram, "Montagnards"] + fulldata[bigram, "prior"]) / 
      ((n2 + nprior) - (fulldata[bigram, "Montagnards"] + fulldata[bigram, "prior"]))
    intermediate_value <- 1/(as.double(fulldata[bigram, "Girondins"]) + as.double(fulldata[bigram, "prior"])) + 
      1/(as.double(fulldata[bigram, "Montagnards"]) + as.double(fulldata[bigram, "prior"]))
    fulldata[bigram, "sigmasquared"] <- intermediate_value
    fulldata[bigram, "sigma"] <- sqrt(intermediate_value)
    fulldata[bigram, "delta"] <- (log(l1) - log(l2))/fulldata[bigram, "sigma"]
  }
  return(fulldata)
}

log_post_odds_data <- logpostodds(data)
write.xlsx(log_post_odds_data, "log_post_odds_r.xlsx", row.names = TRUE)





