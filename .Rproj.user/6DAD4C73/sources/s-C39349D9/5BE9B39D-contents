---
title: "PA1_Template"
author: "Bola Lawal"
date: '2022-04-21'
output:
  html_document: default
---
```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```
###Activity Monitoring Data
##Install the packages
```{r}
destination <- getwd()
r = getOption("repos")
r["CRAN"] = "http://cran.us.r-project.org"
options(repos = r)
install.packages("data.table")
install.packages("ggplot2")
install.packages("data.table")
install.packages("ggplot2")
```

##Load the libraries
```{r}
library("data.table")
library("ggplot2")
```

##Load the Data
```{r}
fileUrl <- "https://d396qusza40orc.cloudfront.net/repdata%2Fdata%2Factivity.zip"
download.file(fileUrl, destfile = paste0(getwd(), '/repdata%2Fdata%2Factivity.zip'), method = "curl")
unzip("repdata%2Fdata%2Factivity.zip",exdir = "destination")
```


#Read the files
```{r}
activity_data <- data.table::fread(input = "destination/activity.csv")
Number_of_Observations <- nrow(activity_data)
Number_of_Observations
```


##Questions:
##1. Calculate the mean total number of steps taken per day
#1.1. Calculate the total number of steps taken per day
##Load the libraries
```{r}
Total_Steps <- activity_data[, c(lapply(.SD, sum, na.rm = FALSE)), .SDcols = c("steps"), by = .(date)] 
head(Total_Steps, 10)
```

#1.2. Histogram of the total number of steps taken per day
```{r}
ggplot(Total_Steps, aes(x = steps)) +
  geom_histogram(fill = "green", binwidth = 500) +
  labs(title = "Steps per Day", x = "# of Steps", y = "Frequency")
```

#1.3. Mean and median of the total number of steps taken per day
```{r}
Total_Steps[, .(Mean_Steps = mean(steps, na.rm = TRUE), Median_Steps = median(steps, na.rm = TRUE))]
```

##2. What is the average daily activity pattern?
#2.1. Time series plot
```{r}
Interval <- activity_data[, c(lapply(.SD, mean, na.rm = TRUE)), .SDcols = c("steps"), by = .(interval)] 
ggplot(Interval, aes(x = interval , y = steps)) + geom_line(color="blue", size=1) + labs(title = "Average Daily Steps", x = "Interval", y = "Average Steps per day")
```

#2.2. 5-minute interval, on average across all the days in the dataset with the maximum number of steps
```{r}
Interval[steps == max(steps), .(max_interval = interval)]
```

##3. Inputting missing values
#3.1 Calculate and report the total number of missing values in the dataset
```{r}
nrow(activity_data[is.na(steps),])
```

#3.2. Devise a strategy for filling in all of the missing values in the dataset.
#Strategy used is to find the mean of the existing values.
```{r}
activity_data[is.na(steps), "steps"] <- activity_data[, c(lapply(.SD, mean, na.rm = TRUE)), .SDcols = c("steps")]
```

#3.3. Create a new dataset that is equal to the original dataset but with the missing data filled in.
```{r}
data.table::fwrite(x = activity_data, file = "destination/No_NULL.csv", quote = FALSE)
```


##4. General Questions:
#Make a histogram of the total number of steps taken each day.
#Calculate and report the mean and median total number of steps taken per day. 
#Do these values differ from the estimates from the first part of the assignment? 
#What is the impact of inputing missing data on the estimates of the total daily number of steps?

#4.1 Total number of steps taken per day
```{r}
Total_Steps <- activity_data[, c(lapply(.SD, sum)), .SDcols = c("steps"), by = .(date)] 
```

#4.2 Mean and median total number of steps taken per day
```{r}
Total_Steps[, .(Mean_Steps = mean(steps), Median_Steps = median(steps))]
ggplot(Total_Steps, aes(x = steps)) + geom_histogram(fill = "green",binwidth = 500) + labs(title = "Daily Steps", x = "Steps", y = "Frequency")
```

#4.3
#   Mean_Steps      Median_Steps
#1:   10766.19        10765
#2:   10751.74        10656


##5.Are there differences in activity patterns between weekdays and weekends?
```{r}
activity_data_week <- data.table::fread(input = "destination/activity.csv")
activity_data_week[, date := as.POSIXct(date, format = "%Y-%m-%d")]
activity_data_week[, `Day of Week`:= weekdays(x = date)]
activity_data_week[grepl(pattern = "Monday|Tuesday|Wednesday|Thursday|Friday", x = `Day of Week`), "Weekday/Weekend"] <- "Weekday"
activity_data_week[grepl(pattern = "Saturday|Sunday", x = `Day of Week`), "Weekday/Weekend"] <- "Weekend"
activity_data_week[, `Weekday/Weekend` := as.factor(`Weekday/Weekend`)]
```

#5.1 A panel plot containing a time series plot (i.e. type = "𝚕") of the 5-minute interval (x-axis) and the average number of steps taken, averaged across all weekday days or weekend days (y-axis).
```{r}
activity_data_week[is.na(steps), "steps"] <- activity_data_week[, c(lapply(.SD, median, na.rm = TRUE)), .SDcols = c("steps")]
Interval <- activity_data_week[, c(lapply(.SD, mean, na.rm = TRUE)), .SDcols = c("steps"), by = .(interval, `Weekday/Weekend`)] 
ggplot(Interval , aes(x = interval , y = steps, color=`Weekday/Weekend`)) + geom_line() + labs(title = "Average Daily Steps by Day Type", x = "Interval", y = "# of Steps") + facet_wrap(~`Weekday/Weekend` , ncol = 1, nrow=2)
```
