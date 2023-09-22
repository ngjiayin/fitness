library(dplyr)
library(lubridate)
df = read.csv("HKQuantityTypeIdentifierStepCount.csv", skip=1, header=TRUE, sep=";")

# Q1: Which day of the week in 2023 did I walk the most?
day_2023_most = df %>%
  transmute(date=ymd(substring(creationdate, 1, 10)), value) %>%
  mutate(day = wday(date, label=TRUE)) %>%
  filter(year(date) == 2023) %>%
  group_by(day) %>%
  summarise(day_steps = sum(value)) %>%
  arrange(desc(day_steps))
day_2023_most
# A1: Day with most steps: Friday
