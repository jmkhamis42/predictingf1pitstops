# F1 Pit-Stop Prediction: Data and Exploratory Analysis
# Source: https://www.kaggle.com/competitions/playground-series-s6e5/data

# 1. Load the packages and data
# The listed NA strings are recognized if present
library(dplyr)
library(ggplot2)

trainRaw <- read.csv("train.csv", fileEncoding = "UTF-8",
                     na.strings = c("", "NA", "NaN"))
testRaw <- read.csv("test.csv", fileEncoding = "UTF-8",
                    na.strings = c("", "NA", "NaN"))


# 2. Inspect the size, variables, and values
glimpse(trainRaw)
glimpse(testRaw)
summary(trainRaw)

datasetSummary <- data.frame(
  Dataset = c("train.csv", "test.csv"),
  Rows = c(nrow(trainRaw), nrow(testRaw)),
  Columns = c(ncol(trainRaw), ncol(testRaw)),
  MissingCells = c(sum(is.na(trainRaw)), sum(is.na(testRaw))),
  RowsWithMissingValues = c(sum(!complete.cases(trainRaw)),
                            sum(!complete.cases(testRaw)))
)
print(datasetSummary)

categorySummary <- data.frame(
  Dataset = c("train.csv", "test.csv"),
  DriverLabels = c(n_distinct(trainRaw$Driver), n_distinct(testRaw$Driver)),
  RaceLabels = c(n_distinct(trainRaw$Race), n_distinct(testRaw$Race)),
  TyreCompounds = c(n_distinct(trainRaw$Compound),
                    n_distinct(testRaw$Compound))
)
print(categorySummary)
print(trainRaw |> count(Compound))
print(trainRaw |> count(Year))
print(trainRaw |> count(PitNextLap))


# 3. Check missing data and duplicates 
# Check duplicates without id as well
duplicateSummary <- data.frame(
  Dataset = c("train.csv", "test.csv"),
  RepeatedIds = c(nrow(trainRaw) - n_distinct(trainRaw$id),
                  nrow(testRaw) - n_distinct(testRaw$id)),
  RepeatedRowsWithoutId = c(
    nrow(trainRaw) - nrow(trainRaw |> select(-id) |> distinct()),
    nrow(testRaw) - nrow(testRaw |> select(-id) |> distinct())
  ),
  BlankCategoricalCells = c(
    sum(trainRaw$Driver == "", na.rm = TRUE) +
      sum(trainRaw$Compound == "", na.rm = TRUE) +
      sum(trainRaw$Race == "", na.rm = TRUE),
    sum(testRaw$Driver == "", na.rm = TRUE) +
      sum(testRaw$Compound == "", na.rm = TRUE) +
      sum(testRaw$Race == "", na.rm = TRUE)
  )
)
print(duplicateSummary)


# 4. Check numeric ranges of original variables to make sure they make sense 
rangeChecks <- data.frame(
  Check = c("Lap number below 1", "Stint below 1", "Negative tyre age",
            "Position outside 1-20", "Nonpositive lap time",
            "Race progress outside 0-1", "PitStop outside 0 or 1"),
  TrainFlagged = c(
    sum(trainRaw$LapNumber < 1, na.rm = TRUE),
    sum(trainRaw$Stint < 1, na.rm = TRUE),
    sum(trainRaw$TyreLife < 0, na.rm = TRUE),
    sum(trainRaw$Position < 1 | trainRaw$Position > 20, na.rm = TRUE),
    sum(trainRaw$LapTime..s. <= 0, na.rm = TRUE),
    sum(trainRaw$RaceProgress < 0 | trainRaw$RaceProgress > 1, na.rm = TRUE),
    sum(!(trainRaw$PitStop == 0 | trainRaw$PitStop == 1), na.rm = TRUE)
  ),
  TestFlagged = c(
    sum(testRaw$LapNumber < 1, na.rm = TRUE),
    sum(testRaw$Stint < 1, na.rm = TRUE),
    sum(testRaw$TyreLife < 0, na.rm = TRUE),
    sum(testRaw$Position < 1 | testRaw$Position > 20, na.rm = TRUE),
    sum(testRaw$LapTime..s. <= 0, na.rm = TRUE),
    sum(testRaw$RaceProgress < 0 | testRaw$RaceProgress > 1, na.rm = TRUE),
    sum(!(testRaw$PitStop == 0 | testRaw$PitStop == 1), na.rm = TRUE)
  )
)
print(rangeChecks)


# 5. Prepare readable columns and categorical variables without removing rows
# read.csv() changes the original heading "LapTime (s)" to LapTime..s, so
# use LapTimeSeconds instead, so later formulas and graphs are easier to read
# Compound, Year, and PitStop represent categories rather than measurements
trainData <- trainRaw |>
  mutate(LapTimeSeconds = LapTime..s.,
         Compound = as.factor(Compound),
         Year = as.factor(Year),
         PitStop = as.factor(PitStop)) |>
  select(-id, -LapTime..s.)

testData <- testRaw |>
  mutate(LapTimeSeconds = LapTime..s.,
         Compound = as.factor(Compound),
         Year = as.factor(Year),
         PitStop = as.factor(PitStop)) |>
  select(-id, -LapTime..s.)

preparationSummary <- data.frame(
  Dataset = c("train.csv", "test.csv"),
  OriginalRows = c(nrow(trainRaw), nrow(testRaw)),
  PreparedRows = c(nrow(trainData), nrow(testData)),
  RemovedRows = c(nrow(trainRaw) - nrow(trainData),
                  nrow(testRaw) - nrow(testData)),
  PredictorColumns = c(ncol(trainData) - 1, ncol(testData))
)
print(preparationSummary)


# 6. Flag unusual observations for review for lapTimeSeconds
# summary() returns Q1 in position 2 and Q3 in position 5
lapTimeSummary <- summary(trainData$LapTimeSeconds)
lapTimeIQR <- IQR(trainData$LapTimeSeconds)
lapTimeLowerFence <- lapTimeSummary[2] - 1.5 * lapTimeIQR
lapTimeUpperFence <- lapTimeSummary[5] + 1.5 * lapTimeIQR

print(lapTimeSummary)
print(lapTimeLowerFence)
print(lapTimeUpperFence)

# Use 60 seconds * 5 = 300 seconds for long lap threshold
# Tyre age greater than the current lap is also a review flag, not proof of error
trainAudit <- trainData |>
  mutate(LapTimeIQRFlag = LapTimeSeconds < lapTimeLowerFence |
                           LapTimeSeconds > lapTimeUpperFence,
         LongLapFlag = LapTimeSeconds > 300,
         TyreAgeReviewFlag = TyreLife > LapNumber,
         PitOutcome = as.factor(ifelse(PitNextLap == 1,
                                       "Pit next lap", "No pit next lap")))

testAudit <- testData |>
  mutate(LapTimeIQRFlag = LapTimeSeconds < lapTimeLowerFence |
                           LapTimeSeconds > lapTimeUpperFence,
         LongLapFlag = LapTimeSeconds > 300,
         TyreAgeReviewFlag = TyreLife > LapNumber)

outlierSummary <- data.frame(
  Dataset = c("train.csv", "test.csv"),
  LapTimeIQRFlags = c(sum(trainAudit$LapTimeIQRFlag),
                      sum(testAudit$LapTimeIQRFlag)),
  LapsAbove300Seconds = c(sum(trainAudit$LongLapFlag),
                          sum(testAudit$LongLapFlag)),
  MaximumLapTimeSeconds = c(max(trainData$LapTimeSeconds),
                            max(testData$LapTimeSeconds)),
  TyreAgeGreaterThanLap = c(sum(trainAudit$TyreAgeReviewFlag),
                            sum(testAudit$TyreAgeReviewFlag))
)
print(outlierSummary)

# Make a compact table of the most extreme training lap times for review
longLapRecords <- trainAudit |>
  filter(LongLapFlag) |>
  select(Race, Year, LapNumber, PitStop, TyreLife, LapTimeSeconds, PitNextLap) |>
  arrange(LapTimeSeconds)
print(longLapRecords)


# 7. Summarize  main relationships 
targetSummary <- trainAudit |>
  group_by(PitOutcome) |>
  summarize(Laps = n(), .groups = "drop") |>
  mutate(Percent = 100 * Laps / sum(Laps))
print(targetSummary)

tyreAgeSummary <- trainAudit |>
  group_by(PitOutcome) |>
  summarize(Laps = n(),
            MeanTyreAge = mean(TyreLife),
            MedianTyreAge = median(TyreLife),
            .groups = "drop")
print(tyreAgeSummary)

compoundSummary <- trainData |>
  group_by(Compound) |>
  summarize(Laps = n(),
            PitLaps = sum(PitNextLap),
            PitRatePercent = 100 * mean(PitNextLap),
            .groups = "drop")
print(compoundSummary)

# Find unusual patterns in pit stops based on year
yearSummary <- trainData |>
  group_by(Year) |>
  summarize(Laps = n(),
            PitLaps = sum(PitNextLap),
            PitRatePercent = 100 * mean(PitNextLap),
            .groups = "drop")
print(yearSummary)


# 8. Exploratoyr Graphs
# Use counts for class balance and percentages for comparisons between groups
classBalancePlot <- targetSummary |>
  ggplot(aes(x = PitOutcome, y = Laps / 1000)) +
  geom_bar(stat = "identity", fill = "#386CB0") +
  labs(title = "Number of Training Laps Followed By Pit Stop",
       x = "Next-lap Outcome", y = "Training Laps (thousands)") +
  theme(plot.title = element_text(hjust=0.5))
print(classBalancePlot)

# Boxplot medians and spreads are readable without plotting 439,140 individual laps
tyreAgePlot <- trainAudit |>
  ggplot(aes(x = PitOutcome, y = TyreLife)) +
  geom_boxplot(fill = "#80B1D3", outlier.alpha = 0.12) +
  labs(title = "Laps Followed by a Pit Stop",
       x = "Next-Lap Outcome", y = "Tyre Age (Laps)") +
  theme(plot.title = element_text(hjust=0.5))
print(tyreAgePlot)

# Compare within-compound pit percentages rather than raw group counts
compoundPitRatePlot <- compoundSummary |>
  ggplot(aes(x = Compound, y = PitRatePercent)) +
  geom_bar(stat = "identity", fill = "#386CB0") +
  labs(title = "Next-Lap Pit Rate by Tire Compound",
       x = "Tire Compound", y = "Laps Followed by a Pit Stop (%)") +
  theme(plot.title = element_text(hjust=0.5))
print(compoundPitRatePlot)

# Display the unusual 2023 label distribution
yearPitRatePlot <- yearSummary |>
  ggplot(aes(x = Year, y = PitRatePercent)) +
  geom_bar(stat = "identity", fill = "#386CB0") +
  labs(title = "Next-Lap Pit Rate by Year",
       x = "Year", y = "Laps Followed by a Pit Stop (%)") +
  theme(plot.title = element_text(hjust=0.5))
print(yearPitRatePlot)

# Zoom the histogram to 150 seconds so rare extreme values do not compress the plot
lapTimePlotData <- trainData |> filter(LapTimeSeconds <= 150)
lapTimePlot <- lapTimePlotData |>
  ggplot(aes(x = LapTimeSeconds)) +
  geom_histogram(binwidth = 2, fill = "#80B1D3", color = "white") +
  labs(title = "Training Lap-Time Distribution",
       subtitle = paste0(nrow(trainData) - nrow(lapTimePlotData),
                         " longer outlier laps are not shown"),
       x = "Lap Time (Seconds)", y = "Training Laps") +
  theme(plot.title = element_text(hjust=0.5),plot.subtitle = element_text(hjust = 0.5))
print(lapTimePlot)