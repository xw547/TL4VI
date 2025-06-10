# Define the named numeric vector
data_values <- c(
  season = 166.34,
  yr = 698.25,
  month = 83.66,
  holiday = 540.32,
  Weekday = 352.92,
  workingday = 788.71,
  weathersit = 408.32,
  Temp = 139.73,
  atemp = 146.84,
  hum = 201.07,
  wdspd = 97.70,
  causal = 345.95,
  regis = 270.67
)

# Convert to a data frame for ggplot
library(ggplot2)
data_df <- data.frame(
  Feature = names(data_values),
  TL_Estimate = data_values
)

# Create the bar plot
real_plot = ggplot(data_df, aes(x = reorder(Feature, TL_Estimate), y = TL_Estimate)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +  # Flip the coordinates for better readability
  labs(x = "Feature", y = "Conditional Permuation Importance", title = "Bikesharing Variable Importance using TL") +
  theme_minimal()
#center title
real_plot <- real_plot + theme(plot.title = element_text(hjust = 0.5))
# use larger fonts for title


ggsave("~/Working/Ning/Giles1/Code/Python_Implementation/plots/real_plot.png", 
       real_plot, width = 6.27, height = 6.27, units = "in", dpi = 500)

