import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt

right_bat = pd.read_csv('right_bat.csv', header = 1)
left_bat = pd.read_csv('left_bat.csv', header = 1)

right_bat['handedness'] = 0  
left_bat['handedness'] = 1   

data = pd.concat([right_bat, left_bat], ignore_index=True)
features = ['Height', 'Weight']
X = data[features]
print(X)
#normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled)
lin_reg = LinearRegression()
lin_reg.fit(X_scaled, data['bat_speed'])
data['bat_speed_residual'] = data['bat_speed'] - lin_reg.predict(X_scaled)
print(data['bat_speed_residual'])
# Compare the residuals of bat_speed between left and right-handed batters
left_handed = data[data['handedness'] == 1]
right_handed = data[data['handedness'] == 0]

# Print mean residual bat speeds
print("Left-handed batters mean residual bat speed:\n", left_handed['bat_speed_residual'].mean())
print("Right-handed batters mean residual bat speed:\n", right_handed['bat_speed_residual'].mean())

# Perform a t-test on the residuals
t_stat, p_val = ttest_ind(left_handed['bat_speed_residual'], right_handed['bat_speed_residual'], equal_var=False)
print(f"Residual Bat Speed: T-statistic = {t_stat}, P-value = {p_val}")

# Plot the residuals
plt.figure(figsize=(12, 6))
sns.boxplot(x='handedness', y='bat_speed_residual', data=data)
plt.title('Residual Bat Speed for Left vs. Right-Handed Batters (Adjusted for Height and Weight)')
plt.xlabel('Handedness (0: Right, 1: Left)')
plt.ylabel('Residual Bat Speed')
plt.savefig('bat_speed_residual_comparison.png')
plt.show()

'''
# Compare the means of bat_speed between left and right-handed batters
left_handed = data[data['handedness'] == 1]
right_handed = data[data['handedness'] == 0]

print("Left-handed batters mean performance:\n", left_handed['bat_speed'].mean())
print("Right-handed batters mean performance:\n", right_handed['bat_speed'].mean())

# Perform a t-test to compare bat_speed between the two groups
t_stat, p_val = ttest_ind(left_handed['bat_speed'], right_handed['bat_speed'], equal_var=False)
print(f"Bat Speed: T-statistic = {t_stat}, P-value = {p_val}")

# plot
plt.figure(figsize=(12, 6))
sns.boxplot(x='handedness', y='bat_speed', data=data)
plt.title('Bat Speed for Left vs. Right-Handed Batters')
plt.xlabel('Handedness (0: Right, 1: Left)')
plt.ylabel('Bat Speed')
plt.savefig('bat_speed_comparison.png')
plt.show()

'''
