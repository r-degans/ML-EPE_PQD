import numpy as np
import matplotlib.pyplot as plt

# ------------------------
# 1. Data
# ------------------------
labels = ['Swell', 'Transient', 'Swell harmonics', 'Sag harmonics', 
          'Normal', 'Sag', 'Interruption', 'Harmonics', 'Flicker']
num_vars = len(labels)

# Example data for 3 models
model1 = [0.6741573033707865,0.584070796460177,0.6881720430107527,0.6666666666666666,0.5409836065573771,0.625,0.9705882352941176,0.5076923076923077,0.7272727272727273,] 
model2 = [0.9705882352941176,1.0,0.9041095890410958,0.825,1.0,0.927536231884058,0.9705882352941176,0.9850746268656716,0.868421052631579,] 
model3 = [0.5739130434782609,0.5,0.5517241379310345,0.6336633663366337,0.4583333333333333,0.6326530612244898,0.9552238805970149,0.3875,0.7272727272727273,] 

model4 = [0.6741573033707865,0.584070796460177,0.6881720430107527,0.6666666666666666,0.5409836065573771,0.625,0.9705882352941176,0.5076923076923077,0.7272727272727273,] 
model5 = [0.9705882352941176,1.0,0.9041095890410958,0.825,1.0,0.927536231884058,0.9705882352941176,0.9850746268656716,0.868421052631579,] 
model6 = [0.5739130434782609,0.5,0.5517241379310345,0.6336633663366337,0.4583333333333333,0.6326530612244898,0.9552238805970149,0.3875,0.7272727272727273,] 

# ------------------------
# 2. Prepare angles
# ------------------------
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # close the circle

# Close the data loops
model1 += model1[:1]
model2 += model2[:1]
model3 += model3[:1]

model4 += model4[:1]
model5 += model5[:1]
model6 += model6[:1]

# ------------------------
# 3. Create radar plot
# ------------------------
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

# Plot each model
ax.plot(angles, model1, label='Linear - corr', marker='o')
ax.fill(angles, model1, alpha=0.1)

ax.plot(angles, model2, label='Linear L1 - corr', marker='o')
ax.fill(angles, model2, alpha=0.1)

ax.plot(angles, model3, label='SVM - corr', marker='o')
ax.fill(angles, model3, alpha=0.1)

#######

ax.plot(angles, model4, label='Linear - man', marker='o')
ax.fill(angles, model4, alpha=0.1)

ax.plot(angles, model5, label='Linear L1 - man', marker='o')
ax.fill(angles, model5, alpha=0.1)

ax.plot(angles, model6, label='SVM - man', marker='o')
ax.fill(angles, model6, alpha=0.1)

# ------------------------
# 4. Style the plot
# ------------------------
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=10)
ax.set_yticklabels([])  # optional: hide radial labels
ax.set_title("Model performance, 40dB", size=14, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax.set_ylim(0, 1)  # forces radial axis from 0 to 1

plt.tight_layout()
plt.show()
