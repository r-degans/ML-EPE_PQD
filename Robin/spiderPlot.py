import numpy as np
import matplotlib.pyplot as plt

# ------------------------
# 1. Data
# ------------------------
labels = ['Swell', 'Transient', 'Swell harmonics', 'Sag harmonics', 
          'Normal', 'Sag', 'Interruption', 'Harmonics', 'Flicker']
num_vars = len(labels)

# Example data for 3 models

baseCorr = "Task1-F1"
baseMan = "Task1-manual-F1"
noiseList = ["noiseless", "50db", "40db", "30db", "20db"]

# model1 = [0.6813186813186813,0.8813559322033898,0.6896551724137931,0.5909090909090909,0.5517241379310345,0.6823529411764706,0.9565217391304348,0.5365853658536586,0.9538461538461539,] 
# model2 = [0.9850746268656716,1.0,0.9041095890410958,0.8354430379746836,0.9850746268656716,1.0,0.9850746268656716,0.9850746268656716,0.9850746268656716,] 
# model3 = [0.9166666666666666,0.9696969696969697,0.673469387755102,0.8767123287671232,0.7096774193548387,0.825,0.9850746268656716,0.9041095890410958,0.9538461538461539,] 

# model4 = [0.6105263157894737,0.825,0.6666666666666666,0.5769230769230769,0.6153846153846154,0.64,1.0,0.775,0.4444444444444444,] 
# model5 = [0.9850746268656716,1.0,0.8823529411764706,0.7804878048780488,0.9295774647887324,0.9696969696969697,1.0,1.0,0.7586206896551724,] 
# model6 = [0.3127962085308057,0.30985915492957744,0.29464285714285715,0.29464285714285715,0.6666666666666666,0.28193832599118945,1.0,0.7536231884057971,0.30275229357798167,] 

for noise in noiseList:
    corrFile = str(baseCorr + noise + ".txt")
    manFile = str(baseMan + noise + ".txt")

    with open(corrFile) as f:
        exec(f.read())

    with open(manFile) as f:
        exec(f.read())
    
    cor = np.average(model1) + np.average(model2) + np.average(model3)
    man = np.average(model4) + np.average(model5) + np.average(model6)
    print(noise)
    print(cor)
    print(man)
    # print(str("Linear corr: " + str(np.average(model1))))
    # print(str("Linear L1/L2: " + str(np.average(model2))))
    # print(str("SVM corr: " + str(np.average(model3))))
    # print(str("Linear man: " + str(np.average(model4))))
    # print(str("Linear L1/l2 man: " + str(np.average(model5))))
    # print(str("SVM man: " + str(np.average(model6))))





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
    ax.set_title(str("F1 scores, " + noise), size=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.set_ylim(0, 1)  # forces radial axis from 0 to 1
    ax.legend(fontsize=12)  # increase number as needed


    plt.tight_layout()
    plt.savefig(str(noise) + "F1.png")
    del model1, model2, model3, model4, model5, model6
