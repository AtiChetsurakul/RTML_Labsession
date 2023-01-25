import pickle

loss = [1.3887, 1.0712, 0.8531, 0.8357,
        0.7797, .6904, .681, .6844, .6919, .6404, .6690, 0.7100, 0.6933, 0.6634]
acc = [0.4787, 0.6304, 0.7061, 0.7148,
       0.7368, .7646, .7742, .7713, .7759, .7813, 0.7684, .7867, 0.7914]
# Training complete in 35m 44s
# Best val Acc: 0.791400
with open('/root/models/lab2alexacc.atikeep', 'wb') as handle:
    pickle.dump((loss, acc), handle)

loss = [1.4758, 1.04, .91, .854, .752, .7048, .6268, .6873, .6316,
        0.7129, .6234, 0.6463, 0.7032, 0.6972, 0.6623]
acc = [.4606, 0.63, .68, .7, .74, .76, .7894, .77, .7914,
       0.7616, .791, .798, 0.7873, 0.7980, 0.7998]

# Training complete in 36m 49s
# best val  # Acc: 0.799800
with open('/root/models/lab2alexacc_lrn.atikeep', 'wb') as handle:
    pickle.dump((loss, acc), handle)
